//! A Hugging Face checkpoint as the GGUF container rllm's loaders read: the
//! metadata keys and tensor names llama.cpp's `convert_hf_to_gguf.py` would
//! write (see `research_notes/safetensors/conversion_rules.md`), built in
//! memory. Matrices whose bytes are already what GGUF stores (F32, F16, BF16)
//! are views into the mapped shards; only tensors that change are copied.
//!
//! Supported so far: Qwen3 (`Qwen3ForCausalLM`). The other families follow
//! the plan in `docs/safetensors.md`.

use std::collections::HashMap;

use rllm_core::DType;
use rllm_gguf::{Backing, Gguf, TensorInfo, Value};
use serde_json::Value as Json;

use crate::{Checkpoint, Dtype, Error, Result};

/// GGUF token types (llama.cpp's `TokenType`).
const NORMAL: i32 = 1;
const CONTROL: i32 = 3;
const USER_DEFINED: i32 = 4;
const UNUSED: i32 = 5;
const BYTE: i32 = 6;

fn bad(what: impl Into<String>) -> Error {
    Error::Malformed("checkpoint".into(), what.into())
}

/// `config.json` as transformers sees it for the text model: `text_config`
/// merged over the root.
fn text_config(c: &Checkpoint) -> Result<serde_json::Map<String, Json>> {
    let cfg = c.json("config.json")?.ok_or_else(|| bad(format!("no config.json in {}", c.dir.display())))?;
    let mut root = cfg.as_object().cloned().ok_or_else(|| bad("config.json is not an object"))?;
    if let Some(Json::Object(t)) = root.get("text_config").cloned() {
        root.extend(t);
    }
    Ok(root)
}

/// The first of `keys` present in the config, as an integer.
fn int(cfg: &serde_json::Map<String, Json>, keys: &[&str]) -> Option<u64> {
    keys.iter().find_map(|k| cfg.get(*k).and_then(Json::as_u64))
}

fn float(cfg: &serde_json::Map<String, Json>, keys: &[&str]) -> Option<f64> {
    keys.iter().find_map(|k| cfg.get(*k).and_then(Json::as_f64))
}

/// How to convert.
#[derive(Debug, Clone, Copy, Default)]
pub struct Options {
    /// Quantize the weight matrices as they load: `DType::Q8_0`, or
    /// `DType::Q4K` for llama.cpp's Q4_K_M mix (Q4_K, with Q6_K for the
    /// output and for `attn_v` / `ffn_down` on the layers it gives more
    /// bits). Norms, routers and other vectors stay F32.
    pub quantize: Option<DType>,
}

impl Options {
    /// From `RLLM_QUANTIZE` (`q8_0`), for commands without a flag for it.
    pub fn from_env() -> Result<Self> {
        match std::env::var("RLLM_QUANTIZE").ok().as_deref().map(str::to_ascii_lowercase).as_deref() {
            None | Some("") | Some("none") => Ok(Self::default()),
            Some(other) => Ok(Self { quantize: Some(parse_quantize(other).map_err(|e| bad(format!("RLLM_QUANTIZE: {e}")))?) }),
        }
    }
}

/// `q8_0` or `q4_k_m` (as `--quantize` takes them).
pub fn parse_quantize(s: &str) -> std::result::Result<DType, String> {
    match s.to_ascii_lowercase().as_str() {
        "q8_0" => Ok(DType::Q8_0),
        "q4_k_m" | "q4_k" => Ok(DType::Q4K),
        other => Err(format!("{other}: supported are q8_0 and q4_k_m")),
    }
}

/// llama.cpp's `use_more_bits`: the first and last eighth of the layers and
/// every third one between get more bits in its *_M mixes.
fn use_more_bits(i: usize, n: usize) -> bool {
    i < n / 8 || i >= 7 * n / 8 || (i as isize - (n / 8) as isize).rem_euclid(3) == 2
}

/// What the Q4_K_M mix needs to know about the model.
#[derive(Debug, Clone, Default)]
struct Plan {
    /// The blocks with attention (`attn_v`), in order.
    attn_layers: Vec<usize>,
    /// All blocks (for `ffn_down`).
    n_blocks: usize,
    /// The output matrix is the token embedding.
    tied: bool,
}

thread_local! {
    static OPTS: std::cell::Cell<Options> = const { std::cell::Cell::new(Options { quantize: None }) };
}

/// Convert a checkpoint into a GGUF container (see the module docs).
pub fn to_gguf(c: Checkpoint) -> Result<Gguf> {
    to_gguf_with(c, Options::default())
}

/// [`to_gguf`] with options.
pub fn to_gguf_with(c: Checkpoint, opts: Options) -> Result<Gguf> {
    if opts.quantize.is_some_and(|q| !matches!(q, DType::Q8_0 | DType::Q4K)) {
        return Err(bad("only Q8_0 and Q4_K_M quantization at load are supported"));
    }
    OPTS.with(|o| o.set(opts));
    let r = convert(c);
    OPTS.with(|o| o.set(Options::default()));
    r
}

fn convert(c: Checkpoint) -> Result<Gguf> {
    let cfg = text_config(&c)?;
    let arch_class = cfg
        .get("architectures")
        .and_then(Json::as_array)
        .and_then(|a| a.first())
        .and_then(Json::as_str)
        .ok_or_else(|| bad("config.json has no architectures"))?
        .to_string();
    match arch_class.as_str() {
        "Qwen3ForCausalLM" | "Qwen3Model" => qwen3(c, &cfg),
        "LlamaForCausalLM" | "MistralForCausalLM" | "LlamaModel" => llama(c, &cfg),
        "Gemma4ForConditionalGeneration" | "Gemma4ForCausalLM" => gemma4(c, &cfg),
        "Qwen3_5ForConditionalGeneration" | "Qwen3_5ForCausalLM" | "Qwen3_5TextModel" if c.dir.join("joint_head_config.json").is_file() => {
            let head = c.json("joint_head_config.json")?.ok_or_else(|| bad("joint_head_config.json"))?;
            qwen35_arch(c, &cfg, false, Some(&head))
        }
        "Qwen3_5ForConditionalGeneration" | "Qwen3_5ForCausalLM" | "Qwen3_5TextModel" => qwen35(c, &cfg, false),
        "Qwen3_5MoeForConditionalGeneration" | "Qwen3_5MoeForCausalLM" => qwen35(c, &cfg, true),
        other => Err(bad(format!(
            "architecture {other} can't be loaded from safetensors yet (supported: Qwen3, Qwen3.5/3.6 dense and MoE, Clef, Llama 3, Gemma 4 dense; see docs/safetensors.md)"
        ))),
    }
}

/// Tensors being collected: views into shards, and bytes built here.
struct Builder {
    tensors: Vec<TensorInfo>,
    owned: Vec<u8>,
    n_shards: usize,
    /// Weight matrices are quantized to this as they are added.
    quantize: Option<DType>,
    plan: Plan,
}

impl Builder {
    fn new(n_shards: usize) -> Self {
        Self { tensors: Vec::new(), owned: Vec::new(), n_shards, quantize: OPTS.with(|o| o.get().quantize), plan: Plan::default() }
    }

    /// The layers for the Q4_K_M mix (set before adding tensors).
    fn plan(&mut self, attn_layers: Vec<usize>, n_blocks: usize, tied: bool) {
        self.plan = Plan { attn_layers, n_blocks, tied };
    }

    /// The type a matrix gets: llama.cpp's choice for its Q4_K_M files.
    fn q4_k_m_type(&self, name: &str, row: usize) -> DType {
        let layer = name.strip_prefix("blk.").and_then(|r| r.split('.').next()).and_then(|i| i.parse::<usize>().ok());
        // The decision head (read once, on the CPU) and the token types keep 8 bits.
        let mut t = if name.starts_with("dec.") || name.starts_with("decision.") || name.starts_with("token_types") {
            DType::Q8_0
        } else if name == "output.weight" || (self.plan.tied && name == "token_embd.weight") {
            DType::Q6K
        } else if name.ends_with("attn_v.weight") {
            let rank = layer.and_then(|l| self.plan.attn_layers.iter().position(|&a| a == l));
            if rank.is_some_and(|r| use_more_bits(r, self.plan.attn_layers.len())) { DType::Q6K } else { DType::Q4K }
        } else if name.contains("ffn_down") {
            if layer.is_some_and(|l| use_more_bits(l, self.plan.n_blocks)) { DType::Q6K } else { DType::Q4K }
        } else {
            DType::Q4K
        };
        // Rows that aren't whole super-blocks fall back to Q8_0 (as llama.cpp does).
        if matches!(t, DType::Q4K | DType::Q6K) && !row.is_multiple_of(256) {
            t = DType::Q8_0;
        }
        t
    }

    /// A weight matrix (2-D or more) in `dtype` with GGUF `dims`: kept, or
    /// quantized when asked to (rows of whole blocks only).
    fn matrix(&mut self, name: &str, dims: Vec<u64>, dtype: DType, bytes: &[u8]) {
        let row = dims.first().copied().unwrap_or(1) as usize;
        // Routers stay as they are (llama.cpp never quantizes them).
        let target = match self.quantize {
            _ if name.contains("ffn_gate_inp") || name.contains("ssm_conv1d") || name.ends_with("_norm.weight") => None,
            Some(DType::Q4K) => Some(self.q4_k_m_type(name, row)),
            q => q,
        };
        match target {
            Some(target) if dims.len() >= 2 && row.is_multiple_of(target.block_size()) && matches!(dtype, DType::F32 | DType::F16 | DType::BF16) => {
                let rows = bytes.len() / dtype.row_bytes(row);
                let (rb, qb) = (dtype.row_bytes(row), target.row_bytes(row));
                // Rows are independent: quantize them on all cores.
                use rayon::prelude::*;
                let mut out = vec![0u8; rows * qb];
                out.par_chunks_mut(qb * 64).zip(bytes.par_chunks(rb * 64)).for_each(|(o, src)| {
                    let (mut f, mut q) = (vec![0f32; row], Vec::with_capacity(qb));
                    for (o, src) in o.chunks_exact_mut(qb).zip(src.chunks_exact(rb)) {
                        rllm_core::quant::dequantize_row(dtype, src, &mut f);
                        q.clear();
                        match target {
                            DType::Q4K => rllm_core::quant::quantize_q4_k(&f, &mut q),
                            DType::Q6K => rllm_core::quant::quantize_q6_k(&f, &mut q),
                            _ => rllm_core::quant::quantize_q8_0(&f, &mut q),
                        }
                        o.copy_from_slice(&q);
                    }
                });
                self.owned_tensor(name, dims, target, &out);
            }
            _ => self.owned_tensor(name, dims, dtype, bytes),
        }
    }

    /// GGUF dims: the shape reversed (innermost first).
    fn dims(shape: &[usize]) -> Vec<u64> {
        shape.iter().rev().map(|&d| d as u64).collect()
    }

    /// The tensor's bytes as they are (F32/F16/BF16): a view into its shard,
    /// or quantized when asked to.
    fn view(&mut self, name: &str, t: &crate::Tensor, data: &[u8]) -> Result<()> {
        let dtype = t.dtype.rllm().ok_or_else(|| bad(format!("{}: {:?} weights aren't supported yet", t.name, t.dtype)))?;
        if self.quantize.is_some() && t.shape.len() >= 2 {
            self.matrix(name, Self::dims(&t.shape), dtype, data);
            return Ok(());
        }
        self.tensors.push(TensorInfo { name: name.into(), dims: Self::dims(&t.shape), dtype, offset: t.offset, size: t.size, buffer: t.file });
        Ok(())
    }

    /// The tensor widened to F32 (norms and other vectors, as llama.cpp
    /// stores them).
    fn f32(&mut self, name: &str, t: &crate::Tensor, data: &[u8]) -> Result<()> {
        let vals = to_f32(t, data)?;
        self.f32_vals(name, Self::dims(&t.shape), &vals);
        Ok(())
    }

    /// F32 values built here, with GGUF `dims`.
    fn f32_vals(&mut self, name: &str, dims: Vec<u64>, vals: &[f32]) {
        let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.owned_tensor(name, dims, DType::F32, &bytes);
    }

    /// Bytes built here (32-byte aligned in the owned buffer).
    fn owned_tensor(&mut self, name: &str, dims: Vec<u64>, dtype: DType, bytes: &[u8]) {
        let at = self.owned.len().next_multiple_of(32);
        self.owned.resize(at, 0);
        self.owned.extend_from_slice(bytes);
        self.tensors.push(TensorInfo { name: name.into(), dims, dtype, offset: at, size: bytes.len(), buffer: self.n_shards });
    }
}

fn to_f32(t: &crate::Tensor, data: &[u8]) -> Result<Vec<f32>> {
    Ok(match t.dtype {
        Dtype::F32 => data.as_chunks::<4>().0.iter().map(|&b| f32::from_le_bytes(b)).collect(),
        Dtype::BF16 => data.as_chunks::<2>().0.iter().map(|&b| f32::from_bits((u16::from_le_bytes(b) as u32) << 16)).collect(),
        Dtype::F16 => data.as_chunks::<2>().0.iter().map(|&b| f16_to_f32(u16::from_le_bytes(b))).collect(),
        other => return Err(bad(format!("{}: {other:?} can't be converted to F32", t.name))),
    })
}

/// llama.cpp's linear-attention value-head reorder on raw bytes: `bytes`
/// holds `nk * r` blocks of `block` bytes, grouped by key head (block
/// `k * r + j`); they come out tiled (block `j * nk + k`), the order the
/// GDN kernels read (value head h uses key head h % nk).
fn tile_heads(bytes: &mut [u8], block: usize, nk: usize, r: usize) {
    debug_assert_eq!(bytes.len(), block * nk * r);
    let old = bytes.to_vec();
    for k in 0..nk {
        for j in 0..r {
            let (from, to) = ((k * r + j) * block, (j * nk + k) * block);
            bytes[to..to + block].copy_from_slice(&old[from..from + block]);
        }
    }
}

fn f16_to_f32(h: u16) -> f32 {
    let (s, e, m) = ((h >> 15) as u32, ((h >> 10) & 0x1f) as u32, (h & 0x3ff) as u32);
    let bits = match (e, m) {
        (0, 0) => s << 31,
        (0, _) => {
            // Subnormal: normalize.
            let mut e = 127 - 15 + 1;
            let mut m = m;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            (s << 31) | (e << 23) | ((m & 0x3ff) << 13)
        }
        (0x1f, _) => (s << 31) | (0xff << 23) | (m << 13),
        _ => (s << 31) | ((e + 127 - 15) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

/// Qwen3 dense (`qwen3`): no tensor transforms; norms to F32.
fn qwen3(c: Checkpoint, cfg: &serde_json::Map<String, Json>) -> Result<Gguf> {
    let need = |keys: &[&str]| int(cfg, keys).ok_or_else(|| bad(format!("config.json: no {}", keys[0])));
    let n_layers = need(&["num_hidden_layers", "n_layers", "n_layer", "num_layers"])?;
    let heads = need(&["num_attention_heads", "n_head", "n_heads"])?;
    let mut meta: Vec<(String, Value)> = vec![
        ("general.architecture".into(), Value::String("qwen3".into())),
        ("general.name".into(), Value::String(c.dir.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default())),
        ("qwen3.block_count".into(), Value::U32(n_layers as u32)),
        ("qwen3.context_length".into(), Value::U32(int(cfg, &["max_position_embeddings", "n_ctx", "n_positions"]).unwrap_or(40960) as u32)),
        ("qwen3.embedding_length".into(), Value::U32(need(&["hidden_size", "n_embd", "dim"])? as u32)),
        ("qwen3.feed_forward_length".into(), Value::U32(need(&["intermediate_size", "n_inner", "hidden_dim"])? as u32)),
        ("qwen3.attention.head_count".into(), Value::U32(heads as u32)),
        ("qwen3.attention.head_count_kv".into(), Value::U32(int(cfg, &["num_key_value_heads", "n_kv_heads"]).unwrap_or(heads) as u32)),
        // Qwen3Config's defaults (transformers), when the file leaves them out.
        ("qwen3.attention.layer_norm_rms_epsilon".into(), Value::F32(float(cfg, &["rms_norm_eps", "norm_eps"]).unwrap_or(1e-6) as f32)),
    ];
    // rope_theta: top level, or in rope_parameters (transformers v5).
    let theta = float(cfg, &["rope_theta"])
        .or_else(|| cfg.get("rope_parameters").and_then(|r| r.get("rope_theta")).and_then(Json::as_f64))
        .unwrap_or(10_000.0);
    meta.push(("qwen3.rope.freq_base".into(), Value::F32(theta as f32)));
    if let Some(hd) = int(cfg, &["head_dim"]) {
        meta.push(("qwen3.attention.key_length".into(), Value::U32(hd as u32)));
        meta.push(("qwen3.attention.value_length".into(), Value::U32(hd as u32)));
    }
    let scaling = cfg.get("rope_scaling").filter(|v| !v.is_null()).or_else(|| cfg.get("rope_parameters").filter(|r| r.get("rope_type").and_then(Json::as_str).is_some_and(|t| t != "default")));
    if let Some(s) = scaling {
        return Err(bad(format!("rope scaling {s} isn't supported for Qwen3 (rllm runs none)")));
    }
    let vocab_size = int(cfg, &["vocab_size"]);
    tokenizer(&c, cfg, vocab_size, "qwen2", &mut meta)?;

    // Tensors.
    let mut b = Builder::new(c.files.len());
    let tied = cfg.get("tie_word_embeddings").and_then(Json::as_bool).unwrap_or(false);
    b.plan((0..n_layers as usize).collect(), n_layers as usize, tied);
    for t in &c.tensors {
        let name = t.name.strip_prefix("model.language_model.").map(|r| format!("model.{r}")).unwrap_or_else(|| t.name.clone());
        if name.ends_with(".rotary_emb.inv_freq") {
            continue;
        }
        if tied && name == "lm_head.weight" {
            continue;
        }
        let gguf = qwen3_name(&name).ok_or_else(|| bad(format!("unexpected tensor {:?} for Qwen3", t.name)))?;
        if t.shape.len() <= 1 || gguf.ends_with("_norm.weight") {
            b.f32(&gguf, t, c.data(t))?;
        } else {
            b.view(&gguf, t, c.data(t))?;
        }
    }
    let Builder { tensors, owned, .. } = b;
    let mut buffers = c.into_backings();
    buffers.push(Backing::owned(owned));
    Ok(Gguf::from_parts(meta, tensors, buffers)?)
}

/// What a Qwen3.5 tensor becomes.
enum Q35 {
    /// As is (a view), under this GGUF name.
    View(String),
    /// Widened to F32; `plus1`: RMSNorm weights stored as w + 1 (Qwen3.5's
    /// zero-centred norms, as llama.cpp's converter does).
    F32 { name: String, plus1: bool },
    /// Linear-attention tensors with value heads to tile (see `tile_heads`).
    Lin { name: String, part: LinPart },
    /// One expert of a stacked tensor: (GGUF name, layer, expert).
    Expert(String, usize),
}

#[derive(Clone, Copy, PartialEq)]
enum LinPart {
    Qkv,
    Z,
    A,
    B,
    Conv,
    Dt,
    ALog,
    Out,
}

/// HF → GGUF names for one layer of Qwen3.5 (and its MTP block's layers).
fn qwen35_layer(n: usize, part: &str) -> Option<Q35> {
    let g = |s: &str| format!("blk.{n}.{s}");
    let norm = |s: &str| Some(Q35::F32 { name: g(s), plus1: true });
    let lin = |s: &str, part| Some(Q35::Lin { name: g(s), part });
    match part {
        "input_layernorm.weight" => norm("attn_norm.weight"),
        "post_attention_layernorm.weight" => norm("post_attention_norm.weight"),
        "self_attn.q_norm.weight" => norm("attn_q_norm.weight"),
        "self_attn.k_norm.weight" => norm("attn_k_norm.weight"),
        "self_attn.q_proj.weight" => Some(Q35::View(g("attn_q.weight"))),
        "self_attn.k_proj.weight" => Some(Q35::View(g("attn_k.weight"))),
        "self_attn.v_proj.weight" => Some(Q35::View(g("attn_v.weight"))),
        "self_attn.o_proj.weight" => Some(Q35::View(g("attn_output.weight"))),
        "linear_attn.in_proj_qkv.weight" => lin("attn_qkv.weight", LinPart::Qkv),
        "linear_attn.in_proj_z.weight" => lin("attn_gate.weight", LinPart::Z),
        "linear_attn.in_proj_a.weight" => lin("ssm_alpha.weight", LinPart::A),
        "linear_attn.in_proj_b.weight" => lin("ssm_beta.weight", LinPart::B),
        "linear_attn.conv1d.weight" => lin("ssm_conv1d.weight", LinPart::Conv),
        "linear_attn.dt_bias" => lin("ssm_dt.bias", LinPart::Dt),
        "linear_attn.A_log" => lin("ssm_a", LinPart::ALog),
        "linear_attn.norm.weight" => Some(Q35::F32 { name: g("ssm_norm.weight"), plus1: false }),
        "linear_attn.out_proj.weight" => lin("ssm_out.weight", LinPart::Out),
        "mlp.gate_proj.weight" => Some(Q35::View(g("ffn_gate.weight"))),
        "mlp.up_proj.weight" => Some(Q35::View(g("ffn_up.weight"))),
        "mlp.down_proj.weight" => Some(Q35::View(g("ffn_down.weight"))),
        "mlp.gate.weight" => Some(Q35::F32 { name: g("ffn_gate_inp.weight"), plus1: false }),
        "mlp.experts.gate_up_proj" | "mlp.experts.gate_up_proj.weight" => Some(Q35::View(g("ffn_gate_up_exps.weight"))),
        "mlp.experts.down_proj" | "mlp.experts.down_proj.weight" => Some(Q35::View(g("ffn_down_exps.weight"))),
        "mlp.shared_expert.gate_proj.weight" => Some(Q35::View(g("ffn_gate_shexp.weight"))),
        "mlp.shared_expert.up_proj.weight" => Some(Q35::View(g("ffn_up_shexp.weight"))),
        "mlp.shared_expert.down_proj.weight" => Some(Q35::View(g("ffn_down_shexp.weight"))),
        "mlp.shared_expert_gate.weight" => Some(Q35::F32 { name: g("ffn_gate_inp_shexp.weight"), plus1: false }),
        _ => {
            // Per-expert tensors: mlp.experts.X.{gate,up,down}_proj.weight.
            let rest = part.strip_prefix("mlp.experts.")?;
            let (x, w) = rest.split_once('.')?;
            let x: usize = x.parse().ok()?;
            let name = match w {
                "gate_proj.weight" => "ffn_gate_exps.weight",
                "up_proj.weight" => "ffn_up_exps.weight",
                "down_proj.weight" => "ffn_down_exps.weight",
                _ => return None,
            };
            Some(Q35::Expert(g(name), x))
        }
    }
}

/// Qwen3.5 / 3.6 hybrids (`qwen35`, `qwen35moe`), with their MTP block.
fn qwen35(c: Checkpoint, cfg: &serde_json::Map<String, Json>, moe: bool) -> Result<Gguf> {
    qwen35_arch(c, cfg, moe, None)
}

/// llama.cpp's `ClefModel._systemone_template`: the decision prompt, its
/// pieces separated by the server's `sep` and span marks.
fn clef_systemone_template() -> String {
    let text = |v: &str| format!("{{{{ {} }}}}", serde_json::to_string(v).unwrap());
    let render = |n: &str| format!("{{{{ {n} if {n} is string else {n} | tojson(separators=[',', ':']) }}}}");
    let option = concat!(
        "{% set d = o.description %}",
        "{% if q.type == 'noul' and d is none %}",
        "{% set d = 'The proposition is true or the answer is yes.' if o.key == 'true' else 'The proposition is false or the answer is no.' %}",
        "{% endif %}",
        "{{ ({'option_id': o.key} if d is none else {'description': d, 'option_id': o.key}) | tojson(separators=[',', ':']) }}",
    );
    let system = "Read the complete state and schema. Decide every field jointly. Each answer must be exactly one of that field's allowed options.";
    [
        text(&format!("<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\nSTATE:\n")),
        "{% if images %}{{ sep }}{% for image in images %}{{ image }}{% endfor %}".into(),
        text("\n"),
        "{% endif %}".into(),
        "{{ sep }}".into(),
        render("state"),
        "{{ sep }}".into(),
        text("\n\nSCHEMA FIELDS:\n"),
        "{% for q in questions %}".into(),
        "{{ sep }}".into(),
        text("\nFIELD "),
        "{{ loop.index }}".into(),
        text("\nID: "),
        "{{ q.id }}".into(),
        text("\nTYPE: "),
        "{{ q.type }}".into(),
        text("\nINSTRUCTION: "),
        "{{ sep }}{{ mark_question }}".into(),
        render("q.instructions"),
        "{{ sep }}".into(),
        text("\nALLOWED OPTIONS:\n"),
        "{% for o in q.options %}".into(),
        "{{ sep }}".into(),
        text("OPTION "),
        "{{ loop.index }}".into(),
        text(": "),
        "{{ sep }}{{ mark_option }}".into(),
        option.into(),
        "{{ sep }}".into(),
        text("\n"),
        "{% endfor %}".into(),
        "{{ sep }}".into(),
        text("END FIELD\n"),
        "{% endfor %}".into(),
        "{{ sep }}".into(),
        text("\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:"),
    ]
    .concat()
}

/// Clef's decision head (`joint_head.*`): GGUF name, and for nn.MultiheadAttention's
/// packed `in_proj_{weight,bias}` the q/k/v prefix to split it into.
enum Head {
    Tensor(String),
    InProj(String, &'static str),
    Scalar(String),
}

fn clef_head_name(name: &str, n_routing: usize) -> Option<Head> {
    let rest = name.strip_prefix("joint_head.")?;
    let parts: Vec<&str> = rest.split('.').collect();
    if parts.len() == 1 {
        return Some(Head::Scalar(parts[0].to_string()));
    }
    let suffix = *parts.last()?;
    let fixed = |g: &str| Some(Head::Tensor(format!("{g}.{suffix}")));
    let stem = parts[..parts.len() - 1].join(".");
    match stem.as_str() {
        "type_embedding" => return fixed("token_types"),
        "hidden_norm" => return fixed("decision.hidden_norm"),
        "memory_projection" => return fixed("decision.proj_memory"),
        "question_projection" => return fixed("decision.proj_question"),
        "option_question_projection" => return fixed("decision.proj_option_question"),
        "global_projection" => return fixed("decision.proj_global"),
        "option_context_projection" => return fixed("decision.proj_option_context"),
        "option_lexical_projection" => return fixed("decision.proj_option_lexical"),
        "option_summary_norm" => return fixed("decision.option_summary_norm"),
        "field_norm" => return fixed("decision.field_norm"),
        "option_norm" => return fixed("decision.option_norm"),
        "residual_scorer.0" => return fixed("decision.scorer"),
        "residual_scorer.3" => return fixed("decision.scorer_out"),
        _ => {}
    }
    // Blocks: evidence (routing) layers first, then the joint layers.
    let (bid, part) = match parts[..] {
        ["layers", i, ..] => (i.parse::<usize>().ok()? + n_routing, parts[2..parts.len() - 1].join(".")),
        ["evidence_layers", i, ..] => (i.parse::<usize>().ok()?, parts[2..parts.len() - 1].join(".")),
        _ => return None,
    };
    let p = |g: &str| format!("dec.blk.{bid}.{g}");
    let g = match part.as_str() {
        "self_attn" | "multihead_attn" | "attention" if suffix.starts_with("in_proj_") => {
            let kind = if part == "self_attn" { "attn" } else { "cross_attn" };
            let s = if suffix == "in_proj_weight" { "weight" } else { "bias" };
            return Some(Head::InProj(p(kind), s));
        }
        "norm1" => "attn_norm",
        "self_attn.out_proj" => "attn_o",
        "norm2" | "query_norm" => "cross_attn_norm",
        "memory_norm" => "cross_attn_norm_kv",
        "multihead_attn.out_proj" | "attention.out_proj" => "cross_attn_o",
        "norm3" | "feedforward_norm" => "ffn_norm",
        "linear1" | "feedforward.0" => "ffn_up",
        "linear2" | "feedforward.3" => "ffn_down",
        _ => return None,
    };
    Some(Head::Tensor(format!("{}.{suffix}", p(g))))
}

/// Qwen3.5 / 3.6 as `qwen35`/`qwen35moe`, or as `clef` with its decision head.
fn qwen35_arch(c: Checkpoint, cfg: &serde_json::Map<String, Json>, moe: bool, clef: Option<&Json>) -> Result<Gguf> {
    let arch = if clef.is_some() { "clef" } else if moe { "qwen35moe" } else { "qwen35" };
    let k = |s: &str| format!("{arch}.{s}");
    let need = |keys: &[&str]| int(cfg, keys).ok_or_else(|| bad(format!("config.json: no {}", keys[0])));
    let n_layers = need(&["num_hidden_layers"])? as usize;
    let hidden = need(&["hidden_size"])?;
    let heads = need(&["num_attention_heads"])?;
    let head_dim = int(cfg, &["head_dim"]).unwrap_or(hidden / heads.max(1));
    // HF names, normalized: `model.language_model.` -> `model.`, `model.mtp.` -> `mtp.`.
    let norm_name = |n: &str| {
        let n = n.strip_prefix("model.language_model.").map(|r| format!("model.{r}")).unwrap_or_else(|| n.to_string());
        n.strip_prefix("model.mtp.").map(|r| format!("mtp.{r}")).unwrap_or(n)
    };
    // MTP layers: the config's count, else as many as the tensors show.
    let mtp_seen = c.tensors.iter().filter_map(|t| norm_name(&t.name).strip_prefix("mtp.layers.")?.split('.').next()?.parse::<usize>().ok()).max().map_or(0, |m| m + 1);
    let n_mtp = if mtp_seen == 0 { 0 } else { int(cfg, &["mtp_num_hidden_layers"]).map(|v| v as usize).filter(|&v| v > 0).unwrap_or(mtp_seen) };
    let rope = cfg.get("rope_parameters").or_else(|| cfg.get("rope_scaling")).cloned().unwrap_or(Json::Null);
    if let Some(t) = rope.get("rope_type").or_else(|| rope.get("type")).and_then(Json::as_str).filter(|t| *t != "default") {
        return Err(bad(format!("rope type {t} isn't supported for Qwen3.5")));
    }
    let theta = rope.get("rope_theta").and_then(Json::as_f64).or_else(|| float(cfg, &["rope_theta"])).unwrap_or(10_000_000.0);
    let partial = rope.get("partial_rotary_factor").and_then(Json::as_f64).or_else(|| float(cfg, &["partial_rotary_factor"])).unwrap_or(0.25);
    let (nk, nv) = (need(&["linear_num_key_heads"])? as usize, need(&["linear_num_value_heads"])? as usize);
    let (dk, dv) = (need(&["linear_key_head_dim"])? as usize, need(&["linear_value_head_dim"])? as usize);
    let mut meta: Vec<(String, Value)> = vec![
        ("general.architecture".into(), Value::String(arch.into())),
        ("general.name".into(), Value::String(c.dir.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default())),
        (k("block_count"), Value::U32((n_layers + n_mtp) as u32)),
        (k("context_length"), Value::U32(int(cfg, &["max_position_embeddings"]).unwrap_or(262_144) as u32)),
        (k("embedding_length"), Value::U32(hidden as u32)),
        (k("attention.head_count"), Value::U32(heads as u32)),
        (k("attention.head_count_kv"), Value::U32(int(cfg, &["num_key_value_heads"]).unwrap_or(heads) as u32)),
        (k("attention.key_length"), Value::U32(head_dim as u32)),
        (k("attention.value_length"), Value::U32(head_dim as u32)),
        (k("attention.layer_norm_rms_epsilon"), Value::F32(float(cfg, &["rms_norm_eps"]).unwrap_or(1e-6) as f32)),
        (k("rope.freq_base"), Value::F32(theta as f32)),
        (k("rope.dimension_count"), Value::U32((head_dim as f64 * partial) as u32)),
        (k("ssm.conv_kernel"), Value::U32(need(&["linear_conv_kernel_dim"])? as u32)),
        (k("ssm.state_size"), Value::U32(dk as u32)),
        (k("ssm.group_count"), Value::U32(nk as u32)),
        (k("ssm.time_step_rank"), Value::U32(nv as u32)),
        (k("ssm.inner_size"), Value::U32((dv * nv) as u32)),
        (k("full_attention_interval"), Value::U32(int(cfg, &["full_attention_interval"]).unwrap_or(4) as u32)),
    ];
    // M-RoPE sections (vision positions; text-only rllm ignores them), padded to 4.
    let mut sections: Vec<Value> = rope.get("mrope_section").and_then(Json::as_array).map_or(vec![11, 11, 10], |a| a.iter().filter_map(Json::as_u64).collect()).into_iter().map(|v| Value::U32(v as u32)).collect();
    sections.resize(4, Value::U32(0));
    meta.push((k("rope.dimension_sections"), Value::Array(sections)));
    if let Some(f) = int(cfg, &["intermediate_size"]) {
        meta.push((k("feed_forward_length"), Value::U32(f as u32)));
    }
    if n_mtp > 0 {
        meta.push((k("nextn_predict_layers"), Value::U32(n_mtp as u32)));
    }
    if let Some(types) = cfg.get("layer_types").and_then(Json::as_array) {
        if types.len() != n_layers {
            return Err(bad(format!("{} layer_types for {n_layers} layers", types.len())));
        }
        let mut r: Vec<Value> = types.iter().map(|t| Value::Bool(t.as_str() == Some("linear_attention"))).collect();
        r.extend((0..n_mtp).map(|_| Value::Bool(false)));
        meta.push((k("attention.recurrent_layers"), Value::Array(r)));
    }
    if moe {
        meta.push((k("expert_count"), Value::U32(need(&["num_experts"])? as u32)));
        meta.push((k("expert_used_count"), Value::U32(need(&["num_experts_per_tok"])? as u32)));
        meta.push((k("expert_feed_forward_length"), Value::U32(need(&["moe_intermediate_size"])? as u32)));
        meta.push((k("expert_shared_feed_forward_length"), Value::U32(int(cfg, &["shared_expert_intermediate_size"]).unwrap_or(0) as u32)));
    }
    tokenizer(&c, cfg, int(cfg, &["vocab_size"]), "qwen35", &mut meta)?;
    let n_routing = match clef {
        Some(h) => {
            let u = |key: &str| h.get(key).and_then(Json::as_u64).ok_or_else(|| bad(format!("joint_head_config.json: no {key}")));
            let n_routing = u("routing_layers")?;
            meta.push((k("decision.type"), Value::String("clef".into())));
            meta.push((k("decision.routing_block_count"), Value::U32(n_routing as u32)));
            meta.push((k("decision.block_count"), Value::U32(u("layers")? as u32)));
            meta.push((k("decision.head_count"), Value::U32(u("heads")? as u32)));
            // torch.nn.LayerNorm's default, used by the head.
            meta.push((k("attention.layer_norm_epsilon"), Value::F32(1e-5)));
            meta.push(("tokenizer.chat_template.systemone".into(), Value::String(clef_systemone_template())));
            n_routing as usize
        }
        None => 0,
    };
    let mut scales = std::collections::BTreeMap::new();

    let tied = cfg.get("tie_word_embeddings").and_then(Json::as_bool).unwrap_or(false);
    let r = if nk > 0 && nv > nk && nv % nk == 0 { nv / nk } else { 1 };
    let mut b = Builder::new(c.files.len());
    {
        let interval = int(cfg, &["full_attention_interval"]).unwrap_or(4).max(1) as usize;
        let types = cfg.get("layer_types").and_then(Json::as_array);
        let attn = (0..n_layers)
            .filter(|&i| match types {
                Some(t) => t.get(i).and_then(Json::as_str) != Some("linear_attention"),
                None => (i + 1) % interval == 0,
            })
            .chain(n_layers..n_layers + n_mtp)
            .collect();
        b.plan(attn, n_layers + n_mtp, tied);
    }
    // Per-expert matrices, stacked once all are seen: (GGUF name) -> experts.
    let mut experts: std::collections::BTreeMap<String, Vec<Option<usize>>> = std::collections::BTreeMap::new();
    for (ti, t) in c.tensors.iter().enumerate() {
        let name = norm_name(&t.name);
        if clef.is_some() && name.starts_with("joint_head.") {
            let data = c.data(t);
            match clef_head_name(&name, n_routing).ok_or_else(|| bad(format!("unexpected tensor {name:?} in Clef's head")))? {
                Head::Scalar(s) => {
                    let v = to_f32(t, data)?.first().copied().ok_or_else(|| bad(format!("{name}: empty")))? as f64;
                    // Stored as the values used at inference.
                    let v = if s == "residual_gate" { 1.0 / (1.0 + (-v).exp()) } else { v.min(100f64.ln()).exp() };
                    scales.insert(s, v as f32);
                }
                Head::Tensor(g) if t.shape.len() <= 1 => b.f32(&g, t, data)?,
                // A one-row matrix (the scorer's output) is stored as a vector, as llama.cpp does.
                Head::Tensor(g) if t.shape.len() == 2 && t.shape[0] == 1 => b.f32_vals(&g, vec![t.shape[1] as u64], &to_f32(t, data)?),
                Head::Tensor(g) => b.view(&g, t, data)?,
                Head::InProj(prefix, suffix) => {
                    // q, k, v stacked on the rows.
                    let rows = t.shape[0] / 3;
                    let row_bytes = data.len() / t.shape[0].max(1);
                    for (i, x) in ["q", "k", "v"].iter().enumerate() {
                        let mut shape = t.shape.clone();
                        shape[0] = rows;
                        let part = &data[i * rows * row_bytes..(i + 1) * rows * row_bytes];
                        let g = format!("{prefix}_{x}.{suffix}");
                        if shape.len() <= 1 {
                            let tt = crate::Tensor { shape, size: part.len(), ..t.clone() };
                            b.f32(&g, &tt, part)?;
                        } else {
                            let dtype = t.dtype.rllm().ok_or_else(|| bad(format!("{}: {:?} weights aren't supported yet", t.name, t.dtype)))?;
                            b.matrix(&g, Builder::dims(&shape), dtype, part);
                        }
                    }
                }
            }
            continue;
        }
        if name.starts_with("model.visual.") || name.contains("visual.") || name.ends_with(".rotary_emb.inv_freq") || (tied && name == "lm_head.weight") {
            continue;
        }
        let what = match name.as_str() {
            "model.embed_tokens.weight" => Q35::View("token_embd.weight".into()),
            "model.norm.weight" => Q35::F32 { name: "output_norm.weight".into(), plus1: true },
            "lm_head.weight" => Q35::View("output.weight".into()),
            "mtp.fc.weight" => Q35::View(format!("blk.{n_layers}.nextn.eh_proj.weight")),
            "mtp.pre_fc_norm_embedding.weight" => Q35::F32 { name: format!("blk.{n_layers}.nextn.enorm.weight"), plus1: true },
            "mtp.pre_fc_norm_hidden.weight" => Q35::F32 { name: format!("blk.{n_layers}.nextn.hnorm.weight"), plus1: true },
            "mtp.norm.weight" => Q35::F32 { name: format!("blk.{n_layers}.nextn.shared_head_norm.weight"), plus1: true },
            other => {
                let (n, part) = if let Some(rest) = other.strip_prefix("mtp.layers.") {
                    let (i, part) = rest.split_once('.').ok_or_else(|| bad(format!("tensor {other}")))?;
                    (n_layers + i.parse::<usize>().map_err(|_| bad(format!("tensor {other}")))?, part)
                } else {
                    let rest = other.strip_prefix("model.layers.").ok_or_else(|| bad(format!("unexpected tensor {:?} for Qwen3.5", t.name)))?;
                    let (i, part) = rest.split_once('.').ok_or_else(|| bad(format!("tensor {other}")))?;
                    (i.parse::<usize>().map_err(|_| bad(format!("tensor {other}")))?, part)
                };
                qwen35_layer(n, part).ok_or_else(|| bad(format!("unexpected tensor {:?} for Qwen3.5", t.name)))?
            }
        };
        let data = c.data(t);
        match what {
            Q35::View(g) => b.view(&g, t, data)?,
            Q35::F32 { name: g, plus1 } => {
                let mut v = to_f32(t, data)?;
                if plus1 {
                    v.iter_mut().for_each(|x| *x += 1.0);
                }
                // A [1, d] gate is stored as the vector [d] (as rllm reads it).
                let dims = if g.ends_with("ffn_gate_inp_shexp.weight") { vec![v.len() as u64] } else { Builder::dims(&t.shape) };
                b.f32_vals(&g, dims, &v);
            }
            Q35::Expert(g, x) => {
                let e = experts.entry(g).or_default();
                if e.len() <= x {
                    e.resize(x + 1, None);
                }
                e[x] = Some(ti);
            }
            Q35::Lin { name: g, part } => {
                let qk_rows = 2 * nk * dk;
                match part {
                    LinPart::Qkv | LinPart::Z | LinPart::A | LinPart::B | LinPart::Out => {
                        let dtype = t.dtype.rllm().ok_or_else(|| bad(format!("{}: {:?} weights aren't supported yet", t.name, t.dtype)))?;
                        let mut bytes = data.to_vec();
                        if r > 1 {
                            let (rows, cols) = (t.shape[0], t.shape.get(1).copied().unwrap_or(1));
                            let row = cols * t.dtype.size();
                            match part {
                                LinPart::Qkv => tile_heads(&mut bytes[qk_rows * row..], dv * row, nk, r),
                                LinPart::Z => tile_heads(&mut bytes, dv * row, nk, r),
                                LinPart::A | LinPart::B => tile_heads(&mut bytes, row, nk, r),
                                _ => {
                                    // out_proj: the columns (each row's nv * dv inputs).
                                    for rr in 0..rows {
                                        tile_heads(&mut bytes[rr * row..(rr + 1) * row], dv * t.dtype.size(), nk, r);
                                    }
                                }
                            }
                        }
                        b.matrix(&g, Builder::dims(&t.shape), dtype, &bytes);
                    }
                    LinPart::Conv => {
                        // [C, 1, K] -> [C, K], F32, value channels tiled.
                        let kk = *t.shape.last().unwrap_or(&1);
                        let ch = t.numel() / kk.max(1);
                        let mut v = to_f32(t, data)?;
                        if r > 1 {
                            let mut bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                            let row = kk * 4;
                            tile_heads(&mut bytes[qk_rows * row..], dv * row, nk, r);
                            v = bytes.as_chunks::<4>().0.iter().map(|&b| f32::from_le_bytes(b)).collect();
                        }
                        b.f32_vals(&g, vec![kk as u64, ch as u64], &v);
                    }
                    LinPart::Dt | LinPart::ALog => {
                        let mut v = to_f32(t, data)?;
                        if part == LinPart::ALog {
                            v.iter_mut().for_each(|x| *x = -x.exp());
                        }
                        if r > 1 {
                            let mut bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                            tile_heads(&mut bytes, 4, nk, r);
                            v = bytes.as_chunks::<4>().0.iter().map(|&b| f32::from_le_bytes(b)).collect();
                        }
                        b.f32_vals(&g, vec![v.len() as u64], &v);
                    }
                }
            }
        }
    }
    // Stacked experts: [E, out, in] (GGUF dims [in, out, E]), in expert order.
    for (g, ids) in experts {
        let ids: Vec<usize> = ids.into_iter().collect::<Option<_>>().ok_or_else(|| bad(format!("{g}: experts missing")))?;
        let first = &c.tensors[ids[0]];
        let dtype = first.dtype.rllm().ok_or_else(|| bad(format!("{}: {:?} experts aren't supported yet", first.name, first.dtype)))?;
        if ids.iter().any(|&i| c.tensors[i].shape != first.shape || c.tensors[i].dtype != first.dtype) {
            return Err(bad(format!("{g}: experts of different shapes")));
        }
        let mut bytes = Vec::with_capacity(first.size * ids.len());
        for &i in &ids {
            bytes.extend_from_slice(c.data(&c.tensors[i]));
        }
        let mut dims = Builder::dims(&first.shape);
        dims.push(ids.len() as u64);
        b.matrix(&g, dims, dtype, &bytes);
    }
    if clef.is_some() {
        let v: Vec<f32> = ["prior_logit_scale", "joint_logit_scale", "residual_gate"]
            .iter()
            .map(|k| scales.get(*k).copied().ok_or_else(|| bad(format!("Clef's head has no {k}"))))
            .collect::<Result<_>>()?;
        b.f32_vals("decision.scales", vec![3], &v);
    }
    let Builder { tensors, owned, .. } = b;
    let mut buffers = c.into_backings();
    buffers.push(Backing::owned(owned));
    Ok(Gguf::from_parts(meta, tensors, buffers)?)
}

/// llama.cpp's `LlamaModel.permute` of Q/K rows (NeoX halves to
/// interleaved pairs), on raw bytes: row `hd*h + 2j + t` takes row
/// `hd*h + t*hd/2 + j`. rllm's llama loader undoes it, so converting as
/// llama.cpp does keeps one path (and byte-identical tensors).
fn permute_qk(bytes: &[u8], rows: usize, row_bytes: usize, heads: usize) -> Vec<u8> {
    let hd = rows / heads.max(1);
    let mut out = vec![0u8; bytes.len()];
    for h in 0..heads {
        for j in 0..hd / 2 {
            for t in 0..2 {
                let (to, from) = (hd * h + 2 * j + t, hd * h + t * (hd / 2) + j);
                out[to * row_bytes..(to + 1) * row_bytes].copy_from_slice(&bytes[from * row_bytes..(from + 1) * row_bytes]);
            }
        }
    }
    out
}

/// The `tokenizer.ggml.pre` of a byte-level BPE tokenizer.json, from its
/// pre-tokenizer regex (llama.cpp hashes the tokenizer's output instead; the
/// regex names the same split and fails loudly on unknown ones).
fn bpe_pre(tj: &Json) -> Result<&'static str> {
    const LLAMA3: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    const TEKKEN: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    let pt = tj.get("pre_tokenizer").cloned().unwrap_or(Json::Null);
    let steps = match pt.get("type").and_then(Json::as_str) {
        Some("Sequence") => pt.get("pretokenizers").and_then(Json::as_array).cloned().unwrap_or_default(),
        _ => vec![pt],
    };
    let regex = steps.iter().find_map(|p| p.get("pattern")?.get("Regex")?.as_str().map(str::to_owned));
    match regex.as_deref() {
        Some(LLAMA3) => Ok("llama-bpe"),
        Some(TEKKEN) => Ok("tekken"),
        Some(other) => Err(bad(format!("tokenizer.json: unknown pre-tokenizer regex {other:?}"))),
        None => Err(bad("tokenizer.json: no pre-tokenizer regex")),
    }
}

/// Llama 3 (`llama`): Q/K permuted as llama.cpp does; Llama 3.1+ RoPE
/// scaling as a generated `rope_freqs` tensor.
fn llama(c: Checkpoint, cfg: &serde_json::Map<String, Json>) -> Result<Gguf> {
    let need = |keys: &[&str]| int(cfg, keys).ok_or_else(|| bad(format!("config.json: no {}", keys[0])));
    let n_layers = need(&["num_hidden_layers"])?;
    let hidden = need(&["hidden_size"])?;
    let heads = need(&["num_attention_heads"])?;
    let kv_heads = int(cfg, &["num_key_value_heads"]).unwrap_or(heads);
    let head_dim = int(cfg, &["head_dim"]).unwrap_or(hidden / heads.max(1));
    let theta = float(cfg, &["rope_theta"]).or_else(|| cfg.get("rope_parameters").and_then(|r| r.get("rope_theta")).and_then(Json::as_f64)).unwrap_or(10_000.0);
    let vocab = int(cfg, &["vocab_size"]);
    let mut meta: Vec<(String, Value)> = vec![
        ("general.architecture".into(), Value::String("llama".into())),
        ("general.name".into(), Value::String(c.dir.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default())),
        ("llama.block_count".into(), Value::U32(n_layers as u32)),
        ("llama.context_length".into(), Value::U32(int(cfg, &["max_position_embeddings"]).unwrap_or(2048) as u32)),
        ("llama.embedding_length".into(), Value::U32(hidden as u32)),
        ("llama.feed_forward_length".into(), Value::U32(need(&["intermediate_size"])? as u32)),
        ("llama.attention.head_count".into(), Value::U32(heads as u32)),
        ("llama.attention.head_count_kv".into(), Value::U32(kv_heads as u32)),
        ("llama.attention.layer_norm_rms_epsilon".into(), Value::F32(float(cfg, &["rms_norm_eps"]).unwrap_or(1e-6) as f32)),
        ("llama.rope.freq_base".into(), Value::F32(theta as f32)),
        ("llama.attention.key_length".into(), Value::U32(head_dim as u32)),
        ("llama.attention.value_length".into(), Value::U32(head_dim as u32)),
        ("llama.rope.dimension_count".into(), Value::U32(head_dim as u32)),
    ];
    if let Some(v) = vocab {
        meta.push(("llama.vocab_size".into(), Value::U32(v as u32)));
    }
    let rope = cfg.get("rope_scaling").filter(|v| !v.is_null()).or_else(|| cfg.get("rope_parameters")).cloned().unwrap_or(Json::Null);
    let rope_type = rope.get("rope_type").or_else(|| rope.get("type")).and_then(Json::as_str).unwrap_or("default").to_lowercase();
    let mut b = Builder::new(c.files.len());
    match rope_type.as_str() {
        "default" => {}
        "llama3" => {
            // llama.cpp's generate_extra_tensors: per-frequency divisors.
            let f = |k: &str, d: f64| rope.get(k).and_then(Json::as_f64).unwrap_or(d);
            let (factor, lf, hf, old) = (f("factor", 8.0), f("low_freq_factor", 1.0), f("high_freq_factor", 4.0), f("original_max_position_embeddings", 8192.0));
            let (low_wl, high_wl) = (old / lf, old / hf);
            let dim = head_dim as usize;
            let v: Vec<f32> = (0..dim / 2)
                .map(|i| {
                    // In f32, as torch computes it there.
                    let freq = (1.0f32 / (theta as f32).powf(2.0 * i as f32 / dim as f32)) as f64;
                    let wl = 2.0 * std::f64::consts::PI / freq;
                    (if wl < high_wl {
                        1.0
                    } else if wl > low_wl {
                        factor
                    } else {
                        let s = (old / wl - lf) / (hf - lf);
                        1.0 / ((1.0 - s) / factor + s)
                    }) as f32
                })
                .collect();
            b.f32_vals("rope_freqs.weight", vec![v.len() as u64], &v);
        }
        other => return Err(bad(format!("rope type {other} isn't supported for Llama"))),
    }
    if c.dir.join("tokenizer.model").is_file() {
        // Llama 2, Mistral 7B, TinyLlama: SentencePiece, as llama.cpp prefers.
        spm_tokenizer(&c, cfg, vocab, &mut meta)?;
    } else {
        // Byte-level BPE: Llama 3's, or Mistral's Tekken (Nemo, Small 3).
        let tj = c.json("tokenizer.json")?.ok_or_else(|| bad("no tokenizer.json"))?;
        tokenizer(&c, cfg, vocab, bpe_pre(&tj)?, &mut meta)?;
        // As llama.cpp's LlamaModel.set_vocab, for any tokenizer.
        if let Some(p) = c.json("tokenizer_config.json")?.and_then(|t| t.get("add_prefix_space").and_then(Json::as_bool)) {
            meta.push(("tokenizer.ggml.add_space_prefix".into(), Value::Bool(p)));
        }
    }
    let tied = cfg.get("tie_word_embeddings").and_then(Json::as_bool).unwrap_or(false);
    b.plan((0..n_layers as usize).collect(), n_layers as usize, tied);
    for t in &c.tensors {
        let name = t.name.as_str();
        if name.ends_with(".rotary_emb.inv_freq") || (tied && name == "lm_head.weight") {
            continue;
        }
        let gguf = match name {
            "model.embed_tokens.weight" => "token_embd.weight".to_string(),
            "model.norm.weight" => "output_norm.weight".to_string(),
            "lm_head.weight" => "output.weight".to_string(),
            _ => {
                let rest = name.strip_prefix("model.layers.").ok_or_else(|| bad(format!("unexpected tensor {name:?} for Llama")))?;
                let (n, part) = rest.split_once('.').ok_or_else(|| bad(format!("tensor {name}")))?;
                let n: usize = n.parse().map_err(|_| bad(format!("tensor {name}")))?;
                let g = match part {
                    "input_layernorm.weight" => "attn_norm.weight",
                    "self_attn.q_proj.weight" => "attn_q.weight",
                    "self_attn.k_proj.weight" => "attn_k.weight",
                    "self_attn.v_proj.weight" => "attn_v.weight",
                    "self_attn.o_proj.weight" => "attn_output.weight",
                    "post_attention_layernorm.weight" => "ffn_norm.weight",
                    "mlp.gate_proj.weight" => "ffn_gate.weight",
                    "mlp.up_proj.weight" => "ffn_up.weight",
                    "mlp.down_proj.weight" => "ffn_down.weight",
                    _ => return Err(bad(format!("unexpected tensor {name:?} for Llama"))),
                };
                format!("blk.{n}.{g}")
            }
        };
        if t.shape.len() <= 1 || gguf.ends_with("_norm.weight") {
            b.f32(&gguf, t, c.data(t))?;
        } else if gguf.ends_with("attn_q.weight") || gguf.ends_with("attn_k.weight") {
            let h = if gguf.ends_with("attn_q.weight") { heads } else { kv_heads } as usize;
            let dtype = t.dtype.rllm().ok_or_else(|| bad(format!("{}: {:?} weights aren't supported yet", t.name, t.dtype)))?;
            let row = t.shape[1] * t.dtype.size();
            b.matrix(&gguf, Builder::dims(&t.shape), dtype, &permute_qk(c.data(t), t.shape[0], row, h));
        } else {
            b.view(&gguf, t, c.data(t))?;
        }
    }
    let Builder { tensors, owned, .. } = b;
    let mut buffers = c.into_backings();
    buffers.push(Backing::owned(owned));
    Ok(Gguf::from_parts(meta, tensors, buffers)?)
}

/// Gemma 4 dense (`gemma4`, llama.cpp's `Gemma4Model`): norms as they are
/// (no +1, unlike Gemma 1–3), `rope_freqs` generated for the global layers'
/// partial rotary, a SentencePiece-style BPE vocabulary.
fn gemma4(c: Checkpoint, cfg: &serde_json::Map<String, Json>) -> Result<Gguf> {
    let k = |s: &str| format!("gemma4.{s}");
    let need = |keys: &[&str]| int(cfg, keys).ok_or_else(|| bad(format!("config.json: no {}", keys[0])));
    let n_layers = need(&["num_hidden_layers"])? as usize;
    let hidden = need(&["hidden_size"])?;
    let head_dim = need(&["head_dim"])?;
    let global_head_dim = int(cfg, &["global_head_dim"]).unwrap_or(head_dim);
    let shared = need(&["num_kv_shared_layers"])? as usize;
    let ffn = need(&["intermediate_size"])?;
    let types: Vec<bool> = cfg
        .get("layer_types")
        .and_then(Json::as_array)
        .ok_or_else(|| bad("config.json: no layer_types"))?
        .iter()
        .map(|t| t.as_str() == Some("sliding_attention"))
        .collect();
    if types.len() != n_layers {
        return Err(bad(format!("{} layer_types for {n_layers} layers", types.len())));
    }
    let rope = cfg.get("rope_parameters").cloned().unwrap_or(Json::Null);
    let (full, swa) = (rope.get("full_attention").cloned().unwrap_or(Json::Null), rope.get("sliding_attention").cloned().unwrap_or(Json::Null));
    if full.get("rope_type").and_then(Json::as_str).is_some_and(|t| t != "proportional") {
        return Err(bad("Gemma 4: full-attention rope type must be proportional"));
    }
    let theta = |r: &Json, d: f64| r.get("rope_theta").and_then(Json::as_f64).unwrap_or(d);
    let mut meta: Vec<(String, Value)> = vec![
        ("general.architecture".into(), Value::String("gemma4".into())),
        ("general.name".into(), Value::String(c.dir.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default())),
        (k("block_count"), Value::U32(n_layers as u32)),
        (k("context_length"), Value::U32(int(cfg, &["max_position_embeddings"]).unwrap_or(131_072) as u32)),
        (k("embedding_length"), Value::U32(hidden as u32)),
        (k("attention.head_count"), Value::U32(int(cfg, &["num_attention_heads"]).unwrap_or(8) as u32)),
        (k("attention.layer_norm_rms_epsilon"), Value::F32(float(cfg, &["rms_norm_eps"]).unwrap_or(1e-6) as f32)),
        (k("attention.key_length"), Value::U32(global_head_dim as u32)),
        (k("attention.value_length"), Value::U32(global_head_dim as u32)),
        (k("attention.key_length_swa"), Value::U32(head_dim as u32)),
        (k("attention.value_length_swa"), Value::U32(head_dim as u32)),
        (k("rope.freq_base"), Value::F32(theta(&full, 1e6) as f32)),
        (k("rope.freq_base_swa"), Value::F32(theta(&swa, 10_000.0) as f32)),
        (k("rope.dimension_count"), Value::U32(global_head_dim as u32)),
        (k("rope.dimension_count_swa"), Value::U32((head_dim as f64 * rope.get("partial_rotary_factor").and_then(Json::as_f64).unwrap_or(1.0)) as u32)),
        (k("attention.sliding_window"), Value::U32(need(&["sliding_window"])? as u32)),
        (k("attention.sliding_window_pattern"), Value::Array(types.iter().map(|&b| Value::Bool(b)).collect())),
        (k("attention.shared_kv_layers"), Value::U32(shared as u32)),
        (k("embedding_length_per_layer_input"), Value::U32(int(cfg, &["hidden_size_per_layer_input"]).unwrap_or(0) as u32)),
    ];
    // Double-wide MLP: the KV-sharing layers have twice the FFN width.
    if cfg.get("use_double_wide_mlp").and_then(Json::as_bool).unwrap_or(false) {
        let v = (0..n_layers).map(|i| Value::U32(if i < n_layers - shared { ffn } else { 2 * ffn } as u32)).collect();
        meta.push((k("feed_forward_length"), Value::Array(v)));
    } else {
        meta.push((k("feed_forward_length"), Value::U32(ffn as u32)));
    }
    let kv = int(cfg, &["num_key_value_heads"]).unwrap_or(4);
    match int(cfg, &["num_global_key_value_heads"]) {
        Some(g) => meta.push((k("attention.head_count_kv"), Value::Array(types.iter().map(|&s| Value::U32(if s { kv } else { g } as u32)).collect()))),
        None => meta.push((k("attention.head_count_kv"), Value::U32(kv as u32))),
    }
    if let Some(cap) = float(cfg, &["final_logit_softcapping"]).filter(|&v| v != 0.0) {
        meta.push((k("final_logit_softcapping"), Value::F32(cap as f32)));
    }
    // 26B-A4B: routed experts beside each layer's dense FFN.
    if let Some(n) = int(cfg, &["num_local_experts", "num_experts", "n_routed_experts"]) {
        meta.push((k("expert_count"), Value::U32(n as u32)));
    }
    if let Some(n) = int(cfg, &["num_experts_per_tok", "num_experts_per_token", "top_k_experts"]) {
        meta.push((k("expert_used_count"), Value::U32(n as u32)));
    }
    if let Some(n) = int(cfg, &["expert_intermediate_size", "moe_intermediate_size"]) {
        meta.push((k("expert_feed_forward_length"), Value::U32(n as u32)));
    }
    gemma4_tokenizer(&c, cfg, &mut meta)?;

    let tied = cfg.get("tie_word_embeddings").and_then(Json::as_bool).unwrap_or(true);
    let mut b = Builder::new(c.files.len());
    b.plan((0..n_layers).filter(|&i| i < n_layers - shared).collect(), n_layers, tied || c.tensor("lm_head.weight").is_none());
    // rope_freqs: 1 for the rotated pairs of a global head, 1e30 (no rotation) for the rest.
    let n_rot = (global_head_dim as f64 * full.get("partial_rotary_factor").and_then(Json::as_f64).unwrap_or(1.0) / 2.0) as usize;
    let half = global_head_dim as usize / 2;
    let freqs: Vec<f32> = (0..half).map(|i| if i < n_rot { 1.0 } else { 1e30 }).collect();
    b.f32_vals("rope_freqs.weight", vec![half as u64], &freqs);
    for t in &c.tensors {
        let name = t.name.strip_prefix("model.language_model.").map(|r| format!("model.{r}")).unwrap_or_else(|| t.name.clone());
        if name.contains("vision") || name.contains("audio") || (tied && name == "lm_head.weight") {
            continue;
        }
        let gguf = match name.as_str() {
            "model.embed_tokens.weight" => "token_embd.weight".to_string(),
            "model.embed_tokens_per_layer.weight" => "per_layer_token_embd.weight".to_string(),
            "model.per_layer_model_projection.weight" => "per_layer_model_proj.weight".to_string(),
            "model.per_layer_projection_norm.weight" => "per_layer_proj_norm.weight".to_string(),
            "model.norm.weight" => "output_norm.weight".to_string(),
            "lm_head.weight" => "output.weight".to_string(),
            _ => {
                let rest = name.strip_prefix("model.layers.").ok_or_else(|| bad(format!("unexpected tensor {:?} for Gemma 4", t.name)))?;
                let (n, part) = rest.split_once('.').ok_or_else(|| bad(format!("tensor {name}")))?;
                let n: usize = n.parse().map_err(|_| bad(format!("tensor {name}")))?;
                let g = match part {
                    "input_layernorm.weight" => "attn_norm.weight",
                    "self_attn.q_proj.weight" => "attn_q.weight",
                    "self_attn.k_proj.weight" => "attn_k.weight",
                    "self_attn.v_proj.weight" => "attn_v.weight",
                    "self_attn.o_proj.weight" => "attn_output.weight",
                    "self_attn.q_norm.weight" => "attn_q_norm.weight",
                    "self_attn.k_norm.weight" => "attn_k_norm.weight",
                    "post_attention_layernorm.weight" => "post_attention_norm.weight",
                    "pre_feedforward_layernorm.weight" => "ffn_norm.weight",
                    "post_feedforward_layernorm.weight" => "post_ffw_norm.weight",
                    "mlp.gate_proj.weight" => "ffn_gate.weight",
                    "mlp.up_proj.weight" => "ffn_up.weight",
                    "mlp.down_proj.weight" => "ffn_down.weight",
                    "per_layer_input_gate.weight" => "inp_gate.weight",
                    "per_layer_projection.weight" => "proj.weight",
                    "post_per_layer_input_norm.weight" => "post_norm.weight",
                    "layer_scalar" => "layer_output_scale.weight",
                    // MoE (26B-A4B); the fused gate_up stays fused.
                    "router.proj.weight" => "ffn_gate_inp.weight",
                    "router.scale" => "ffn_gate_inp.scale",
                    "router.per_expert_scale" => "ffn_down_exps.scale",
                    "experts.gate_up_proj" | "experts.gate_up_proj.weight" => "ffn_gate_up_exps.weight",
                    "experts.down_proj" | "experts.down_proj.weight" => "ffn_down_exps.weight",
                    "pre_feedforward_layernorm_2.weight" => "pre_ffw_norm_2.weight",
                    "post_feedforward_layernorm_1.weight" => "post_ffw_norm_1.weight",
                    "post_feedforward_layernorm_2.weight" => "post_ffw_norm_2.weight",
                    _ => return Err(bad(format!("unexpected tensor {:?} for Gemma 4", t.name))),
                };
                format!("blk.{n}.{g}")
            }
        };
        if t.shape.len() <= 1 || gguf.ends_with("norm.weight") || gguf.ends_with(".scale") {
            b.f32(&gguf, t, c.data(t))?;
        } else {
            b.view(&gguf, t, c.data(t))?;
        }
    }
    let Builder { tensors, owned, .. } = b;
    let mut buffers = c.into_backings();
    buffers.push(Backing::owned(owned));
    Ok(Gguf::from_parts(meta, tensors, buffers)?)
}

/// The pieces of a SentencePiece `tokenizer.model` (a protobuf
/// `ModelProto`): (text, score, type), type as sentencepiece numbers it
/// (1 normal, 2 unknown, 3 control, 4 user-defined, 5 unused, 6 byte).
fn spm_pieces(bytes: &[u8]) -> Result<Vec<(String, f32, i32)>> {
    fn varint(b: &[u8], i: &mut usize) -> Result<u64> {
        let mut v = 0u64;
        for shift in (0..64).step_by(7) {
            let x = *b.get(*i).ok_or_else(|| bad("tokenizer.model: truncated"))?;
            *i += 1;
            v |= u64::from(x & 0x7f) << shift;
            if x & 0x80 == 0 {
                return Ok(v);
            }
        }
        Err(bad("tokenizer.model: bad varint"))
    }
    /// One field: (number, wire type, payload for length-delimited / fixed32 / varint).
    fn field<'a>(b: &'a [u8], i: &mut usize) -> Result<(u64, u64, &'a [u8], u64)> {
        let key = varint(b, i)?;
        let (num, wt) = (key >> 3, key & 7);
        match wt {
            0 => Ok((num, wt, &[][..], varint(b, i)?)),
            1 | 5 => {
                let n = if wt == 1 { 8 } else { 4 };
                let p = b.get(*i..*i + n).ok_or_else(|| bad("tokenizer.model: truncated"))?;
                *i += n;
                Ok((num, wt, p, 0))
            }
            2 => {
                let n = varint(b, i)? as usize;
                let p = b.get(*i..i.checked_add(n).ok_or_else(|| bad("tokenizer.model: truncated"))?).ok_or_else(|| bad("tokenizer.model: truncated"))?;
                *i += n;
                Ok((num, wt, p, 0))
            }
            _ => Err(bad(format!("tokenizer.model: wire type {wt}"))),
        }
    }
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let (num, wt, p, _) = field(bytes, &mut i)?;
        if num != 1 || wt != 2 {
            continue;
        }
        let (mut text, mut score, mut ty) = (String::new(), 0f32, 1i32);
        let mut j = 0;
        while j < p.len() {
            match field(p, &mut j)? {
                (1, 2, s, _) => text = String::from_utf8(s.to_vec()).map_err(|_| bad("tokenizer.model: piece is not UTF-8"))?,
                (2, 5, f, _) => score = f32::from_le_bytes(f.try_into().unwrap()),
                (3, 0, _, v) => ty = v as i32,
                _ => {}
            }
        }
        out.push((text, score, ty));
    }
    if out.is_empty() {
        return Err(bad("tokenizer.model: no pieces"));
    }
    Ok(out)
}

/// llama.cpp's `_set_vocab_sentencepiece` (`llama` tokenizer): the pieces'
/// text, scores and types (user-defined pieces count as normal, as there),
/// then `added_tokens.json` and `added_tokens_decoder` (CONTROL when special
/// or looking special), padded to `vocab_size`.
fn spm_tokenizer(c: &Checkpoint, cfg: &serde_json::Map<String, Json>, vocab_size: Option<u64>, meta: &mut Vec<(String, Value)>) -> Result<()> {
    let pieces = spm_pieces(&std::fs::read(c.dir.join("tokenizer.model"))?)?;
    let n = vocab_size.map_or(pieces.len(), |v| v as usize);
    if n > 4 << 20 {
        return Err(bad(format!("vocabulary of {n} tokens")));
    }
    let mut tokens: Vec<(String, f32, i32)> = (0..n).map(|i| (format!("[PAD{i}]"), -10_000.0, UNUSED)).collect();
    for (i, (text, score, ty)) in pieces.into_iter().enumerate().take(n) {
        let ty = match ty {
            2 => 2,
            3 => CONTROL,
            5 => UNUSED,
            6 => BYTE,
            _ => NORMAL,
        };
        tokens[i] = (text, score, ty);
    }
    if let Some(Json::Object(a)) = c.json("added_tokens.json")? {
        for (text, id) in a {
            if let Some(id) = id.as_u64().map(|v| v as usize).filter(|&v| v < n) {
                tokens[id] = (text.clone(), -1000.0, USER_DEFINED);
            }
        }
    }
    let tc = c.json("tokenizer_config.json")?.unwrap_or(Json::Null);
    for (id, a) in tc.get("added_tokens_decoder").and_then(Json::as_object).into_iter().flatten() {
        let (Ok(id), Some(content)) = (id.parse::<usize>(), a.get("content").and_then(Json::as_str)) else { continue };
        if id >= n {
            continue;
        }
        tokens[id] = if a.get("special").and_then(Json::as_bool).unwrap_or(false) || looks_special(content) {
            (content.to_string(), -1000.0, CONTROL)
        } else {
            (content.replace('\u{2581}', " "), -1000.0, USER_DEFINED)
        };
    }
    let (mut names, mut scores, mut types) = (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
    for (t, s, ty) in tokens {
        names.push(Value::String(t));
        scores.push(Value::F32(s));
        types.push(Value::I32(ty));
    }
    meta.push(("tokenizer.ggml.model".into(), Value::String("llama".into())));
    meta.push(("tokenizer.ggml.pre".into(), Value::String("default".into())));
    meta.push(("tokenizer.ggml.tokens".into(), Value::Array(names)));
    meta.push(("tokenizer.ggml.scores".into(), Value::Array(scores)));
    meta.push(("tokenizer.ggml.token_type".into(), Value::Array(types)));
    if let Some(p) = tc.get("add_prefix_space").and_then(Json::as_bool) {
        meta.push(("tokenizer.ggml.add_space_prefix".into(), Value::Bool(p)));
    }
    let tj = c.json("tokenizer.json")?.unwrap_or(Json::Null);
    let added: Vec<(u64, String, bool)> = tj
        .get("added_tokens")
        .and_then(Json::as_array)
        .into_iter()
        .flatten()
        .filter_map(|a| Some((a.get("id")?.as_u64()?, a.get("content")?.as_str()?.to_string(), a.get("special").and_then(Json::as_bool).unwrap_or(false))))
        .collect();
    special_vocab(c, cfg, &tj, &tc, &added, Some(n), meta)
}

/// Gemma 4's tokenizer keys (llama.cpp's `LlamaHfVocab` for `gemma4`): the
/// vocabulary as stored, scores of -1000, CONTROL for transformers'
/// `all_special_ids`, BYTE for `<0xNN>`, a few tool/channel markers forced
/// to USER_DEFINED; BOS always added.
fn gemma4_tokenizer(c: &Checkpoint, cfg: &serde_json::Map<String, Json>, meta: &mut Vec<(String, Value)>) -> Result<()> {
    let tj = c.json("tokenizer.json")?.ok_or_else(|| bad("no tokenizer.json"))?;
    let tc = c.json("tokenizer_config.json")?.unwrap_or(Json::Null);
    let model = tj.get("model").ok_or_else(|| bad("tokenizer.json: no model"))?;
    let vocab = model.get("vocab").and_then(Json::as_object).ok_or_else(|| bad("tokenizer.json: no vocab"))?;
    let added: Vec<(u64, String, bool)> = tj
        .get("added_tokens")
        .and_then(Json::as_array)
        .into_iter()
        .flatten()
        .filter_map(|a| Some((a.get("id")?.as_u64()?, a.get("content")?.as_str()?.to_string(), a.get("special").and_then(Json::as_bool).unwrap_or(false))))
        .collect();
    let base = vocab.len();
    let n = vocab.values().filter_map(Json::as_u64).chain(added.iter().map(|a| a.0)).max().map_or(0, |m| m as usize + 1).max(base);
    if n > 4 << 20 {
        return Err(bad(format!("vocabulary of {n} tokens")));
    }
    let mut tokens: Vec<Option<String>> = vec![None; n];
    for (t, id) in vocab {
        tokens[id.as_u64().ok_or_else(|| bad("tokenizer.json: non-integer vocab id"))? as usize] = Some(t.clone());
    }
    // Added tokens win over base entries of the same id (transformers' get_vocab).
    for (id, t, _) in &added {
        tokens[*id as usize] = Some(t.clone());
    }
    // transformers' all_special_ids: every named `*_token` (bos, eos, …, and
    // model-specific ones such as Gemma 4's `sot_token`) and the extra ones.
    let text = |v: &Json| v.as_str().map(str::to_owned).or_else(|| v.get("content").and_then(Json::as_str).map(str::to_owned));
    let mut special: Vec<String> = tc.as_object().into_iter().flatten().filter(|(k, _)| k.ends_with("_token")).filter_map(|(_, v)| text(v)).collect();
    for key in ["additional_special_tokens", "extra_special_tokens"] {
        match tc.get(key) {
            Some(Json::Array(a)) => special.extend(a.iter().filter_map(text)),
            Some(Json::Object(o)) => special.extend(o.values().filter_map(text)),
            _ => {}
        }
    }
    let id_of: HashMap<&str, usize> = tokens.iter().enumerate().filter_map(|(i, t)| Some((t.as_deref()?, i))).collect();
    let special_ids: std::collections::HashSet<usize> = special.iter().filter_map(|s| id_of.get(s.as_str()).copied()).collect();
    const FORCED: [&str; 7] = ["<|channel>", "<channel|>", "<|tool_call>", "<tool_call|>", "<|tool_response>", "<tool_response|>", "<|\"|>"];
    let is_byte = |s: &str| s.len() == 6 && s.starts_with("<0x") && s.ends_with('>') && s[3..5].chars().all(|c| c.is_ascii_hexdigit());
    let (names, types): (Vec<Value>, Vec<Value>) = tokens
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let s = t.clone().unwrap_or_else(|| format!("[PAD{i}]"));
            let ty = if t.is_none() {
                UNUSED
            } else if FORCED.contains(&s.as_str()) {
                USER_DEFINED
            } else if is_byte(&s) {
                BYTE
            } else if special_ids.contains(&i) {
                CONTROL
            } else if i >= base {
                USER_DEFINED
            } else {
                NORMAL
            };
            (Value::String(s), Value::I32(ty))
        })
        .unzip();
    let merges: Vec<Value> = model
        .get("merges")
        .and_then(Json::as_array)
        .into_iter()
        .flatten()
        .filter_map(|m| match m {
            Json::String(s) => Some(s.clone()),
            Json::Array(p) if p.len() == 2 => Some(format!("{} {}", p[0].as_str()?, p[1].as_str()?)),
            _ => None,
        })
        .map(Value::String)
        .collect();
    meta.push(("tokenizer.ggml.model".into(), Value::String("gemma4".into())));
    meta.push(("tokenizer.ggml.scores".into(), Value::Array(vec![Value::F32(-1000.0); names.len()])));
    meta.push(("tokenizer.ggml.tokens".into(), Value::Array(names)));
    meta.push(("tokenizer.ggml.token_type".into(), Value::Array(types)));
    meta.push(("tokenizer.ggml.merges".into(), Value::Array(merges)));
    for (typ, key) in [("bos", "bos_token_id"), ("eos", "eos_token_id"), ("unk", "unknown_token_id"), ("pad", "padding_token_id"), ("mask", "mask_token_id")] {
        let id = tc.get(format!("{typ}_token")).and_then(text).and_then(|s| id_of.get(s.as_str()).copied()).or_else(|| int(cfg, &[&format!("{typ}_token_id")]).map(|v| v as usize));
        if let Some(id) = id {
            meta.push((format!("tokenizer.ggml.{key}"), Value::U32(id as u32)));
        }
    }
    meta.push(("tokenizer.ggml.add_bos_token".into(), Value::Bool(true)));
    meta.push(("tokenizer.ggml.add_space_prefix".into(), Value::Bool(false)));
    let template = match tc.get("chat_template") {
        Some(Json::String(s)) => Some(s.clone()),
        _ => {
            let p = c.dir.join("chat_template.jinja");
            p.is_file().then(|| std::fs::read_to_string(p)).transpose()?
        }
    };
    if let Some(t) = template {
        meta.push(("tokenizer.chat_template".into(), Value::String(t)));
    }
    Ok(())
}

/// HF → GGUF names for Qwen3 (llama.cpp's tensor_mapping).
fn qwen3_name(hf: &str) -> Option<String> {
    Some(match hf {
        "model.embed_tokens.weight" => "token_embd.weight".into(),
        "model.norm.weight" => "output_norm.weight".into(),
        "lm_head.weight" => "output.weight".into(),
        _ => {
            let rest = hf.strip_prefix("model.layers.")?;
            let (n, part) = rest.split_once('.')?;
            let n: usize = n.parse().ok()?;
            let g = match part {
                "input_layernorm.weight" => "attn_norm.weight",
                "self_attn.q_proj.weight" => "attn_q.weight",
                "self_attn.k_proj.weight" => "attn_k.weight",
                "self_attn.v_proj.weight" => "attn_v.weight",
                "self_attn.o_proj.weight" => "attn_output.weight",
                "self_attn.q_norm.weight" => "attn_q_norm.weight",
                "self_attn.k_norm.weight" => "attn_k_norm.weight",
                "post_attention_layernorm.weight" => "ffn_norm.weight",
                "mlp.gate_proj.weight" => "ffn_gate.weight",
                "mlp.up_proj.weight" => "ffn_up.weight",
                "mlp.down_proj.weight" => "ffn_down.weight",
                _ => return None,
            };
            format!("blk.{n}.{g}")
        }
    })
}

/// llama.cpp's `looks_special`: names that are control tokens by form.
fn looks_special(t: &str) -> bool {
    matches!(t, "<pad>" | "<mask>" | "<2mass>" | "[@BOS@]")
        || (t.starts_with("<|") && t.ends_with("|>"))
        || (t.starts_with("<｜") && t.ends_with("｜>"))
        || t.starts_with("<unused")
}

/// The GPT-2/BPE tokenizer keys from tokenizer.json and
/// tokenizer_config.json (llama.cpp's `_set_vocab_gpt2` and `SpecialVocab`).
fn tokenizer(c: &Checkpoint, cfg: &serde_json::Map<String, Json>, vocab_size: Option<u64>, pre: &str, meta: &mut Vec<(String, Value)>) -> Result<()> {
    let tj = c.json("tokenizer.json")?.ok_or_else(|| bad("no tokenizer.json"))?;
    let tc = c.json("tokenizer_config.json")?.unwrap_or(Json::Null);
    let model = tj.get("model").ok_or_else(|| bad("tokenizer.json: no model"))?;
    if model.get("type").and_then(Json::as_str) != Some("BPE") {
        return Err(bad("tokenizer.json: only BPE tokenizers are supported"));
    }
    let vocab = model.get("vocab").and_then(Json::as_object).ok_or_else(|| bad("tokenizer.json: no vocab"))?;
    // Added tokens: tokenizer.json's, and tokenizer_config.json's
    // added_tokens_decoder (transformers merges both; some, e.g. Qwen3.5's
    // audio tokens, are only in the latter).
    let mut added: Vec<(u64, String, bool)> = tj
        .get("added_tokens")
        .and_then(Json::as_array)
        .into_iter()
        .flatten()
        .filter_map(|a| Some((a.get("id")?.as_u64()?, a.get("content")?.as_str()?.to_string(), a.get("special").and_then(Json::as_bool).unwrap_or(false))))
        .collect();
    for (id, a) in tc.get("added_tokens_decoder").and_then(Json::as_object).into_iter().flatten() {
        let (Ok(id), Some(content)) = (id.parse::<u64>(), a.get("content").and_then(Json::as_str)) else { continue };
        if !added.iter().any(|x| x.0 == id) {
            added.push((id, content.to_string(), a.get("special").and_then(Json::as_bool).unwrap_or(false)));
        }
    }
    let max_id = vocab.values().filter_map(Json::as_u64).chain(added.iter().map(|a| a.0)).max().unwrap_or(0);
    let n = vocab_size.unwrap_or(max_id + 1).max(max_id + 1) as usize;
    if n > 4 << 20 {
        return Err(bad(format!("vocabulary of {n} tokens")));
    }
    let mut tokens: Vec<Option<(String, i32)>> = vec![None; n];
    for (t, id) in vocab {
        let id = id.as_u64().ok_or_else(|| bad("tokenizer.json: non-integer vocab id"))? as usize;
        tokens[id] = Some((t.clone(), NORMAL));
    }
    for (id, t, special) in &added {
        let ty = if *special || looks_special(t) { CONTROL } else { USER_DEFINED };
        tokens[*id as usize] = Some((t.clone(), ty));
    }
    let (names, types): (Vec<Value>, Vec<Value>) = tokens
        .into_iter()
        .enumerate()
        .map(|(i, t)| match t {
            Some((s, ty)) => (Value::String(s), Value::I32(ty)),
            None => (Value::String(format!("[PAD{i}]")), Value::I32(UNUSED)),
        })
        .unzip();
    // Merges: "a b" strings, or [a, b] pairs (spaces inside a part as Ġ).
    let merges: Vec<Value> = model
        .get("merges")
        .and_then(Json::as_array)
        .into_iter()
        .flatten()
        .filter_map(|m| match m {
            Json::String(s) => Some(s.clone()),
            Json::Array(p) if p.len() == 2 => {
                let part = |v: &Json| v.as_str().map(|s| s.replace(' ', "\u{0120}"));
                Some(format!("{} {}", part(&p[0])?, part(&p[1])?))
            }
            _ => None,
        })
        .map(Value::String)
        .collect();
    meta.push(("tokenizer.ggml.model".into(), Value::String("gpt2".into())));
    meta.push(("tokenizer.ggml.pre".into(), Value::String(pre.into())));
    meta.push(("tokenizer.ggml.tokens".into(), Value::Array(names)));
    meta.push(("tokenizer.ggml.token_type".into(), Value::Array(types)));
    meta.push(("tokenizer.ggml.merges".into(), Value::Array(merges)));
    special_vocab(c, cfg, &tj, &tc, &added, None, meta)
}

/// llama.cpp's `SpecialVocab`: bos/eos/pad ids (tokenizer_config's token,
/// found among the added tokens, else config.json's id; ids at or past
/// `n_vocab` skipped), `add_bos_token`, and the chat template.
fn special_vocab(c: &Checkpoint, cfg: &serde_json::Map<String, Json>, tj: &Json, tc: &Json, added: &[(u64, String, bool)], n_vocab: Option<usize>, meta: &mut Vec<(String, Value)>) -> Result<()> {
    // Special tokens: tokenizer_config's string (or {content}), found among
    // the added tokens; else config.json's id.
    let by_content: HashMap<&str, u64> = added.iter().map(|(id, t, _)| (t.as_str(), *id)).collect();
    for (typ, key) in [("bos", "bos_token_id"), ("eos", "eos_token_id"), ("pad", "padding_token_id")] {
        let from_tc = tc
            .get(format!("{typ}_token"))
            .and_then(|v| v.as_str().map(str::to_owned).or_else(|| v.get("content").and_then(Json::as_str).map(str::to_owned)))
            .and_then(|s| by_content.get(s.as_str()).copied());
        if let Some(id) = from_tc.or_else(|| int(cfg, &[&format!("{typ}_token_id")])).filter(|&id| n_vocab.is_none_or(|n| (id as usize) < n)) {
            meta.push((format!("tokenizer.ggml.{key}"), Value::U32(id as u32)));
        }
    }
    // The post-processor decides (llama.cpp's SpecialVocab): ByteLevel adds
    // nothing; a TemplateProcessing (alone or in a Sequence) adds the BOS if
    // its single-sequence template starts with it; else tokenizer_config.
    let pp = tj.get("post_processor").cloned().unwrap_or(Json::Null);
    let procs: Vec<Json> = match pp.get("type").and_then(Json::as_str) {
        Some("Sequence") => pp.get("processors").and_then(Json::as_array).cloned().unwrap_or_default(),
        _ => vec![pp.clone()],
    };
    let template = procs.iter().find(|p| p.get("type").and_then(Json::as_str) == Some("TemplateProcessing"));
    let bos_text = tc.get("bos_token").and_then(|v| v.as_str().map(str::to_owned).or_else(|| v.get("content").and_then(Json::as_str).map(str::to_owned)));
    let add_bos = match template {
        Some(t) => {
            let first = t.get("single").and_then(Json::as_array).and_then(|s| s.first()).and_then(|f| f.get("SpecialToken")).and_then(|s| s.get("id")).and_then(Json::as_str);
            first.is_some() && first == bos_text.as_deref()
        }
        None if procs.iter().any(|p| p.get("type").and_then(Json::as_str) == Some("ByteLevel")) => false,
        None => tc.get("add_bos_token").and_then(Json::as_bool).unwrap_or(false),
    };
    meta.push(("tokenizer.ggml.add_bos_token".into(), Value::Bool(add_bos)));
    let template = match tc.get("chat_template") {
        Some(Json::String(s)) => Some(s.clone()),
        _ => {
            let p = c.dir.join("chat_template.jinja");
            p.is_file().then(|| std::fs::read_to_string(p)).transpose()?
        }
    };
    if let Some(t) = template {
        meta.push(("tokenizer.chat_template".into(), Value::String(t)));
    }
    Ok(())
}

/// Open a model: a GGUF file, or a safetensors checkpoint (a directory with
/// config.json, or a `.safetensors` file next to one) seen as GGUF.
/// Quantizing at load as `RLLM_QUANTIZE` says (see [`Options::from_env`]).
pub fn open_model(path: impl AsRef<std::path::Path>) -> Result<Gguf> {
    open_model_with(path, Options::from_env()?)
}

/// [`open_model`] with options (they apply to safetensors checkpoints; a
/// GGUF file is read as it is).
pub fn open_model_with(path: impl AsRef<std::path::Path>, opts: Options) -> Result<Gguf> {
    let path = path.as_ref();
    let safetensors = path.is_dir() || path.extension().is_some_and(|e| e == "safetensors");
    if safetensors { to_gguf_with(Checkpoint::open(path)?, opts) } else { Ok(Gguf::open(path)?) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_and_halves() {
        assert_eq!(qwen3_name("model.layers.12.self_attn.k_norm.weight").as_deref(), Some("blk.12.attn_k_norm.weight"));
        assert_eq!(qwen3_name("model.embed_tokens.weight").as_deref(), Some("token_embd.weight"));
        assert!(qwen3_name("model.layers.x.mlp.up_proj.weight").is_none());
        for v in [0.0f32, 1.0, -2.5, 65504.0, 6.1035156e-5, 5.9604645e-8] {
            let h = half_bits(v);
            assert_eq!(f16_to_f32(h), v, "{v}");
        }
        assert!(looks_special("<|im_start|>") && !looks_special("<think>"));
    }

    /// llama.cpp's reorder: heads grouped by key head (h = k * r + j) to
    /// tiled (h' = j * nk + k).
    #[test]
    fn sentencepiece_pieces_parse() {
        // ModelProto { pieces: [{piece "<unk>", score 0, type UNKNOWN}, {piece "▁a", score -1.5}], trainer_spec: {...} }
        fn piece(text: &str, score: f32, ty: Option<u8>) -> Vec<u8> {
            let mut p = vec![0x0a, text.len() as u8];
            p.extend_from_slice(text.as_bytes());
            p.push(0x15);
            p.extend_from_slice(&score.to_le_bytes());
            if let Some(t) = ty {
                p.extend_from_slice(&[0x18, t]);
            }
            let mut out = vec![0x0a, p.len() as u8];
            out.extend(p);
            out
        }
        let mut m = piece("<unk>", 0.0, Some(2));
        m.extend(piece("\u{2581}a", -1.5, None));
        m.extend_from_slice(&[0x12, 0x02, 0x08, 0x01]); // another field: skipped
        let p = spm_pieces(&m).unwrap();
        assert_eq!(p, vec![("<unk>".to_string(), 0.0, 2), ("\u{2581}a".to_string(), -1.5, 1)]);
        assert!(spm_pieces(&m[..m.len() - 5]).is_err());
    }

    #[test]
    fn clef_head_names_are_llama_cpp_s() {
        let n = |s: &str| match clef_head_name(s, 2) {
            Some(Head::Tensor(g)) => g,
            Some(Head::InProj(p, x)) => format!("{p}_{{q,k,v}}.{x}"),
            Some(Head::Scalar(s)) => format!("scalar {s}"),
            None => "none".into(),
        };
        assert_eq!(n("joint_head.layers.0.norm1.weight"), "dec.blk.2.attn_norm.weight");
        assert_eq!(n("joint_head.evidence_layers.1.feedforward.3.bias"), "dec.blk.1.ffn_down.bias");
        assert_eq!(n("joint_head.layers.3.multihead_attn.in_proj_weight"), "dec.blk.5.cross_attn_{q,k,v}.weight");
        assert_eq!(n("joint_head.layers.3.self_attn.in_proj_bias"), "dec.blk.5.attn_{q,k,v}.bias");
        assert_eq!(n("joint_head.evidence_layers.0.attention.out_proj.weight"), "dec.blk.0.cross_attn_o.weight");
        assert_eq!(n("joint_head.residual_scorer.3.weight"), "decision.scorer_out.weight");
        assert_eq!(n("joint_head.type_embedding.weight"), "token_types.weight");
        assert_eq!(n("joint_head.residual_gate"), "scalar residual_gate");
    }

    #[test]
    fn systemone_template_matches_llama_cpp_s() {
        let t = clef_systemone_template();
        assert!(t.starts_with(r#"{{ "<|im_start|>system\nRead the complete state"#), "{}", &t[..80]);
        assert!(t.ends_with(r#"{{ "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:" }}"#));
        assert!(t.contains("{{ q.instructions if q.instructions is string else q.instructions | tojson(separators=[',', ':']) }}"));
    }

    #[test]
    fn qk_permute_is_llama_cpp_s_and_rllm_undoes_it() {
        // 2 heads of 4 rows, one byte per row: per head [0,1,2,3] -> [0,2,1,3].
        let rows: Vec<u8> = (0..8).collect();
        assert_eq!(permute_qk(&rows, 8, 1, 2), vec![0, 2, 1, 3, 4, 6, 5, 7]);
    }

    #[test]
    fn value_heads_are_tiled() {
        let (nk, r) = (2, 3);
        // Head h holds the byte h.
        let mut b: Vec<u8> = (0..(nk * r) as u8).collect();
        tile_heads(&mut b, 1, nk, r);
        // h' = j * nk + k takes old h = k * r + j.
        assert_eq!(b, vec![0, 3, 1, 4, 2, 5]);
        // Value head h' then reads key head h' % nk: old head k*r+j belonged to key head k.
        for (hp, &old) in b.iter().enumerate() {
            assert_eq!(hp % nk, old as usize / r);
        }
    }

    /// f16 bits of exactly representable values (test helper).
    fn half_bits(v: f32) -> u16 {
        if v == 0.0 {
            return 0;
        }
        let b = v.to_bits();
        let (s, e, m) = ((b >> 31) as u16, ((b >> 23) & 0xff) as i32 - 127, b & 0x7f_ffff);
        if e < -14 {
            // Subnormal: value = m' * 2^-24.
            let m = ((m | 0x80_0000) >> (-e - 14 + 13)) as u16;
            return (s << 15) | m;
        }
        (s << 15) | (((e + 15) as u16) << 10) | (m >> 13) as u16
    }
}
