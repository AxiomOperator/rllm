//! Qwen3.5 / 3.6 / 3.8 hybrids: dense (GGUF architecture `qwen35`) and
//! mixture-of-experts (`qwen35moe`, e.g. Qwen3.6-35B-A3B).
//!
//! Three of every four layers are Gated DeltaNet (linear attention with a
//! recurrent state per sequence); every fourth is gated full attention.
//! Every layer has a SwiGLU MLP: dense, or (MoE) the top-k of `n_expert`
//! routed experts (softmax router, weights renormalized over the chosen
//! ones) plus a shared expert scaled by `sigmoid(x · w_gate)`.
//!
//! - **DeltaNet layer:** RMSNorm → `qkv` and `z` projections, `alpha` and
//!   `beta` → causal conv (kernel 4) + SiLU over q|k|v → per head
//!   `S = g S + k (b (v - Sᵀk))ᵀ`, `o = Sᵀq` with L2-normalized q, k →
//!   RMSNorm(o) · SiLU(z) → output projection.
//! - **Attention layer:** RMSNorm → Q (with a per-head output gate packed
//!   in the same projection), K, V → per-head RMSNorm on Q and K → RoPE on
//!   the first `rope.dimension_count` dims of each head → GQA attention →
//!   `· sigmoid(gate)` → output projection.
//! - Then `x += layer`, `x += MLP(RMSNorm(x))`.
//!
//! RoPE: Qwen3.5 uses multi-section RoPE, which for text (the same
//! position in every section) is NeoX RoPE over pairs `(i, i + n_rot/2)`,
//! `i < n_rot/2`, at `theta^(-2i/n_rot)`. rllm's RoPE rotates pairs
//! `(i, i + head_dim/2)` at `theta^(-2i/head_dim) / factor[i]`, so Q and K
//! dims are reordered on load (rotated pairs moved to `(i, i + head_dim/2)`,
//! the rest after them; attention only sees q·k, so a shared order is
//! free) and the factors set the frequencies, or ~no rotation for the rest.
//!
//! MTP: files with `nextn_predict_layers` carry one extra block that
//! predicts the token after next from the main model's last hidden state
//! and the next token's embedding (pair `(h_p, x_{p+1})` at position
//! `p + 1`). With MTP enabled every step also runs that block's K/V over
//! the step's tokens (so its attention sees the whole sequence), and
//! drafting chains MTP steps. Verification steps snapshot the recurrent
//! state after each token so rejected drafts can be rolled back.
//! Recurrent state can't be rebuilt from cached KV, so prefix reuse is off.

use anyhow::{Context, Result, bail, ensure};
use rllm_core::{Activation, AttnMeta, Backend, GdnShape, HostTensor, KvLayout, KvType, MoeFfn, SeqRuns};
use rllm_gguf::Gguf;

use crate::loader::{
    MAX_CONTEXT, MAX_DIM, check_shape, check_shape_opt, load_weight, load_weight_cols, load_weight_rows, meta_f32_or, meta_usize, meta_usize_or, read_vector,
};
use crate::{Batch, Model, MtpBatch, StepOut};

/// The expert layout of a `qwen35moe` model.
#[derive(Debug, Clone, Copy)]
pub struct MoeConfig {
    pub n_expert: usize,
    /// Experts used per token.
    pub n_used: usize,
    /// Hidden width of one routed expert.
    pub n_ff: usize,
    /// Hidden width of the shared expert (0: none).
    pub n_ff_shared: usize,
    /// Multiplier on the routing weights.
    pub scale: f32,
}

#[derive(Debug, Clone)]
pub struct Qwen35Config {
    /// `qwen35` or `qwen35moe`.
    pub arch: String,
    /// Decoder layers run in a forward pass (MTP blocks excluded).
    pub n_layers: usize,
    pub hidden: usize,
    /// Dense MLP width (MoE: the shared expert's).
    pub ffn: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    /// Rotated dims per head (the rest are not rotated).
    pub n_rot: usize,
    pub vocab: usize,
    pub rope_theta: f32,
    pub rms_eps: f32,
    pub max_context: usize,
    /// Per layer: true for Gated DeltaNet, false for full attention.
    pub recurrent: Vec<bool>,
    pub conv_kernel: usize,
    pub gdn: GdnShape,
    /// The file carries an MTP block (`blk.{n_layers}`).
    pub mtp: bool,
    /// Routed experts (`qwen35moe`).
    pub moe: Option<MoeConfig>,
}

impl Qwen35Config {
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        let arch = g.get_str("general.architecture")?;
        // Clef (a decision model) is a dense Qwen3.5 under its own name.
        if arch != "qwen35" && arch != "qwen35moe" && arch != "clef" {
            bail!("unsupported architecture {arch:?} (expected qwen35, qwen35moe or clef)");
        }
        let k = |s: &str| format!("{arch}.{s}");
        let dim = |s: &str| meta_usize(g, &k(s), MAX_DIM, false);
        let blocks = dim("block_count")?;
        let nextn = meta_usize_or(g, &k("nextn_predict_layers"), 0, 16, true)?;
        let n_layers = blocks.checked_sub(nextn).filter(|&n| n > 0).context("nextn_predict_layers must be smaller than block_count")?;
        let head_dim = dim("attention.key_length")?;
        let v_len = meta_usize_or(g, &k("attention.value_length"), head_dim, MAX_DIM, false)?;
        ensure!(v_len == head_dim, "attention.value_length {v_len} differs from key_length {head_dim}: not supported");
        let embd = g.require_tensor("token_embd.weight")?;
        let [_, vocab] = embd.dims[..] else { bail!("token_embd.weight has shape {:?}, expected 2 dims", embd.dims) };
        let vocab = usize::try_from(vocab).ok().filter(|&v| (1..=MAX_DIM).contains(&v)).context("token_embd.weight: vocabulary size out of range")?;
        if let Some(t) = g.get(&k("rope.scaling.type")).and_then(|v| v.as_str())
            && t != "none"
        {
            bail!("RoPE scaling {t:?} is not supported");
        }
        let interval = meta_usize_or(g, &k("full_attention_interval"), 4, 1024, false)?;
        let recurrent = match g.get(&k("attention.recurrent_layers")).and_then(|v| v.as_array()) {
            Some(a) => {
                ensure!(a.len() >= n_layers, "attention.recurrent_layers has {} entries for {n_layers} layers", a.len());
                a[..n_layers]
                    .iter()
                    .map(|v| match v {
                        rllm_gguf::Value::Bool(b) => Ok(*b),
                        v => v.as_u64().map(|x| x != 0).context("attention.recurrent_layers: expected bools or integers"),
                    })
                    .collect::<Result<Vec<_>>>()?
            }
            None => (0..n_layers).map(|i| (i + 1) % interval != 0).collect(),
        };
        let state = dim("ssm.state_size")?;
        let n_v_heads = dim("ssm.time_step_rank")?;
        let inner = dim("ssm.inner_size")?;
        ensure!(inner == n_v_heads * state, "ssm.inner_size {inner} is not time_step_rank {n_v_heads} x state_size {state}");
        let moe = if arch == "qwen35moe" {
            let n_expert = dim("expert_count")?;
            let n_used = dim("expert_used_count")?;
            let n_ff = dim("expert_feed_forward_length")?;
            let n_ff_shared = meta_usize_or(g, &k("expert_shared_feed_forward_length"), 0, MAX_DIM, true)?;
            // 0 (llama.cpp's default) means no scaling.
            let scale = match g.get(&k("expert_weights_scale")).and_then(|v| v.as_f64()) {
                Some(s) if s != 0.0 => s as f32,
                _ => 1.0,
            };
            Some(MoeConfig { n_expert, n_used, n_ff, n_ff_shared, scale })
        } else {
            None
        };
        let ffn = match moe {
            Some(m) => m.n_ff_shared,
            None => dim("feed_forward_length")?,
        };
        let cfg = Self {
            arch: arch.to_string(),
            n_layers,
            hidden: dim("embedding_length")?,
            ffn,
            n_heads: dim("attention.head_count")?,
            n_kv_heads: dim("attention.head_count_kv")?,
            head_dim,
            n_rot: meta_usize_or(g, &k("rope.dimension_count"), head_dim, MAX_DIM, false)?,
            vocab,
            rope_theta: meta_f32_or(g, &k("rope.freq_base"), 10_000_000.0)?,
            rms_eps: meta_f32_or(g, &k("attention.layer_norm_rms_epsilon"), 1e-6)?,
            max_context: meta_usize_or(g, &k("context_length"), 32768, MAX_CONTEXT, false)?,
            recurrent,
            conv_kernel: dim("ssm.conv_kernel")?,
            gdn: GdnShape { n_k_heads: dim("ssm.group_count")?, n_v_heads, head_dim: state },
            mtp: nextn > 0 && g.tensor(&format!("blk.{n_layers}.nextn.eh_proj.weight")).is_some(),
            moe,
        };
        cfg.validate()?;
        cfg.check_tensors(g)?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<()> {
        let c = self;
        ensure!(c.n_heads.is_multiple_of(c.n_kv_heads), "head_count {} is not a multiple of head_count_kv {}", c.n_heads, c.n_kv_heads);
        ensure!(c.head_dim.is_multiple_of(2) && c.n_rot.is_multiple_of(2) && c.n_rot <= c.head_dim, "RoPE over {} of {} dims", c.n_rot, c.head_dim);
        ensure!(c.gdn.n_v_heads.is_multiple_of(c.gdn.n_k_heads), "ssm.time_step_rank {} is not a multiple of ssm.group_count {}", c.gdn.n_v_heads, c.gdn.n_k_heads);
        ensure!((2..=16).contains(&c.conv_kernel), "ssm.conv_kernel {} out of range", c.conv_kernel);
        ensure!(c.gdn.head_dim <= 256, "ssm.state_size {} above 256 is not supported", c.gdn.head_dim);
        ensure!(c.recurrent.iter().any(|r| !r), "no full-attention layer");
        ensure!(c.n_layers <= 4096 && c.n_heads <= 4096 && c.head_dim <= 4096, "implausible model size: {c:?}");
        ensure!(c.rope_theta.is_finite() && c.rope_theta > 0.0 && c.rms_eps.is_finite() && c.rms_eps > 0.0, "invalid RoPE base or RMS epsilon");
        if let Some(m) = c.moe {
            ensure!(m.n_used >= 1 && m.n_used <= m.n_expert && m.n_used <= 32, "{} of {} experts per token not supported", m.n_used, m.n_expert);
            ensure!(m.n_expert <= 1024, "{} experts not supported (at most 1024)", m.n_expert);
            ensure!(m.scale.is_finite(), "invalid expert_weights_scale");
        }
        Ok(())
    }

    /// GGUF names and dims (innermost first) of a decoder block's MLP tensors.
    fn ffn_tensors(&self, g: &Gguf, p: &dyn Fn(&str) -> String) -> Vec<(String, Vec<usize>)> {
        let d = self.hidden;
        match self.moe {
            None => vec![
                (p("ffn_gate.weight"), vec![d, self.ffn]),
                (p("ffn_up.weight"), vec![d, self.ffn]),
                (p("ffn_down.weight"), vec![self.ffn, d]),
            ],
            Some(m) => {
                let (e, f) = (m.n_expert, m.n_ff);
                let mut v = vec![(p("ffn_gate_inp.weight"), vec![d, e]), (p("ffn_down_exps.weight"), vec![f, d, e])];
                if g.tensor(&p("ffn_gate_up_exps.weight")).is_some() {
                    v.push((p("ffn_gate_up_exps.weight"), vec![d, 2 * f, e]));
                } else {
                    v.push((p("ffn_gate_exps.weight"), vec![d, f, e]));
                    v.push((p("ffn_up_exps.weight"), vec![d, f, e]));
                }
                if m.n_ff_shared > 0 {
                    let s = m.n_ff_shared;
                    v.extend([
                        (p("ffn_gate_inp_shexp.weight"), vec![d]),
                        (p("ffn_gate_shexp.weight"), vec![d, s]),
                        (p("ffn_up_shexp.weight"), vec![d, s]),
                        (p("ffn_down_shexp.weight"), vec![s, d]),
                    ]);
                }
                v
            }
        }
    }

    fn check_tensors(&self, g: &Gguf) -> Result<()> {
        let c = self;
        let (d, hd) = (c.hidden, c.head_dim);
        let (qw, kvw) = (c.n_heads * hd, c.n_kv_heads * hd);
        let (conv, inner) = (c.gdn.qkv_width(), c.gdn.n_v_heads * c.gdn.head_dim);
        check_shape(g, "token_embd.weight", &[d, c.vocab])?;
        check_shape(g, "output_norm.weight", &[d])?;
        check_shape_opt(g, "output.weight", &[d, c.vocab])?;
        for (i, &rec) in c.recurrent.iter().enumerate() {
            let p = |s: &str| format!("blk.{i}.{s}");
            let mut want: Vec<(String, Vec<usize>)> = vec![(p("attn_norm.weight"), vec![d]), (p("post_attention_norm.weight"), vec![d])];
            want.extend(c.ffn_tensors(g, &p));
            if rec {
                let hv = c.gdn.n_v_heads;
                want.extend([
                    (p("attn_qkv.weight"), vec![d, conv]),
                    (p("attn_gate.weight"), vec![d, inner]),
                    (p("ssm_conv1d.weight"), vec![c.conv_kernel, conv]),
                    (p("ssm_dt.bias"), vec![hv]),
                    (p("ssm_a"), vec![hv]),
                    (p("ssm_alpha.weight"), vec![d, hv]),
                    (p("ssm_beta.weight"), vec![d, hv]),
                    (p("ssm_norm.weight"), vec![c.gdn.head_dim]),
                    (p("ssm_out.weight"), vec![inner, d]),
                ]);
            } else {
                want.extend([
                    (p("attn_q.weight"), vec![d, 2 * qw]),
                    (p("attn_k.weight"), vec![d, kvw]),
                    (p("attn_v.weight"), vec![d, kvw]),
                    (p("attn_output.weight"), vec![qw, d]),
                    (p("attn_q_norm.weight"), vec![hd]),
                    (p("attn_k_norm.weight"), vec![hd]),
                ]);
            }
            for (name, dims) in &want {
                check_shape(g, name, dims)?;
            }
        }
        if c.mtp {
            let p = |s: &str| format!("blk.{}.{s}", c.n_layers);
            for (name, dims) in [
                (p("nextn.eh_proj.weight"), vec![2 * d, d]),
                (p("nextn.enorm.weight"), vec![d]),
                (p("nextn.hnorm.weight"), vec![d]),
                (p("attn_norm.weight"), vec![d]),
                (p("post_attention_norm.weight"), vec![d]),
                (p("attn_q.weight"), vec![d, 2 * qw]),
                (p("attn_k.weight"), vec![d, kvw]),
                (p("attn_v.weight"), vec![d, kvw]),
                (p("attn_output.weight"), vec![qw, d]),
                (p("attn_q_norm.weight"), vec![hd]),
                (p("attn_k_norm.weight"), vec![hd]),
            ]
            .into_iter()
            .chain(c.ffn_tensors(g, &p))
            {
                check_shape(g, &name, &dims)?;
            }
            check_shape_opt(g, &p("nextn.shared_head_norm.weight"), &[d])?;
            check_shape_opt(g, &p("nextn.embed_tokens.weight"), &[d, c.vocab])?;
            check_shape_opt(g, &p("nextn.shared_head_head.weight"), &[d, c.vocab])?;
        }
        Ok(())
    }

    pub fn n_attn_layers(&self) -> usize {
        self.recurrent.iter().filter(|r| !**r).count()
    }

    pub fn n_recurrent_layers(&self) -> usize {
        self.recurrent.iter().filter(|r| **r).count()
    }

    /// KV cache: the attention layers, then the MTP block's.
    pub fn kv_layout(&self, n_blocks: usize, block_size: usize, dtype: KvType) -> KvLayout {
        let n_layers = self.n_attn_layers() + self.mtp as usize;
        KvLayout { n_layers, n_blocks, block_size, n_kv_heads: self.n_kv_heads, head_dim: self.head_dim, dtype }
    }

    /// Floats of recurrent state per sequence and recurrent layer: conv inputs, DeltaNet state.
    fn state_floats(&self) -> (usize, usize) {
        (self.gdn.qkv_width() * (self.conv_kernel - 1), self.gdn.state_len())
    }

    /// Recurrent state bytes per sequence (all layers).
    pub fn state_bytes_per_seq(&self) -> usize {
        let (conv, ssm) = self.state_floats();
        4 * (conv + ssm) * self.n_recurrent_layers()
    }

    /// MoE activation memory per token of a step: router logits, and per
    /// (token, expert) pair the expert's hidden row (f32 and int8) and its
    /// output row, plus the routing indices (see the GPU `moe_ffn`).
    pub fn moe_bytes_per_token(&self) -> usize {
        match self.moe {
            Some(m) => 4 * m.n_expert + m.n_used * (5 * m.n_ff + 4 * self.hidden + 32),
            None => 0,
        }
    }

    /// Where head dim `j` of the reordered Q/K comes from (see the module docs).
    fn dim_order(&self) -> Vec<usize> {
        let (d, half, r2) = (self.head_dim, self.head_dim / 2, self.n_rot / 2);
        let mut order = vec![usize::MAX; d];
        for i in 0..r2 {
            order[i] = i;
            order[half + i] = r2 + i;
        }
        let mut rest = self.n_rot..d;
        for o in order.iter_mut().filter(|o| **o == usize::MAX) {
            *o = rest.next().expect("dims add up");
        }
        order
    }

    /// RoPE factors for the reordered dims: pair `i < n_rot/2` at
    /// `theta^(-2i/n_rot)`, the rest left (practically) unrotated.
    fn rope_factors(&self) -> Vec<f32> {
        let (d, r) = (self.head_dim as f64, self.n_rot as f64);
        let theta = self.rope_theta as f64;
        (0..self.head_dim / 2).map(|i| if i < self.n_rot / 2 { theta.powf(2.0 * i as f64 / r - 2.0 * i as f64 / d) as f32 } else { 1e30 }).collect()
    }
}

struct Ffn<B: Backend> {
    post_norm: B::Weight,
    kind: FfnKind<B>,
}

enum FfnKind<B: Backend> {
    Dense {
        gate: B::Weight,
        up: B::Weight,
        down: B::Weight,
    },
    Moe {
        router: B::Weight,
        /// Stacked experts (see [`MoeFfn`]).
        gate: B::Weight,
        up: B::Weight,
        down: B::Weight,
        shared: Option<SharedExpert<B>>,
    },
}

struct SharedExpert<B: Backend> {
    /// `[1, hidden]`: the shared expert's gate logit.
    gate_inp: B::Weight,
    gate: B::Weight,
    up: B::Weight,
    down: B::Weight,
}

enum Mixer<B: Backend> {
    Attn {
        wq: B::Weight,
        wgate: B::Weight,
        wk: B::Weight,
        wv: B::Weight,
        wo: B::Weight,
        q_norm: B::Weight,
        k_norm: B::Weight,
        /// Index among attention layers (KV cache layer).
        kv_layer: usize,
    },
    Gdn {
        wqkv: B::Weight,
        wz: B::Weight,
        walpha: B::Weight,
        wbeta: B::Weight,
        conv: B::Weight,
        dt_bias: B::Weight,
        a: B::Weight,
        norm: B::Weight,
        wout: B::Weight,
        /// Index among recurrent layers (state buffers).
        state_layer: usize,
    },
}

struct Layer<B: Backend> {
    attn_norm: B::Weight,
    mixer: Mixer<B>,
    ffn: Ffn<B>,
}

/// The MTP block: `x = eh_proj([enorm(embed(x_{p+1})) | hnorm(h_p)])`, then a
/// gated-attention decoder block (its own KV layer), then `head_norm` and
/// the LM head.
struct Mtp<B: Backend> {
    /// `eh_proj` split by input: the embedding half and the hidden half.
    eh_e: B::Weight,
    eh_h: B::Weight,
    enorm: B::Weight,
    hnorm: B::Weight,
    head_norm: B::Weight,
    layer: Layer<B>,
    embed: Option<B::Weight>,
    head: Option<B::Weight>,
    /// The output head's rows for a subset of the vocabulary (`ids`):
    /// drafting scores only these tokens, reading a fraction of the full
    /// head. Drafts only affect speed (verification uses the full head), and
    /// a draft distribution with no mass outside the subset is still exact
    /// for speculative sampling. None: draft over the whole vocabulary.
    draft_head: Option<DraftHead<B>>,
}

struct DraftHead<B: Backend> {
    w: B::Weight,
    ids: Vec<u32>,
}

/// Default draft vocabulary: the first this many token ids (BPE merge
/// order, a proxy for frequency) plus every control/user-defined token.
/// Qwen3.6-35B-A3B on Strix Halo, long greedy generations: 131072 of the
/// 248320 tokens keep the acceptance rate (71.7% vs 72.3% with the full
/// head) and are 7% faster end to end; 32768 lose too many drafts (62.5%).
const DRAFT_VOCAB: usize = 131072;

/// The draft vocabulary: ids below `n`, then the special tokens above it.
fn draft_vocab(g: &Gguf, vocab: usize, n: usize) -> Vec<u32> {
    let types: Vec<i64> = g
        .get("tokenizer.ggml.token_type")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().map(|t| t.as_f64().unwrap_or(1.0) as i64).collect())
        .unwrap_or_default();
    // GGUF token types: 1 normal, 2 unknown, 3 control, 4 user-defined, 5 unused, 6 byte.
    (0..vocab).filter(|&i| i < n || matches!(types.get(i), Some(3 | 4))).map(|i| i as u32).collect()
}

/// Per sequence slot, what MTP drafting carries between steps, and the
/// snapshots that roll back a verification step.
pub struct MtpState<B: Backend> {
    /// `[slots + 1, hidden]`: each slot's last (normalized) main hidden
    /// state; the extra last row stays zero (fresh sequences).
    pending: B::Buffer,
    /// `[slots, hidden]`: the previous MTP step's hidden state (draft chains).
    chain: B::Buffer,
    /// Per recurrent layer: `n_snap` state snapshots shared by the step's
    /// sequences (each verifying sequence takes as many entries as drafts).
    conv_snap: Vec<B::Buffer>,
    ssm_snap: Vec<B::Buffer>,
    n_snap: usize,
    /// The last step's normalized hidden states, and per sequence (state
    /// slot, first row, rows, first snapshot entry).
    step_h: Option<B::Buffer>,
    step_rows: Vec<(u32, usize, usize, usize)>,
}

pub struct Qwen35<B: Backend> {
    pub cfg: Qwen35Config,
    embed: B::Weight,
    layers: Vec<Layer<B>>,
    out_norm: B::Weight,
    output: Option<B::Weight>,
    rope_factors: B::Weight,
    mtp: Option<Mtp<B>>,
    /// `[1, hidden]` of zeros (fresh sequences' "previous hidden state").
    zero_row: B::Weight,
}

/// KV cache of the attention layers plus each sequence slot's recurrent
/// state, per recurrent layer.
pub struct Cache<B: Backend> {
    kv: B::KvCache,
    conv: Vec<B::Buffer>,
    ssm: Vec<B::Buffer>,
    n_slots: usize,
    mtp: Option<MtpState<B>>,
}

fn upload_vec<B: Backend>(b: &B, name: &str, v: &[f32]) -> Result<B::Weight> {
    b.upload(&HostTensor::from_f32(name, vec![v.len()], v)).with_context(|| format!("uploading {name}"))
}

impl<B: Backend> Qwen35<B> {
    pub fn load(backend: &B, g: &Gguf) -> Result<Self> {
        let cfg = Qwen35Config::from_gguf(g)?;
        let w = |name: &str| load_weight(backend, g, name);
        let hd = cfg.head_dim;
        let order = cfg.dim_order();
        // Rows of a [heads x head_dim]-row projection with each head's dims reordered.
        let qk_rows = |heads: usize, stride: usize| -> Vec<usize> { (0..heads).flat_map(|h| order.iter().map(move |&j| h * stride + j)).collect() };
        let reordered = |name: &str| -> Result<B::Weight> {
            let v = read_vector(g, name)?;
            upload_vec(backend, name, &order.iter().map(|&j| v[j]).collect::<Vec<_>>())
        };
        // Conv taps as one f32 vector ([channels, kernel] row-major).
        let conv_weight = |name: &str| -> Result<B::Weight> {
            let t = crate::loader::host_tensor(g, name)?;
            ensure!(t.dtype == rllm_core::DType::F32, "{name}: {} not supported (needs F32)", t.dtype);
            let v: Vec<f32> = t.bytes.as_slice().as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
            upload_vec(backend, name, &v)
        };
        let attn = |p: &dyn Fn(&str) -> String, kv_layer: usize| -> Result<Mixer<B>> {
            // attn_q holds per head [query (head_dim) | gate (head_dim)].
            let q_rows = qk_rows(cfg.n_heads, 2 * hd);
            let gate_rows: Vec<usize> = (0..cfg.n_heads).flat_map(|h| (0..hd).map(move |j| h * 2 * hd + hd + j)).collect();
            Ok(Mixer::Attn {
                wq: load_weight_rows(backend, g, &p("attn_q.weight"), &q_rows)?,
                wgate: load_weight_rows(backend, g, &p("attn_q.weight"), &gate_rows)?,
                wk: load_weight_rows(backend, g, &p("attn_k.weight"), &qk_rows(cfg.n_kv_heads, hd))?,
                wv: w(&p("attn_v.weight"))?,
                wo: w(&p("attn_output.weight"))?,
                q_norm: reordered(&p("attn_q_norm.weight"))?,
                k_norm: reordered(&p("attn_k_norm.weight"))?,
                kv_layer,
            })
        };
        let ffn = |p: &dyn Fn(&str) -> String| -> Result<Ffn<B>> {
            let kind = match cfg.moe {
                None => FfnKind::Dense { gate: w(&p("ffn_gate.weight"))?, up: w(&p("ffn_up.weight"))?, down: w(&p("ffn_down.weight"))? },
                Some(m) => {
                    let (gate, up) = if g.tensor(&p("ffn_gate_up_exps.weight")).is_some() {
                        // Per expert: n_ff gate rows, then n_ff up rows.
                        let f = m.n_ff;
                        let rows = |off: usize| -> Vec<usize> { (0..m.n_expert).flat_map(|e| (0..f).map(move |r| e * 2 * f + off + r)).collect() };
                        let name = p("ffn_gate_up_exps.weight");
                        (load_weight_rows(backend, g, &name, &rows(0))?, load_weight_rows(backend, g, &name, &rows(f))?)
                    } else {
                        (w(&p("ffn_gate_exps.weight"))?, w(&p("ffn_up_exps.weight"))?)
                    };
                    let shared = if m.n_ff_shared > 0 {
                        let gi = p("ffn_gate_inp_shexp.weight");
                        let v = read_vector(g, &gi)?;
                        Some(SharedExpert {
                            gate_inp: backend.upload(&HostTensor::from_f32(&gi, vec![1, v.len()], &v)).with_context(|| format!("uploading {gi}"))?,
                            gate: w(&p("ffn_gate_shexp.weight"))?,
                            up: w(&p("ffn_up_shexp.weight"))?,
                            down: w(&p("ffn_down_shexp.weight"))?,
                        })
                    } else {
                        None
                    };
                    FfnKind::Moe { router: w(&p("ffn_gate_inp.weight"))?, gate, up, down: w(&p("ffn_down_exps.weight"))?, shared }
                }
            };
            Ok(Ffn { post_norm: w(&p("post_attention_norm.weight"))?, kind })
        };
        let (mut n_attn, mut n_rec) = (0, 0);
        let layers = (0..cfg.n_layers)
            .map(|i| {
                let p = |s: &str| format!("blk.{i}.{s}");
                let mixer = if cfg.recurrent[i] {
                    n_rec += 1;
                    Mixer::Gdn {
                        wqkv: w(&p("attn_qkv.weight"))?,
                        wz: w(&p("attn_gate.weight"))?,
                        walpha: w(&p("ssm_alpha.weight"))?,
                        wbeta: w(&p("ssm_beta.weight"))?,
                        conv: conv_weight(&p("ssm_conv1d.weight"))?,
                        dt_bias: w(&p("ssm_dt.bias"))?,
                        a: w(&p("ssm_a"))?,
                        norm: w(&p("ssm_norm.weight"))?,
                        wout: w(&p("ssm_out.weight"))?,
                        state_layer: n_rec - 1,
                    }
                } else {
                    n_attn += 1;
                    attn(&p, n_attn - 1)?
                };
                Ok(Layer { attn_norm: w(&p("attn_norm.weight"))?, mixer, ffn: ffn(&p)? })
            })
            .collect::<Result<Vec<_>>>()?;
        let output = match g.tensor("output.weight") {
            Some(_) => Some(w("output.weight")?),
            None => None,
        };
        let rope_factors = upload_vec(backend, "rope_factors", &cfg.rope_factors())?;
        let mtp = if cfg.mtp {
            let p = |s: &str| format!("blk.{}.{s}", cfg.n_layers);
            let opt = |name: String| -> Result<Option<B::Weight>> { if g.tensor(&name).is_some() { Ok(Some(w(&name)?)) } else { Ok(None) } };
            let d = cfg.hidden;
            Some(Mtp {
                eh_e: load_weight_cols(backend, g, &p("nextn.eh_proj.weight"), 0, d)?,
                eh_h: load_weight_cols(backend, g, &p("nextn.eh_proj.weight"), d, d)?,
                enorm: w(&p("nextn.enorm.weight"))?,
                hnorm: w(&p("nextn.hnorm.weight"))?,
                head_norm: match opt(p("nextn.shared_head_norm.weight"))? {
                    Some(x) => x,
                    None => w("output_norm.weight")?,
                },
                layer: Layer { attn_norm: w(&p("attn_norm.weight"))?, mixer: attn(&p, n_attn)?, ffn: ffn(&p)? },
                embed: opt(p("nextn.embed_tokens.weight"))?,
                head: opt(p("nextn.shared_head_head.weight"))?,
                draft_head: {
                    let n = std::env::var("RLLM_MTP_VOCAB").ok().and_then(|v| v.parse().ok()).unwrap_or(DRAFT_VOCAB);
                    let ids = draft_vocab(g, cfg.vocab, n);
                    if n == 0 || ids.len() >= cfg.vocab {
                        None
                    } else {
                        let src = [p("nextn.shared_head_head.weight"), "output.weight".into(), "token_embd.weight".into()]
                            .into_iter()
                            .find(|t| g.tensor(t).is_some())
                            .context("no output head")?;
                        let rows: Vec<usize> = ids.iter().map(|&i| i as usize).collect();
                        Some(DraftHead { w: load_weight_rows(backend, g, &src, &rows)?, ids })
                    }
                },
            })
        } else {
            None
        };
        let zero_row = backend.upload(&HostTensor::from_f32("zero_row", vec![1, cfg.hidden], &vec![0.0; cfg.hidden]))?;
        Ok(Self { embed: w("token_embd.weight")?, out_norm: w("output_norm.weight")?, layers, output, rope_factors, mtp, zero_row, cfg })
    }

    fn forward_impl(&self, b: &B, cache: &mut Cache<B>, batch: &Batch, block_size: usize) -> Result<StepOut> {
        let c = &self.cfg;
        let n = batch.tokens.len();
        let n_seqs = batch.block_tables.len();
        ensure!(batch.state_slots.len() == n_seqs, "{} state slots for {n_seqs} sequences", batch.state_slots.len());
        // Each sequence's tokens are contiguous and in order (as KvPool builds steps).
        let mut starts = Vec::with_capacity(n_seqs + 1);
        for (t, &s) in batch.token_seq.iter().enumerate() {
            if t == 0 || s != batch.token_seq[t - 1] {
                ensure!(s as usize == starts.len(), "sequence tokens are not contiguous in the step");
                starts.push(t as u32);
            }
        }
        ensure!(starts.len() == n_seqs, "{} sequences in the step, {n_seqs} block tables", starts.len());
        let fresh: Vec<u32> = starts.iter().map(|&t| (batch.positions[t as usize] == 0) as u32).collect();
        starts.push(n as u32);
        let mtp_on = batch.mtp && self.mtp.is_some() && cache.mtp.is_some();
        // Snapshots only exist with MTP enabled (they serve its verification
        // steps); sequences take consecutive entries of the shared buffers.
        let snap: &[u32] = if cache.mtp.is_some() { batch.snap } else { &[] };
        let snap_at: Vec<u32> = snap.iter().scan(0u32, |acc, &k| { let at = *acc; *acc += k; Some(at) }).collect();
        if let Some(st) = &cache.mtp {
            let total: u32 = snap.iter().sum();
            ensure!(total as usize <= st.n_snap, "step needs {total} recurrent-state snapshots, {} allocated", st.n_snap);
        }
        let runs = SeqRuns { starts: &starts, slots: batch.state_slots, fresh: &fresh, snap, snap_at: &snap_at };
        runs.check(n, cache.n_slots)?;
        let snapping = snap.iter().any(|&k| k > 0);

        let (d, hd) = (c.hidden, c.head_dim);
        let qw = c.n_heads * hd;
        let kvw = c.n_kv_heads * hd;
        let (conv_w, inner, hv) = (c.gdn.qkv_width(), c.gdn.n_v_heads * c.gdn.head_dim, c.gdn.n_v_heads);
        // Decode-sized steps replay a recorded graph. What varies between
        // steps with one key (tokens, positions, KV and state slots, fresh
        // flags, snapshot counts, MTP row choices) is index data the replay
        // refreshes; whether MTP runs and snapshots are taken changes the
        // launches themselves, so they are part of the key.
        let key = batch.graph_key(self as *const Self as usize).map(|k| {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            (k, mtp_on, snapping).hash(&mut h);
            h.finish()
        });
        // The previous step's hidden states (kept for `keep_step`, which has
        // run by now) are freed here: freeing an eagerly allocated buffer
        // inside a graph capture would break the capture.
        if let Some(st) = cache.mtp.as_mut() {
            st.step_h = None;
        }
        b.begin_step(key)?;

        let mut x = b.alloc(n * d)?;
        let mut h = b.alloc(n * d)?;
        let mut o = b.alloc(n * d)?;
        let mut a1 = b.alloc(n * qw.max(conv_w))?;
        let mut a2 = b.alloc(n * qw.max(inner))?;
        let mut a3 = b.alloc(n * kvw.max(hv).max(conv_w))?;
        let mut a4 = b.alloc(n * kvw.max(hv))?;
        let mut a5 = b.alloc(n * qw.max(inner))?;

        let meta = AttnMeta {
            n_heads: c.n_heads,
            n_kv_heads: c.n_kv_heads,
            head_dim: hd,
            scale: 1.0 / (hd as f32).sqrt(),
            window: 0,
            token_seq: batch.token_seq,
            positions: batch.positions,
            block_tables: batch.block_tables,
            block_size,
        };

        b.embed(&self.embed, batch.tokens, &mut x)?;
        b.rms_norm(&x, &self.layers[0].attn_norm, &mut h, n, c.rms_eps)?;
        for (li, l) in self.layers.iter().enumerate() {
            // h = attn_norm(x) on entry.
            match &l.mixer {
                Mixer::Attn { wq, wgate, wk, wv, wo, q_norm, k_norm, kv_layer } => {
                    // a1 = q, a2 = gate, a3 = k, a4 = v, a5 = normed q, then attention out.
                    b.matmul_many(&h, &[wq, wgate, wk, wv], &mut [&mut a1, &mut a2, &mut a3, &mut a4], n)?;
                    b.qkv_norm_rope_store(
                        &a1, &a3, &a4, Some((q_norm, k_norm)), None, &mut a5, &mut cache.kv, *kv_layer, batch.positions, batch.slots,
                        c.n_heads, c.n_kv_heads, hd, c.rms_eps, c.rope_theta, Some(&self.rope_factors),
                    )?;
                    b.attention(&a5, &cache.kv, *kv_layer, &meta, &mut a1)?;
                    b.sigmoid_mul(&mut a1, &a2, n * qw)?;
                    b.matmul(&a1, wo, &mut o, n)?;
                }
                Mixer::Gdn { wqkv, wz, walpha, wbeta, conv, dt_bias, a, norm, wout, state_layer } => {
                    // a1 = mixed qkv, a2 = z, a3 = alpha, a4 = beta.
                    b.matmul_many(&h, &[wqkv, wz, walpha, wbeta], &mut [&mut a1, &mut a2, &mut a3, &mut a4], n)?;
                    // a5 = conv output (q|k|v), then normalized core output.
                    let mut conv_out = b.alloc(n * conv_w)?;
                    let sl = *state_layer;
                    let (conv_snap, ssm_snap) = match cache.mtp.as_mut().filter(|_| snapping) {
                        Some(m) => (Some((&mut m.conv_snap[sl], m.n_snap)), Some((&mut m.ssm_snap[sl], m.n_snap))),
                        None => (None, None),
                    };
                    b.conv1d_silu(&a1, conv, &mut cache.conv[sl], &runs, &mut conv_out, conv_w, c.conv_kernel, conv_snap)?;
                    b.gated_delta_net(&conv_out, &a3, &a4, dt_bias, a, &mut cache.ssm[sl], &runs, &mut a1, c.gdn, c.rms_eps, ssm_snap)?;
                    b.rms_norm(&a1, norm, &mut a5, n * hv, c.rms_eps)?;
                    // z = silu(z) * norm(o)
                    b.silu_mul(&mut a2, &a5, n * inner)?;
                    b.matmul(&a2, wout, &mut o, n)?;
                }
            }
            b.add_rms_norm(&mut x, &o, &l.ffn.post_norm, &mut h, n, d, c.rms_eps)?;
            self.run_ffn(b, &l.ffn.kind, &h, &mut o, n)?;
            match self.layers.get(li + 1) {
                Some(next) => b.add_rms_norm(&mut x, &o, &next.attn_norm, &mut h, n, d, c.rms_eps)?,
                None => b.add(&mut x, &o, n * d)?,
            }
        }

        let m = batch.logit_rows.len();
        let normed = if mtp_on {
            // Every token's normalized hidden state feeds the MTP block.
            let mut hn = b.alloc(n * d)?;
            b.rms_norm(&x, &self.out_norm, &mut hn, n, c.rms_eps)?;
            self.mtp_fill(b, cache, batch, &runs, &hn)?;
            let mut rows = b.alloc(m * d)?;
            b.gather_rows(&hn, batch.logit_rows, d, &mut rows)?;
            let st = cache.mtp.as_mut().expect("mtp_on");
            st.step_rows = (0..n_seqs)
                .map(|s| (batch.state_slots[s], starts[s] as usize, (starts[s + 1] - starts[s]) as usize, snap_at.get(s).map_or(0, |&a| a as usize)))
                .collect();
            st.step_h = Some(hn);
            rows
        } else {
            let all_rows = m == n && batch.logit_rows.iter().enumerate().all(|(i, &r)| r as usize == i);
            let gathered;
            let last = if all_rows {
                &x
            } else {
                let mut g = b.alloc(m * d)?;
                b.gather_rows(&x, batch.logit_rows, d, &mut g)?;
                gathered = g;
                &gathered
            };
            let mut normed = b.alloc(m * d)?;
            b.rms_norm(last, &self.out_norm, &mut normed, m, c.rms_eps)?;
            normed
        };
        if batch.hidden {
            let mut host = vec![0f32; m * d];
            b.to_host(&normed, &mut host)?;
            return Ok(StepOut { rows: host, ids: Vec::new() });
        }
        let mut logits = b.alloc(m * c.vocab)?;
        b.matmul(&normed, self.output.as_ref().unwrap_or(&self.embed), &mut logits, m)?;
        crate::finish_logits(b, &logits, m, c.vocab, batch)
    }
}

impl<B: Backend> Qwen35<B> {
    /// `out = MLP(h)` over `n` rows (dense, or routed experts plus the gated shared expert).
    fn run_ffn(&self, b: &B, f: &FfnKind<B>, h: &B::Buffer, out: &mut B::Buffer, n: usize) -> Result<()> {
        let c = &self.cfg;
        let d = c.hidden;
        match f {
            FfnKind::Dense { gate, up, down } => {
                let (mut g, mut u) = (b.alloc(n * c.ffn)?, b.alloc(n * c.ffn)?);
                b.gated_ffn(h, gate, up, down, Activation::Silu, out, &mut g, &mut u, n, c.ffn)?;
            }
            FfnKind::Moe { router, gate, up, down, shared } => {
                let m = c.moe.expect("MoE layers only in MoE models");
                let mut logits = b.alloc(n * m.n_expert)?;
                let mut sg = b.alloc(n)?;
                match shared {
                    Some(s) => b.matmul_many(h, &[router, &s.gate_inp], &mut [&mut logits, &mut sg], n)?,
                    None => b.matmul(h, router, &mut logits, n)?,
                }
                let moe = MoeFfn { gate, up, down, n_expert: m.n_expert, n_used: m.n_used, n_ff: m.n_ff, norm_topk: true, scale: m.scale };
                let so = match shared {
                    Some(s) => {
                        let f = m.n_ff_shared;
                        let (mut g, mut u, mut so) = (b.alloc(n * f)?, b.alloc(n * f)?, b.alloc(n * d)?);
                        b.gated_ffn(h, &s.gate, &s.up, &s.down, Activation::Silu, &mut so, &mut g, &mut u, n, f)?;
                        Some(so)
                    }
                    None => None,
                };
                b.moe_ffn(h, &logits, &moe, so.as_ref().map(|so| (so, &sg)), out, n)?;
            }
        }
        Ok(())
    }

    pub fn has_mtp(&self) -> bool {
        self.mtp.is_some()
    }

    /// `eh_proj([enorm(embed(tokens)) | hnorm(src)])`: the MTP block's input.
    fn mtp_input(&self, b: &B, mtp: &Mtp<B>, tokens: &[u32], src: &B::Buffer) -> Result<B::Buffer> {
        let (n, d, eps) = (tokens.len(), self.cfg.hidden, self.cfg.rms_eps);
        let mut e = b.alloc(n * d)?;
        b.embed(mtp.embed.as_ref().unwrap_or(&self.embed), tokens, &mut e)?;
        let (mut en, mut hn) = (b.alloc(n * d)?, b.alloc(n * d)?);
        b.rms_norm(&e, &mtp.enorm, &mut en, n, eps)?;
        b.rms_norm(src, &mtp.hnorm, &mut hn, n, eps)?;
        let (mut xm, mut t) = (b.alloc(n * d)?, b.alloc(n * d)?);
        b.matmul(&en, &mtp.eh_e, &mut xm, n)?;
        b.matmul(&hn, &mtp.eh_h, &mut t, n)?;
        b.add(&mut xm, &t, n * d)?;
        Ok(xm)
    }

    /// The MTP part of a main step: each token, paired with the previous
    /// token's hidden state (`hn`; a sequence's first token uses the one
    /// saved from its previous step), writes the MTP block's K/V at its
    /// position. Then each sequence's last hidden state is saved.
    fn mtp_fill(&self, b: &B, cache: &mut Cache<B>, batch: &Batch, runs: &SeqRuns, hn: &B::Buffer) -> Result<()> {
        let c = &self.cfg;
        let mtp = self.mtp.as_ref().expect("checked");
        let Mixer::Attn { wk, wv, k_norm, kv_layer, .. } = &mtp.layer.mixer else { unreachable!("MTP block is attention") };
        let (n, d, kvw) = (batch.tokens.len(), c.hidden, c.n_kv_heads * c.head_dim);
        let st = cache.mtp.as_mut().expect("checked");
        // Rows of `cat = [pending | hn]`, chosen by gathers (not copies at
        // host-computed offsets, which a recorded graph would freeze).
        let np = cache.n_slots + 1;
        let mut cat = b.alloc((np + n) * d)?;
        b.copy_span(&st.pending, 0, &mut cat, 0, np * d)?;
        b.copy_span(hn, 0, &mut cat, np * d, n * d)?;
        // Each token pairs with the previous token's hidden state; a
        // sequence's first token with its saved one (zeros if fresh).
        let mut prev: Vec<u32> = (0..n).map(|t| (np + t.saturating_sub(1)) as u32).collect();
        for s in 0..runs.n_seqs() {
            prev[runs.starts[s] as usize] = if runs.fresh[s] != 0 { cache.n_slots as u32 } else { runs.slots[s] };
        }
        let mut src = b.alloc(n * d)?;
        b.gather_rows(&cat, &prev, d, &mut src)?;
        let xm = self.mtp_input(b, mtp, batch.tokens, &src)?;
        let mut h = b.alloc(n * d)?;
        b.rms_norm(&xm, &mtp.layer.attn_norm, &mut h, n, c.rms_eps)?;
        let (mut k, mut v, mut kn) = (b.alloc(n * kvw)?, b.alloc(n * kvw)?, b.alloc(n * kvw)?);
        b.matmul_many(&h, &[wk, wv], &mut [&mut k, &mut v], n)?;
        b.rms_norm_rope(&k, k_norm, &mut kn, batch.positions, c.n_kv_heads, c.head_dim, c.rms_eps, c.rope_theta, Some(&self.rope_factors))?;
        b.kv_store(&mut cache.kv, *kv_layer, &kn, &v, batch.slots)?;
        // Save each sequence's last hidden state in its pending row.
        let mut keep: Vec<u32> = (0..np as u32).collect();
        for s in 0..runs.n_seqs() {
            keep[runs.slots[s] as usize] = np as u32 + runs.starts[s + 1] - 1;
        }
        let mut next = b.alloc(np * d)?;
        b.gather_rows(&cat, &keep, d, &mut next)?;
        b.copy_span(&next, 0, &mut st.pending, 0, np * d)?;
        Ok(())
    }

    pub fn enable_mtp(&self, b: &B, cache: &mut Cache<B>, max_snapshots: usize) -> Result<bool> {
        if self.mtp.is_none() {
            return Ok(false);
        }
        let (slots, d, n_snap) = (cache.n_slots, self.cfg.hidden, max_snapshots.max(1));
        let (conv, ssm) = self.cfg.state_floats();
        let n_rec = self.cfg.n_recurrent_layers();
        let mut pending = b.alloc((slots + 1) * d)?;
        let mut zero = b.alloc(d)?;
        b.embed(&self.zero_row, &[0], &mut zero)?;
        b.copy_span(&zero, 0, &mut pending, slots * d, d)?;
        cache.mtp = Some(MtpState {
            pending,
            chain: b.alloc(slots * d)?,
            conv_snap: (0..n_rec).map(|_| b.alloc(n_snap * conv)).collect::<std::result::Result<_, _>>()?,
            ssm_snap: (0..n_rec).map(|_| b.alloc(n_snap * ssm)).collect::<std::result::Result<_, _>>()?,
            n_snap,
            step_h: None,
            step_rows: Vec::new(),
        });
        Ok(true)
    }

    pub fn keep_step(&self, b: &B, cache: &mut Cache<B>, slot: u32, kept: usize) -> Result<()> {
        let d = self.cfg.hidden;
        let (conv, ssm) = self.cfg.state_floats();
        let Cache { conv: conv_state, ssm: ssm_state, mtp, .. } = cache;
        let st = mtp.as_mut().context("keep_step without MTP state")?;
        let &(_, row0, n_rows, at) = st.step_rows.iter().find(|r| r.0 == slot).context("keep_step: sequence not in the last step")?;
        if kept >= n_rows {
            return Ok(());
        }
        ensure!(kept >= 1 && at + kept <= st.n_snap, "keep_step: keeping {kept} of {n_rows} tokens (snapshot entries from {at} of {})", st.n_snap);
        let (e, s) = (at + kept - 1, slot as usize);
        for l in 0..conv_state.len() {
            b.copy_span(&st.conv_snap[l], e * conv, &mut conv_state[l], s * conv, conv)?;
            b.copy_span(&st.ssm_snap[l], e * ssm, &mut ssm_state[l], s * ssm, ssm)?;
        }
        let h = st.step_h.as_ref().context("keep_step: no hidden states")?;
        b.copy_span(h, (row0 + kept - 1) * d, &mut st.pending, s * d, d)?;
        Ok(())
    }

    pub fn mtp_draft(&self, b: &B, cache: &mut Cache<B>, dr: &MtpBatch, block_size: usize) -> Result<Vec<f32>> {
        let c = &self.cfg;
        let mtp = self.mtp.as_ref().context("this model has no MTP head")?;
        let Mixer::Attn { wq, wgate, wk, wv, wo, q_norm, k_norm, kv_layer } = &mtp.layer.mixer else { unreachable!("MTP block is attention") };
        let n = dr.tokens.len();
        ensure!(n > 0 && dr.positions.len() == n && dr.slots.len() == n && dr.block_tables.len() == n && dr.state_slots.len() == n, "bad MTP batch");
        let (d, hd) = (c.hidden, c.head_dim);
        let (qw, kvw) = (c.n_heads * hd, c.n_kv_heads * hd);
        b.begin_step(dr.graph_key(self as *const Self as usize))?;
        let n_slots = cache.n_slots;
        let Cache { kv, mtp: st, .. } = cache;
        let st = st.as_mut().context("MTP not enabled")?;
        let mut src = b.alloc(n * d)?;
        b.gather_rows(if dr.first { &st.pending } else { &st.chain }, dr.state_slots, d, &mut src)?;
        let mut xm = self.mtp_input(b, mtp, dr.tokens, &src)?;
        let mut h = b.alloc(n * d)?;
        b.rms_norm(&xm, &mtp.layer.attn_norm, &mut h, n, c.rms_eps)?;
        let (mut q, mut g, mut k, mut v) = (b.alloc(n * qw)?, b.alloc(n * qw)?, b.alloc(n * kvw)?, b.alloc(n * kvw)?);
        b.matmul_many(&h, &[wq, wgate, wk, wv], &mut [&mut q, &mut g, &mut k, &mut v], n)?;
        let mut qn = b.alloc(n * qw)?;
        b.qkv_norm_rope_store(
            &q, &k, &v, Some((q_norm, k_norm)), None, &mut qn, kv, *kv_layer, dr.positions, dr.slots, c.n_heads, c.n_kv_heads, hd, c.rms_eps,
            c.rope_theta, Some(&self.rope_factors),
        )?;
        let token_seq: Vec<u32> = (0..n as u32).collect();
        let meta = AttnMeta {
            n_heads: c.n_heads,
            n_kv_heads: c.n_kv_heads,
            head_dim: hd,
            scale: 1.0 / (hd as f32).sqrt(),
            window: 0,
            token_seq: &token_seq,
            positions: dr.positions,
            block_tables: dr.block_tables,
            block_size,
        };
        let mut att = b.alloc(n * qw)?;
        b.attention(&qn, kv, *kv_layer, &meta, &mut att)?;
        b.sigmoid_mul(&mut att, &g, n * qw)?;
        let mut o = b.alloc(n * d)?;
        b.matmul(&att, wo, &mut o, n)?;
        let ffn = &mtp.layer.ffn;
        b.add_rms_norm(&mut xm, &o, &ffn.post_norm, &mut h, n, d, c.rms_eps)?;
        self.run_ffn(b, &ffn.kind, &h, &mut o, n)?;
        b.add(&mut xm, &o, n * d)?;
        let mut hh = b.alloc(n * d)?;
        b.rms_norm(&xm, &mtp.head_norm, &mut hh, n, c.rms_eps)?;
        // chain[slot] = hh[i], by a gather over `[chain | hh]` (graph-safe).
        let mut cat = b.alloc((n_slots + n) * d)?;
        b.copy_span(&st.chain, 0, &mut cat, 0, n_slots * d)?;
        b.copy_span(&hh, 0, &mut cat, n_slots * d, n * d)?;
        let mut keep: Vec<u32> = (0..n_slots as u32).collect();
        for (i, &s) in dr.state_slots.iter().enumerate() {
            keep[s as usize] = (n_slots + i) as u32;
        }
        let mut next = b.alloc(n_slots * d)?;
        b.gather_rows(&cat, &keep, d, &mut next)?;
        b.copy_span(&next, 0, &mut st.chain, 0, n_slots * d)?;
        if let Some(dh) = &mtp.draft_head {
            // Score the draft vocabulary; every other token gets -inf.
            let k = dh.ids.len();
            let mut logits = b.alloc(n * k)?;
            b.matmul(&hh, &dh.w, &mut logits, n)?;
            let mut sub = vec![0f32; n * k];
            b.to_host(&logits, &mut sub)?;
            let mut host = vec![f32::NEG_INFINITY; n * c.vocab];
            for (row, s) in host.chunks_exact_mut(c.vocab).zip(sub.chunks_exact(k)) {
                for (&id, &v) in dh.ids.iter().zip(s) {
                    row[id as usize] = v;
                }
            }
            return Ok(host);
        }
        let mut logits = b.alloc(n * c.vocab)?;
        b.matmul(&hh, mtp.head.as_ref().or(self.output.as_ref()).unwrap_or(&self.embed), &mut logits, n)?;
        let mut host = vec![0f32; n * c.vocab];
        b.to_host(&logits, &mut host)?;
        Ok(host)
    }
}

impl<B: Backend> Model<B> for Qwen35<B> {
    type Cache = Cache<B>;

    fn alloc_cache(&self, b: &B, n_blocks: usize, block_size: usize, kv: KvType, max_seqs: usize) -> Result<Cache<B>> {
        let c = &self.cfg;
        ensure!(max_seqs > 0, "recurrent state needs at least one sequence slot");
        let (conv, ssm) = c.state_floats();
        let n_rec = c.n_recurrent_layers();
        Ok(Cache {
            kv: b.alloc_kv(c.kv_layout(n_blocks, block_size, kv))?,
            conv: (0..n_rec).map(|_| b.alloc(max_seqs * conv)).collect::<std::result::Result<_, _>>()?,
            ssm: (0..n_rec).map(|_| b.alloc(max_seqs * ssm)).collect::<std::result::Result<_, _>>()?,
            n_slots: max_seqs,
            mtp: None,
        })
    }

    fn recurrent(&self) -> bool {
        true
    }

    fn has_mtp(&self) -> bool {
        self.mtp.is_some()
    }

    fn enable_mtp(&self, b: &B, cache: &mut Cache<B>, max_snapshots: usize) -> Result<bool> {
        Qwen35::enable_mtp(self, b, cache, max_snapshots)
    }

    fn keep_step(&self, b: &B, cache: &mut Cache<B>, slot: u32, kept: usize) -> Result<()> {
        Qwen35::keep_step(self, b, cache, slot, kept)
    }

    fn mtp_draft(&self, b: &B, cache: &mut Cache<B>, d: &MtpBatch, block_size: usize) -> Result<Vec<f32>> {
        Qwen35::mtp_draft(self, b, cache, d, block_size)
    }

    type HostCache = B::HostKv;

    fn kv_block_bytes(&self, block_size: usize, kv: KvType) -> usize {
        self.cfg.kv_layout(1, block_size, kv).block_bytes()
    }

    fn alloc_host_cache(&self, b: &B, n_blocks: usize, block_size: usize, kv: KvType) -> Result<B::HostKv> {
        Ok(b.alloc_host_kv(self.cfg.kv_layout(n_blocks, block_size, kv), n_blocks)?)
    }

    fn kv_to_host(&self, b: &B, cache: &Cache<B>, store: &mut B::HostKv, dev: &[u32], host: &[u32]) -> Result<()> {
        Ok(b.kv_to_host(&cache.kv, store, dev, host)?)
    }

    fn kv_from_host(&self, b: &B, cache: &mut Cache<B>, store: &B::HostKv, host: &[u32], dev: &[u32]) -> Result<()> {
        Ok(b.kv_from_host(&mut cache.kv, store, host, dev)?)
    }

    fn forward(&self, b: &B, cache: &mut Cache<B>, batch: &Batch, block_size: usize) -> Result<StepOut> {
        self.forward_impl(b, cache, batch, block_size)
    }

    fn vocab_size(&self) -> usize {
        self.cfg.vocab
    }

    fn summary(&self) -> String {
        let c = &self.cfg;
        let experts = match c.moe {
            Some(m) => format!(", {} experts (top {}) of {}{}", m.n_expert, m.n_used, m.n_ff, if m.n_ff_shared > 0 { " + shared" } else { "" }),
            None => String::new(),
        };
        format!(
            "{}: {} layers ({} Gated DeltaNet, {} attention), hidden {}, {} heads / {} kv{experts}",
            c.arch,
            c.n_layers,
            c.n_recurrent_layers(),
            c.n_attn_layers(),
            c.hidden,
            c.n_heads,
            c.n_kv_heads
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(head_dim: usize, n_rot: usize) -> Qwen35Config {
        Qwen35Config {
            n_layers: 4,
            hidden: 8,
            ffn: 16,
            n_heads: 2,
            n_kv_heads: 1,
            head_dim,
            n_rot,
            vocab: 10,
            rope_theta: 10_000_000.0,
            rms_eps: 1e-6,
            max_context: 64,
            recurrent: vec![true, true, true, false],
            conv_kernel: 4,
            gdn: GdnShape { n_k_heads: 1, n_v_heads: 2, head_dim: 4 },
            mtp: false,
            moe: None,
            arch: "qwen35".into(),
        }
    }

    /// Reordered dims rotated by rllm's RoPE match Qwen3.5's partial NeoX RoPE.
    #[test]
    fn reordered_rope_matches_partial_rotary() {
        let c = cfg(16, 8);
        let order = c.dim_order();
        let mut sorted = order.clone();
        sorted.sort();
        assert_eq!(sorted, (0..16).collect::<Vec<_>>());
        let f = c.rope_factors();
        let x: Vec<f32> = (0..16).map(|i| (i as f32 * 0.37).sin()).collect();
        let pos = 1234.0f64;
        // Reference: rotate pairs (i, i + 4) for i < 4 at theta^(-2i/8), rest unchanged.
        let mut want = x.clone();
        for i in 0..4 {
            let a = pos * (c.rope_theta as f64).powf(-2.0 * i as f64 / 8.0);
            let (s, co) = a.sin_cos();
            want[i] = (x[i] as f64 * co - x[i + 4] as f64 * s) as f32;
            want[i + 4] = (x[i] as f64 * s + x[i + 4] as f64 * co) as f32;
        }
        // rllm: reorder, rotate (i, i + 8) at theta^(-2i/16) / f[i].
        let mut y: Vec<f32> = order.iter().map(|&j| x[j]).collect();
        for i in 0..8 {
            let a = pos * (c.rope_theta as f64).powf(-2.0 * i as f64 / 16.0) / f[i] as f64;
            let (s, co) = a.sin_cos();
            let (p, q) = (y[i] as f64, y[i + 8] as f64);
            y[i] = (p * co - q * s) as f32;
            y[i + 8] = (p * s + q * co) as f32;
        }
        for (j, &src) in order.iter().enumerate() {
            assert!((y[j] - want[src]).abs() < 1e-4, "dim {src}: {} vs {}", y[j], want[src]);
        }
    }
}
