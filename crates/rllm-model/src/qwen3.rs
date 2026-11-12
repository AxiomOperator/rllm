//! Dense decoders: Qwen3 and the Llama family (GGUF architectures `qwen3`
//! and `llama`, which covers Llama 2/3 and Mistral).
//!
//! Per layer: RMSNorm → Q/K/V projections → per-head RMSNorm on Q and K
//! (Qwen3 only) → NeoX RoPE → paged GQA attention → output projection →
//! residual, then RMSNorm → SwiGLU MLP → residual. Llama's Q/K rows are
//! stored for ggml's adjacent-pair RoPE and are reordered on load (see
//! `load_weight_rope_neox`).

use anyhow::{Context, Result, bail};
use rllm_core::{Activation, AttnMeta, Backend, KvLayout, KvType};
use rllm_gguf::Gguf;

use crate::loader::{
    MAX_CONTEXT, MAX_DIM, check_same_as_head_dim, check_shape, check_shape_opt, load_weight, load_weight_rope_neox, meta_f32_or,
    meta_usize, meta_usize_or,
};
use crate::{Batch, Model, StepOut};

#[derive(Debug, Clone)]
pub struct Qwen3Config {
    /// `qwen3` or `llama`.
    pub arch: String,
    pub n_layers: usize,
    pub hidden: usize,
    pub ffn: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub vocab: usize,
    pub rope_theta: f32,
    pub rms_eps: f32,
    pub max_context: usize,
    /// Sliding-window attention over every layer (Mistral v0.1); 0: none.
    pub sliding_window: usize,
}

impl Qwen3Config {
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        let arch = g.get_str("general.architecture")?;
        if !matches!(arch, "qwen3" | "llama") {
            bail!("unsupported architecture {arch:?} (dense: qwen3, llama)");
        }
        let llama = arch == "llama";
        let k = |s: &str| format!("{arch}.{s}");
        let dim = |s: &str| meta_usize(g, &k(s), MAX_DIM, false);
        let n_heads = dim("attention.head_count")?;
        let hidden = dim("embedding_length")?;
        let head_dim = match g.get(&k("attention.key_length")) {
            Some(_) => dim("attention.key_length")?,
            None => {
                anyhow::ensure!(hidden.is_multiple_of(n_heads), "embedding_length {hidden} is not a multiple of head_count {n_heads}");
                hidden / n_heads
            }
        };
        let embd = g.require_tensor("token_embd.weight")?;
        let [_, vocab] = embd.dims[..] else { bail!("token_embd.weight has shape {:?}, expected 2 dims", embd.dims) };
        let vocab = usize::try_from(vocab).ok().filter(|&v| (1..=MAX_DIM).contains(&v)).context("token_embd.weight: vocabulary size out of range")?;
        // Llama 3 scaling arrives as `rope_freqs.weight`; others would need
        // positions or frequencies changed, which isn't implemented.
        if let Some(t) = g.get(&k("rope.scaling.type")).and_then(|v| v.as_str())
            && !matches!(t, "none" | "llama3")
        {
            bail!("RoPE scaling {t:?} is not supported");
        }
        if g.get(&k("rope.scaling.factor")).and_then(|v| v.as_f64()).is_some_and(|f| f != 1.0)
            && g.tensor("rope_freqs.weight").is_none()
        {
            bail!("RoPE scaling (rope.scaling.factor) is not supported");
        }
        // Partial rotary embeddings and V heads narrower or wider than K
        // heads would load fine and compute garbage: refuse them.
        check_same_as_head_dim(g, &k("rope.dimension_count"), head_dim, "partial rotary embedding")?;
        check_same_as_head_dim(g, &k("attention.value_length"), head_dim, "a value head size different from the key head size")?;
        // Per-layer head counts or sizes (arrays) aren't supported.
        for key in ["feed_forward_length", "attention.head_count", "attention.head_count_kv"] {
            if g.get(&k(key)).is_some_and(|v| v.as_array().is_some()) {
                bail!("per-layer {key} not supported");
            }
        }
        let cfg = Self {
            arch: arch.to_string(),
            n_layers: dim("block_count")?,
            hidden,
            ffn: dim("feed_forward_length")?,
            n_heads,
            n_kv_heads: dim("attention.head_count_kv")?,
            head_dim,
            vocab,
            rope_theta: meta_f32_or(g, &k("rope.freq_base"), if llama { 10_000.0 } else { 1_000_000.0 })?,
            rms_eps: meta_f32_or(g, &k("attention.layer_norm_rms_epsilon"), if llama { 1e-5 } else { 1e-6 })?,
            max_context: meta_usize_or(g, &k("context_length"), 32768, MAX_CONTEXT, false)?,
            sliding_window: meta_usize_or(g, &k("attention.sliding_window"), 0, MAX_CONTEXT, true)?,
        };
        cfg.validate()?;
        cfg.check_tensors(g)?;
        Ok(cfg)
    }

    /// Check the hyperparameters are consistent (every count positive, KV
    /// heads dividing query heads, RoPE on whole pairs).
    pub fn validate(&self) -> Result<()> {
        let c = self;
        for (what, v) in [
            ("block_count", c.n_layers),
            ("embedding_length", c.hidden),
            ("feed_forward_length", c.ffn),
            ("head_count", c.n_heads),
            ("head_count_kv", c.n_kv_heads),
            ("head size", c.head_dim),
            ("vocabulary", c.vocab),
            ("context_length", c.max_context),
        ] {
            anyhow::ensure!(v > 0, "{what} must be positive");
        }
        anyhow::ensure!(
            c.n_heads.is_multiple_of(c.n_kv_heads),
            "head_count {} is not a multiple of head_count_kv {}",
            c.n_heads,
            c.n_kv_heads
        );
        anyhow::ensure!(c.head_dim.is_multiple_of(2), "head size {} must be even (RoPE rotates pairs)", c.head_dim);
        anyhow::ensure!(c.n_layers <= 4096 && c.n_heads <= 4096 && c.head_dim <= 4096, "implausible model size: {c:?}");
        anyhow::ensure!(c.rope_theta.is_finite() && c.rope_theta > 0.0 && c.rms_eps.is_finite() && c.rms_eps > 0.0, "invalid RoPE base or RMS epsilon");
        Ok(())
    }

    /// Check every tensor the model loads has the shape the config implies.
    fn check_tensors(&self, g: &Gguf) -> Result<()> {
        let c = self;
        let (d, qw, kvw) = (c.hidden, c.q_width(), c.kv_width());
        check_shape(g, "token_embd.weight", &[d, c.vocab])?;
        check_shape(g, "output_norm.weight", &[d])?;
        check_shape_opt(g, "output.weight", &[d, c.vocab])?;
        check_shape_opt(g, "rope_freqs.weight", &[c.head_dim / 2])?;
        for i in 0..c.n_layers {
            let p = |s: &str| format!("blk.{i}.{s}.weight");
            for (name, dims) in [
                ("attn_norm", &[d][..]),
                ("attn_q", &[d, qw]),
                ("attn_k", &[d, kvw]),
                ("attn_v", &[d, kvw]),
                ("attn_output", &[qw, d]),
                ("ffn_norm", &[d]),
                ("ffn_gate", &[d, c.ffn]),
                ("ffn_up", &[d, c.ffn]),
                ("ffn_down", &[c.ffn, d]),
            ] {
                check_shape(g, &p(name), dims)?;
            }
            if g.tensor(&p("attn_q_norm")).is_some() {
                check_shape(g, &p("attn_q_norm"), &[c.head_dim])?;
                check_shape(g, &p("attn_k_norm"), &[c.head_dim])?;
            }
        }
        Ok(())
    }

    pub fn q_width(&self) -> usize {
        self.n_heads * self.head_dim
    }

    pub fn kv_width(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }

    pub fn kv_layout(&self, n_blocks: usize, block_size: usize, dtype: KvType) -> KvLayout {
        KvLayout {
            n_layers: self.n_layers,
            n_blocks,
            block_size,
            n_kv_heads: self.n_kv_heads,
            head_dim: self.head_dim,
            dtype,
        }
    }
}

struct Layer<B: Backend> {
    attn_norm: B::Weight,
    wq: B::Weight,
    wk: B::Weight,
    wv: B::Weight,
    wo: B::Weight,
    /// Per-head Q and K RMSNorm weights (Qwen3).
    qk_norm: Option<(B::Weight, B::Weight)>,
    ffn_norm: B::Weight,
    gate: B::Weight,
    up: B::Weight,
    down: B::Weight,
}

pub struct Qwen3<B: Backend> {
    pub cfg: Qwen3Config,
    embed: B::Weight,
    layers: Vec<Layer<B>>,
    out_norm: B::Weight,
    /// `None` when the output projection is tied to the embedding table.
    output: Option<B::Weight>,
    /// Llama 3.1+ RoPE scaling: per-frequency divisors (`rope_freqs.weight`).
    rope_freqs: Option<B::Weight>,
}

impl<B: Backend> Qwen3<B> {
    pub fn load(backend: &B, g: &Gguf) -> Result<Self> {
        let cfg = Qwen3Config::from_gguf(g)?;
        let w = |name: &str| load_weight(backend, g, name);
        let llama = cfg.arch == "llama";
        // Q and K in NeoX order (see the module docs).
        let wqk = |name: &str| if llama { load_weight_rope_neox(backend, g, name, cfg.head_dim) } else { w(name) };
        let layers = (0..cfg.n_layers)
            .map(|i| {
                let p = |s: &str| format!("blk.{i}.{s}.weight");
                let qk_norm = match g.tensor(&p("attn_q_norm")) {
                    Some(_) => Some((w(&p("attn_q_norm"))?, w(&p("attn_k_norm"))?)),
                    None => None,
                };
                Ok(Layer {
                    attn_norm: w(&p("attn_norm"))?,
                    wq: wqk(&p("attn_q"))?,
                    wk: wqk(&p("attn_k"))?,
                    wv: w(&p("attn_v"))?,
                    wo: w(&p("attn_output"))?,
                    qk_norm,
                    ffn_norm: w(&p("ffn_norm"))?,
                    gate: w(&p("ffn_gate"))?,
                    up: w(&p("ffn_up"))?,
                    down: w(&p("ffn_down"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let output = match g.tensor("output.weight") {
            Some(_) => Some(w("output.weight")?),
            None => None,
        };
        let rope_freqs = match g.tensor("rope_freqs.weight") {
            Some(_) => Some(w("rope_freqs.weight")?),
            None => None,
        };
        Ok(Self { embed: w("token_embd.weight")?, out_norm: w("output_norm.weight")?, layers, output, rope_freqs, cfg })
    }

    fn forward_impl(&self, b: &B, kv: &mut B::KvCache, batch: &Batch, block_size: usize) -> Result<StepOut> {
        let c = &self.cfg;
        let n = batch.tokens.len();
        let (d, qw, kvw) = (c.hidden, c.q_width(), c.kv_width());
        b.begin_step(batch.graph_key(self as *const Self as usize))?;

        let mut x = b.alloc(n * d)?;
        let mut h = b.alloc(n * d)?;
        // q and attn double as [n, d] scratch for the MLP.
        let mut q = b.alloc(n * qw.max(d))?;
        let mut k = b.alloc(n * kvw)?;
        let mut v = b.alloc(n * kvw)?;
        let mut qn = b.alloc(n * qw)?;
        let mut attn = b.alloc(n * qw.max(d))?;
        let mut gate = b.alloc(n * c.ffn)?;
        let mut up = b.alloc(n * c.ffn)?;

        let meta = AttnMeta {
            n_heads: c.n_heads,
            n_kv_heads: c.n_kv_heads,
            head_dim: c.head_dim,
            scale: 1.0 / (c.head_dim as f32).sqrt(),
            window: c.sliding_window,
            token_seq: batch.token_seq,
            positions: batch.positions,
            block_tables: batch.block_tables,
            block_size,
        };

        b.embed(&self.embed, batch.tokens, &mut x)?;
        b.rms_norm(&x, &self.layers[0].attn_norm, &mut h, n, c.rms_eps)?;
        for (li, l) in self.layers.iter().enumerate() {
            // h = attn_norm(x) on entry (fused into the previous layer's residual add).
            b.matmul_many(&h, &[&l.wq, &l.wk, &l.wv], &mut [&mut q, &mut k, &mut v], n)?;
            b.qkv_norm_rope_store(
                &q, &k, &v, l.qk_norm.as_ref().map(|(a, b)| (a, b)), None, &mut qn, kv, li, batch.positions, batch.slots, c.n_heads,
                c.n_kv_heads, c.head_dim, c.rms_eps, c.rope_theta, self.rope_freqs.as_ref(),
            )?;
            b.attention(&qn, kv, li, &meta, &mut attn)?;
            b.matmul(&attn, &l.wo, &mut h, n)?;
            b.add_rms_norm(&mut x, &h, &l.ffn_norm, &mut q, n, d, c.rms_eps)?;

            b.gated_ffn(&q, &l.gate, &l.up, &l.down, Activation::Silu, &mut attn, &mut gate, &mut up, n, c.ffn)?;
            match self.layers.get(li + 1) {
                Some(next) => b.add_rms_norm(&mut x, &attn, &next.attn_norm, &mut h, n, d, c.rms_eps)?,
                None => b.add(&mut x, &attn, n * d)?,
            }
        }

        let m = batch.logit_rows.len();
        // Every row wanted, in order (decode): no gather needed. Skipping it
        // also keeps the step's last index upload early (see CUDA graph replay).
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

impl<B: Backend> Model<B> for Qwen3<B> {
    type Cache = B::KvCache;

    fn alloc_cache(&self, b: &B, n_blocks: usize, block_size: usize, kv: KvType, _max_seqs: usize) -> Result<B::KvCache> {
        Ok(b.alloc_kv(self.cfg.kv_layout(n_blocks, block_size, kv))?)
    }

    type HostCache = B::HostKv;

    fn kv_block_bytes(&self, block_size: usize, kv: KvType) -> usize {
        self.cfg.kv_layout(1, block_size, kv).block_bytes()
    }

    fn alloc_host_cache(&self, b: &B, n_blocks: usize, block_size: usize, kv: KvType) -> Result<B::HostKv> {
        Ok(b.alloc_host_kv(self.cfg.kv_layout(n_blocks, block_size, kv), n_blocks)?)
    }

    fn kv_to_host(&self, b: &B, cache: &B::KvCache, store: &mut B::HostKv, dev: &[u32], host: &[u32]) -> Result<()> {
        Ok(b.kv_to_host(cache, store, dev, host)?)
    }

    fn kv_from_host(&self, b: &B, cache: &mut B::KvCache, store: &B::HostKv, host: &[u32], dev: &[u32]) -> Result<()> {
        Ok(b.kv_from_host(cache, store, host, dev)?)
    }

    fn forward(&self, b: &B, cache: &mut B::KvCache, batch: &Batch, block_size: usize) -> Result<StepOut> {
        self.forward_impl(b, cache, batch, block_size)
    }

    fn vocab_size(&self) -> usize {
        self.cfg.vocab
    }

    fn summary(&self) -> String {
        let c = &self.cfg;
        format!("{}: {} layers, hidden {}, {} heads / {} kv", c.arch, c.n_layers, c.hidden, c.n_heads, c.n_kv_heads)
    }
}

#[cfg(test)]
mod tests {
    use rllm_gguf::Value;

    use super::*;
    use crate::test_gguf::Spec;

    /// Qwen3-shaped: 2 layers, hidden 8, 2 heads of 4 over 1 KV head, FFN 16, vocab 10.
    fn tiny() -> Spec {
        let mut s = Spec::default();
        s.set("general.architecture", Value::String("qwen3".into()));
        for (k, v) in [("block_count", 2), ("embedding_length", 8), ("feed_forward_length", 16), ("attention.head_count", 2), ("attention.head_count_kv", 1), ("attention.key_length", 4), ("context_length", 64)] {
            s.set(&format!("qwen3.{k}"), Value::U32(v));
        }
        s.tensor("token_embd.weight", &[8, 10]).tensor("output_norm.weight", &[8]);
        for i in 0..2 {
            for (n, d) in [
                ("attn_norm", &[8][..]),
                ("attn_q", &[8, 8]),
                ("attn_k", &[8, 4]),
                ("attn_v", &[8, 4]),
                ("attn_output", &[8, 8]),
                ("attn_q_norm", &[4]),
                ("attn_k_norm", &[4]),
                ("ffn_norm", &[8]),
                ("ffn_gate", &[8, 16]),
                ("ffn_up", &[8, 16]),
                ("ffn_down", &[16, 8]),
            ] {
                s.tensor(&format!("blk.{i}.{n}.weight"), d);
            }
        }
        s
    }

    fn err(s: &Spec) -> String {
        format!("{:#}", Qwen3Config::from_gguf(&s.build().unwrap()).unwrap_err())
    }

    #[test]
    fn valid_config_loads() {
        let g = tiny().build().unwrap();
        let c = Qwen3Config::from_gguf(&g).unwrap();
        assert_eq!((c.n_layers, c.vocab, c.head_dim, c.max_context), (2, 10, 4, 64));
        assert!(crate::ModelProfile::read(&g).is_ok());
    }

    #[test]
    fn malformed_configs_are_errors() {
        // 1-D embedding table.
        assert!(err(tiny().tensor("token_embd.weight", &[80])).contains("token_embd"));
        // Zero heads / KV heads, KV heads not dividing heads.
        assert!(err(tiny().set("qwen3.attention.head_count", Value::U32(0))).contains("head_count"));
        assert!(err(tiny().set("qwen3.attention.head_count_kv", Value::U32(0))).contains("head_count_kv"));
        assert!(err(tiny().set("qwen3.attention.head_count_kv", Value::U32(3))).contains("multiple"));
        // No key_length and a hidden size heads don't divide.
        assert!(err(tiny().remove("qwen3.attention.key_length").set("qwen3.attention.head_count", Value::U32(3))).contains("multiple"));
        // Zero layers, context, absurd sizes.
        assert!(err(tiny().set("qwen3.block_count", Value::U32(0))).contains("block_count"));
        assert!(err(tiny().set("qwen3.context_length", Value::U32(0))).contains("context_length"));
        assert!(err(tiny().set("qwen3.embedding_length", Value::U64(u64::MAX))).contains("out of range"));
        // A tensor whose shape doesn't match the config.
        assert!(err(tiny().tensor("blk.1.attn_k.weight", &[8, 8])).contains("blk.1.attn_k"));
        assert!(err(tiny().tensor("blk.0.ffn_down.weight", &[8, 16])).contains("ffn_down"));
        assert!(err(tiny().tensor("output.weight", &[8, 11])).contains("output.weight"));
        // Bad floats.
        assert!(err(tiny().set("qwen3.rope.freq_base", Value::F32(f32::NAN))).contains("freq_base"));
        assert!(err(tiny().set("qwen3.attention.layer_norm_rms_epsilon", Value::F32(-1.0))).contains("epsilon"));
    }

    #[test]
    fn unsupported_head_layouts_are_rejected() {
        // Partial rotary.
        assert!(err(tiny().set("qwen3.rope.dimension_count", Value::U32(2))).contains("partial rotary"));
        // V heads wider than K heads.
        assert!(err(tiny().set("qwen3.attention.value_length", Value::U32(8))).contains("value head"));
        // Matching values are fine.
        let mut s = tiny();
        s.set("qwen3.rope.dimension_count", Value::U32(4)).set("qwen3.attention.value_length", Value::U32(4));
        assert!(Qwen3Config::from_gguf(&s.build().unwrap()).is_ok());
    }
}
