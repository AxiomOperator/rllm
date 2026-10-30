//! Qwen3 dense decoder (GGUF architecture `qwen3`).
//!
//! Per layer: RMSNorm → Q/K/V projections → per-head RMSNorm on Q and K →
//! NeoX RoPE → paged GQA attention → output projection → residual, then
//! RMSNorm → SwiGLU MLP → residual.

use anyhow::{Context, Result, bail};
use rllm_core::{Activation, AttnMeta, Backend, KvLayout, KvType};
use rllm_gguf::Gguf;

use crate::loader::load_weight;
use crate::{Batch, Model};

#[derive(Debug, Clone)]
pub struct Qwen3Config {
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
}

impl Qwen3Config {
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        let arch = g.get_str("general.architecture")?;
        if arch != "qwen3" {
            bail!("unsupported architecture {arch:?} (only qwen3 for now)");
        }
        let k = |s: &str| format!("{arch}.{s}");
        let n_heads = g.get_u64(&k("attention.head_count"))? as usize;
        let hidden = g.get_u64(&k("embedding_length"))? as usize;
        let head_dim = match g.get(&k("attention.key_length")) {
            Some(v) => v.as_u64().context("attention.key_length")? as usize,
            None => hidden / n_heads,
        };
        let vocab = g.require_tensor("token_embd.weight")?.dims[1] as usize;
        Ok(Self {
            n_layers: g.get_u64(&k("block_count"))? as usize,
            hidden,
            ffn: g.get_u64(&k("feed_forward_length"))? as usize,
            n_heads,
            n_kv_heads: g.get_u64(&k("attention.head_count_kv"))? as usize,
            head_dim,
            vocab,
            rope_theta: g.get_f64(&k("rope.freq_base")).unwrap_or(1_000_000.0) as f32,
            rms_eps: g.get_f64(&k("attention.layer_norm_rms_epsilon")).unwrap_or(1e-6) as f32,
            max_context: g.get_u64(&k("context_length")).unwrap_or(32768) as usize,
        })
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
    q_norm: B::Weight,
    k_norm: B::Weight,
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
}

impl<B: Backend> Qwen3<B> {
    pub fn load(backend: &B, g: &Gguf) -> Result<Self> {
        let cfg = Qwen3Config::from_gguf(g)?;
        let w = |name: &str| load_weight(backend, g, name);
        let layers = (0..cfg.n_layers)
            .map(|i| {
                let p = |s: &str| format!("blk.{i}.{s}.weight");
                Ok(Layer {
                    attn_norm: w(&p("attn_norm"))?,
                    wq: w(&p("attn_q"))?,
                    wk: w(&p("attn_k"))?,
                    wv: w(&p("attn_v"))?,
                    wo: w(&p("attn_output"))?,
                    q_norm: w(&p("attn_q_norm"))?,
                    k_norm: w(&p("attn_k_norm"))?,
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
        Ok(Self { embed: w("token_embd.weight")?, out_norm: w("output_norm.weight")?, layers, output, cfg })
    }

    fn forward_impl(&self, b: &B, kv: &mut B::KvCache, batch: &Batch, block_size: usize) -> Result<Vec<f32>> {
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
            window: 0,
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
                &q, &k, &v, &l.q_norm, &l.k_norm, None, &mut qn, kv, li, batch.positions, batch.slots, c.n_heads,
                c.n_kv_heads, c.head_dim, c.rms_eps, c.rope_theta, None,
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
            return Ok(host);
        }
        let mut logits = b.alloc(m * c.vocab)?;
        b.matmul(&normed, self.output.as_ref().unwrap_or(&self.embed), &mut logits, m)?;
        let mut host = vec![0f32; m * c.vocab];
        b.to_host(&logits, &mut host)?;
        Ok(host)
    }
}

impl<B: Backend> Model<B> for Qwen3<B> {
    type Cache = B::KvCache;

    fn alloc_cache(&self, b: &B, n_blocks: usize, block_size: usize, kv: KvType) -> Result<B::KvCache> {
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

    fn forward(&self, b: &B, cache: &mut B::KvCache, batch: &Batch, block_size: usize) -> Result<Vec<f32>> {
        self.forward_impl(b, cache, batch, block_size)
    }

    fn vocab_size(&self) -> usize {
        self.cfg.vocab
    }

    fn summary(&self) -> String {
        let c = &self.cfg;
        format!("qwen3: {} layers, hidden {}, {} heads / {} kv", c.n_layers, c.hidden, c.n_heads, c.n_kv_heads)
    }
}
