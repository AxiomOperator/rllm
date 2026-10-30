//! Gemma 4 dense decoder (GGUF architecture `gemma4`), e.g. Gemma 4 E2B/E4B
//! and fine-tunes such as Nemotron 3.5 Bolt.
//!
//! Ported from llama.cpp `src/models/gemma4.cpp` (MIT). Differences from a
//! plain Llama-style decoder:
//! - Interleaved sliding-window (local) and global attention layers, with
//!   different head sizes and RoPE bases. Global layers use per-frequency
//!   RoPE factors (partial rotary).
//! - The last `shared_kv_layers` layers compute no K/V; they attend over the
//!   cache of the last earlier layer of the same kind.
//! - Q/K get per-head RMSNorm, V gets a weightless RMSNorm, attention scale is 1.
//! - Extra RMSNorms after attention and after the MLP; GELU (tanh) MLP.
//! - Per-layer embeddings (PLE): a second embedding table gives each layer a
//!   small input vector that gates a residual branch after the MLP.
//! - Each layer's output is multiplied by a learned scalar.
//! - Final logits are soft-capped: `cap * tanh(logits / cap)`.

use anyhow::{Context, Result, bail};
use rllm_core::{Activation, AttnMeta, Backend, HostTensor, KvLayout, KvType};
use rllm_gguf::{Gguf, Value};

use crate::loader::{host_tensor, load_weight};
use crate::{Batch, Model, StepOut};

#[derive(Debug, Clone)]
pub struct Gemma4Config {
    pub n_layers: usize,
    pub hidden: usize,
    pub ffn: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim_global: usize,
    pub head_dim_swa: usize,
    pub rope_theta_global: f32,
    pub rope_theta_swa: f32,
    pub sliding_window: usize,
    /// `is_swa[i]`: layer `i` uses sliding-window attention.
    pub is_swa: Vec<bool>,
    /// Layers `>= n_kv_layers` reuse an earlier layer's KV cache.
    pub n_kv_layers: usize,
    pub per_layer_dim: usize,
    pub vocab: usize,
    pub rms_eps: f32,
    pub logit_softcap: Option<f32>,
    pub max_context: usize,
}

impl Gemma4Config {
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        let k = |s: &str| format!("gemma4.{s}");
        let u = |s: &str| -> Result<usize> { Ok(g.get_u64(&k(s))? as usize) };
        let n_layers = u("block_count")?;
        let is_swa: Vec<bool> = g
            .get_array(&k("attention.sliding_window_pattern"))?
            .iter()
            .map(|v| matches!(v, Value::Bool(true)))
            .collect();
        if is_swa.len() != n_layers {
            bail!("sliding_window_pattern has {} entries for {n_layers} layers", is_swa.len());
        }
        let shared = g.get_u64(&k("attention.shared_kv_layers")).unwrap_or(0) as usize;
        let n_kv_layers = n_layers - shared;
        if shared > 0 && (n_kv_layers < 2 || is_swa[n_kv_layers - 1] || !is_swa[n_kv_layers - 2]) {
            // llama.cpp hard-codes the reuse sources as the last global (n-1) and
            // last sliding (n-2) layer; check the layout actually matches that.
            bail!("unexpected layer layout for KV sharing");
        }
        for key in ["feed_forward_length", "attention.head_count", "attention.head_count_kv"] {
            if g.get(&k(key)).is_some_and(|v| v.as_array().is_some()) {
                bail!("per-layer {key} not supported yet");
            }
        }
        if g.tensor("blk.0.ffn_gate_inp.weight").is_some() {
            bail!("Gemma 4 MoE variants not supported yet");
        }
        Ok(Self {
            n_layers,
            hidden: u("embedding_length")?,
            ffn: u("feed_forward_length")?,
            n_heads: u("attention.head_count")?,
            n_kv_heads: u("attention.head_count_kv")?,
            head_dim_global: u("attention.key_length")?,
            head_dim_swa: u("attention.key_length_swa")?,
            rope_theta_global: g.get_f64(&k("rope.freq_base"))? as f32,
            rope_theta_swa: g.get_f64(&k("rope.freq_base_swa"))? as f32,
            sliding_window: u("attention.sliding_window")?,
            is_swa,
            n_kv_layers,
            per_layer_dim: g.get_u64(&k("embedding_length_per_layer_input")).unwrap_or(0) as usize,
            vocab: g.require_tensor("token_embd.weight")?.dims[1] as usize,
            rms_eps: g.get_f64(&k("attention.layer_norm_rms_epsilon"))? as f32,
            logit_softcap: g.get_f64(&k("final_logit_softcapping")).ok().map(|v| v as f32).filter(|&v| v > 0.0),
            max_context: u("context_length").unwrap_or(8192),
        })
    }

    /// Layers with their own sliding-window and global KV cache.
    pub fn kv_layer_counts(&self) -> (usize, usize) {
        let swa = self.is_swa[..self.n_kv_layers].iter().filter(|&&s| s).count();
        (swa, self.n_kv_layers - swa)
    }

    fn head_dim(&self, layer: usize) -> usize {
        if self.is_swa[layer] { self.head_dim_swa } else { self.head_dim_global }
    }

    /// Which layer's KV cache `layer` attends over (itself unless shared).
    fn kv_source(&self, layer: usize) -> usize {
        if layer < self.n_kv_layers {
            layer
        } else if self.is_swa[layer] {
            self.n_kv_layers - 2
        } else {
            self.n_kv_layers - 1
        }
    }
}

struct Layer<B: Backend> {
    is_swa: bool,
    attn_norm: B::Weight,
    wq: B::Weight,
    /// K/V projections exist only for layers that own KV.
    wk: Option<B::Weight>,
    /// Absent in some checkpoints: V then reuses the K projection.
    wv: Option<B::Weight>,
    q_norm: B::Weight,
    k_norm: Option<B::Weight>,
    wo: B::Weight,
    post_attn_norm: B::Weight,
    ffn_norm: B::Weight,
    gate: B::Weight,
    up: B::Weight,
    down: B::Weight,
    post_ffw_norm: B::Weight,
    /// Per-layer-embedding branch: hidden -> per_layer_dim gate, back to hidden.
    ple_gate: Option<B::Weight>,
    ple_proj: Option<B::Weight>,
    ple_post_norm: Option<B::Weight>,
    out_scale: f32,
}

struct PerLayerEmbed<B: Backend> {
    table: B::Weight,
    model_proj: B::Weight,
    proj_norm: B::Weight,
}

pub struct Gemma4<B: Backend> {
    pub cfg: Gemma4Config,
    embed: B::Weight,
    output: Option<B::Weight>,
    out_norm: B::Weight,
    rope_freqs: Option<B::Weight>,
    ple: Option<PerLayerEmbed<B>>,
    /// Weightless RMSNorm for V, as all-ones weights (global, swa head sizes).
    ones_global: B::Weight,
    ones_swa: B::Weight,
    layers: Vec<Layer<B>>,
    /// For each layer: (is_swa cache, index within that cache) of the KV it attends to.
    kv_index: Vec<(bool, usize)>,
    n_swa_kv: usize,
    n_global_kv: usize,
}

/// Separate caches for sliding-window and global layers (different head sizes).
/// Both use the same block size and count, so one block table serves both.
pub struct Cache<B: Backend> {
    swa: B::KvCache,
    global: B::KvCache,
}

pub struct HostCache<B: Backend> {
    swa: B::HostKv,
    global: B::HostKv,
}

fn scalar_f32(g: &Gguf, name: &str) -> Result<Option<f32>> {
    let Some(info) = g.tensor(name) else { return Ok(None) };
    let t = host_tensor(g, &info.name)?;
    let mut v = vec![0f32; t.numel()];
    rllm_core::quant::dequantize_row(t.dtype, t.bytes.as_slice(), &mut v);
    Ok(v.first().copied())
}

impl<B: Backend> Gemma4<B> {
    /// Sliding-window and global cache layouts (separate pools: their head
    /// sizes differ). Both use the same block ids.
    fn kv_layouts(&self, n_blocks: usize, block_size: usize, kv: KvType) -> [KvLayout; 2] {
        let c = &self.cfg;
        let layout = |n_layers, head_dim| KvLayout { n_layers, n_blocks, block_size, n_kv_heads: c.n_kv_heads, head_dim, dtype: kv };
        [layout(self.n_swa_kv.max(1), c.head_dim_swa), layout(self.n_global_kv.max(1), c.head_dim_global)]
    }

    pub fn load(backend: &B, g: &Gguf) -> Result<Self> {
        let cfg = Gemma4Config::from_gguf(g)?;
        let w = |name: &str| load_weight(backend, g, name);
        let opt = |name: &str| -> Result<Option<B::Weight>> {
            if g.tensor(name).is_some() { Ok(Some(w(name)?)) } else { Ok(None) }
        };

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let p = |s: &str| format!("blk.{i}.{s}.weight");
            let has_kv = i < cfg.n_kv_layers;
            let has_ple = cfg.per_layer_dim > 0;
            layers.push(Layer {
                is_swa: cfg.is_swa[i],
                attn_norm: w(&p("attn_norm"))?,
                wq: w(&p("attn_q"))?,
                wk: if has_kv { Some(w(&p("attn_k"))?) } else { None },
                wv: if has_kv { opt(&p("attn_v"))? } else { None },
                q_norm: w(&p("attn_q_norm"))?,
                k_norm: if has_kv { Some(w(&p("attn_k_norm"))?) } else { None },
                wo: w(&p("attn_output"))?,
                post_attn_norm: w(&p("post_attention_norm"))?,
                ffn_norm: w(&p("ffn_norm"))?,
                gate: w(&p("ffn_gate"))?,
                up: w(&p("ffn_up"))?,
                down: w(&p("ffn_down"))?,
                post_ffw_norm: w(&p("post_ffw_norm"))?,
                ple_gate: if has_ple { Some(w(&p("inp_gate"))?) } else { None },
                ple_proj: if has_ple { Some(w(&p("proj"))?) } else { None },
                ple_post_norm: if has_ple { Some(w(&p("post_norm"))?) } else { None },
                out_scale: scalar_f32(g, &p("layer_output_scale"))?.unwrap_or(1.0),
            });
        }

        // Assign each KV-owning layer a slot in its kind's cache, then point
        // shared layers at their source layer's slot.
        let (mut n_swa_kv, mut n_global_kv) = (0, 0);
        let mut own = vec![(false, 0); cfg.n_layers];
        for i in 0..cfg.n_kv_layers {
            own[i] = if cfg.is_swa[i] {
                n_swa_kv += 1;
                (true, n_swa_kv - 1)
            } else {
                n_global_kv += 1;
                (false, n_global_kv - 1)
            };
        }
        let kv_index = (0..cfg.n_layers).map(|i| own[cfg.kv_source(i)]).collect();

        let ple = if cfg.per_layer_dim > 0 {
            Some(PerLayerEmbed {
                table: w("per_layer_token_embd.weight")?,
                model_proj: w("per_layer_model_proj.weight")?,
                proj_norm: w("per_layer_proj_norm.weight")?,
            })
        } else {
            None
        };

        let ones = |d: usize| -> Result<B::Weight> {
            Ok(backend.upload(&HostTensor::from_f32("v_norm_ones", vec![d], &vec![1.0; d]))?)
        };

        Ok(Self {
            embed: w("token_embd.weight")?,
            output: opt("output.weight")?,
            out_norm: w("output_norm.weight")?,
            rope_freqs: opt("rope_freqs.weight")?,
            ple,
            ones_global: ones(cfg.head_dim_global)?,
            ones_swa: ones(cfg.head_dim_swa)?,
            layers,
            kv_index,
            n_swa_kv,
            n_global_kv,
            cfg,
        })
    }

    fn forward_impl(&self, b: &B, cache: &mut Cache<B>, batch: &Batch, block_size: usize) -> Result<StepOut> {
        let c = &self.cfg;
        let n = batch.tokens.len();
        let d = c.hidden;
        b.begin_step(batch.graph_key(self as *const Self as usize))?;
        let hd_max = c.head_dim_global.max(c.head_dim_swa);
        let (qw_max, kvw_max) = (c.n_heads * hd_max, c.n_kv_heads * hd_max);
        let (n_l, p_dim) = (c.n_layers, c.per_layer_dim);

        let mut x = b.alloc(n * d)?;
        let mut h = b.alloc(n * d)?;
        let mut q = b.alloc(n * qw_max)?;
        let mut qn = b.alloc(n * qw_max)?;
        let mut k = b.alloc(n * kvw_max)?;
        let mut v = b.alloc(n * kvw_max)?;
        let mut attn = b.alloc(n * qw_max)?;
        let mut gate = b.alloc(n * c.ffn)?;
        let mut up = b.alloc(n * c.ffn)?;

        b.embed(&self.embed, batch.tokens, &mut x)?;
        b.scale(&mut x, (d as f32).sqrt(), n * d)?;

        // Per-layer inputs, [n, n_layers, per_layer_dim]:
        //   (rmsnorm(x · proj / sqrt(d)) + ple_embed(tok) * sqrt(p)) / sqrt(2)
        let ple_inputs = match &self.ple {
            Some(ple) => {
                let len = n * n_l * p_dim;
                let mut from_tok = b.alloc(len)?;
                b.embed(&ple.table, batch.tokens, &mut from_tok)?;
                b.scale(&mut from_tok, (p_dim as f32).sqrt(), len)?;
                let mut proj = b.alloc(len)?;
                b.matmul(&x, &ple.model_proj, &mut proj, n)?;
                b.scale(&mut proj, 1.0 / (d as f32).sqrt(), len)?;
                let mut inputs = b.alloc(len)?;
                b.rms_norm(&proj, &ple.proj_norm, &mut inputs, n * n_l, c.rms_eps)?;
                b.add(&mut inputs, &from_tok, len)?;
                b.scale(&mut inputs, std::f32::consts::FRAC_1_SQRT_2, len)?;
                Some(inputs)
            }
            None => None,
        };
        let mut ple_gate = b.alloc(n * p_dim.max(1))?;
        // Second residual buffer: ple_gate writes the updated residual here
        // (it can't update x in place), then the two swap.
        let mut xn = if p_dim > 0 { Some(b.alloc(n * d)?) } else { None };
        let mut ple_slice = b.alloc(n * p_dim.max(1))?;

        let meta = |swa: bool| AttnMeta {
            n_heads: c.n_heads,
            n_kv_heads: c.n_kv_heads,
            head_dim: if swa { c.head_dim_swa } else { c.head_dim_global },
            scale: 1.0,
            window: if swa { c.sliding_window } else { 0 },
            token_seq: batch.token_seq,
            positions: batch.positions,
            block_tables: batch.block_tables,
            block_size,
        };
        let (meta_swa, meta_global) = (meta(true), meta(false));

        b.rms_norm(&x, &self.layers[0].attn_norm, &mut h, n, c.rms_eps)?;
        for (li, l) in self.layers.iter().enumerate() {
            let hd = c.head_dim(li);
            let (theta, factors) =
                if l.is_swa { (c.rope_theta_swa, None) } else { (c.rope_theta_global, self.rope_freqs.as_ref()) };
            // The next layer's attention norm, and the matmuls that consume it.
            let next_layer = self.layers.get(li + 1);
            let next_consumers: Vec<&B::Weight> = next_layer
                .map(|nl| [Some(&nl.wq), nl.wk.as_ref(), nl.wv.as_ref().or(nl.wk.as_ref())].into_iter().flatten().collect())
                .unwrap_or_default();
            let next_norm = next_layer.map(|nl| &nl.attn_norm);

            // Attention (h = attn_norm(x) on entry).
            if let (Some(wk), Some(k_norm)) = (&l.wk, &l.k_norm) {
                let wv = l.wv.as_ref().unwrap_or(wk);
                b.matmul_many(&h, &[&l.wq, wk, wv], &mut [&mut q, &mut k, &mut v], n)?;
                let ones = if l.is_swa { &self.ones_swa } else { &self.ones_global };
                let (swa, idx) = self.kv_index[li];
                let kv = if swa { &mut cache.swa } else { &mut cache.global };
                b.qkv_norm_rope_store(
                    &q, &k, &v, Some((&l.q_norm, k_norm)), Some(ones), &mut qn, kv, idx, batch.positions, batch.slots,
                    c.n_heads, c.n_kv_heads, hd, c.rms_eps, theta, factors,
                )?;
            } else {
                b.matmul(&h, &l.wq, &mut q, n)?;
                b.rms_norm_rope(&q, &l.q_norm, &mut qn, batch.positions, c.n_heads, hd, c.rms_eps, theta, factors)?;
            }

            let (swa, idx) = self.kv_index[li];
            let (kv, m) = if swa { (&cache.swa, &meta_swa) } else { (&cache.global, &meta_global) };
            b.attention(&qn, kv, idx, m, &mut attn)?;
            b.matmul(&attn, &l.wo, &mut h, n)?;
            // x += post_attn_norm(o); MLP on ffn_norm(x)  (q is free scratch here).
            // With per-layer inputs the residual moves x -> xn here and back to
            // x in ple_gate below (fused backends can't update it in place).
            let ffn = (&l.gate, &l.up, &l.down, Activation::GeluTanh);
            if let Some(xn) = xn.as_mut().filter(|_| l.ple_gate.is_some()) {
                b.norm_add_gated_ffn(&h, &l.post_attn_norm, &x, xn, n, d, c.rms_eps, 1.0, &l.ffn_norm, ffn, &mut q, &mut attn, &mut gate, &mut up, c.ffn)?;
                std::mem::swap(&mut x, xn);
            } else {
                b.rms_norm_add(&h, &l.post_attn_norm, &mut x, n, d, c.rms_eps, 1.0, Some((&l.ffn_norm, &mut q, &[&l.gate, &l.up])))?;
                b.gated_ffn(&q, &l.gate, &l.up, &l.down, Activation::GeluTanh, &mut attn, &mut gate, &mut up, n, c.ffn)?;
            }

            // Per-layer embedding branch:
            //   x += rmsnorm(proj(gelu(gate(x)) * ple_inputs[:, li, :]))
            // Each path ends with x *= out_scale and h = next attn_norm(x).
            let ple = (&ple_inputs, &l.ple_gate, &l.ple_proj, &l.ple_post_norm);
            if let (Some(inputs), Some(pg), Some(pp), Some(pn), Some(xn)) = (ple.0, ple.1, ple.2, ple.3, xn.as_mut()) {
                b.ple_gate(&attn, &l.post_ffw_norm, &x, xn, n, d, c.rms_eps, pg, inputs, li, n_l, p_dim, &mut ple_gate, &mut ple_slice)?;
                b.matmul(&ple_gate, pp, &mut attn, n)?;
                b.rms_norm_add(&attn, pn, xn, n, d, c.rms_eps, l.out_scale, next_norm.map(|w| (w, &mut h, &next_consumers[..])))?;
                std::mem::swap(&mut x, xn);
            } else {
                b.rms_norm_add(&attn, &l.post_ffw_norm, &mut x, n, d, c.rms_eps, l.out_scale, next_norm.map(|w| (w, &mut h, &next_consumers[..])))?;
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
        if let Some(cap) = c.logit_softcap {
            b.softcap(&mut logits, cap, m * c.vocab)?;
        }
        crate::finish_logits(b, &logits, m, c.vocab, batch)
    }
}

impl<B: Backend> Model<B> for Gemma4<B> {
    type Cache = Cache<B>;

    fn alloc_cache(&self, b: &B, n_blocks: usize, block_size: usize, kv: KvType) -> Result<Cache<B>> {
        let [swa, global] = self.kv_layouts(n_blocks, block_size, kv);
        Ok(Cache {
            swa: b.alloc_kv(swa).context("allocating sliding-window KV cache")?,
            global: b.alloc_kv(global).context("allocating global KV cache")?,
        })
    }

    type HostCache = HostCache<B>;

    fn kv_block_bytes(&self, block_size: usize, kv: KvType) -> usize {
        self.kv_layouts(1, block_size, kv).iter().map(KvLayout::block_bytes).sum()
    }

    fn alloc_host_cache(&self, b: &B, n_blocks: usize, block_size: usize, kv: KvType) -> Result<HostCache<B>> {
        let [swa, global] = self.kv_layouts(n_blocks, block_size, kv);
        Ok(HostCache { swa: b.alloc_host_kv(swa, n_blocks)?, global: b.alloc_host_kv(global, n_blocks)? })
    }

    fn kv_to_host(&self, b: &B, cache: &Cache<B>, store: &mut HostCache<B>, dev: &[u32], host: &[u32]) -> Result<()> {
        b.kv_to_host(&cache.swa, &mut store.swa, dev, host)?;
        Ok(b.kv_to_host(&cache.global, &mut store.global, dev, host)?)
    }

    fn kv_from_host(&self, b: &B, cache: &mut Cache<B>, store: &HostCache<B>, host: &[u32], dev: &[u32]) -> Result<()> {
        b.kv_from_host(&mut cache.swa, &store.swa, host, dev)?;
        Ok(b.kv_from_host(&mut cache.global, &store.global, host, dev)?)
    }

    fn forward(&self, b: &B, cache: &mut Cache<B>, batch: &Batch, block_size: usize) -> Result<StepOut> {
        self.forward_impl(b, cache, batch, block_size)
    }

    fn vocab_size(&self) -> usize {
        self.cfg.vocab
    }

    fn summary(&self) -> String {
        let c = &self.cfg;
        format!(
            "gemma4: {} layers ({} sliding / {} global, KV shared after layer {}), hidden {}, {} heads / {} kv, head {}/{}, PLE {}",
            c.n_layers,
            c.is_swa.iter().filter(|s| **s).count(),
            c.is_swa.iter().filter(|s| !**s).count(),
            c.n_kv_layers - 1,
            c.hidden,
            c.n_heads,
            c.n_kv_heads,
            c.head_dim_swa,
            c.head_dim_global,
            c.per_layer_dim,
        )
    }
}
