//! Qwen3.5 / 3.6 / 3.8 dense hybrids (GGUF architecture `qwen35`).
//!
//! Three of every four layers are Gated DeltaNet (linear attention with a
//! recurrent state per sequence); every fourth is gated full attention.
//! Every layer has a SwiGLU MLP.
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
//! The MTP (next-token prediction) block some files carry is not used yet.
//! Recurrent state can't be rewound or rebuilt from cached KV, so prefix
//! reuse and speculative decoding are off for these models.

use anyhow::{Context, Result, bail, ensure};
use rllm_core::{Activation, AttnMeta, Backend, GdnShape, HostTensor, KvLayout, KvType, SeqRuns};
use rllm_gguf::Gguf;

use crate::loader::{MAX_CONTEXT, MAX_DIM, check_shape, check_shape_opt, load_weight, load_weight_rows, meta_f32_or, meta_usize, meta_usize_or, read_vector};
use crate::{Batch, Model, StepOut};

#[derive(Debug, Clone)]
pub struct Qwen35Config {
    /// Decoder layers run in a forward pass (MTP blocks excluded).
    pub n_layers: usize,
    pub hidden: usize,
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
}

impl Qwen35Config {
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        let arch = g.get_str("general.architecture")?;
        if arch != "qwen35" {
            bail!("unsupported architecture {arch:?} (expected qwen35)");
        }
        let k = |s: &str| format!("qwen35.{s}");
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
        let cfg = Self {
            n_layers,
            hidden: dim("embedding_length")?,
            ffn: dim("feed_forward_length")?,
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
        Ok(())
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
            let mut want: Vec<(String, Vec<usize>)> = vec![
                (p("attn_norm.weight"), vec![d]),
                (p("post_attention_norm.weight"), vec![d]),
                (p("ffn_gate.weight"), vec![d, c.ffn]),
                (p("ffn_up.weight"), vec![d, c.ffn]),
                (p("ffn_down.weight"), vec![c.ffn, d]),
            ];
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
        Ok(())
    }

    pub fn n_attn_layers(&self) -> usize {
        self.recurrent.iter().filter(|r| !**r).count()
    }

    pub fn n_recurrent_layers(&self) -> usize {
        self.recurrent.iter().filter(|r| **r).count()
    }

    pub fn kv_layout(&self, n_blocks: usize, block_size: usize, dtype: KvType) -> KvLayout {
        KvLayout { n_layers: self.n_attn_layers(), n_blocks, block_size, n_kv_heads: self.n_kv_heads, head_dim: self.head_dim, dtype }
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

pub struct Qwen35<B: Backend> {
    pub cfg: Qwen35Config,
    embed: B::Weight,
    layers: Vec<Layer<B>>,
    out_norm: B::Weight,
    output: Option<B::Weight>,
    rope_factors: B::Weight,
}

/// KV cache of the attention layers plus each sequence slot's recurrent
/// state, per recurrent layer.
pub struct Cache<B: Backend> {
    kv: B::KvCache,
    conv: Vec<B::Buffer>,
    ssm: Vec<B::Buffer>,
    n_slots: usize,
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
                    // attn_q holds per head [query (head_dim) | gate (head_dim)].
                    let q_rows = qk_rows(cfg.n_heads, 2 * hd);
                    let gate_rows: Vec<usize> = (0..cfg.n_heads).flat_map(|h| (0..hd).map(move |j| h * 2 * hd + hd + j)).collect();
                    Mixer::Attn {
                        wq: load_weight_rows(backend, g, &p("attn_q.weight"), &q_rows)?,
                        wgate: load_weight_rows(backend, g, &p("attn_q.weight"), &gate_rows)?,
                        wk: load_weight_rows(backend, g, &p("attn_k.weight"), &qk_rows(cfg.n_kv_heads, hd))?,
                        wv: w(&p("attn_v.weight"))?,
                        wo: w(&p("attn_output.weight"))?,
                        q_norm: reordered(&p("attn_q_norm.weight"))?,
                        k_norm: reordered(&p("attn_k_norm.weight"))?,
                        kv_layer: n_attn - 1,
                    }
                };
                Ok(Layer {
                    attn_norm: w(&p("attn_norm.weight"))?,
                    mixer,
                    ffn: Ffn {
                        post_norm: w(&p("post_attention_norm.weight"))?,
                        gate: w(&p("ffn_gate.weight"))?,
                        up: w(&p("ffn_up.weight"))?,
                        down: w(&p("ffn_down.weight"))?,
                    },
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let output = match g.tensor("output.weight") {
            Some(_) => Some(w("output.weight")?),
            None => None,
        };
        let rope_factors = upload_vec(backend, "rope_factors", &cfg.rope_factors())?;
        Ok(Self { embed: w("token_embd.weight")?, out_norm: w("output_norm.weight")?, layers, output, rope_factors, cfg })
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
        let runs = SeqRuns { starts: &starts, slots: batch.state_slots, fresh: &fresh };
        runs.check(n, cache.n_slots)?;

        let (d, hd) = (c.hidden, c.head_dim);
        let qw = c.n_heads * hd;
        let kvw = c.n_kv_heads * hd;
        let (conv_w, inner, hv) = (c.gdn.qkv_width(), c.gdn.n_v_heads * c.gdn.head_dim, c.gdn.n_v_heads);
        // No graph capture yet: the recurrent ops' index data isn't graph-safe.
        b.begin_step(None)?;

        let mut x = b.alloc(n * d)?;
        let mut h = b.alloc(n * d)?;
        let mut o = b.alloc(n * d)?;
        let mut a1 = b.alloc(n * qw.max(conv_w))?;
        let mut a2 = b.alloc(n * qw.max(inner))?;
        let mut a3 = b.alloc(n * kvw.max(hv).max(conv_w))?;
        let mut a4 = b.alloc(n * kvw.max(hv))?;
        let mut a5 = b.alloc(n * qw.max(inner))?;
        let mut gate = b.alloc(n * c.ffn)?;
        let mut up = b.alloc(n * c.ffn)?;

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
                    b.conv1d_silu(&a1, conv, &mut cache.conv[*state_layer], &runs, &mut conv_out, conv_w, c.conv_kernel)?;
                    b.gated_delta_net(&conv_out, &a3, &a4, dt_bias, a, &mut cache.ssm[*state_layer], &runs, &mut a1, c.gdn, c.rms_eps)?;
                    b.rms_norm(&a1, norm, &mut a5, n * hv, c.rms_eps)?;
                    // z = silu(z) * norm(o)
                    b.silu_mul(&mut a2, &a5, n * inner)?;
                    b.matmul(&a2, wout, &mut o, n)?;
                }
            }
            b.add_rms_norm(&mut x, &o, &l.ffn.post_norm, &mut h, n, d, c.rms_eps)?;
            b.gated_ffn(&h, &l.ffn.gate, &l.ffn.up, &l.ffn.down, Activation::Silu, &mut o, &mut gate, &mut up, n, c.ffn)?;
            match self.layers.get(li + 1) {
                Some(next) => b.add_rms_norm(&mut x, &o, &next.attn_norm, &mut h, n, d, c.rms_eps)?,
                None => b.add(&mut x, &o, n * d)?,
            }
        }

        let m = batch.logit_rows.len();
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
        })
    }

    fn recurrent(&self) -> bool {
        true
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
        format!(
            "qwen35: {} layers ({} Gated DeltaNet, {} attention), hidden {}, {} heads / {} kv",
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
