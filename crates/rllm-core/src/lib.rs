//! Backend-independent core: dtypes, quantization formats and the `Backend`
//! trait every compute backend (CPU, CUDA, later HIP) implements.
//!
//! Nothing vendor-specific lives here. The model code only talks to a
//! `Backend`, so adding a backend never touches the model.

mod dtype;
pub mod quant;
pub mod sample;

use std::ops::Range;
use std::sync::Arc;

pub use dtype::DType;
pub use sample::RowSample;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("unsupported: {0}")]
    Unsupported(String),
    #[error("shape mismatch: {0}")]
    Shape(String),
    #[error("backend: {0}")]
    Backend(String),
}

pub type Result<T> = std::result::Result<T, Error>;

/// A byte range inside some shared owner (typically a memory-mapped file).
#[derive(Clone)]
pub struct HostBytes {
    owner: Arc<dyn AsRef<[u8]> + Send + Sync>,
    range: Range<usize>,
}

impl HostBytes {
    pub fn new(owner: Arc<dyn AsRef<[u8]> + Send + Sync>, range: Range<usize>) -> Self {
        assert!(range.end <= (*owner).as_ref().len());
        Self { owner, range }
    }

    pub fn as_slice(&self) -> &[u8] {
        &(*self.owner).as_ref()[self.range.clone()]
    }
}

/// A weight tensor in host memory, as read from a model file.
///
/// `shape` is row-major (outermost first), the reverse of GGUF's dims.
/// A 2-D weight `[rows, cols]` has quantization blocks running along `cols`.
#[derive(Clone)]
pub struct HostTensor {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub bytes: HostBytes,
}

impl HostTensor {
    /// An owned F32 tensor, for weights a model synthesizes rather than loads.
    pub fn from_f32(name: &str, shape: Vec<usize>, data: &[f32]) -> Self {
        assert_eq!(shape.iter().product::<usize>(), data.len());
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let len = bytes.len();
        let owner: Arc<dyn AsRef<[u8]> + Send + Sync> = Arc::new(bytes);
        Self { name: name.to_string(), dtype: DType::F32, shape, bytes: HostBytes::new(owner, 0..len) }
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// `(rows, cols)` treating everything but the last axis as rows.
    pub fn as_2d(&self) -> (usize, usize) {
        let cols = *self.shape.last().unwrap_or(&1);
        (self.numel() / cols.max(1), cols)
    }
}

/// How the KV cache stores keys and values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KvType {
    /// 16-bit floats.
    #[default]
    F16,
    /// 8-bit integers with one f16 scale per 32 values (GGUF's q8_0 layout):
    /// about half the memory of f16.
    Q8,
}

impl std::str::FromStr for KvType {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "f16" => Ok(Self::F16),
            "q8" | "q8_0" => Ok(Self::Q8),
            _ => Err(format!("unknown KV cache type {s:?} (expected f16 or q8)")),
        }
    }
}

impl std::fmt::Display for KvType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::F16 => "f16",
            Self::Q8 => "q8",
        })
    }
}

/// Shape of the paged KV cache.
///
/// Each layer has a K and a V pool of `n_blocks * block_size` slots; a slot
/// holds one token's keys (or values) for all KV heads: `n_kv_heads * head_dim`.
#[derive(Debug, Clone, Copy)]
pub struct KvLayout {
    pub n_layers: usize,
    pub n_blocks: usize,
    pub block_size: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub dtype: KvType,
}

impl KvLayout {
    pub fn slot_width(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }

    /// Bytes one block takes across all layers, K and V (and q8 scales).
    pub fn block_bytes(&self) -> usize {
        let n = self.block_size * self.slot_width();
        let per_layer = match self.dtype {
            KvType::F16 => 2 * n,
            KvType::Q8 => n + n / 32 * 2,
        };
        2 * per_layer * self.n_layers
    }
}

/// Per-step attention metadata for a flattened batch of tokens.
///
/// Tokens from all sequences are concatenated (prefill and decode can mix).
/// Token `t` belongs to sequence `token_seq[t]`, sits at `positions[t]`, and
/// attends to positions `0..=positions[t]` of that sequence through its block
/// table (or only the last `window` of them for sliding-window attention).
/// All KV for those positions must already be stored.
pub struct AttnMeta<'a> {
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub scale: f32,
    /// Sliding window size: attend to keys with `pos - key_pos < window`. 0 = unlimited.
    pub window: usize,
    pub token_seq: &'a [u32],
    pub positions: &'a [u32],
    pub block_tables: &'a [Vec<u32>],
    pub block_size: usize,
}

/// Activation of a gated FFN (see [`Backend::glu_matmul`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    Silu,
    GeluTanh,
}

/// A compute backend.
///
/// Activations are always f32 `Buffer`s laid out row-major `[tokens, width]`.
/// Index data (token ids, positions, slots, block tables) is passed as host
/// slices; device backends upload it themselves.
pub trait Backend: Send + Sync {
    type Buffer: Send;
    type Weight: Send + Sync;
    type KvCache: Send;
    /// Host memory holding KV blocks copied out of a cache (the prefix
    /// cache's host tier).
    type HostKv: Send;

    fn name(&self) -> String;

    fn upload(&self, t: &HostTensor) -> Result<Self::Weight>;
    fn alloc(&self, len: usize) -> Result<Self::Buffer>;
    fn alloc_kv(&self, layout: KvLayout) -> Result<Self::KvCache>;
    fn to_host(&self, buf: &Self::Buffer, out: &mut [f32]) -> Result<()>;

    /// Sample one token from each listed row of `logits` (`[n_rows, vocab]`),
    /// as [`sample::sample_row`] does, into `out` (one id per entry of
    /// `rows`). Ends a recorded step like `to_host`. GPU backends sample on
    /// the device and copy back only the ids.
    fn sample_rows(&self, logits: &Self::Buffer, n_rows: usize, vocab: usize, rows: &[(u32, &RowSample)], out: &mut [u32]) -> Result<()> {
        let mut host = vec![0f32; n_rows * vocab];
        self.to_host(logits, &mut host)?;
        for ((r, s), o) in rows.iter().zip(out.iter_mut()) {
            let r = *r as usize;
            *o = sample::sample_row(&host[r * vocab..(r + 1) * vocab], s);
        }
        Ok(())
    }

    /// `out[i, :] = table[tokens[i], :]`
    fn embed(&self, table: &Self::Weight, tokens: &[u32], out: &mut Self::Buffer) -> Result<()>;

    /// `out[m, n] = x[m, k] · w[n, k]ᵀ`
    fn matmul(&self, x: &Self::Buffer, w: &Self::Weight, out: &mut Self::Buffer, m: usize) -> Result<()>;

    /// Several matmuls sharing one input: `outs[i][m, n_i] = x[m, k] · ws[i]ᵀ`.
    /// Backends can quantize `x` once and batch the launches (Q/K/V, gate/up).
    fn matmul_many(&self, x: &Self::Buffer, ws: &[&Self::Weight], outs: &mut [&mut Self::Buffer], m: usize) -> Result<()> {
        if ws.len() != outs.len() {
            return Err(Error::Shape(format!("matmul_many: {} weights but {} outputs", ws.len(), outs.len())));
        }
        for (w, out) in ws.iter().zip(outs.iter_mut()) {
            self.matmul(x, w, out, m)?;
        }
        Ok(())
    }

    /// Gated FFN input: `out[m, n] = act(x · gateᵀ) * (x · upᵀ)`. `up_scratch`
    /// (at least `m * n`) may be used for the up projection.
    #[allow(clippy::too_many_arguments)]
    fn glu_matmul(
        &self,
        x: &Self::Buffer,
        gate: &Self::Weight,
        up: &Self::Weight,
        out: &mut Self::Buffer,
        up_scratch: &mut Self::Buffer,
        m: usize,
        n: usize,
        act: Activation,
    ) -> Result<()> {
        self.matmul_many(x, &[gate, up], &mut [&mut *out, &mut *up_scratch], m)?;
        match act {
            Activation::Silu => self.silu_mul(out, up_scratch, m * n),
            Activation::GeluTanh => self.gelu_mul(out, up_scratch, m * n),
        }
    }

    /// Whole gated FFN: `out[m, d] = (act(x · gateᵀ) * (x · upᵀ)) · downᵀ`.
    /// `gate_buf`/`up_buf` (at least `m * n_ffn`) are scratch. Lets a backend
    /// skip materializing the activation when the down projection doesn't
    /// need it in f32.
    #[allow(clippy::too_many_arguments)]
    fn gated_ffn(
        &self,
        x: &Self::Buffer,
        gate: &Self::Weight,
        up: &Self::Weight,
        down: &Self::Weight,
        act: Activation,
        out: &mut Self::Buffer,
        gate_buf: &mut Self::Buffer,
        up_buf: &mut Self::Buffer,
        m: usize,
        n_ffn: usize,
    ) -> Result<()> {
        self.glu_matmul(x, gate, up, gate_buf, up_buf, m, n_ffn, act)?;
        self.matmul(gate_buf, down, out, m)
    }

    /// RMSNorm over rows of width `w.len()`: `out = x / rms(x) * w`. `rows * width` elements.
    fn rms_norm(&self, x: &Self::Buffer, w: &Self::Weight, out: &mut Self::Buffer, rows: usize, eps: f32) -> Result<()>;

    /// In-place rotary embedding, NeoX layout (rotate pairs `(i, i + head_dim/2)`).
    /// `x` is `[tokens, n_heads * head_dim]`. Pair `i` rotates by
    /// `pos * theta^(-2i/head_dim) / freq_factors[i]` (factor 1 when `None`);
    /// a huge factor leaves that pair unrotated (partial rotary).
    fn rope(
        &self,
        x: &mut Self::Buffer,
        positions: &[u32],
        n_heads: usize,
        head_dim: usize,
        theta: f32,
        freq_factors: Option<&Self::Weight>,
    ) -> Result<()>;

    /// `gate = silu(gate) * up`, over `len` elements.
    fn silu_mul(&self, gate: &mut Self::Buffer, up: &Self::Buffer, len: usize) -> Result<()>;

    /// `gate = gelu_tanh(gate) * up`, over `len` elements.
    fn gelu_mul(&self, gate: &mut Self::Buffer, up: &Self::Buffer, len: usize) -> Result<()>;

    /// `x = gelu_tanh(x)`, over `len` elements.
    fn gelu(&self, x: &mut Self::Buffer, len: usize) -> Result<()>;

    /// `x += y`, over `len` elements.
    fn add(&self, x: &mut Self::Buffer, y: &Self::Buffer, len: usize) -> Result<()>;

    /// `x *= y` elementwise, over `len` elements.
    fn mul(&self, x: &mut Self::Buffer, y: &Self::Buffer, len: usize) -> Result<()>;

    /// `x *= s`, over `len` elements.
    fn scale(&self, x: &mut Self::Buffer, s: f32, len: usize) -> Result<()>;

    /// `x = cap * tanh(x / cap)`, over `len` elements (Gemma's logit softcap).
    fn softcap(&self, x: &mut Self::Buffer, cap: f32, len: usize) -> Result<()>;

    /// `out[i, :] = x[first + i * stride, :]` for `i < count`, rows of `width`.
    #[allow(clippy::too_many_arguments)]
    fn gather_strided(
        &self,
        x: &Self::Buffer,
        first: usize,
        stride: usize,
        count: usize,
        width: usize,
        out: &mut Self::Buffer,
    ) -> Result<()> {
        let rows: Vec<u32> = (0..count).map(|i| (first + i * stride) as u32).collect();
        self.gather_rows(x, &rows, width, out)
    }

    /// `gate[i, :] = gelu_tanh(gate[i, :]) * x[first + i * stride, :]` for
    /// `i < count`, rows of `width` (Gemma's per-layer-input gate). `scratch`
    /// holds `count * width` values for backends that gather first.
    #[allow(clippy::too_many_arguments)]
    fn gelu_mul_strided(
        &self,
        gate: &mut Self::Buffer,
        x: &Self::Buffer,
        first: usize,
        stride: usize,
        count: usize,
        width: usize,
        scratch: &mut Self::Buffer,
    ) -> Result<()> {
        self.gather_strided(x, first, stride, count, width, scratch)?;
        self.gelu_mul(gate, scratch, count * width)
    }

    /// Residual update, then a gated FFN on its normalized form:
    ///   x_out = (x_in + rmsnorm(src) * w1) * scale
    ///   out   = gated_ffn(rmsnorm(x_out) * w2)
    /// `x_out` must not alias `x_in` (a fused backend reads x_in while writing
    /// x_out). `normed` holds `rows * width` values.
    #[allow(clippy::too_many_arguments)]
    fn norm_add_gated_ffn(
        &self,
        src: &Self::Buffer,
        w1: &Self::Weight,
        x_in: &Self::Buffer,
        x_out: &mut Self::Buffer,
        rows: usize,
        width: usize,
        eps: f32,
        scale: f32,
        w2: &Self::Weight,
        ffn: (&Self::Weight, &Self::Weight, &Self::Weight, Activation),
        normed: &mut Self::Buffer,
        out: &mut Self::Buffer,
        gate_buf: &mut Self::Buffer,
        up_buf: &mut Self::Buffer,
        n_ffn: usize,
    ) -> Result<()> {
        let (gate, up, down, act) = ffn;
        self.gather_strided(x_in, 0, 1, rows, width, x_out)?;
        self.rms_norm_add(src, w1, x_out, rows, width, eps, scale, Some((w2, normed, &[gate, up])))?;
        self.gated_ffn(normed, gate, up, down, act, out, gate_buf, up_buf, rows, n_ffn)
    }

    /// Gemma's per-layer-input gate, start to finish:
    ///   x_out = x_in + rmsnorm(src) * w            (rows of `width`)
    ///   gate  = gelu_tanh(x_out · gate_wᵀ) * inputs[i * n_layers + layer, :]
    /// `x_out` must not alias `x_in` (a fused backend reads x_in while writing
    /// x_out). `scratch` holds `rows * gate_w.rows` values.
    #[allow(clippy::too_many_arguments)]
    fn ple_gate(
        &self,
        src: &Self::Buffer,
        w: &Self::Weight,
        x_in: &Self::Buffer,
        x_out: &mut Self::Buffer,
        rows: usize,
        width: usize,
        eps: f32,
        gate_w: &Self::Weight,
        inputs: &Self::Buffer,
        layer: usize,
        n_layers: usize,
        p_dim: usize,
        gate: &mut Self::Buffer,
        scratch: &mut Self::Buffer,
    ) -> Result<()> {
        self.gather_strided(x_in, 0, 1, rows, width, x_out)?;
        self.rms_norm_add(src, w, x_out, rows, width, eps, 1.0, None)?;
        self.matmul(x_out, gate_w, gate, rows)?;
        self.gelu_mul_strided(gate, inputs, layer, n_layers, rows, p_dim, scratch)
    }

    /// `out[i, :] = x[rows[i], :]`, rows of `width`.
    fn gather_rows(&self, x: &Self::Buffer, rows: &[u32], width: usize, out: &mut Self::Buffer) -> Result<()>;

    /// Host storage for `n_blocks` blocks of a cache with `layout`
    /// (pinned memory on GPUs, so copies run at full PCIe speed).
    fn alloc_host_kv(&self, layout: KvLayout, n_blocks: usize) -> Result<Self::HostKv>;

    /// Copy cache blocks `dev[i]` (every layer) to host blocks `host[i]`.
    /// Ordered with the backend's other work: later steps may overwrite the
    /// device blocks right away.
    fn kv_to_host(&self, kv: &Self::KvCache, store: &mut Self::HostKv, dev: &[u32], host: &[u32]) -> Result<()>;

    /// Copy host blocks `host[i]` into cache blocks `dev[i]` (every layer).
    fn kv_from_host(&self, kv: &mut Self::KvCache, store: &Self::HostKv, host: &[u32], dev: &[u32]) -> Result<()>;

    /// Write K and V (`[tokens, slot_width]`) for `layer` into `slots`.
    fn kv_store(&self, kv: &mut Self::KvCache, layer: usize, k: &Self::Buffer, v: &Self::Buffer, slots: &[u32]) -> Result<()>;

    /// Paged causal attention for `layer`. `q` and `out` are `[tokens, n_heads * head_dim]`.
    fn attention(&self, q: &Self::Buffer, kv: &Self::KvCache, layer: usize, meta: &AttnMeta, out: &mut Self::Buffer) -> Result<()>;

    /// Fused `x += y; out = rmsnorm(x) * w` over `rows` rows of width `w.len()`.
    fn add_rms_norm(
        &self,
        x: &mut Self::Buffer,
        y: &Self::Buffer,
        w: &Self::Weight,
        out: &mut Self::Buffer,
        rows: usize,
        width: usize,
        eps: f32,
    ) -> Result<()> {
        self.add(x, y, rows * width)?;
        self.rms_norm(x, w, out, rows, eps)
    }

    /// `x = (x + rmsnorm(src) * w) * scale`, then optionally
    /// `next.1 = rmsnorm(x) * next.0`. Rows of width `w.len()`. (Gemma-style
    /// post-norm residual add, followed by the next block's pre-norm.)
    /// `next.2` lists the weights that will consume `next.1` (all through
    /// matmuls), so a backend can skip forms of the output nobody reads.
    #[allow(clippy::too_many_arguments)]
    fn rms_norm_add(
        &self,
        src: &Self::Buffer,
        w: &Self::Weight,
        x: &mut Self::Buffer,
        rows: usize,
        width: usize,
        eps: f32,
        scale: f32,
        next: Option<(&Self::Weight, &mut Self::Buffer, &[&Self::Weight])>,
    ) -> Result<()> {
        let mut t = self.alloc(rows * width)?;
        self.rms_norm(src, w, &mut t, rows, eps)?;
        self.add(x, &t, rows * width)?;
        if scale != 1.0 {
            self.scale(x, scale, rows * width)?;
        }
        if let Some((w2, out, _)) = next {
            self.rms_norm(x, w2, out, rows, eps)?;
        }
        Ok(())
    }

    /// Fused per-head RMSNorm then RoPE: `out = rope(rmsnorm(x) * w)`, where
    /// `x` is `[tokens, n_heads * head_dim]` and `w` has `head_dim` entries.
    #[allow(clippy::too_many_arguments)]
    fn rms_norm_rope(
        &self,
        x: &Self::Buffer,
        w: &Self::Weight,
        out: &mut Self::Buffer,
        positions: &[u32],
        n_heads: usize,
        head_dim: usize,
        eps: f32,
        theta: f32,
        freq_factors: Option<&Self::Weight>,
    ) -> Result<()> {
        self.rms_norm(x, w, out, positions.len() * n_heads, eps)?;
        self.rope(out, positions, n_heads, head_dim, theta, freq_factors)
    }

    /// Start of a forward step, before its first op. Steps passing the same
    /// `graph_key` must issue the same ops with the same shapes (only index
    /// data such as positions may differ); a backend may then record the step
    /// once and replay it. The step ends at the next `to_host`.
    fn begin_step(&self, graph_key: Option<u64>) -> Result<()> {
        let _ = graph_key;
        Ok(())
    }

    /// Fused attention input prep for one layer: `qn = rope(rmsnorm(q) * q_norm)`,
    /// and into the KV cache at `slots`: `rope(rmsnorm(k) * k_norm)` and
    /// `v` (RMS-normalized with `v_norm` when given). `qk_norm` = (q_norm,
    /// k_norm); without it q and k are only rotated (Llama).
    #[allow(clippy::too_many_arguments)]
    fn qkv_norm_rope_store(
        &self,
        q: &Self::Buffer,
        k: &Self::Buffer,
        v: &Self::Buffer,
        qk_norm: Option<(&Self::Weight, &Self::Weight)>,
        v_norm: Option<&Self::Weight>,
        qn: &mut Self::Buffer,
        kv: &mut Self::KvCache,
        layer: usize,
        positions: &[u32],
        slots: &[u32],
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        eps: f32,
        theta: f32,
        freq_factors: Option<&Self::Weight>,
    ) -> Result<()> {
        let n = positions.len();
        let mut kn = self.alloc(n * n_kv_heads * head_dim)?;
        match qk_norm {
            Some((q_norm, k_norm)) => {
                self.rms_norm_rope(q, q_norm, qn, positions, n_heads, head_dim, eps, theta, freq_factors)?;
                self.rms_norm_rope(k, k_norm, &mut kn, positions, n_kv_heads, head_dim, eps, theta, freq_factors)?;
            }
            None => {
                let rows = |heads: usize| (0..(n * heads) as u32).collect::<Vec<_>>();
                self.gather_rows(q, &rows(n_heads), head_dim, qn)?;
                self.rope(qn, positions, n_heads, head_dim, theta, freq_factors)?;
                self.gather_rows(k, &rows(n_kv_heads), head_dim, &mut kn)?;
                self.rope(&mut kn, positions, n_kv_heads, head_dim, theta, freq_factors)?;
            }
        }
        match v_norm {
            Some(w) => {
                let mut vn = self.alloc(n * n_kv_heads * head_dim)?;
                self.rms_norm(v, w, &mut vn, n * n_kv_heads, eps)?;
                self.kv_store(kv, layer, &kn, &vn, slots)
            }
            None => self.kv_store(kv, layer, &kn, v, slots),
        }
    }

    /// Block until queued work is finished (no-op on synchronous backends).
    fn synchronize(&self) -> Result<()> {
        Ok(())
    }
}
