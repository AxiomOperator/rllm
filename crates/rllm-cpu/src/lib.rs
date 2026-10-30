//! CPU backend.
//!
//! Its first job is to be obviously correct: every GPU kernel is tested
//! against it. It is multithreaded with rayon so models are usable, but not
//! tuned beyond that.

use rayon::prelude::*;
use rllm_core::quant::dequantize_row;
use rllm_core::{AttnMeta, Backend, DType, Error, HostBytes, HostTensor, KvLayout, KvType, Result};

pub struct CpuBackend;

impl CpuBackend {
    pub fn new() -> Self {
        CpuBackend
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

pub struct CpuWeight {
    dtype: DType,
    rows: usize,
    cols: usize,
    bytes: HostBytes,
    /// Small 1-D weights (norms) are dequantized once on upload.
    dense: Option<Vec<f32>>,
}

impl CpuWeight {
    fn row_bytes(&self) -> usize {
        self.dtype.row_bytes(self.cols)
    }

    fn row(&self, r: usize) -> &[u8] {
        let rb = self.row_bytes();
        &self.bytes.as_slice()[r * rb..(r + 1) * rb]
    }
}

pub struct CpuKvCache {
    layout: KvLayout,
    /// Per layer, `[n_blocks * block_size, slot_width]`.
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

/// Copy blocks `from[i]` of `src` to blocks `to[i]` of `dst` (every layer).
fn copy_blocks(src: &CpuKvCache, dst: &mut CpuKvCache, from: &[u32], to: &[u32]) -> Result<()> {
    let n = src.layout.block_size * src.layout.slot_width();
    for (&f, &t) in from.iter().zip(to) {
        let (f, t) = (f as usize * n, t as usize * n);
        if f + n > src.k[0].len() || t + n > dst.k[0].len() {
            return Err(Error::Shape(format!("kv block copy {f} -> {t} out of range")));
        }
        for (s, d) in src.k.iter().zip(&mut dst.k).chain(src.v.iter().zip(&mut dst.v)) {
            d[t..t + n].copy_from_slice(&s[f..f + n]);
        }
    }
    Ok(())
}

fn check_len(what: &str, have: usize, need: usize) -> Result<()> {
    if have < need {
        return Err(Error::Shape(format!("{what}: buffer has {have} elements, need {need}")));
    }
    Ok(())
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    // Eight independent accumulators so the compiler can vectorize.
    let mut acc = [0f32; 8];
    let (ac, ar) = a.as_chunks::<8>();
    let (bc, br) = b.as_chunks::<8>();
    for (x, y) in ac.iter().zip(bc) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    let mut s: f32 = acc.iter().sum();
    for (x, y) in ar.iter().zip(br) {
        s += x * y;
    }
    s
}

/// `RLLM_CPU_LLAMACPP_NUMERICS=1`: round matmul activations the way llama.cpp's
/// CPU backend does before its dot products (8-bit blocks for quantized
/// weights, f16/bf16 for half-precision weights). Only for validating against
/// llama.cpp; it makes results slightly *less* accurate.
fn llamacpp_numerics() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("RLLM_CPU_LLAMACPP_NUMERICS").is_some_and(|v| v != "0"))
}

/// Round-trip one row of activations through llama.cpp's `vec_dot_type` for
/// weights of `dtype` (see ggml-cpu.c `type_traits_cpu`, ggml-quants.c).
fn llamacpp_round_activations(dtype: DType, x: &mut [f32]) {
    use half::{bf16, f16};
    match dtype {
        DType::F32 => {}
        DType::F16 => x.iter_mut().for_each(|v| *v = f16::from_f32(*v).to_f32()),
        DType::BF16 => x.iter_mut().for_each(|v| *v = bf16::from_f32(*v).to_f32()),
        // quantize_row_q8_0_ref: d = amax/127 stored as f16, q = roundf(x/d)
        DType::Q8_0 => {
            for b in x.chunks_exact_mut(32) {
                let amax = b.iter().fold(0f32, |m, v| m.max(v.abs()));
                let d = amax / 127.0;
                let id = if d != 0.0 { 1.0 / d } else { 0.0 };
                let d16 = f16::from_f32(d).to_f32();
                b.iter_mut().for_each(|v| *v = (*v * id).round() * d16);
            }
        }
        // quantize_row_q8_K_ref: iscale = -127/max (signed max), q = nearest_int, d = 1/iscale (f32)
        DType::Q4K | DType::Q5K | DType::Q6K | DType::IQ4XS => {
            for b in x.chunks_exact_mut(256) {
                let (mut max, mut amax) = (0f32, 0f32);
                for &v in b.iter() {
                    if v.abs() > amax {
                        amax = v.abs();
                        max = v;
                    }
                }
                if amax == 0.0 {
                    b.fill(0.0);
                    continue;
                }
                let iscale = -127.0 / max;
                let d = 1.0 / iscale;
                b.iter_mut().for_each(|v| *v = ((iscale * *v).round_ties_even().min(127.0)) * d);
            }
        }
    }
}

/// GELU, tanh approximation (`gelu_pytorch_tanh`).
#[inline]
fn gelu_tanh(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)).tanh())
}

impl Backend for CpuBackend {
    type Buffer = Vec<f32>;
    type Weight = CpuWeight;
    type KvCache = CpuKvCache;
    type HostKv = CpuKvCache;

    fn name(&self) -> String {
        format!("cpu ({} threads)", rayon::current_num_threads())
    }

    fn upload(&self, t: &HostTensor) -> Result<CpuWeight> {
        let (rows, cols) = t.as_2d();
        if cols % t.dtype.block_size() != 0 {
            return Err(Error::Shape(format!("{}: {cols} columns not a multiple of {} block", t.name, t.dtype)));
        }
        let dense = (t.shape.len() == 1).then(|| {
            let mut v = vec![0f32; cols];
            dequantize_row(t.dtype, t.bytes.as_slice(), &mut v);
            v
        });
        Ok(CpuWeight { dtype: t.dtype, rows, cols, bytes: t.bytes.clone(), dense })
    }

    fn alloc(&self, len: usize) -> Result<Vec<f32>> {
        Ok(vec![0f32; len])
    }

    fn alloc_kv(&self, layout: KvLayout) -> Result<CpuKvCache> {
        let n = layout.n_blocks * layout.block_size * layout.slot_width();
        Ok(CpuKvCache {
            layout,
            k: (0..layout.n_layers).map(|_| vec![0f32; n]).collect(),
            v: (0..layout.n_layers).map(|_| vec![0f32; n]).collect(),
        })
    }

    fn to_host(&self, buf: &Vec<f32>, out: &mut [f32]) -> Result<()> {
        check_len("to_host", buf.len(), out.len())?;
        out.copy_from_slice(&buf[..out.len()]);
        Ok(())
    }

    fn embed(&self, table: &CpuWeight, tokens: &[u32], out: &mut Vec<f32>) -> Result<()> {
        let d = table.cols;
        check_len("embed", out.len(), tokens.len() * d)?;
        for (&tok, o) in tokens.iter().zip(out.chunks_exact_mut(d)) {
            if tok as usize >= table.rows {
                return Err(Error::Shape(format!("token id {tok} out of range ({})", table.rows)));
            }
            dequantize_row(table.dtype, table.row(tok as usize), o);
        }
        Ok(())
    }

    fn matmul(&self, x: &Vec<f32>, w: &CpuWeight, out: &mut Vec<f32>, m: usize) -> Result<()> {
        let (n, k) = (w.rows, w.cols);
        check_len("matmul x", x.len(), m * k)?;
        check_len("matmul out", out.len(), m * n)?;
        let rounded;
        let x: &[f32] = if llamacpp_numerics() {
            let mut v = x[..m * k].to_vec();
            v.chunks_exact_mut(k).for_each(|row| llamacpp_round_activations(w.dtype, row));
            rounded = v;
            &rounded
        } else {
            x
        };
        // Each task dequantizes a run of weight rows once and dots them with
        // every input row. Results land transposed ([n, m]) so tasks write
        // disjoint memory; for decode (m = 1) that is already the final layout.
        let mut out_t = if m == 1 { Vec::new() } else { vec![0f32; n * m] };
        let dst: &mut [f32] = if m == 1 { &mut out[..n] } else { &mut out_t };
        const ROWS_PER_TASK: usize = 16;
        dst.par_chunks_mut(ROWS_PER_TASK * m).enumerate().for_each(|(chunk, dst)| {
            let mut wrow = vec![0f32; k];
            for (i, d) in dst.chunks_exact_mut(m).enumerate() {
                let r = chunk * ROWS_PER_TASK + i;
                dequantize_row(w.dtype, w.row(r), &mut wrow);
                for (j, d) in d.iter_mut().enumerate() {
                    *d = dot(&x[j * k..(j + 1) * k], &wrow);
                }
            }
        });
        if m > 1 {
            for r in 0..n {
                for j in 0..m {
                    out[j * n + r] = out_t[r * m + j];
                }
            }
        }
        Ok(())
    }

    fn rms_norm(&self, x: &Vec<f32>, w: &CpuWeight, out: &mut Vec<f32>, rows: usize, eps: f32) -> Result<()> {
        let w = w.dense.as_deref().ok_or_else(|| Error::Shape("rms_norm weight must be 1-D".into()))?;
        let d = w.len();
        check_len("rms_norm x", x.len(), rows * d)?;
        check_len("rms_norm out", out.len(), rows * d)?;
        out[..rows * d].par_chunks_mut(d).zip(x[..rows * d].par_chunks(d)).for_each(|(o, x)| {
            let ms = dot(x, x) / d as f32;
            let s = 1.0 / (ms + eps).sqrt();
            for ((o, x), w) in o.iter_mut().zip(x).zip(w) {
                *o = x * s * w;
            }
        });
        Ok(())
    }

    fn rope(
        &self,
        x: &mut Vec<f32>,
        positions: &[u32],
        n_heads: usize,
        head_dim: usize,
        theta: f32,
        freq_factors: Option<&CpuWeight>,
    ) -> Result<()> {
        let width = n_heads * head_dim;
        check_len("rope", x.len(), positions.len() * width)?;
        let half = head_dim / 2;
        let factors = match freq_factors {
            Some(w) => {
                let f = w.dense.as_deref().ok_or_else(|| Error::Shape("rope freq factors must be 1-D".into()))?;
                check_len("rope freq factors", f.len(), half)?;
                Some(f)
            }
            None => None,
        };
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32) / factors.map_or(1.0, |f| f[i]))
            .collect();
        x[..positions.len() * width].par_chunks_mut(width).zip(positions).for_each(|(row, &pos)| {
            for head in row.chunks_exact_mut(head_dim) {
                for i in 0..half {
                    let (sin, cos) = (pos as f32 * inv_freq[i]).sin_cos();
                    let (a, b) = (head[i], head[i + half]);
                    head[i] = a * cos - b * sin;
                    head[i + half] = a * sin + b * cos;
                }
            }
        });
        Ok(())
    }

    fn silu_mul(&self, gate: &mut Vec<f32>, up: &Vec<f32>, len: usize) -> Result<()> {
        check_len("silu_mul", gate.len().min(up.len()), len)?;
        gate[..len].par_iter_mut().zip(&up[..len]).for_each(|(g, u)| {
            *g = *g / (1.0 + (-*g).exp()) * u;
        });
        Ok(())
    }

    fn gelu_mul(&self, gate: &mut Vec<f32>, up: &Vec<f32>, len: usize) -> Result<()> {
        check_len("gelu_mul", gate.len().min(up.len()), len)?;
        gate[..len].par_iter_mut().zip(&up[..len]).for_each(|(g, u)| *g = gelu_tanh(*g) * u);
        Ok(())
    }

    fn gelu(&self, x: &mut Vec<f32>, len: usize) -> Result<()> {
        check_len("gelu", x.len(), len)?;
        x[..len].par_iter_mut().for_each(|v| *v = gelu_tanh(*v));
        Ok(())
    }

    fn add(&self, x: &mut Vec<f32>, y: &Vec<f32>, len: usize) -> Result<()> {
        check_len("add", x.len().min(y.len()), len)?;
        x[..len].iter_mut().zip(&y[..len]).for_each(|(x, y)| *x += y);
        Ok(())
    }

    fn mul(&self, x: &mut Vec<f32>, y: &Vec<f32>, len: usize) -> Result<()> {
        check_len("mul", x.len().min(y.len()), len)?;
        x[..len].iter_mut().zip(&y[..len]).for_each(|(x, y)| *x *= y);
        Ok(())
    }

    fn scale(&self, x: &mut Vec<f32>, s: f32, len: usize) -> Result<()> {
        check_len("scale", x.len(), len)?;
        x[..len].iter_mut().for_each(|v| *v *= s);
        Ok(())
    }

    fn softcap(&self, x: &mut Vec<f32>, cap: f32, len: usize) -> Result<()> {
        check_len("softcap", x.len(), len)?;
        x[..len].iter_mut().for_each(|v| *v = cap * (*v / cap).tanh());
        Ok(())
    }

    fn gather_rows(&self, x: &Vec<f32>, rows: &[u32], width: usize, out: &mut Vec<f32>) -> Result<()> {
        check_len("gather_rows out", out.len(), rows.len() * width)?;
        for (&r, o) in rows.iter().zip(out.chunks_exact_mut(width)) {
            let r = r as usize;
            check_len("gather_rows x", x.len(), (r + 1) * width)?;
            o.copy_from_slice(&x[r * width..(r + 1) * width]);
        }
        Ok(())
    }

    fn alloc_host_kv(&self, layout: KvLayout, n_blocks: usize) -> Result<CpuKvCache> {
        self.alloc_kv(KvLayout { n_blocks, ..layout })
    }

    fn kv_to_host(&self, kv: &CpuKvCache, store: &mut CpuKvCache, dev: &[u32], host: &[u32]) -> Result<()> {
        copy_blocks(kv, store, dev, host)
    }

    fn kv_from_host(&self, kv: &mut CpuKvCache, store: &CpuKvCache, host: &[u32], dev: &[u32]) -> Result<()> {
        copy_blocks(store, kv, host, dev)
    }

    fn kv_store(&self, kv: &mut CpuKvCache, layer: usize, k: &Vec<f32>, v: &Vec<f32>, slots: &[u32]) -> Result<()> {
        let w = kv.layout.slot_width();
        let n_slots = kv.layout.n_blocks * kv.layout.block_size;
        check_len("kv_store k", k.len(), slots.len() * w)?;
        check_len("kv_store v", v.len(), slots.len() * w)?;
        let (kc, vc) = (&mut kv.k[layer], &mut kv.v[layer]);
        for (i, &s) in slots.iter().enumerate() {
            let s = s as usize;
            if s >= n_slots {
                return Err(Error::Shape(format!("kv slot {s} out of range ({n_slots})")));
            }
            kc[s * w..(s + 1) * w].copy_from_slice(&k[i * w..(i + 1) * w]);
            vc[s * w..(s + 1) * w].copy_from_slice(&v[i * w..(i + 1) * w]);
            if kv.layout.dtype == KvType::Q8 {
                // Same rounding as a q8 cache (values stay f32 here).
                // Keys are quantized rotated and rotated back (H is its own
                // inverse): what attention sees through a rotated q8 cache.
                let hd = kv.layout.head_dim;
                let rot = 1 << hd.trailing_zeros();
                for chunk in kc[s * w..(s + 1) * w].chunks_exact_mut(rot) {
                    fwht(chunk);
                    chunk.chunks_exact_mut(32).for_each(q8_round);
                    fwht(chunk);
                }
                vc[s * w..(s + 1) * w].chunks_exact_mut(32).for_each(q8_round);
            }
        }
        Ok(())
    }

    fn attention(&self, q: &Vec<f32>, kv: &CpuKvCache, layer: usize, meta: &AttnMeta, out: &mut Vec<f32>) -> Result<()> {
        let (nh, nkv, hd) = (meta.n_heads, meta.n_kv_heads, meta.head_dim);
        let n_tok = meta.positions.len();
        let w = nh * hd;
        let kv_w = nkv * hd;
        let group = nh / nkv;
        let bs = meta.block_size;
        check_len("attention q", q.len(), n_tok * w)?;
        check_len("attention out", out.len(), n_tok * w)?;
        let (kc, vc) = (&kv.k[layer], &kv.v[layer]);

        out[..n_tok * w].par_chunks_mut(hd).enumerate().for_each(|(idx, o)| {
            let (t, h) = (idx / nh, idx % nh);
            let ctx = meta.positions[t] as usize + 1;
            let start = if meta.window > 0 { ctx.saturating_sub(meta.window) } else { 0 };
            let table = &meta.block_tables[meta.token_seq[t] as usize];
            let qh = &q[t * w + h * hd..t * w + (h + 1) * hd];
            let off = (h / group) * hd;
            let slot = |p: usize| (table[p / bs] as usize * bs + p % bs) * kv_w + off;

            let mut scores: Vec<f32> = (start..ctx).map(|p| dot(qh, &kc[slot(p)..slot(p) + hd]) * meta.scale).collect();
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0;
            for s in &mut scores {
                *s = (*s - max).exp();
                sum += *s;
            }
            o.fill(0.0);
            for (p, s) in (start..ctx).zip(&scores) {
                let wgt = s / sum;
                for (o, v) in o.iter_mut().zip(&vc[slot(p)..slot(p) + hd]) {
                    *o += wgt * v;
                }
            }
        });
        Ok(())
    }
}

/// Orthonormal fast Walsh-Hadamard transform (length a power of two).
fn fwht(x: &mut [f32]) {
    let n = x.len();
    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(2 * h) {
            for j in i..i + h {
                let (a, b) = (x[j], x[j + h]);
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
        h *= 2;
    }
    let sc = 1.0 / (n as f32).sqrt();
    x.iter_mut().for_each(|v| *v *= sc);
}

/// Round a group of 32 values as a q8 KV cache stores them: int8 times an
/// f16 scale of max|x| / 127.
fn q8_round(g: &mut [f32]) {
    let d = g.iter().fold(0f32, |a, x| a.max(x.abs())) / 127.0;
    let dh = half::f16::from_f32(d).to_f32();
    for x in g {
        *x = if d > 0.0 { (*x / d).round_ties_even() * dh } else { 0.0 };
    }
}
