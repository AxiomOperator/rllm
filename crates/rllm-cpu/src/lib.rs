//! CPU backend.
//!
//! Its first job is to be obviously correct: every GPU kernel is tested
//! against it. It is multithreaded with rayon so models are usable, but not
//! tuned beyond that.

use rayon::prelude::*;
use rllm_core::quant::dequantize_row;
use rllm_core::{AttnMeta, Backend, DType, Error, GdnShape, HostBytes, HostTensor, KvLayout, KvType, MoeFfn, Result, SeqRuns};

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

/// A raw pointer shared across rayon jobs that write disjoint parts of one buffer.
#[derive(Clone, Copy)]
struct SyncPtr(*mut f32);
unsafe impl Send for SyncPtr {}
unsafe impl Sync for SyncPtr {}

impl SyncPtr {
    fn get(self) -> *mut f32 {
        self.0
    }
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
        DType::Q3K | DType::Q4K | DType::Q5K | DType::Q6K | DType::IQ4XS => {
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

/// `out[m, n] = x[m, k] · w[r0..r0 + n, :]ᵀ` (a row range of `w`).
fn rows_matmul(w: &CpuWeight, r0: usize, n: usize, x: &[f32], m: usize, out: &mut [f32]) {
    let k = w.cols;
    if m == 0 || n == 0 || k == 0 {
        out.fill(0.0);
        return;
    }
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
            let r = r0 + chunk * ROWS_PER_TASK + i;
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

    fn keeps_host_weights(&self) -> bool {
        true
    }

    fn upload(&self, t: &HostTensor) -> Result<CpuWeight> {
        let (rows, cols) = t.as_2d();
        if cols % t.dtype.block_size() != 0 {
            return Err(Error::Shape(format!("{}: {cols} columns not a multiple of {} block", t.name, t.dtype)));
        }
        if rows.checked_mul(t.dtype.row_bytes(cols)) != Some(t.bytes.as_slice().len()) {
            return Err(Error::Shape(format!("{}: {} bytes for {rows} x {cols} {}", t.name, t.bytes.as_slice().len(), t.dtype)));
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
        let n = layout
            .n_blocks
            .checked_mul(layout.block_size)
            .and_then(|n| n.checked_mul(layout.slot_width()))
            .ok_or_else(|| Error::Shape(format!("KV cache size overflows: {layout:?}")))?;
        if layout.block_size == 0 || layout.slot_width() == 0 {
            return Err(Error::Shape(format!("empty KV cache layout: {layout:?}")));
        }
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
        if d == 0 {
            return Err(Error::Shape("embed: empty embedding rows".into()));
        }
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
        rows_matmul(w, 0, n, &x[..m * k], m, &mut out[..m * n]);
        Ok(())
    }

    fn rms_norm(&self, x: &Vec<f32>, w: &CpuWeight, out: &mut Vec<f32>, rows: usize, eps: f32) -> Result<()> {
        let w = w.dense.as_deref().ok_or_else(|| Error::Shape("rms_norm weight must be 1-D".into()))?;
        let d = w.len();
        if d == 0 {
            return Err(Error::Shape("rms_norm: empty weight".into()));
        }
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
        if width == 0 || !head_dim.is_multiple_of(2) {
            return Err(Error::Shape(format!("rope: {n_heads} heads of size {head_dim}")));
        }
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

    fn sigmoid_mul(&self, x: &mut Vec<f32>, g: &Vec<f32>, len: usize) -> Result<()> {
        check_len("sigmoid_mul", x.len().min(g.len()), len)?;
        x[..len].par_iter_mut().zip(&g[..len]).for_each(|(x, g)| *x *= 1.0 / (1.0 + (-*g).exp()));
        Ok(())
    }

    fn moe_ffn(&self, x: &Vec<f32>, logits: &Vec<f32>, moe: &MoeFfn<CpuWeight>, shared: Option<(&Vec<f32>, &Vec<f32>)>, out: &mut Vec<f32>, m: usize) -> Result<()> {
        let MoeFfn { gate, up, down, n_expert: ne, n_used: k, n_ff, norm_topk, scale } = *moe;
        let d = gate.cols;
        if k == 0 || k > ne || gate.rows != ne * n_ff || up.rows != ne * n_ff || up.cols != d || down.rows != ne * d || down.cols != n_ff {
            return Err(Error::Shape(format!(
                "moe_ffn: {ne} experts (top {k}) of {n_ff}: gate {}x{}, up {}x{}, down {}x{}",
                gate.rows, gate.cols, up.rows, up.cols, down.rows, down.cols
            )));
        }
        check_len("moe_ffn x", x.len(), m * d)?;
        check_len("moe_ffn logits", logits.len(), m * ne)?;
        check_len("moe_ffn out", out.len(), m * d)?;
        let (mut ids, mut wts) = (vec![0u32; m * k], vec![0f32; m * k]);
        for t in 0..m {
            rllm_core::moe_route(&logits[t * ne..(t + 1) * ne], k, norm_topk, scale, &mut ids[t * k..(t + 1) * k], &mut wts[t * k..(t + 1) * k]);
        }
        // Per expert, its (token, slot) pairs in order; each pair's output row.
        let mut pairs: Vec<Vec<usize>> = vec![Vec::new(); ne];
        for (p, &e) in ids.iter().enumerate() {
            pairs[e as usize].push(p);
        }
        let mut y = vec![0f32; m * k * d];
        for (e, ps) in pairs.iter().enumerate().filter(|(_, ps)| !ps.is_empty()) {
            let c = ps.len();
            let xg: Vec<f32> = ps.iter().flat_map(|&p| x[p / k * d..(p / k + 1) * d].iter().copied()).collect();
            let (mut g, mut u) = (vec![0f32; c * n_ff], vec![0f32; c * n_ff]);
            rows_matmul(gate, e * n_ff, n_ff, &xg, c, &mut g);
            rows_matmul(up, e * n_ff, n_ff, &xg, c, &mut u);
            for (g, u) in g.iter_mut().zip(&u) {
                *g = *g / (1.0 + (-*g).exp()) * u;
            }
            let mut o = vec![0f32; c * d];
            rows_matmul(down, e * d, d, &g, c, &mut o);
            for (i, &p) in ps.iter().enumerate() {
                y[p * d..(p + 1) * d].copy_from_slice(&o[i * d..(i + 1) * d]);
            }
        }
        out[..m * d].par_chunks_mut(d).enumerate().for_each(|(t, o)| {
            o.fill(0.0);
            for j in 0..k {
                let (w, row) = (wts[t * k + j], &y[(t * k + j) * d..(t * k + j + 1) * d]);
                o.iter_mut().zip(row).for_each(|(o, v)| *o += w * v);
            }
        });
        if let Some((y, g)) = shared {
            self.add_sigmoid_scaled(out, y, g, m, d)?;
        }
        Ok(())
    }

    fn add_sigmoid_scaled(&self, x: &mut Vec<f32>, y: &Vec<f32>, g: &Vec<f32>, rows: usize, width: usize) -> Result<()> {
        check_len("add_sigmoid_scaled", x.len().min(y.len()), rows * width)?;
        check_len("add_sigmoid_scaled gate", g.len(), rows)?;
        x[..rows * width].par_chunks_mut(width.max(1)).zip(y[..rows * width].par_chunks(width.max(1))).enumerate().for_each(|(r, (x, y))| {
            let s = 1.0 / (1.0 + (-g[r]).exp());
            x.iter_mut().zip(y).for_each(|(x, y)| *x += s * y);
        });
        Ok(())
    }

    fn conv1d_silu(
        &self,
        x: &Vec<f32>,
        w: &CpuWeight,
        state: &mut Vec<f32>,
        runs: &SeqRuns,
        out: &mut Vec<f32>,
        channels: usize,
        kernel: usize,
        snaps: Option<(&mut Vec<f32>, usize)>,
    ) -> Result<()> {
        let wf = w.dense.as_deref().filter(|v| kernel >= 2 && channels > 0 && v.len() == channels * kernel).ok_or_else(|| {
            Error::Shape(format!("conv1d_silu: weight must be 1-D with {channels} x {kernel} values"))
        })?;
        let n = *runs.starts.last().unwrap_or(&0) as usize;
        let keep = kernel - 1;
        let n_slots = state.len() / (channels * keep);
        runs.check(n, n_slots)?;
        check_len("conv1d_silu x", x.len(), n * channels)?;
        check_len("conv1d_silu out", out.len(), n * channels)?;
        // Per sequence, then per channel: the channel's window slides over its tokens.
        let per_seq: Vec<(usize, usize, usize, bool)> = (0..runs.n_seqs())
            .map(|s| (runs.starts[s] as usize, runs.starts[s + 1] as usize, runs.slots[s] as usize, runs.fresh[s] != 0))
            .collect();
        let out_ptr = SyncPtr(out.as_mut_ptr());
        let state_ptr = SyncPtr(state.as_mut_ptr());
        let (snap_ptr, n_snap) = match snaps {
            Some((buf, n)) => {
                check_len("conv1d_silu snapshots", buf.len(), n * channels * keep)?;
                (Some(SyncPtr(buf.as_mut_ptr())), n)
            }
            None => (None, 0),
        };
        let snaps_of = |s: usize| -> Result<(usize, usize)> {
            let (n, at) = (runs.snap.get(s).map_or(0, |&n| n as usize), runs.snap_at.get(s).map_or(0, |&a| a as usize));
            if n > 0 && at + n > n_snap {
                return Err(Error::Shape(format!("snapshots {at}..{} of {n_snap}", at + n)));
            }
            Ok((n, at))
        };
        let per_seq: Vec<(usize, usize, usize, bool, (usize, usize))> =
            per_seq.iter().enumerate().map(|(i, &(a, b, c, d))| Ok((a, b, c, d, snaps_of(i)?))).collect::<Result<_>>()?;
        let _ = n_slots;
        per_seq.par_iter().for_each(|&(t0, t1, slot, fresh, (n_keep, snap_at))| {
            (0..channels).into_par_iter().for_each(|c| {
                // SAFETY: each (sequence, channel) touches only its own state
                // entries and output column; sequences have distinct slots
                // (checked by the allocator) and disjoint token ranges.
                let st = unsafe { std::slice::from_raw_parts_mut(state_ptr.get().add((slot * channels + c) * keep), keep) };
                // win[1..] holds the previous inputs; each token shifts it
                // left and lands in the last place.
                let mut win = vec![0f32; kernel];
                if !fresh {
                    win[1..].copy_from_slice(st);
                }
                let wc = &wf[c * kernel..(c + 1) * kernel];
                for t in t0..t1 {
                    win.copy_within(1.., 0);
                    win[keep] = x[t * channels + c];
                    let v: f32 = win.iter().zip(wc).map(|(a, b)| a * b).sum();
                    unsafe { *out_ptr.get().add(t * channels + c) = v / (1.0 + (-v).exp()) };
                    let j = t - t0;
                    if let Some(p) = snap_ptr
                        && j < n_keep
                    {
                        // SAFETY: (snapshot entry, channel) pairs are this job's own.
                        let dst = unsafe { std::slice::from_raw_parts_mut(p.get().add(((snap_at + j) * channels + c) * keep), keep) };
                        dst.copy_from_slice(&win[1..]);
                    }
                }
                st.copy_from_slice(&win[1..]);
            });
        });
        Ok(())
    }

    fn gated_delta_net(
        &self,
        qkv: &Vec<f32>,
        alpha: &Vec<f32>,
        beta: &Vec<f32>,
        dt_bias: &CpuWeight,
        a: &CpuWeight,
        state: &mut Vec<f32>,
        runs: &SeqRuns,
        out: &mut Vec<f32>,
        shape: GdnShape,
        eps: f32,
        snaps: Option<(&mut Vec<f32>, usize)>,
    ) -> Result<()> {
        let GdnShape { n_k_heads: hk, n_v_heads: hv, head_dim: d } = shape;
        if hk == 0 || d == 0 || !hv.is_multiple_of(hk) {
            return Err(Error::Shape(format!("gated_delta_net: {hk} key heads, {hv} value heads of {d}")));
        }
        let dt = dt_bias.dense.as_deref().filter(|v| v.len() == hv).ok_or_else(|| Error::Shape("gated_delta_net: dt_bias".into()))?;
        let av = a.dense.as_deref().filter(|v| v.len() == hv).ok_or_else(|| Error::Shape("gated_delta_net: a".into()))?;
        let n = *runs.starts.last().unwrap_or(&0) as usize;
        let width = shape.qkv_width();
        let n_slots = state.len() / shape.state_len();
        runs.check(n, n_slots)?;
        check_len("gated_delta_net qkv", qkv.len(), n * width)?;
        check_len("gated_delta_net alpha", alpha.len().min(beta.len()), n * hv)?;
        check_len("gated_delta_net out", out.len(), n * hv * d)?;
        let scale = 1.0 / (d as f32).sqrt();
        let out_ptr = SyncPtr(out.as_mut_ptr());
        let state_ptr = SyncPtr(state.as_mut_ptr());
        let (snap_ptr, n_snap) = match snaps {
            Some((buf, n)) => {
                check_len("gated_delta_net snapshots", buf.len(), n * shape.state_len())?;
                for s in 0..runs.n_seqs() {
                    let (k, at) = (runs.snap.get(s).map_or(0, |&k| k as usize), runs.snap_at.get(s).map_or(0, |&a| a as usize));
                    if k > 0 && at + k > n {
                        return Err(Error::Shape(format!("snapshots {at}..{} of {n}", at + k)));
                    }
                }
                (Some(SyncPtr(buf.as_mut_ptr())), n)
            }
            None => (None, 0),
        };
        let jobs: Vec<(usize, usize)> = (0..runs.n_seqs()).flat_map(|s| (0..hv).map(move |h| (s, h))).collect();
        jobs.par_iter().for_each(|&(s, h)| {
            let (t0, t1, slot) = (runs.starts[s] as usize, runs.starts[s + 1] as usize, runs.slots[s] as usize);
            // State layout: column j of S at st[j * d..]. SAFETY: one job per
            // (sequence, head): disjoint state and output.
            let st = unsafe { std::slice::from_raw_parts_mut(state_ptr.get().add((slot * hv + h) * d * d), d * d) };
            if runs.fresh[s] != 0 {
                st.fill(0.0);
            }
            let kh = h % hk;
            let (mut q, mut k) = (vec![0f32; d], vec![0f32; d]);
            for t in t0..t1 {
                let row = &qkv[t * width..(t + 1) * width];
                q.copy_from_slice(&row[kh * d..(kh + 1) * d]);
                k.copy_from_slice(&row[(hk + kh) * d..(hk + kh + 1) * d]);
                let v = &row[(2 * hk + h) * d..(2 * hk + h + 1) * d];
                for x in [&mut q, &mut k] {
                    let r = 1.0 / (dot(x, x) + eps).sqrt();
                    x.iter_mut().for_each(|e| *e *= r);
                }
                let sp = {
                    let z = alpha[t * hv + h] + dt[h];
                    if z > 20.0 { z } else { z.exp().ln_1p() }
                };
                let g = (sp * av[h]).exp();
                let b = 1.0 / (1.0 + (-beta[t * hv + h]).exp());
                let o = unsafe { std::slice::from_raw_parts_mut(out_ptr.get().add(t * hv * d + h * d), d) };
                for j in 0..d {
                    let col = &mut st[j * d..(j + 1) * d];
                    col.iter_mut().for_each(|e| *e *= g);
                    let delta = (v[j] - dot(col, &k)) * b;
                    col.iter_mut().zip(&k).for_each(|(e, k)| *e += k * delta);
                    o[j] = dot(col, &q) * scale;
                }
                let k_t = t - t0;
                if let Some(p) = snap_ptr
                    && k_t < runs.snap.get(s).map_or(0, |&n| n as usize).min(n_snap)
                {
                    // SAFETY: (snapshot entry, head) blocks are this job's own.
                    let at = runs.snap_at[s] as usize;
                    let dst = unsafe { std::slice::from_raw_parts_mut(p.get().add(((at + k_t) * hv + h) * d * d), d * d) };
                    dst.copy_from_slice(st);
                }
            }
        });
        Ok(())
    }

    fn copy_span(&self, src: &Vec<f32>, src_off: usize, dst: &mut Vec<f32>, dst_off: usize, len: usize) -> Result<()> {
        check_len("copy_span src", src.len(), src_off + len)?;
        check_len("copy_span dst", dst.len(), dst_off + len)?;
        dst[dst_off..dst_off + len].copy_from_slice(&src[src_off..src_off + len]);
        Ok(())
    }

    fn gather_rows(&self, x: &Vec<f32>, rows: &[u32], width: usize, out: &mut Vec<f32>) -> Result<()> {
        if width == 0 {
            return Ok(());
        }
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
        if layer >= kv.k.len() {
            return Err(Error::Shape(format!("kv_store: layer {layer} of {}", kv.k.len())));
        }
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
                // The rotation can be narrower than a 32-value q8 group
                // (head dims 48, 80, 112: 16): rotate, round, rotate back.
                let rot = 1 << hd.trailing_zeros();
                let slot = &mut kc[s * w..(s + 1) * w];
                slot.chunks_exact_mut(rot).for_each(fwht);
                slot.chunks_exact_mut(32).for_each(q8_round);
                slot.chunks_exact_mut(rot).for_each(fwht);
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
        let bs = meta.block_size;
        // Everything the parallel loop below indexes is checked here: a
        // panic inside rayon would take the process down.
        if nkv == 0 || hd == 0 || !nh.is_multiple_of(nkv) {
            return Err(Error::Shape(format!("attention: {nh} heads over {nkv} KV heads of size {hd}")));
        }
        let l = &kv.layout;
        if (l.n_kv_heads, l.head_dim, l.block_size) != (nkv, hd, bs) || layer >= kv.k.len() {
            return Err(Error::Shape(format!("attention: layer {layer}, {nkv} x {hd}, block {bs} vs cache {l:?}")));
        }
        if meta.token_seq.len() < n_tok {
            return Err(Error::Shape(format!("attention: {} sequence ids for {n_tok} tokens", meta.token_seq.len())));
        }
        for t in 0..n_tok {
            let ctx = meta.positions[t] as usize + 1;
            let table = meta.block_tables.get(meta.token_seq[t] as usize);
            let ok = table.is_some_and(|tb| tb.len() >= ctx.div_ceil(bs) && tb[..ctx.div_ceil(bs)].iter().all(|&b| (b as usize) < l.n_blocks));
            if !ok {
                return Err(Error::Shape(format!("attention: token {t} (position {}) has no valid block table", ctx - 1)));
            }
        }
        let group = nh / nkv;
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

#[cfg(test)]
mod tests {
    use super::*;

    fn layout(n_kv_heads: usize, head_dim: usize) -> KvLayout {
        KvLayout { n_layers: 1, n_blocks: 2, block_size: 4, n_kv_heads, head_dim, dtype: KvType::F16 }
    }

    fn meta<'a>(n_heads: usize, n_kv_heads: usize, tables: &'a [Vec<u32>], positions: &'a [u32]) -> AttnMeta<'a> {
        AttnMeta {
            n_heads,
            n_kv_heads,
            head_dim: 4,
            scale: 0.5,
            window: 0,
            token_seq: &[0, 0],
            positions,
            block_tables: tables,
            block_size: 4,
        }
    }

    #[test]
    fn bad_attention_shapes_are_errors_not_panics() {
        let b = CpuBackend::new();
        let kv = b.alloc_kv(layout(2, 4)).unwrap();
        let tables = [vec![0u32, 1]];
        let (q, mut out) = (vec![0f32; 2 * 4 * 4], vec![0f32; 2 * 4 * 4]);
        assert!(b.attention(&q, &kv, 0, &meta(4, 2, &tables, &[0, 1]), &mut out).is_ok());
        // Zero KV heads, KV heads not dividing heads, a layout mismatch, a bad layer.
        assert!(b.attention(&q, &kv, 0, &meta(4, 0, &tables, &[0, 1]), &mut out).is_err());
        assert!(b.attention(&q, &kv, 0, &meta(3, 2, &tables, &[0, 1]), &mut out).is_err());
        assert!(b.attention(&q, &kv, 0, &meta(4, 4, &tables, &[0, 1]), &mut out).is_err());
        assert!(b.attention(&q, &kv, 1, &meta(4, 2, &tables, &[0, 1]), &mut out).is_err());
        // Positions past the block table, block ids past the cache, a missing table.
        assert!(b.attention(&q, &kv, 0, &meta(4, 2, &tables, &[0, 8]), &mut out).is_err());
        assert!(b.attention(&q, &kv, 0, &meta(4, 2, &[vec![0, 5]], &[0, 5]), &mut out).is_err());
        assert!(b.attention(&q, &kv, 0, &meta(4, 2, &[], &[0, 1]), &mut out).is_err());
    }

    #[test]
    fn empty_layouts_and_short_weights_are_errors() {
        let b = CpuBackend::new();
        assert!(b.alloc_kv(layout(0, 4)).is_err());
        assert!(b.alloc_kv(KvLayout { block_size: 0, ..layout(2, 4) }).is_err());
        let t = HostTensor::from_f32("w", vec![2, 4], &[0.0; 8]);
        assert!(b.upload(&t).is_ok());
        let short = HostTensor { shape: vec![3, 4], ..t };
        assert!(b.upload(&short).is_err());
    }
}
