// Every GPU op against the CPU reference backend on random inputs; shared by
// rllm-cuda and rllm-hip (each defines `GpuBackend` and `VENDOR`, then
// includes this file). Skipped (with a message) when no device is available.

use std::sync::Arc;

use half::{bf16, f16};
use rllm_core::{AttnMeta, Backend, DType, HostBytes, HostTensor, KvLayout, KvType};
use rllm_cpu::CpuBackend;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn f32(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
    fn byte(&mut self) -> u8 {
        self.next() as u8
    }
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.f32()).collect()
    }
}

fn gpu_backend() -> Option<GpuBackend> {
    match GpuBackend::new(0) {
        Ok(b) => Some(b),
        Err(e) => {
            eprintln!("skipping {VENDOR} parity tests: {e}");
            None
        }
    }
}

/// Random but well-formed weight data: finite, modest block scales.
fn random_weight(rng: &mut Rng, dtype: DType, rows: usize, cols: usize) -> HostTensor {
    let n = rows * cols;
    let mut bytes = Vec::with_capacity(dtype.row_bytes(cols) * rows);
    let scale = |rng: &mut Rng, lo: f32, hi: f32| f16::from_f32(lo + (rng.f32() * 0.5 + 0.5) * (hi - lo)).to_le_bytes();
    match dtype {
        DType::F32 => (0..n).for_each(|_| bytes.extend(rng.f32().to_le_bytes())),
        DType::F16 => (0..n).for_each(|_| bytes.extend(f16::from_f32(rng.f32()).to_le_bytes())),
        DType::BF16 => (0..n).for_each(|_| bytes.extend(bf16::from_f32(rng.f32()).to_le_bytes())),
        DType::Q8_0 => {
            for _ in 0..n / 32 {
                bytes.extend(scale(rng, 0.001, 0.01));
                (0..32).for_each(|_| bytes.push(rng.byte()));
            }
        }
        DType::Q4K => {
            for _ in 0..n / 256 {
                bytes.extend(scale(rng, 0.001, 0.01));
                bytes.extend(scale(rng, 0.001, 0.01));
                (0..140).for_each(|_| bytes.push(rng.byte()));
            }
        }
        DType::Q5K => {
            // Half Q4_K's scales: 5-bit values are twice as large, and the
            // FFN tests' outputs would otherwise pass the f16 GEMM output
            // range on HIP (see HALF_OUT_SCALE), which real weights never do.
            for _ in 0..n / 256 {
                bytes.extend(scale(rng, 0.0005, 0.005));
                bytes.extend(scale(rng, 0.0005, 0.005));
                (0..172).for_each(|_| bytes.push(rng.byte()));
            }
        }
        DType::Q6K => {
            for _ in 0..n / 256 {
                (0..208).for_each(|_| bytes.push(rng.byte()));
                bytes.extend(scale(rng, 0.0001, 0.001));
            }
        }
        DType::IQ4XS => {
            for _ in 0..n / 256 {
                bytes.extend(scale(rng, 0.00005, 0.0005));
                (0..134).for_each(|_| bytes.push(rng.byte()));
            }
        }
    }
    let len = bytes.len();
    let owner: Arc<dyn AsRef<[u8]> + Send + Sync> = Arc::new(bytes);
    HostTensor { name: format!("{dtype}"), dtype, shape: vec![rows, cols], bytes: HostBytes::new(owner, 0..len) }
}

fn f32_tensor(name: &str, data: &[f32]) -> HostTensor {
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let len = bytes.len();
    let owner: Arc<dyn AsRef<[u8]> + Send + Sync> = Arc::new(bytes);
    HostTensor { name: name.into(), dtype: DType::F32, shape: vec![data.len()], bytes: HostBytes::new(owner, 0..len) }
}

fn upload<B: Backend>(b: &B, data: &[f32]) -> B::Buffer {
    // Backends only expose alloc + compute; route data in via an embedding lookup
    // of an F32 "table" whose rows are the data.
    let width = 32;
    assert_eq!(data.len() % width, 0);
    let rows = data.len() / width;
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let len = bytes.len();
    let owner: Arc<dyn AsRef<[u8]> + Send + Sync> = Arc::new(bytes);
    let t = HostTensor { name: "data".into(), dtype: DType::F32, shape: vec![rows, width], bytes: HostBytes::new(owner, 0..len) };
    let w = b.upload(&t).unwrap();
    let mut buf = b.alloc(data.len()).unwrap();
    let ids: Vec<u32> = (0..rows as u32).collect();
    b.embed(&w, &ids, &mut buf).unwrap();
    buf
}

fn download<B: Backend>(b: &B, buf: &B::Buffer, len: usize) -> Vec<f32> {
    let mut out = vec![0f32; len];
    b.to_host(buf, &mut out).unwrap();
    out
}

#[track_caller]
fn assert_close(what: &str, want: &[f32], got: &[f32], rel_tol: f32) {
    assert_eq!(want.len(), got.len(), "{what}: length");
    let scale = want.iter().map(|v| v.abs()).fold(0f32, f32::max).max(1e-6);
    let (i, diff) = want.iter().zip(got).map(|(a, b)| (a - b).abs()).enumerate().fold((0, 0f32), |m, (i, d)| if d > m.1 { (i, d) } else { m });
    assert!(got.iter().all(|v| v.is_finite()), "{what}: non-finite output");
    assert!(diff / scale <= rel_tol, "{what}: max diff {diff} at {i} (want {}, got {}), rel {} > {rel_tol}", want[i], got[i], diff / scale);
}

#[test]
fn matmul_all_dtypes_both_paths() {
    let Some(gpu) = gpu_backend() else { return };
    let cpu = CpuBackend::new();
    let mut rng = Rng(0x1234_5678);
    let (n, k) = (96, 512);
    for dtype in [DType::F32, DType::F16, DType::BF16, DType::Q8_0, DType::Q4K, DType::Q6K, DType::Q5K, DType::IQ4XS] {
        let w = random_weight(&mut rng, dtype, n, k);
        let (wc, wg) = (cpu.upload(&w).unwrap(), gpu.upload(&w).unwrap());
        // m <= 8 uses the matvec kernel, larger m the f16 cuBLAS path.
        for m in [1, 3, 8, 9, 33] {
            let x = rng.vec(m * k);
            let (xc, xg) = (upload(&cpu, &x), upload(&gpu, &x));
            let (mut oc, mut og) = (cpu.alloc(m * n).unwrap(), gpu.alloc(m * n).unwrap());
            cpu.matmul(&xc, &wc, &mut oc, m).unwrap();
            gpu.matmul(&xg, &wg, &mut og, m).unwrap();
            // m <= 8 on quantized weights runs int8 activations (dp4a), like
            // llama.cpp's q8_1 path: per-32 rounding costs ~0.5% per dot product.
            let quantized = matches!(dtype, DType::Q8_0 | DType::Q4K | DType::Q6K | DType::Q5K | DType::IQ4XS);
            let tol = match (m <= 8, quantized) {
                (true, false) => 1e-5,
                (true, true) => 1.5e-2,
                (false, _) => 5e-3,
            };
            assert_close(&format!("matmul {dtype} m={m}"), &download(&cpu, &oc, m * n), &download(&gpu, &og, m * n), tol);
        }
    }
}

#[test]
fn embed_all_dtypes() {
    let Some(gpu) = gpu_backend() else { return };
    let cpu = CpuBackend::new();
    let mut rng = Rng(42);
    let (vocab, d) = (50, 256);
    let ids = [0u32, 7, 49, 7, 3];
    for dtype in [DType::F32, DType::F16, DType::BF16, DType::Q8_0, DType::Q4K, DType::Q6K, DType::Q5K, DType::IQ4XS] {
        let t = random_weight(&mut rng, dtype, vocab, d);
        let (tc, tg) = (cpu.upload(&t).unwrap(), gpu.upload(&t).unwrap());
        let (mut oc, mut og) = (cpu.alloc(ids.len() * d).unwrap(), gpu.alloc(ids.len() * d).unwrap());
        cpu.embed(&tc, &ids, &mut oc).unwrap();
        gpu.embed(&tg, &ids, &mut og).unwrap();
        // K-quants are repacked with per-unit f16 scales (d * sub-scale), which
        // rounds them by up to ~5e-4 relative.
        let tol = if matches!(dtype, DType::Q4K | DType::Q6K | DType::Q5K | DType::IQ4XS) { 1e-3 } else { 1e-6 };
        assert_close(&format!("embed {dtype}"), &download(&cpu, &oc, ids.len() * d), &download(&gpu, &og, ids.len() * d), tol);
    }
}

#[test]
fn elementwise_ops() {
    let Some(gpu) = gpu_backend() else { return };
    let cpu = CpuBackend::new();
    let mut rng = Rng(7);
    let (rows, d) = (5, 128);
    let n = rows * d;
    let x = rng.vec(n);
    let y = rng.vec(n);
    let norm_w = f32_tensor("norm", &rng.vec(d));
    let (nc, ng) = (cpu.upload(&norm_w).unwrap(), gpu.upload(&norm_w).unwrap());

    // rms_norm
    let (mut oc, mut og) = (cpu.alloc(n).unwrap(), gpu.alloc(n).unwrap());
    cpu.rms_norm(&upload(&cpu, &x), &nc, &mut oc, rows, 1e-6).unwrap();
    gpu.rms_norm(&upload(&gpu, &x), &ng, &mut og, rows, 1e-6).unwrap();
    assert_close("rms_norm", &download(&cpu, &oc, n), &download(&gpu, &og, n), 1e-5);

    // rope: 5 tokens x 2 heads x 64
    let positions = [0u32, 1, 17, 300, 4095];
    let (mut rc, mut rg) = (upload(&cpu, &x), upload(&gpu, &x));
    cpu.rope(&mut rc, &positions, 2, 64, 1_000_000.0, None).unwrap();
    gpu.rope(&mut rg, &positions, 2, 64, 1_000_000.0, None).unwrap();
    // Angles reach ~4000 rad at position 4095, so f32 powf/sincos differences
    // between CPU and GPU math libraries show up around 2e-5.
    assert_close("rope", &download(&cpu, &rc, n), &download(&gpu, &rg, n), 1e-4);

    // rope with frequency factors: half the pairs effectively unrotated (partial rotary)
    let factors: Vec<f32> = (0..32).map(|i| if i < 16 { 1.0 } else { 1e30 }).collect();
    let ff = f32_tensor("rope_freqs", &factors);
    let (fc, fg) = (cpu.upload(&ff).unwrap(), gpu.upload(&ff).unwrap());
    let (mut rc, mut rg) = (upload(&cpu, &x), upload(&gpu, &x));
    cpu.rope(&mut rc, &positions, 2, 64, 10_000.0, Some(&fc)).unwrap();
    gpu.rope(&mut rg, &positions, 2, 64, 10_000.0, Some(&fg)).unwrap();
    let (rc, rg) = (download(&cpu, &rc, n), download(&gpu, &rg, n));
    assert_close("rope freq_factors", &rc, &rg, 1e-4);
    // Pairs 16..32 of each head (elements 16..32 and 48..64) must be untouched.
    for head in 0..10 {
        for i in [16, 31, 48, 63] {
            assert_eq!(rc[head * 64 + i], x[head * 64 + i], "partial rotary changed element {i}");
        }
    }

    // gelu / gelu_mul
    let (mut gc, mut gg) = (upload(&cpu, &x), upload(&gpu, &x));
    cpu.gelu(&mut gc, n).unwrap();
    gpu.gelu(&mut gg, n).unwrap();
    assert_close("gelu", &download(&cpu, &gc, n), &download(&gpu, &gg, n), 1e-6);
    let (mut gc, mut gg) = (upload(&cpu, &x), upload(&gpu, &x));
    cpu.gelu_mul(&mut gc, &upload(&cpu, &y), n).unwrap();
    gpu.gelu_mul(&mut gg, &upload(&gpu, &y), n).unwrap();
    assert_close("gelu_mul", &download(&cpu, &gc, n), &download(&gpu, &gg, n), 1e-6);

    // mul / scale
    let (mut mc, mut mg) = (upload(&cpu, &x), upload(&gpu, &x));
    cpu.mul(&mut mc, &upload(&cpu, &y), n).unwrap();
    gpu.mul(&mut mg, &upload(&gpu, &y), n).unwrap();
    assert_close("mul", &download(&cpu, &mc, n), &download(&gpu, &mg, n), 1e-7);
    let (mut sc, mut sg) = (upload(&cpu, &x), upload(&gpu, &x));
    cpu.scale(&mut sc, 0.37, n).unwrap();
    gpu.scale(&mut sg, 0.37, n).unwrap();
    assert_close("scale", &download(&cpu, &sc, n), &download(&gpu, &sg, n), 1e-7);

    // silu_mul
    let (mut gc, mut gg) = (upload(&cpu, &x), upload(&gpu, &x));
    cpu.silu_mul(&mut gc, &upload(&cpu, &y), n).unwrap();
    gpu.silu_mul(&mut gg, &upload(&gpu, &y), n).unwrap();
    assert_close("silu_mul", &download(&cpu, &gc, n), &download(&gpu, &gg, n), 1e-6);

    // add
    let (mut ac, mut ag) = (upload(&cpu, &x), upload(&gpu, &x));
    cpu.add(&mut ac, &upload(&cpu, &y), n).unwrap();
    gpu.add(&mut ag, &upload(&gpu, &y), n).unwrap();
    assert_close("add", &download(&cpu, &ac, n), &download(&gpu, &ag, n), 1e-7);

    // gather_rows
    let rows_idx = [4u32, 0, 2];
    let (mut oc, mut og) = (cpu.alloc(3 * d).unwrap(), gpu.alloc(3 * d).unwrap());
    cpu.gather_rows(&upload(&cpu, &x), &rows_idx, d, &mut oc).unwrap();
    gpu.gather_rows(&upload(&gpu, &x), &rows_idx, d, &mut og).unwrap();
    assert_close("gather_rows", &download(&cpu, &oc, 3 * d), &download(&gpu, &og, 3 * d), 0.0);
}

/// Two sequences sharing a step: one mid-prefill, one decoding, scattered
/// non-contiguous blocks, GQA (4 query heads per KV head). Covers head sizes
/// up to 512 and sliding-window attention.
#[test]
fn paged_attention_mixed_batch() {
    for (head_dim, window) in [(128, 0), (256, 3), (512, 0), (512, 5)] {
        attention_case(head_dim, window, Step::Mixed, KvType::F16);
    }
}

/// One token per sequence (the decode kernel), including a sliding window.
#[test]
fn paged_attention_decode_only() {
    for (head_dim, window) in [(128, 0), (256, 4), (512, 0)] {
        attention_case(head_dim, window, Step::Decode, KvType::F16);
    }
}

/// A q8 cache through every attention path: decode, a small mixed step (the
/// decode kernel reading int8), and a long prefill (blocks expanded to f16).
/// The CPU emulates q8's rounding (with the key rotation).
#[test]
fn paged_attention_q8_cache() {
    for (head_dim, window) in [(64, 0), (128, 0), (256, 3), (512, 0)] {
        for step in [Step::Decode, Step::Mixed, Step::Long] {
            attention_case(head_dim, window, step, KvType::Q8);
        }
    }
}

/// Head sizes whose dims don't split evenly into 128-wide warp slices.
#[test]
fn paged_attention_odd_head_dims() {
    for head_dim in [320, 352, 416, 480] {
        for step in [Step::Mixed, Step::Long] {
            attention_case(head_dim, 0, step, KvType::F16);
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Step {
    /// One token per sequence.
    Decode,
    /// A 3-token prefill chunk and a decode token.
    Mixed,
    /// An 11-token step (beyond the decode-sized steps).
    Long,
}

fn attention_case(head_dim: usize, window: usize, step: Step, kv: KvType) {
    let Some(gpu) = gpu_backend() else { return };
    let cpu = CpuBackend::new();
    let mut rng = Rng(99);
    let (n_heads, n_kv_heads, block_size) = (8, 2, 4);
    let layout = KvLayout { n_layers: 1, n_blocks: 16, block_size, n_kv_heads, head_dim, dtype: kv };
    let kv_w = layout.slot_width();
    let block_tables = vec![vec![9, 2, 14, 5], vec![0, 11, 3]];
    let slot = |seq: usize, p: usize| block_tables[seq][p / block_size] * block_size as u32 + (p % block_size) as u32;

    let (mut kc, mut kg) = (cpu.alloc_kv(layout).unwrap(), gpu.alloc_kv(layout).unwrap());
    // History: seq 0 has 10 tokens, seq 1 has 9.
    let hist: Vec<(usize, usize)> = (0..10).map(|p| (0, p)).chain((0..9).map(|p| (1, p))).collect();
    let slots: Vec<u32> = hist.iter().map(|&(s, p)| slot(s, p)).collect();
    let k = rng.vec(hist.len() * kv_w);
    let v = rng.vec(hist.len() * kv_w);
    cpu.kv_store(&mut kc, 0, &upload(&cpu, &k), &upload(&cpu, &v), &slots).unwrap();
    gpu.kv_store(&mut kg, 0, &upload(&gpu, &k), &upload(&gpu, &v), &slots).unwrap();

    // Mixed: seq 0 tokens at positions 7, 8, 9 (prefill chunk), seq 1 at position 8 (decode).
    // Decode: one token each, seq 0 at position 9, seq 1 at position 8.
    // Long: all of seq 0 (positions 0..=9), seq 1 at position 8.
    let (token_seq, positions): (Vec<u32>, Vec<u32>) = match step {
        Step::Decode => (vec![0, 1], vec![9, 8]),
        Step::Mixed => (vec![0, 0, 0, 1], vec![7, 8, 9, 8]),
        Step::Long => ((0..10).map(|_| 0).chain([1]).collect(), (0..10).chain([8]).collect()),
    };
    let n_q = token_seq.len();
    let q = rng.vec(n_q * n_heads * head_dim);
    let meta = AttnMeta {
        n_heads,
        n_kv_heads,
        head_dim,
        scale: 1.0 / (head_dim as f32).sqrt(),
        window,
        token_seq: &token_seq,
        positions: &positions,
        block_tables: &block_tables,
        block_size,
    };
    let n = n_q * n_heads * head_dim;
    let (mut oc, mut og) = (cpu.alloc(n).unwrap(), gpu.alloc(n).unwrap());
    cpu.attention(&upload(&cpu, &q), &kc, 0, &meta, &mut oc).unwrap();
    gpu.attention(&upload(&gpu, &q), &kg, 0, &meta, &mut og).unwrap();
    // Prefill tokens go through the tensor-core kernel (f16 Q/K/P/V, f32
    // accumulate). With q8, an int8 step can round differently on each side.
    assert_close(
        &format!("paged_attention hd={head_dim} window={window} {step:?} {kv}"),
        &download(&cpu, &oc, n),
        &download(&gpu, &og, n),
        if kv == KvType::Q8 { 5e-3 } else { 2e-3 },
    );
}

/// Index arrays re-uploaded with new contents in quick succession must not
/// clobber a copy that hasn't run yet (pinned staging buffer reuse).
#[test]
fn index_reupload_back_to_back() {
    let Some(gpu) = gpu_backend() else { return };
    let cpu = CpuBackend::new();
    let mut rng = Rng(5);
    let (rows, width) = (64, 32);
    let x = rng.vec(rows * width);
    let (xc, xg) = (upload(&cpu, &x), upload(&gpu, &x));
    let sets: Vec<Vec<u32>> = (0..8).map(|i| (0..16).map(|j| ((i * 7 + j * 3) % rows) as u32).collect()).collect();
    let mut outs_g = Vec::new();
    for rows_i in &sets {
        let mut o = gpu.alloc(rows_i.len() * width).unwrap();
        gpu.gather_rows(&xg, rows_i, width, &mut o).unwrap();
        outs_g.push(o);
    }
    for (rows_i, og) in sets.iter().zip(&outs_g) {
        let mut oc = cpu.alloc(rows_i.len() * width).unwrap();
        cpu.gather_rows(&xc, rows_i, width, &mut oc).unwrap();
        let n = rows_i.len() * width;
        assert_close("gather_rows back-to-back", &download(&cpu, &oc, n), &download(&gpu, og, n), 0.0);
    }
}

/// Fused Q/K/V norm + RoPE + cache store against the CPU's composed version,
/// checked through Q and an attention pass over the stored K/V.
#[test]
fn qkv_norm_rope_store_matches() {
    let Some(gpu) = gpu_backend() else { return };
    let cpu = CpuBackend::new();
    let mut rng = Rng(11);
    let (n_heads, n_kv, hd, bs) = (8, 2, 128, 4);
    for (qk_norm, v_norm, partial) in [(true, false, false), (true, true, true), (false, false, true)] {
        let layout = KvLayout { n_layers: 1, n_blocks: 4, block_size: bs, n_kv_heads: n_kv, head_dim: hd, dtype: KvType::F16 };
        let (mut kc, mut kg) = (cpu.alloc_kv(layout).unwrap(), gpu.alloc_kv(layout).unwrap());
        let n = 5;
        let positions: Vec<u32> = (0..n as u32).collect();
        let slots: Vec<u32> = vec![2, 3, 8, 9, 10];
        let tables = vec![vec![0u32, 2, 1]];
        let q = rng.vec(n * n_heads * hd);
        let k = rng.vec(n * n_kv * hd);
        let v = rng.vec(n * n_kv * hd);
        let w = |rng: &mut Rng| f32_tensor("w", &rng.vec(hd).iter().map(|x| 1.0 + 0.3 * x).collect::<Vec<_>>());
        let (qw, kw, vw) = (w(&mut rng), w(&mut rng), w(&mut rng));
        let factors: Vec<f32> = (0..hd / 2).map(|i| if i < hd / 4 { 1.0 } else { 1e30 }).collect();
        let ff = f32_tensor("ff", &factors);
        let (qwc, kwc, vwc, ffc) = (cpu.upload(&qw).unwrap(), cpu.upload(&kw).unwrap(), cpu.upload(&vw).unwrap(), cpu.upload(&ff).unwrap());
        let (qwg, kwg, vwg, ffg) = (gpu.upload(&qw).unwrap(), gpu.upload(&kw).unwrap(), gpu.upload(&vw).unwrap(), gpu.upload(&ff).unwrap());
        let nq = n * n_heads * hd;
        let (mut qnc, mut qng) = (cpu.alloc(nq).unwrap(), gpu.alloc(nq).unwrap());
        cpu.qkv_norm_rope_store(
            &upload(&cpu, &q), &upload(&cpu, &k), &upload(&cpu, &v), qk_norm.then_some((&qwc, &kwc)), v_norm.then_some(&vwc), &mut qnc,
            &mut kc, 0, &positions, &slots, n_heads, n_kv, hd, 1e-6, 10_000.0, partial.then_some(&ffc),
        )
        .unwrap();
        gpu.qkv_norm_rope_store(
            &upload(&gpu, &q), &upload(&gpu, &k), &upload(&gpu, &v), qk_norm.then_some((&qwg, &kwg)), v_norm.then_some(&vwg), &mut qng,
            &mut kg, 0, &positions, &slots, n_heads, n_kv, hd, 1e-6, 10_000.0, partial.then_some(&ffg),
        )
        .unwrap();
        assert_close("qkv_norm_rope_store q", &download(&cpu, &qnc, nq), &download(&gpu, &qng, nq), 1e-4);

        // Read the cache back through decode attention for the last token.
        let meta = AttnMeta {
            n_heads,
            n_kv_heads: n_kv,
            head_dim: hd,
            scale: 1.0 / (hd as f32).sqrt(),
            window: 0,
            token_seq: &[0],
            positions: &[4],
            block_tables: &tables,
            block_size: bs,
        };
        let q1 = rng.vec(n_heads * hd);
        let (mut oc, mut og) = (cpu.alloc(n_heads * hd).unwrap(), gpu.alloc(n_heads * hd).unwrap());
        cpu.attention(&upload(&cpu, &q1), &kc, 0, &meta, &mut oc).unwrap();
        gpu.attention(&upload(&gpu, &q1), &kg, 0, &meta, &mut og).unwrap();
        assert_close(
            &format!("qkv_norm_rope_store cache qk_norm={qk_norm} v_norm={v_norm} partial={partial}"),
            &download(&cpu, &oc, n_heads * hd),
            &download(&gpu, &og, n_heads * hd),
            2e-3,
        );
    }
}

#[test]
fn gated_ffn_fused_prefill_and_decode() {
    // 2048 x 2048 weights reach the MMQ threshold, so gate/up/activation run
    // as one fused MMQ (mmq_glu_*) and the down projection reads its int8
    // output. m = 70 leaves a partial token tile.
    let Some(gpu) = gpu_backend() else { return };
    let cpu = CpuBackend::new();
    let mut rng = Rng(0x0bad_cafe);
    let (d, f) = (2048, 2048);
    // m = 1 and 3 take the decode form (matvec_glu, then the down matvec reads
    // its int8 output).
    for (dtype, m) in [DType::Q8_0, DType::Q4K, DType::Q6K, DType::Q5K, DType::IQ4XS].into_iter().flat_map(|t| [(t, 70), (t, 1), (t, 3)]) {
        let (gate, up, down) = (random_weight(&mut rng, dtype, f, d), random_weight(&mut rng, dtype, f, d), random_weight(&mut rng, dtype, d, f));
        let wc = [&gate, &up, &down].map(|w| cpu.upload(w).unwrap());
        let wg = [&gate, &up, &down].map(|w| gpu.upload(w).unwrap());
        let x = rng.vec(m * d);
        let (xc, xg) = (upload(&cpu, &x), upload(&gpu, &x));
        for act in [rllm_core::Activation::Silu, rllm_core::Activation::GeluTanh] {
            let run = |b: &dyn Fn(&mut [f32])| {
                let mut o = vec![0f32; m * d];
                b(&mut o);
                o
            };
            let want = run(&|o| {
                let (mut out, mut g, mut u) = (cpu.alloc(m * d).unwrap(), cpu.alloc(m * f).unwrap(), cpu.alloc(m * f).unwrap());
                cpu.gated_ffn(&xc, &wc[0], &wc[1], &wc[2], act, &mut out, &mut g, &mut u, m, f).unwrap();
                cpu.to_host(&out, o).unwrap();
            });
            let got = run(&|o| {
                let (mut out, mut g, mut u) = (gpu.alloc(m * d).unwrap(), gpu.alloc(m * f).unwrap(), gpu.alloc(m * f).unwrap());
                gpu.gated_ffn(&xg, &wg[0], &wg[1], &wg[2], act, &mut out, &mut g, &mut u, m, f).unwrap();
                gpu.to_host(&out, o).unwrap();
            });
            // Two int8 activation roundings (FFN input and the product). Decode
            // rounds per token rather than across a batch; on these large
            // random Q4_K values that reaches ~4% (the unfused path, with
            // RLLM_NO_GLU_MMQ=1, gives bit-identical results).
            let tol = if m <= 8 { 5e-2 } else { 3e-2 };
            assert_close(&format!("gated_ffn {dtype} {act:?} m={m}"), &want, &got, tol);
        }
    }
}

#[test]
fn matvec_float_split_k() {
    // Few rows and k >= 1024: the float matvec splits each row across warps.
    // 37 rows leaves a partial row group in the last block.
    let Some(gpu) = gpu_backend() else { return };
    let cpu = CpuBackend::new();
    let mut rng = Rng(0x5eed_f00d);
    let (n, k) = (37, 2048);
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        let w = random_weight(&mut rng, dtype, n, k);
        let (wc, wg) = (cpu.upload(&w).unwrap(), gpu.upload(&w).unwrap());
        for m in [1, 2, 8] {
            let x = rng.vec(m * k);
            let (xc, xg) = (upload(&cpu, &x), upload(&gpu, &x));
            let (mut oc, mut og) = (cpu.alloc(m * n).unwrap(), gpu.alloc(m * n).unwrap());
            cpu.matmul(&xc, &wc, &mut oc, m).unwrap();
            gpu.matmul(&xg, &wg, &mut og, m).unwrap();
            assert_close(&format!("matvec split-k {dtype} m={m}"), &download(&cpu, &oc, m * n), &download(&gpu, &og, m * n), 1e-5);
        }
    }
}

#[test]
fn gelu_mul_strided_matches() {
    let Some(gpu) = gpu_backend() else { return };
    let cpu = CpuBackend::new();
    let mut rng = Rng(0x0dd_ba11);
    // x is [count, layers, width]; take layer 3 of 7 for each of 5 rows.
    let (count, layers, width, first) = (5, 7, 64, 3);
    let (gate, x) = (rng.vec(count * width), rng.vec(count * layers * width));
    let run = |b: &dyn Fn(&mut [f32])| {
        let mut o = vec![0f32; count * width];
        b(&mut o);
        o
    };
    let want = run(&|o| {
        let (mut g, xb, mut s) = (upload(&cpu, &gate), upload(&cpu, &x), cpu.alloc(count * width).unwrap());
        cpu.gelu_mul_strided(&mut g, &xb, first, layers, count, width, &mut s).unwrap();
        cpu.to_host(&g, o).unwrap();
    });
    let got = run(&|o| {
        let (mut g, xb, mut s) = (upload(&gpu, &gate), upload(&gpu, &x), gpu.alloc(count * width).unwrap());
        gpu.gelu_mul_strided(&mut g, &xb, first, layers, count, width, &mut s).unwrap();
        gpu.to_host(&g, o).unwrap();
    });
    assert_close("gelu_mul_strided", &want, &got, 1e-5);
}

#[test]
fn gated_ffn_float_decode_fused() {
    // Float gate/up in decode: matvec_glu_* computes both rows and the activation.
    let Some(gpu) = gpu_backend() else { return };
    let cpu = CpuBackend::new();
    let mut rng = Rng(0xf1_0a7);
    let (d, f) = (256, 384);
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        let (gate, up, down) = (random_weight(&mut rng, dtype, f, d), random_weight(&mut rng, dtype, f, d), random_weight(&mut rng, dtype, d, f));
        let wc = [&gate, &up, &down].map(|w| cpu.upload(w).unwrap());
        let wg = [&gate, &up, &down].map(|w| gpu.upload(w).unwrap());
        for m in [1, 3] {
            let x = rng.vec(m * d);
            let (xc, xg) = (upload(&cpu, &x), upload(&gpu, &x));
            for act in [rllm_core::Activation::Silu, rllm_core::Activation::GeluTanh] {
                let (mut oc, mut gc, mut uc) = (cpu.alloc(m * d).unwrap(), cpu.alloc(m * f).unwrap(), cpu.alloc(m * f).unwrap());
                cpu.gated_ffn(&xc, &wc[0], &wc[1], &wc[2], act, &mut oc, &mut gc, &mut uc, m, f).unwrap();
                let (mut og, mut gg, mut ug) = (gpu.alloc(m * d).unwrap(), gpu.alloc(m * f).unwrap(), gpu.alloc(m * f).unwrap());
                gpu.gated_ffn(&xg, &wg[0], &wg[1], &wg[2], act, &mut og, &mut gg, &mut ug, m, f).unwrap();
                assert_close(&format!("float gated_ffn {dtype} {act:?} m={m}"), &download(&cpu, &oc, m * d), &download(&gpu, &og, m * d), 1e-4);
            }
        }
    }
}

#[test]
fn norm_add_gated_ffn_matches() {
    // rows = 1 with quantized weights takes the fused decode kernel
    // (matvec_glu_norm); rows = 70 the split path.
    let Some(gpu) = gpu_backend() else { return };
    let cpu = CpuBackend::new();
    let mut rng = Rng(0x0a0d_d00d);
    let (d, f) = (512, 1024);
    let norm = |rng: &mut Rng, name: &str| f32_tensor(name, &rng.vec(d).iter().map(|v| 1.0 + 0.5 * v).collect::<Vec<_>>());
    for dtype in [DType::Q8_0, DType::Q4K, DType::Q6K, DType::Q5K, DType::IQ4XS] {
        let (gate, up, down) = (random_weight(&mut rng, dtype, f, d), random_weight(&mut rng, dtype, f, d), random_weight(&mut rng, dtype, d, f));
        let (n1, n2) = (norm(&mut rng, "n1"), norm(&mut rng, "n2"));
        let wc = [&gate, &up, &down, &n1, &n2].map(|w| cpu.upload(w).unwrap());
        let wg = [&gate, &up, &down, &n1, &n2].map(|w| gpu.upload(w).unwrap());
        for rows in [1, 70] {
            let (src, x) = (rng.vec(rows * d), rng.vec(rows * d));
            let act = rllm_core::Activation::GeluTanh;
            let run_cpu = || {
                let (s, xi) = (upload(&cpu, &src), upload(&cpu, &x));
                let (mut xo, mut nm, mut o, mut g, mut u) =
                    (cpu.alloc(rows * d).unwrap(), cpu.alloc(rows * d).unwrap(), cpu.alloc(rows * d).unwrap(), cpu.alloc(rows * f).unwrap(), cpu.alloc(rows * f).unwrap());
                cpu.norm_add_gated_ffn(&s, &wc[3], &xi, &mut xo, rows, d, 1e-6, 1.0, &wc[4], (&wc[0], &wc[1], &wc[2], act), &mut nm, &mut o, &mut g, &mut u, f).unwrap();
                (download(&cpu, &xo, rows * d), download(&cpu, &o, rows * d))
            };
            let run_gpu = || {
                let (s, xi) = (upload(&gpu, &src), upload(&gpu, &x));
                let (mut xo, mut nm, mut o, mut g, mut u) =
                    (gpu.alloc(rows * d).unwrap(), gpu.alloc(rows * d).unwrap(), gpu.alloc(rows * d).unwrap(), gpu.alloc(rows * f).unwrap(), gpu.alloc(rows * f).unwrap());
                gpu.norm_add_gated_ffn(&s, &wg[3], &xi, &mut xo, rows, d, 1e-6, 1.0, &wg[4], (&wg[0], &wg[1], &wg[2], act), &mut nm, &mut o, &mut g, &mut u, f).unwrap();
                (download(&gpu, &xo, rows * d), download(&gpu, &o, rows * d))
            };
            let ((xc, oc), (xg, og)) = (run_cpu(), run_gpu());
            assert_close(&format!("norm_add_gated_ffn x {dtype} rows={rows}"), &xc, &xg, 1e-5);
            // Two int8 activation roundings, as in gated_ffn_fused_prefill_and_decode.
            assert_close(&format!("norm_add_gated_ffn out {dtype} rows={rows}"), &oc, &og, 5e-2);
        }
    }
}

/// Device sampling against the reference (rllm-core's sample.rs, which the
/// CPU backend runs): greedy exactly; random draws almost always (a draw
/// can differ when a threshold or Gumbel key lands within float rounding).
#[test]
fn sample_rows_matches_reference() {
    let Some(gpu) = gpu_backend() else { return };
    let cpu = CpuBackend::new();
    let mut rng = Rng(0x5a3_91e7);
    let (rows, vocab) = (6, 5024);
    let logits: Vec<f32> = (0..rows * vocab).map(|_| rng.f32() * 6.0).collect();
    let configs = [
        (0.0, 0, 1.0),
        (1.0, 0, 1.0),
        (0.7, 40, 1.0),
        (1.0, 0, 0.9),
        (0.8, 50, 0.95),
    ];
    for (t, k, p) in configs {
        let (mut same, mut total) = (0, 0);
        for counter in 0..20u64 {
            let specs: Vec<rllm_core::RowSample> = (0..rows)
                .map(|r| rllm_core::RowSample {
                    temperature: t,
                    top_k: k,
                    top_p: p,
                    seed: 42 + r as u64,
                    counter,
                    adjust: if r == 1 { vec![(7, 40.0), (9, -3.0), (7, -40.0), (5, 25.0), (5, 25.0)] } else { Vec::new() },
                })
                .collect();
            // Every other row, in reverse: rows needn't be in order or complete.
            let pick: Vec<(u32, &rllm_core::RowSample)> = (0..rows).rev().step_by(2).map(|r| (r as u32, &specs[r])).collect();
            let (mut a, mut b) = (vec![0u32; pick.len()], vec![0u32; pick.len()]);
            cpu.sample_rows(&upload(&cpu, &logits), rows, vocab, &pick, &mut a).unwrap();
            gpu.sample_rows(&upload(&gpu, &logits), rows, vocab, &pick, &mut b).unwrap();
            if t == 0.0 {
                assert_eq!(a, b, "greedy");
            }
            same += a.iter().zip(&b).filter(|(x, y)| x == y).count();
            total += a.len();
        }
        assert!(same * 100 >= total * 95, "T={t} k={k} p={p}: {same}/{total} draws agree");
    }
}
