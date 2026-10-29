//! Isolated timing of specific prefill matmuls (m = 512), Q8, BF16 and F32
//! weights. FFN=1 times a whole Nemotron-sized gated FFN instead.
use std::sync::Arc;
use std::time::Instant;

use rllm_core::{Backend, DType, HostBytes, HostTensor};
use rllm_cuda::CudaBackend;

fn weight(dtype: DType, rows: usize, cols: usize) -> HostTensor {
    let mut bytes = Vec::new();
    let mut s = 1u32;
    let mut rnd = || {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        s
    };
    match dtype {
        DType::Q8_0 => {
            for _ in 0..rows * cols / 32 {
                bytes.extend(half::f16::from_f32(0.01).to_le_bytes());
                for _ in 0..32 {
                    bytes.push((rnd() >> 24) as u8);
                }
            }
        }
        DType::BF16 => {
            for _ in 0..rows * cols {
                bytes.extend(half::bf16::from_f32(((rnd() >> 8) as f32 / 16777216.0 - 0.5) * 0.1).to_le_bytes());
            }
        }
        DType::F32 => {
            for _ in 0..rows * cols {
                bytes.extend((((rnd() >> 8) as f32 / 16777216.0 - 0.5) * 0.1).to_le_bytes());
            }
        }
        _ => unimplemented!(),
    }
    let len = bytes.len();
    let owner: Arc<dyn AsRef<[u8]> + Send + Sync> = Arc::new(bytes);
    HostTensor { name: "w".into(), dtype, shape: vec![rows, cols], bytes: HostBytes::new(owner, 0..len) }
}

fn main() {
    let b = CudaBackend::new(0).expect("cuda");
    let m: usize = std::env::var("M").ok().and_then(|v| v.parse().ok()).unwrap_or(512);
    if std::env::var_os("FFN").is_some() {
        return ffn_bench(&b, m);
    }
    for (dtype, n, k) in [(DType::Q8_0, 10240, 2560), (DType::F32, 256, 2560), (DType::Q8_0, 512, 2560), (DType::Q8_0, 1024, 2560), (DType::BF16, 10752, 2560), (DType::BF16, 3072, 1024), (DType::BF16, 151936, 1024), (DType::Q8_0, 151936, 1024)] {
        let w = b.upload(&weight(dtype, n, k)).unwrap();
        let x = b.alloc(m * k).unwrap();
        let mut out = b.alloc(m * n).unwrap();
        for _ in 0..3 {
            b.matmul(&x, &w, &mut out, m).unwrap();
        }
        b.synchronize().unwrap();
        let t = Instant::now();
        for _ in 0..10 {
            b.matmul(&x, &w, &mut out, m).unwrap();
        }
        b.synchronize().unwrap();
        let us = t.elapsed().as_secs_f64() * 1e6 / 10.0;
        let bytes = n * k * if dtype == DType::BF16 { 2 } else { 1 };
        println!("{dtype:?} {n}x{k} m={m}: {us:9.1} us  {:6.1} TFLOPS  {:6.1} GB/s(weights)", 2.0 * (m * n * k) as f64 / us / 1e6, bytes as f64 / us / 1e3);
    }
}

/// Whole gated FFN (gate/up 10240x2560, down 2560x10240), m tokens.
#[allow(dead_code)]
fn ffn_bench(b: &CudaBackend, m: usize) {
    let (d, f) = (2560, 10240);
    let gate = b.upload(&weight(DType::Q8_0, f, d)).unwrap();
    let up = b.upload(&weight(DType::Q8_0, f, d)).unwrap();
    let down = b.upload(&weight(DType::Q8_0, d, f)).unwrap();
    // Random activations (zeros would hit slow division paths).
    let table = b.upload(&weight(DType::F32, m, d)).unwrap();
    let mut x = b.alloc(m * d).unwrap();
    b.embed(&table, &(0..m as u32).collect::<Vec<_>>(), &mut x).unwrap();
    let mut out = b.alloc(m * d).unwrap();
    let mut g = b.alloc(m * f).unwrap();
    let mut u = b.alloc(m * f).unwrap();
    let mut run = || b.gated_ffn(&x, &gate, &up, &down, rllm_core::Activation::GeluTanh, &mut out, &mut g, &mut u, m, f).unwrap();
    for _ in 0..3 {
        run();
    }
    b.synchronize().unwrap();
    let t = Instant::now();
    for _ in 0..10 {
        run();
    }
    b.synchronize().unwrap();
    println!("gated_ffn 10240x2560 m={m}: {:9.1} us", t.elapsed().as_secs_f64() * 1e6 / 10.0);
}
