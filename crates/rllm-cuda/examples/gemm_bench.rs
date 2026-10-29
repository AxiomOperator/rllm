//! Times prefill matmuls (m = 512) for several shapes:
//! `cargo run --release -p rllm-cuda --example gemm_bench` (RLLM_MMQ=0|1 forces a path).
use std::sync::Arc;
use std::time::Instant;

use rllm_core::{Backend, DType, HostBytes, HostTensor};
use rllm_cuda::CudaBackend;

fn q8_weight(rows: usize, cols: usize) -> HostTensor {
    let mut bytes = Vec::with_capacity(rows * cols / 32 * 34);
    let mut s = 1u32;
    for _ in 0..rows * cols / 32 {
        bytes.extend(half::f16::from_f32(0.01).to_le_bytes());
        for _ in 0..32 {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            bytes.push((s >> 24) as u8);
        }
    }
    let len = bytes.len();
    let owner: Arc<dyn AsRef<[u8]> + Send + Sync> = Arc::new(bytes);
    HostTensor { name: "w".into(), dtype: DType::Q8_0, shape: vec![rows, cols], bytes: HostBytes::new(owner, 0..len) }
}

fn main() {
    let b = CudaBackend::new(0).expect("cuda");
    let m = 512;
    for (n, k) in [(3072, 1024), (1024, 3072), (1024, 1024), (2048, 1024), (1024, 2048), (2048, 2560), (4096, 2560), (512, 2560), (2560, 2048), (2560, 4096), (10240, 2560), (2560, 10240), (4096, 14336)] {
        let w = b.upload(&q8_weight(n, k)).unwrap();
        let x = b.alloc(m * k).unwrap();
        let mut out = b.alloc(m * n).unwrap();
        for _ in 0..5 {
            b.matmul(&x, &w, &mut out, m).unwrap();
        }
        b.synchronize().unwrap();
        let iters = 50;
        let t = Instant::now();
        for _ in 0..iters {
            b.matmul(&x, &w, &mut out, m).unwrap();
        }
        b.synchronize().unwrap();
        let us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
        println!("{n:>6}x{k:<6} m={m}: {us:8.1} us  {:6.1} TOPS", 2.0 * (m * n * k) as f64 / us / 1e6);
    }
}
