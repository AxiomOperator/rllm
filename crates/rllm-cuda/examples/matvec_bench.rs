//! Times decode matvecs (m = 1) back to back for several Q8 shapes:
//! `cargo run --release -p rllm-cuda --example matvec_bench`
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
    for (n, k) in [(1024, 1024), (2048, 1024), (3072, 1024), (1024, 3072), (10240, 2560), (2560, 10240), (151936, 1024)] {
        let w = b.upload(&q8_weight(n, k)).unwrap();
        let x = b.alloc(k).unwrap();
        let mut out = b.alloc(n).unwrap();
        for _ in 0..10 {
            b.matmul(&x, &w, &mut out, 1).unwrap();
        }
        b.synchronize().unwrap();
        let iters = 200;
        let t = Instant::now();
        for _ in 0..iters {
            b.matmul(&x, &w, &mut out, 1).unwrap();
        }
        b.synchronize().unwrap();
        let us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
        let bytes = n * k * 17 / 16; // int8 + f16 scale per 32
        println!("{n:>6}x{k:<6}: {us:8.2} us  {:6.1} GB/s", bytes as f64 / us / 1e3);
    }
}
