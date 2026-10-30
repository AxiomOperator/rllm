//! Decode matvec (m = 1) time against size, BF16 and Q8_0, plus the cost of
//! a tiny launch: separates per-kernel fixed cost from bandwidth.
//! `cargo run --release -p rllm-hip --example matvec_sweep`
use std::sync::Arc;
use std::time::Instant;

use rllm_core::{Backend, DType, HostBytes, HostTensor};
use rllm_hip::HipBackend;

fn weight(dtype: DType, rows: usize, cols: usize) -> HostTensor {
    let mut bytes = Vec::with_capacity(dtype.row_bytes(cols) * rows);
    let mut s = 1u32;
    let mut rnd = || {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        (s >> 24) as u8
    };
    match dtype {
        DType::BF16 => (0..rows * cols).for_each(|_| bytes.extend(half::bf16::from_f32(rnd() as f32 / 2560.0).to_le_bytes())),
        DType::Q8_0 => (0..rows * cols / 32).for_each(|_| {
            bytes.extend(half::f16::from_f32(0.01).to_le_bytes());
            (0..32).for_each(|_| bytes.push(rnd()));
        }),
        _ => unimplemented!(),
    }
    let len = bytes.len();
    let owner: Arc<dyn AsRef<[u8]> + Send + Sync> = Arc::new(bytes);
    HostTensor { name: "w".into(), dtype, shape: vec![rows, cols], bytes: HostBytes::new(owner, 0..len) }
}

fn time(b: &HipBackend, iters: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..10 {
        f();
    }
    b.synchronize().unwrap();
    let t = Instant::now();
    for _ in 0..iters {
        f();
    }
    b.synchronize().unwrap();
    t.elapsed().as_secs_f64() * 1e6 / iters as f64
}

fn main() {
    let b = HipBackend::new(0).expect("hip");
    let (mut x, y) = (b.alloc(1024).unwrap(), b.alloc(1024).unwrap());
    println!("tiny launch (add, 1024): {:.2} us", time(&b, 2000, || b.add(&mut x, &y, 1024).unwrap()));
    for dtype in [DType::BF16, DType::Q8_0] {
        for (n, k) in [(1024, 1024), (4096, 1024), (16384, 1024), (1024, 4096), (4096, 4096), (12288, 4096), (65536, 1024)] {
            let w = b.upload(&weight(dtype, n, k)).unwrap();
            let x = b.alloc(k).unwrap();
            let mut out = b.alloc(n).unwrap();
            let us = time(&b, 200, || b.matmul(&x, &w, &mut out, 1).unwrap());
            let bytes = dtype.row_bytes(k) * n;
            println!("{dtype:?} {n:>6}x{k:<5}: {us:8.2} us  {:6.1} GB/s", bytes as f64 / us / 1e3);
        }
    }
}
