//! Measures the per-kernel cost of back-to-back tiny launches on one stream:
//! `cargo run --release -p rllm-cuda --example launch_floor`
use std::time::Instant;

use rllm_core::{Backend, HostTensor};
use rllm_cuda::CudaBackend;

fn main() {
    let b = CudaBackend::new(0).expect("cuda");
    let mut x = b.alloc(1024).unwrap();
    let y = b.alloc(1024).unwrap();
    let w = b.upload(&HostTensor::from_f32("w", vec![1024], &[1.0; 1024])).unwrap();
    let mut out = b.alloc(1024).unwrap();
    for (label, n) in [("warmup", 200), ("add x2000", 2000)] {
        b.synchronize().unwrap();
        let t = Instant::now();
        for _ in 0..n {
            b.add(&mut x, &y, 1024).unwrap();
        }
        b.synchronize().unwrap();
        println!("{label}: {:.2} us per launch", t.elapsed().as_secs_f64() * 1e6 / n as f64);
    }
    b.synchronize().unwrap();
    let t = Instant::now();
    for _ in 0..2000 {
        b.rms_norm(&x, &w, &mut out, 1, 1e-6).unwrap();
    }
    b.synchronize().unwrap();
    println!("rms_norm x2000: {:.2} us per launch", t.elapsed().as_secs_f64() * 1e6 / 2000.0);
}
