//! Times prefill attention alone: one sequence of `n` tokens attending causally.
//! `cargo run --release -p rllm-cuda --example attn_bench`
use std::time::Instant;

use rllm_core::{AttnMeta, Backend, KvLayout};
use rllm_cuda::CudaBackend;

fn main() {
    let b = CudaBackend::new(0).expect("cuda");
    for (n_heads, n_kv, hd, n) in [(16usize, 8usize, 128usize, 512usize), (8, 2, 256, 512), (8, 2, 512, 512), (16, 8, 128, 2048)] {
        let bs = 16;
        let layout = KvLayout { n_layers: 1, n_blocks: n.div_ceil(bs), block_size: bs, n_kv_heads: n_kv, head_dim: hd };
        let mut kv = b.alloc_kv(layout).unwrap();
        let q = b.alloc(n * n_heads * hd).unwrap();
        let k = b.alloc(n * n_kv * hd).unwrap();
        let mut out = b.alloc(n * n_heads * hd).unwrap();
        let slots: Vec<u32> = (0..n as u32).collect();
        b.kv_store(&mut kv, 0, &k, &k, &slots).unwrap();
        let positions: Vec<u32> = (0..n as u32).collect();
        let token_seq = vec![0u32; n];
        let tables = vec![(0..n.div_ceil(bs) as u32).collect::<Vec<_>>()];
        let meta = AttnMeta {
            n_heads,
            n_kv_heads: n_kv,
            head_dim: hd,
            scale: 1.0 / (hd as f32).sqrt(),
            window: 0,
            token_seq: &token_seq,
            positions: &positions,
            block_tables: &tables,
            block_size: bs,
        };
        for _ in 0..3 {
            b.attention(&q, &kv, 0, &meta, &mut out).unwrap();
        }
        b.synchronize().unwrap();
        let iters = 20;
        let t = Instant::now();
        for _ in 0..iters {
            b.attention(&q, &kv, 0, &meta, &mut out).unwrap();
        }
        b.synchronize().unwrap();
        let us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
        // causal: ~n^2/2 query-key pairs, 2 matmuls, 2 flops per MAC
        let flops = 4.0 * (n * n / 2) as f64 * (hd * n_heads) as f64;
        println!("heads {n_heads}/{n_kv} hd {hd} n {n}: {us:9.1} us  {:6.2} TFLOPS", flops / us / 1e6);
    }
}
