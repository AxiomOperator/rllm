//! Prints register use and occupancy for a few hot kernels.
use rllm_cuda::CudaBackend;
fn main() {
    let b = CudaBackend::new(0).expect("cuda");
    println!("SMs: {}", b.sm_count());
    for (name, block, smem) in [("mmq_q8", 128, 0), ("mmq_glu_gelu_q8", 128, 0), ("mmq_q4", 128, 0), ("matvec_q8", 128, 0), ("attn_fa_d128", 128, 36864), ("attn_fa_d256", 128, 34000), ("attn_fa_d512", 256, 70000), ("attn_decode_g2_d128", 256, 0)] {
        match b.kernel_info(name, block, smem) {
            Ok((regs, blocks, local)) => println!("{name:<22} regs/thread {regs:>3}  local {local:>4} B  blocks/SM {blocks} (block {block}, smem {smem})"),
            Err(e) => println!("{name}: {e}"),
        }
    }
}
