//! Prints register use and occupancy for a few hot kernels (see rllm-cuda's
//! example of the same name). Kernel names as extra arguments: `name:block:smem`.
use rllm_hip::HipBackend;
fn main() {
    let b = HipBackend::new(0).expect("hip");
    println!("multiprocessors: {}", b.sm_count());
    let mut list: Vec<(String, u32, usize)> = std::env::args()
        .skip(1)
        .filter_map(|a| {
            let mut p = a.split(':');
            Some((p.next()?.to_string(), p.next().unwrap_or("128").parse().ok()?, p.next().unwrap_or("0").parse().ok()?))
        })
        .collect();
    if list.is_empty() {
        list = [("matvec_q4", 128, 0), ("matvec_q6", 128, 0), ("matvec_glu_silu_q4", 1024, 0), ("mmq_q4", 128, 0), ("attn_wmma_d128", 128, 40000)]
            .map(|(n, bl, s)| (n.to_string(), bl, s))
            .to_vec();
    }
    for (name, block, smem) in list {
        match b.kernel_info(&name, block, smem) {
            Ok((regs, blocks, local)) => println!("{name:<22} regs/thread {regs:>3}  local {local:>4} B  blocks/MP {blocks} (block {block}, smem {smem})"),
            Err(e) => println!("{name}: {e}"),
        }
    }
}
