//! Where a converted checkpoint differs from a GGUF (debug aid):
//! `cargo run --release -p rllm-safetensors --example st_diff -- DIR GGUF`
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let ours = rllm_safetensors::open_model(&a[1]).unwrap();
    let theirs = rllm_gguf::Gguf::open(&a[2]).unwrap();
    for n in ["blk.8.ssm_a", "blk.12.ssm_a"] {
        let (o, t) = (ours.tensor(n).unwrap(), theirs.tensor(n).unwrap());
        let f = |b: &[u8]| b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect::<Vec<_>>();
        let (x, y) = (f(ours.tensor_data(o)), f(theirs.tensor_data(t)));
        let d: Vec<_> = x.iter().zip(&y).enumerate().filter(|(_, (p, q))| p != q).map(|(i, (p, q))| (i, *p, *q, (p.to_bits() as i64 - q.to_bits() as i64))).collect();
        println!("{n}: {} differ: {:?}", d.len(), &d[..d.len().min(4)]);
    }
    for key in ["tokenizer.ggml.tokens", "tokenizer.ggml.token_type"] {
        let (x, y) = (ours.get_array(key).unwrap(), theirs.get_array(key).unwrap());
        let d: Vec<_> = x.iter().zip(y).enumerate().filter(|(_, (p, q))| format!("{p:?}") != format!("{q:?}")).take(8).map(|(i, (p, q))| format!("{i}: {p:?} vs {q:?}")).collect();
        println!("{key}: {}", d.join(" | "));
    }
}
