//! Types and bytes of a checkpoint quantized at load against a reference
//! GGUF (debug aid): `q_cmp DIR GGUF q4_k_m`.
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let q = rllm_safetensors::parse_quantize(&a[3]).unwrap();
    let ours = rllm_safetensors::open_model_with(&a[1], rllm_safetensors::Options { quantize: Some(q) }).unwrap();
    let theirs = rllm_gguf::Gguf::open(&a[2]).unwrap();
    let (mut same_type, mut same_bytes, mut n, mut mism) = (0, 0, 0, Vec::new());
    for t in &theirs.tensors {
        let Some(o) = ours.tensor(&t.name) else { continue };
        n += 1;
        if o.dtype == t.dtype {
            same_type += 1;
            if ours.tensor_data(o) == theirs.tensor_data(t) {
                same_bytes += 1;
            } else if mism.len() < 4 {
                let (x, y) = (ours.tensor_data(o), theirs.tensor_data(t));
                let diff = x.iter().zip(y).filter(|(p, q)| p != q).count();
                mism.push(format!("{} {:?}: {diff} of {} bytes differ", t.name, t.dtype, x.len()));
            }
        } else if mism.len() < 12 {
            mism.push(format!("{}: ours {:?}, theirs {:?}", t.name, o.dtype, t.dtype));
        }
    }
    println!("{n} tensors: {same_type} of the same type, {same_bytes} byte-identical");
    for m in mism {
        println!("  {m}");
    }
}
