//! How a Q8_0 tensor differs from a reference (debug aid):
//! `cargo run -p rllm-safetensors --example q8_diff -- DIR GGUF TENSOR`
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let ours = rllm_safetensors::open_model_with(&a[1], rllm_safetensors::Options { quantize: Some(rllm_core::DType::Q8_0) }).unwrap();
    let theirs = rllm_gguf::Gguf::open(&a[2]).unwrap();
    let n = &a[3];
    let (o, t) = (ours.tensor_data(ours.tensor(n).unwrap()), theirs.tensor_data(theirs.tensor(n).unwrap()));
    let deq = |b: &[u8]| { let mut y = vec![0f32; 32]; rllm_core::quant::dequantize_row(rllm_core::DType::Q8_0, b, &mut y); y };
    let (mut worst, mut hist) = (0f32, [0usize; 6]);
    for b in 0..o.len() / 34 {
        let (x, y) = (deq(&o[b * 34..b * 34 + 34]), deq(&t[b * 34..b * 34 + 34]));
        let step = x.iter().fold(0f32, |m, v| m.max(v.abs())) / 127.0;
        for (p, q) in x.iter().zip(&y) {
            let s = (p - q).abs() / step.max(1e-12);
            worst = worst.max(s);
            hist[(s.round() as usize).min(5)] += 1;
        }
    }
    println!("{n}: worst {worst:.2} steps; |diff| in steps 0,1,2,3,4,5+: {hist:?}");
}
