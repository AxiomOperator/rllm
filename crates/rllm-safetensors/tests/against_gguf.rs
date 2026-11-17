//! A checkpoint seen as GGUF against llama.cpp's conversion of the same
//! checkpoint: every tensor (dims, type, bytes) and every metadata key rllm
//! reads. Set RLLM_ST_CHECKPOINT (the HF directory) and RLLM_ST_GGUF (a BF16
//! or F16 GGUF converted from it; norms are F32 in both):
//!
//! `RLLM_ST_CHECKPOINT=models/hf/Qwen3-0.6B RLLM_ST_GGUF=models/Qwen3-0.6B-BF16.gguf cargo test -p rllm-safetensors --test against_gguf`

use rllm_gguf::{Gguf, TensorInfo, Value};

/// A tensor's values as f32.
fn values(g: &Gguf, t: &TensorInfo) -> Vec<f32> {
    let row = t.dims[0] as usize;
    let rows = (t.dims.iter().product::<u64>() as usize) / row.max(1);
    let data = g.tensor_data(t);
    let rb = data.len() / rows.max(1);
    let mut out = vec![0f32; rows * row];
    for (r, o) in out.chunks_exact_mut(row).enumerate() {
        rllm_core::quant::dequantize_row(t.dtype, &data[r * rb..(r + 1) * rb], o);
    }
    out
}

/// Values against a reference's: exact for a float reference, else within
/// its quantization (relative to the largest value).
fn values_match(a: &[f32], b: &[f32], reference: rllm_core::DType) -> Result<(), f32> {
    let scale = a.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
    let err = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max) / scale;
    let tol = match reference {
        rllm_core::DType::F32 | rllm_core::DType::F16 | rllm_core::DType::BF16 => 0.0,
        rllm_core::DType::Q8_0 => 1.0 / 100.0,
        _ => 0.2,
    };
    if a.len() == b.len() && err <= tol { Ok(()) } else { Err(err) }
}

/// The halves of a fused `ffn_gate_up_exps` ([d, 2f, E]): gate rows then up
/// rows per expert, as llama.cpp's split.
fn split_gate_up(v: &[f32], d: usize, f: usize, e: usize) -> (Vec<f32>, Vec<f32>) {
    let (mut gate, mut up) = (Vec::with_capacity(d * f * e), Vec::with_capacity(d * f * e));
    for x in 0..e {
        let base = x * 2 * f * d;
        gate.extend_from_slice(&v[base..base + f * d]);
        up.extend_from_slice(&v[base + f * d..base + 2 * f * d]);
    }
    (gate, up)
}

#[test]
fn matches_llama_cpp_conversion() {
    let (Ok(ckpt), Ok(gguf)) = (std::env::var("RLLM_ST_CHECKPOINT"), std::env::var("RLLM_ST_GGUF")) else {
        eprintln!("RLLM_ST_CHECKPOINT / RLLM_ST_GGUF not set: skipped");
        return;
    };
    let ours = rllm_safetensors::open_model(&ckpt).expect("convert");
    let theirs = Gguf::open(&gguf).expect("open GGUF");
    // Tensors.
    let mut differ = Vec::new();
    for t in &theirs.tensors {
        let Some(o) = ours.tensor(&t.name) else {
            // Kept fused (llama.cpp's --fuse-gate-up-exps layout): compare
            // the matching half.
            let fused = t.name.replace("ffn_gate_exps", "ffn_gate_up_exps").replace("ffn_up_exps", "ffn_gate_up_exps");
            match ours.tensor(&fused).filter(|_| fused != t.name) {
                Some(o) if o.dims.len() == 3 && t.dims.len() == 3 && o.dims[0] == t.dims[0] && o.dims[1] == 2 * t.dims[1] && o.dims[2] == t.dims[2] => {
                    let (gate, up) = split_gate_up(&values(&ours, o), t.dims[0] as usize, t.dims[1] as usize, t.dims[2] as usize);
                    let half = if t.name.contains("ffn_gate_exps") { gate } else { up };
                    if let Err(err) = values_match(&half, &values(&theirs, t), t.dtype) {
                        differ.push(format!("{}: fused half vs {:?}, max error {err} of the largest value", t.name, t.dtype));
                    }
                }
                _ => differ.push(format!("{}: missing", t.name)),
            }
            continue;
        };
        if o.dims != t.dims {
            differ.push(format!("{}: dims {:?} vs {:?}", t.name, o.dims, t.dims));
        } else if o.dtype == t.dtype {
            let (x, y) = (ours.tensor_data(o), theirs.tensor_data(t));
            // F32 values computed at conversion (e.g. -exp(A_log)) may round
            // a last bit differently from PyTorch: 2 ulp.
            let close = t.dtype == rllm_core::DType::F32
                && x.len() == y.len()
                && x.as_chunks::<4>().0.iter().zip(y.as_chunks::<4>().0).all(|(p, q)| (i32::from_le_bytes(*p) - i32::from_le_bytes(*q)).abs() <= 2);
            // Quantized both ways: quantizers may choose scales differently
            // (llama.cpp's reference vs a tuned one): the values, within two
            // quantization steps of the block's largest value.
            let quantized_close = x != y
                && match t.dtype {
                    rllm_core::DType::Q8_0 => {
                        let (a, b) = (values(&ours, o), values(&theirs, t));
                        a.chunks(32).zip(b.chunks(32)).all(|(p, q)| {
                            let step = p.iter().fold(0f32, |m, v| m.max(v.abs())) / 127.0;
                            p.iter().zip(q).all(|(u, v)| (u - v).abs() <= 2.0 * step + 1e-7)
                        })
                    }
                    rllm_core::DType::F32 | rllm_core::DType::F16 | rllm_core::DType::BF16 => false,
                    d => values_match(&values(&ours, o), &values(&theirs, t), d).is_ok(),
                };
            if x != y && !close && !quantized_close {
                differ.push(format!("{}: bytes differ", t.name));
            }
        } else {
            // A quantized reference (e.g. Q8_0): the values, within its rounding.
            if let Err(err) = values_match(&values(&ours, o), &values(&theirs, t), t.dtype) {
                differ.push(format!("{}: {:?} vs {:?}, max error {err} of the largest value", t.name, o.dtype, t.dtype));
            }
        }
    }
    for o in &ours.tensors {
        let split = o.name.contains("ffn_gate_up_exps") && theirs.tensor(&o.name.replace("ffn_gate_up_exps", "ffn_gate_exps")).is_some();
        if theirs.tensor(&o.name).is_none() && !split {
            differ.push(format!("{}: extra", o.name));
        }
    }
    // Metadata rllm reads.
    let arch = theirs.get_str("general.architecture").unwrap().to_string();
    let read = |k: &str| {
        k.starts_with(&format!("{arch}."))
            || matches!(
                k,
                "general.architecture"
                    | "tokenizer.ggml.model"
                    | "tokenizer.ggml.pre"
                    | "tokenizer.ggml.tokens"
                    | "tokenizer.ggml.token_type"
                    | "tokenizer.ggml.merges"
                    | "tokenizer.ggml.bos_token_id"
                    | "tokenizer.ggml.eos_token_id"
                    | "tokenizer.ggml.add_bos_token"
                    | "tokenizer.ggml.scores"
                    | "tokenizer.ggml.add_space_prefix"
                    // Generated by the converter (not the checkpoint's own).
                    | "tokenizer.chat_template.systemone"
            )
    };
    // (tokenizer.chat_template is the checkpoint's own: published GGUFs
    // often ship a rewritten one, e.g. Qwen3's without `messages[::-1]`.)
    // Integer widths and float types may differ (U32 vs I32, F32 vs F64).
    let same = |a: &Value, b: &Value| match (a.as_u64(), b.as_u64(), a.as_f64(), b.as_f64()) {
        (Some(x), Some(y), _, _) => x == y,
        (_, _, Some(x), Some(y)) => (x - y).abs() <= 1e-6 * x.abs().max(1.0),
        _ => match (a.as_array(), b.as_array()) {
            (Some(x), Some(y)) => x.len() == y.len() && x.iter().zip(y).all(|(p, q)| p.as_str() == q.as_str() && p.as_u64() == q.as_u64() && p.as_f64() == q.as_f64()),
            _ => a.as_str() == b.as_str() && a.as_str().is_some() || format!("{a:?}") == format!("{b:?}"),
        },
    };
    for (k, v) in theirs.metadata.iter().filter(|(k, _)| read(k)) {
        match ours.get(k) {
            None => differ.push(format!("metadata {k}: missing")),
            Some(o) if !same(o, v) => differ.push(format!("metadata {k}: {} vs {}", o.summary(), v.summary())),
            _ => {}
        }
    }
    assert!(differ.is_empty(), "{} differences:\n{}", differ.len(), differ.join("\n"));
}
