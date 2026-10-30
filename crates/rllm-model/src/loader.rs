use std::sync::Arc;

use anyhow::{Context, Result};
use rllm_core::{Backend, HostBytes, HostTensor};
use rllm_gguf::Gguf;

/// View a GGUF tensor as a `HostTensor` without copying its data.
pub fn host_tensor(g: &Gguf, name: &str) -> Result<HostTensor> {
    let info = g.require_tensor(name)?;
    let owner: Arc<dyn AsRef<[u8]> + Send + Sync> = g.mmap().clone();
    Ok(HostTensor {
        name: name.to_string(),
        dtype: info.dtype,
        // GGUF lists the innermost axis first; HostTensor is row-major.
        shape: info.dims.iter().rev().map(|&d| d as usize).collect(),
        bytes: HostBytes::new(owner, info.offset..info.offset + info.size),
    })
}

pub fn load_weight<B: Backend>(backend: &B, g: &Gguf, name: &str) -> Result<B::Weight> {
    let t = host_tensor(g, name)?;
    backend.upload(&t).with_context(|| format!("uploading {name}"))
}

/// Load a Q or K projection stored for ggml's "normal" RoPE (rotating
/// adjacent pairs; Llama's GGUF conversion permutes the rows for it) in
/// NeoX order (rotating dims i and i + head_dim / 2), which is what rllm's
/// RoPE kernels do: row `h * head_dim + i` becomes row `2i` of that head for
/// i < head_dim / 2, else row `2(i - head_dim / 2) + 1`. Attention only sees
/// dot products of rotated Q and K, so the order of their dims is free.
pub fn load_weight_rope_neox<B: Backend>(backend: &B, g: &Gguf, name: &str, head_dim: usize) -> Result<B::Weight> {
    let t = host_tensor(g, name)?;
    let (rows, cols) = t.as_2d();
    anyhow::ensure!(rows.is_multiple_of(head_dim) && head_dim.is_multiple_of(2), "{name}: {rows} rows, head dim {head_dim}");
    let row_bytes = t.dtype.row_bytes(cols);
    let src = t.bytes.as_slice();
    let half = head_dim / 2;
    let mut out = Vec::with_capacity(src.len());
    for r in 0..rows {
        let (h, i) = (r / head_dim, r % head_dim);
        let from = h * head_dim + if i < half { 2 * i } else { 2 * (i - half) + 1 };
        out.extend_from_slice(&src[from * row_bytes..(from + 1) * row_bytes]);
    }
    let len = out.len();
    let owner: Arc<dyn AsRef<[u8]> + Send + Sync> = Arc::new(out);
    let t = HostTensor { bytes: HostBytes::new(owner, 0..len), ..t };
    backend.upload(&t).with_context(|| format!("uploading {name}"))
}
