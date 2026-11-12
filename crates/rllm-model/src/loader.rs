use std::sync::Arc;

use anyhow::{Context, Result, bail, ensure};
use rllm_core::{Backend, HostBytes, HostTensor};
use rllm_gguf::Gguf;

/// Largest model dimension (hidden size, heads, vocabulary, ...) accepted
/// from metadata: far beyond real models, small enough that products of a
/// few of them can't overflow.
pub(crate) const MAX_DIM: usize = 1 << 24;
/// Largest context accepted (positions are u32).
pub(crate) const MAX_CONTEXT: usize = 1 << 30;

/// Metadata integer `key` as a usize in `1..=max` (0 allowed if `zero_ok`).
pub(crate) fn meta_usize(g: &Gguf, key: &str, max: usize, zero_ok: bool) -> Result<usize> {
    let v = g.get_u64(key)?;
    bounded(key, v, max, zero_ok)
}

/// Optional metadata integer: `default` when absent, an error when present but invalid.
pub(crate) fn meta_usize_or(g: &Gguf, key: &str, default: usize, max: usize, zero_ok: bool) -> Result<usize> {
    match g.get(key) {
        None => Ok(default),
        Some(v) => bounded(key, v.as_u64().with_context(|| format!("{key}: expected a non-negative integer"))?, max, zero_ok),
    }
}

fn bounded(key: &str, v: u64, max: usize, zero_ok: bool) -> Result<usize> {
    let n = usize::try_from(v).ok().filter(|&n| n <= max).with_context(|| format!("{key} = {v} is out of range (at most {max})"))?;
    ensure!(zero_ok || n > 0, "{key} must be positive");
    Ok(n)
}

/// Optional metadata float: `default` when absent; must be finite and positive.
pub(crate) fn meta_f32_or(g: &Gguf, key: &str, default: f32) -> Result<f32> {
    let v = match g.get(key) {
        None => return Ok(default),
        Some(v) => v.as_f64().with_context(|| format!("{key}: expected a number"))? as f32,
    };
    ensure!(v.is_finite() && v > 0.0, "{key} = {v} must be a positive number");
    Ok(v)
}

/// Check that tensor `name` exists with GGUF dims `dims` (innermost first).
pub(crate) fn check_shape(g: &Gguf, name: &str, dims: &[usize]) -> Result<()> {
    let t = g.require_tensor(name)?;
    if t.dims.len() != dims.len() || t.dims.iter().zip(dims).any(|(&a, &b)| a != b as u64) {
        bail!("tensor {name} has shape {:?}, expected {dims:?}", t.dims);
    }
    Ok(())
}

/// [`check_shape`] for a tensor that may be absent.
pub(crate) fn check_shape_opt(g: &Gguf, name: &str, dims: &[usize]) -> Result<()> {
    if g.tensor(name).is_some() { check_shape(g, name, dims) } else { Ok(()) }
}

/// An optional head-size key (`rope.dimension_count`, `attention.value_length`)
/// must, when present, equal `head_dim`: partial rotary embeddings and
/// value heads of a different width than keys aren't implemented.
pub(crate) fn check_same_as_head_dim(g: &Gguf, key: &str, head_dim: usize, what: &str) -> Result<()> {
    if let Some(v) = g.get(key) {
        let n = v.as_u64().with_context(|| format!("{key}: expected an integer"))?;
        ensure!(n == head_dim as u64, "{key} = {n} differs from the head size {head_dim}: {what} is not supported");
    }
    Ok(())
}

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
/// RoPE kernels do: output row `h * head_dim + i` is taken from stored row
/// `h * head_dim + 2i` for i < head_dim / 2, else from stored row
/// `h * head_dim + 2(i - head_dim / 2) + 1` (each adjacent pair (2j, 2j+1)
/// becomes the pair (j, j + head_dim / 2)). Attention only sees dot products
/// of rotated Q and K, so the order of their dims is free.
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

/// Load rows `rows` of 2-D tensor `name`, in that order (any row of a
/// quantized matrix is whole blocks, so rows can be picked freely).
pub fn load_weight_rows<B: Backend>(backend: &B, g: &Gguf, name: &str, rows: &[usize]) -> Result<B::Weight> {
    let t = host_tensor(g, name)?;
    let (n_rows, cols) = t.as_2d();
    ensure!(t.shape.len() == 2, "{name}: expected a matrix, shape {:?}", t.shape);
    let row_bytes = t.dtype.row_bytes(cols);
    ensure!(n_rows * row_bytes == t.bytes.as_slice().len(), "{name}: {} bytes for {n_rows} rows", t.bytes.as_slice().len());
    let src = t.bytes.as_slice();
    let mut out = Vec::with_capacity(rows.len() * row_bytes);
    for &r in rows {
        ensure!(r < n_rows, "{name}: row {r} of {n_rows}");
        out.extend_from_slice(&src[r * row_bytes..(r + 1) * row_bytes]);
    }
    let len = out.len();
    let owner: Arc<dyn AsRef<[u8]> + Send + Sync> = Arc::new(out);
    let t = HostTensor { shape: vec![rows.len(), cols], bytes: HostBytes::new(owner, 0..len), ..t };
    backend.upload(&t).with_context(|| format!("uploading {name}"))
}

/// A 1-D tensor's values as f32.
pub fn read_vector(g: &Gguf, name: &str) -> Result<Vec<f32>> {
    let t = host_tensor(g, name)?;
    ensure!(t.shape.len() == 1, "{name}: expected a vector, shape {:?}", t.shape);
    let mut v = vec![0f32; t.shape[0]];
    ensure!(t.dtype.row_bytes(v.len()) == t.bytes.as_slice().len(), "{name}: size mismatch");
    rllm_core::quant::dequantize_row(t.dtype, t.bytes.as_slice(), &mut v);
    Ok(v)
}
