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
