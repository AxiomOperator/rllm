//! AMD (ROCm) backend.
//!
//! The backend itself is `rllm-cuda`'s (`backend.rs`, compiled here a
//! second time) over this crate's `gpu` module, which implements the same
//! small vendor API with HIP, hiprtc and hipBLAS. The kernels are the same
//! `kernels.cu`, built with hiprtc for the device's gfx target. ROCm's
//! libraries are loaded at runtime, so building needs no ROCm install.
//!
//! The NVIDIA tensor-core kernels (int8 MMQ, tensor-core attention) are
//! PTX-only, so prefill matmuls take the dequantize + hipBLAS path and
//! prefill attention the plain tiled kernel.

#[path = "../../rllm-cuda/src/backend.rs"]
mod backend;
mod ffi;
mod gpu;

pub use backend::{
    CudaBackend as HipBackend, CudaBuf as HipBuf, CudaHostKv as HipHostKv, CudaKvCache as HipKvCache,
    CudaWeight as HipWeight, DeviceInfo, device_count, probe_devices,
};
