//! CUDA backend.
//!
//! Kernels live in `kernels/*.cu` as plain CUDA C++ and are compiled at
//! startup with NVRTC for the exact device architecture, so building rllm
//! needs no CUDA toolkit. `rllm-hip` compiles the same sources with hiprtc
//! and the same backend (`backend.rs`) over its own `gpu` module.
//!
//! Quantized weights are repacked on upload into GPU-friendly layouts (see
//! `kernels.cu`). Matmuls with few input rows (decode) quantize the
//! activations to int8 and use dp4a matvec kernels. Larger batches (prefill)
//! dequantize the weight to f16 and call cuBLAS. Prefill attention uses a
//! tiled kernel that shares K/V across queries and GQA heads.

mod backend;
mod gpu;
mod libs;

pub use backend::*;
