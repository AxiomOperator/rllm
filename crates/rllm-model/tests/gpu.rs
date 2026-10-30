//! GPU checks against a real model on CUDA:
//!   RLLM_TEST_MODEL=$PWD/models/Qwen3-0.6B-Q8_0.gguf cargo test --release -p rllm-model --test gpu

use rllm_cuda::{CudaBackend as GpuBackend, device_count};

const VENDOR: &str = "CUDA";

include!("common/gpu.rs");
