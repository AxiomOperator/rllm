//! GPU checks against a real model on HIP (the same tests as tests/gpu.rs):
//!   RLLM_TEST_MODEL=$PWD/models/Qwen3-0.6B-Q8_0.gguf cargo test --release -p rllm-model --test gpu_hip

use rllm_hip::{HipBackend as GpuBackend, device_count};

const VENDOR: &str = "HIP";

include!("common/gpu.rs");
