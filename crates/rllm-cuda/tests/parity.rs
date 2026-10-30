//! Every CUDA op against the CPU reference backend on random inputs.

use rllm_cuda::CudaBackend as GpuBackend;

const VENDOR: &str = "CUDA";

include!("common/parity.rs");
