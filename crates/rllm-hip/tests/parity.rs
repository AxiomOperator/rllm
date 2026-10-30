//! Every HIP op against the CPU reference backend on random inputs (the
//! same tests as rllm-cuda's).

use rllm_hip::HipBackend as GpuBackend;

const VENDOR: &str = "HIP";

include!("../../rllm-cuda/tests/common/parity.rs");
