//! The vendor API under the shared backend (`backend.rs`), NVIDIA side:
//! cudarc's driver, NVRTC and cuBLAS. `rllm-hip` has a `gpu` module with the
//! same names over HIP, hiprtc and hipBLAS, so the backend compiles against
//! either. Types keep cudarc's method names; vendor calls the backend makes
//! directly are the free functions below.

use std::sync::Arc;

pub use cudarc::driver::{
    CudaContext as Context, CudaEvent as Event, CudaFunction as Function, CudaGraph as GraphExec,
    CudaModule as Module, CudaSlice as Slice, CudaStream as Stream, DevicePtr, DeviceRepr, DriverError as Error,
    LaunchConfig, PinnedHostSlice as PinnedSlice, PushKernelArg,
};
use cudarc::cublas::{CudaBlas, result as blas, sys as blas_sys};
use cudarc::driver::{result, sys};

/// Lower-case vendor name for messages (`cuda (<device>, <arch>)`).
pub const VENDOR: &str = "cuda";

pub fn preload() {
    crate::libs::preload();
}

/// Whether the driver library loads (cudarc panics on first use otherwise,
/// e.g. on an AMD machine).
fn driver_present() -> bool {
    use std::sync::LazyLock;
    // SAFETY: only tries to dlopen libcuda.
    static PRESENT: LazyLock<bool> = LazyLock::new(|| unsafe { sys::is_culib_present() });
    *PRESENT
}

pub fn device_count() -> usize {
    if !driver_present() {
        return 0;
    }
    Context::device_count().unwrap_or(0).max(0) as usize
}

pub fn new_context(ordinal: usize) -> Result<Arc<Context>, Error> {
    if !driver_present() {
        return Err(cudarc::driver::DriverError(sys::CUresult::CUDA_ERROR_NO_DEVICE));
    }
    let ctx = Context::new(ordinal)?;
    // SAFETY: all work goes through the context's single stream, so stream
    // order already serializes every use; per-buffer event tracking would
    // only add two event calls per kernel argument.
    unsafe { ctx.disable_event_tracking() };
    Ok(ctx)
}

/// The architecture a device reports and how kernels are built for it.
pub struct Arch {
    /// What the kernel compiler takes (`compute_89`).
    pub compile: String,
    /// What rllm reports (`sm_89`).
    pub name: String,
    /// sm_80+: the mma.sync kernels (MMQ, tensor-core attention) can run.
    pub tensor_cores: bool,
    /// RDNA3 WMMA kernels (AMD only).
    pub wmma: bool,
}

pub fn arch(ctx: &Context) -> Result<Arch, Error> {
    let (major, minor) = ctx.compute_capability()?;
    Ok(Arch { compile: format!("compute_{major}{minor}"), name: format!("sm_{major}{minor}"), tensor_cores: major >= 8, wmma: false })
}

/// Compile the kernel source for `arch` (an [`Arch::compile`] string): PTX text.
pub fn compile(src: &str, arch: &str) -> Result<Vec<u8>, String> {
    let opts = cudarc::nvrtc::CompileOptions {
        arch: Some(Box::leak(arch.to_string().into_boxed_str())),
        options: vec!["--std=c++17".into()],
        ..Default::default()
    };
    cudarc::nvrtc::compile_ptx_with_opts(src, opts).map(|p| p.to_src().into_bytes()).map_err(|e| e.to_string())
}

/// File extension for cached [`compile`] output.
pub const CODE_EXT: &str = "ptx";

/// Targets the kernel source must compile for (tested): Ada (the benchmark box) and Ampere.
/// Whether NVRTC can be used (always assumed: the CUDA tests require it).
#[cfg(test)]
pub fn compiler_present() -> bool {
    true
}

#[cfg(test)]
pub const TEST_ARCHES: [&str; 2] = ["compute_89", "compute_86"];

pub fn load_module(ctx: &Arc<Context>, code: Vec<u8>) -> Result<Arc<Module>, Error> {
    ctx.load_module(cudarc::nvrtc::Ptx::from_src(String::from_utf8_lossy(&code)))
}

pub fn sm_count(ctx: &Context) -> Result<usize, Error> {
    Ok(ctx.attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)? as usize)
}

/// Whether the GPU shares system RAM with the CPU.
pub fn integrated(ctx: &Context) -> Result<bool, Error> {
    Ok(ctx.attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_INTEGRATED)? != 0)
}

/// Largest dynamic shared memory a block may opt into.
pub fn max_smem_optin(ctx: &Context) -> Result<usize, Error> {
    Ok(ctx.attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)? as usize)
}

/// Let `f` use up to `bytes` of dynamic shared memory (past the 48 KB default).
pub fn set_max_dynamic_smem(f: &Function, bytes: usize) -> Result<(), Error> {
    // SAFETY: a valid function handle and attribute.
    unsafe {
        result::function::set_function_attribute(
            f.cu_function(),
            sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            bytes as i32,
        )
    }
}

/// (registers per thread, local memory bytes per thread).
pub fn func_resources(f: &Function) -> Result<(i32, i32), Error> {
    use sys::CUfunction_attribute::{CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, CU_FUNC_ATTRIBUTE_NUM_REGS};
    // SAFETY: a valid function handle and attributes.
    unsafe {
        Ok((
            result::function::get_function_attribute(f.cu_function(), CU_FUNC_ATTRIBUTE_NUM_REGS)?,
            result::function::get_function_attribute(f.cu_function(), CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES)?,
        ))
    }
}

pub fn occupancy(f: &Function, block: u32, smem: usize) -> Result<u32, Error> {
    f.occupancy_max_active_blocks_per_multiprocessor(block, smem, None)
}

pub fn new_event(ctx: &Arc<Context>, timing: bool) -> Result<Event, Error> {
    ctx.new_event(timing.then_some(sys::CUevent_flags::CU_EVENT_DEFAULT))
}

/// Zero `bytes` bytes at `ptr`, in stream order.
///
/// # Safety
/// `ptr..ptr + bytes` must be device memory this process owns.
pub unsafe fn memset_zero_async(ptr: u64, bytes: usize, stream: &Stream) -> Result<(), Error> {
    unsafe { result::memset_d8_async(ptr, 0, bytes, stream.cu_stream()) }
}

/// Copy `out.len()` floats from `ptr`, in stream order (synchronize before reading `out`).
///
/// # Safety
/// `ptr` must hold at least `out.len()` floats.
pub unsafe fn memcpy_dtoh_async(out: &mut [f32], ptr: u64, stream: &Stream) -> Result<(), Error> {
    unsafe { result::memcpy_dtoh_async(out, ptr, stream.cu_stream()) }
}

/// Keep memory freed by stream-ordered allocations in the device's pool
/// instead of returning it at each synchronization.
pub fn hold_freed_memory(ctx: &Context) -> Result<(), Error> {
    ctx.bind_to_thread()?;
    let mut pool: sys::CUmemoryPool = std::ptr::null_mut();
    let mut threshold = u64::MAX;
    // SAFETY: plain driver calls on this context's device, with valid out-pointers.
    unsafe {
        sys::cuDeviceGetDefaultMemPool(&mut pool, ctx.cu_device()).result()?;
        sys::cuMemPoolSetAttribute(pool, sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, (&mut threshold as *mut u64).cast())
            .result()
    }
}

/// Pinned host memory mapped into the device's address space: the buffer
/// and its device address.
pub fn alloc_mapped_host(ctx: &Arc<Context>, bytes: usize) -> Result<(PinnedSlice<u8>, u64), Error> {
    let flags = sys::CU_MEMHOSTALLOC_DEVICEMAP | sys::CU_MEMHOSTALLOC_PORTABLE;
    // SAFETY: the memory is only read back after kernels wrote it.
    let mut buf = unsafe { ctx.alloc_pinned_with_flags::<u8>(bytes, flags) }?;
    let host = buf.as_mut_ptr()?;
    let mut dev: sys::CUdeviceptr = 0;
    // SAFETY: `host` is a live allocation made with DEVICEMAP.
    unsafe { sys::cuMemHostGetDevicePointer_v2(&mut dev, host.cast(), 0) }.result()?;
    Ok((buf, dev))
}

pub fn begin_capture(stream: &Stream) -> Result<(), Error> {
    stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
}

pub fn end_capture(stream: &Arc<Stream>) -> Result<Option<GraphExec>, Error> {
    stream.end_capture(sys::CUgraphInstantiate_flags(0))
}

/// Element type of GEMM inputs.
/// GEMMs write f32 directly.
pub const GEMM_HALF_OUT: bool = false;

/// BF16 weights go to cuBLAS as stored (RLLM_BF16_VIA_F16 converts them).
pub const BF16_VIA_F16: bool = false;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum GemmIn {
    F16,
    BF16,
}

/// cuBLAS on the backend's stream, with a workspace.
pub struct Blas {
    blas: CudaBlas,
    _workspace: Slice<u8>,
}

impl Blas {
    pub fn new(stream: &Arc<Stream>) -> Result<Self, String> {
        let blas = CudaBlas::new(stream.clone()).map_err(|e| format!("{e:?}"))?;
        // Without a workspace cuBLAS can't use split-K style algorithms and
        // picks very slow kernels for some shapes (measured 10x on 512x2560).
        const WORKSPACE: usize = 32 << 20;
        let workspace = stream.alloc_zeros::<u8>(WORKSPACE).map_err(|e| format!("{e:?}"))?;
        let (ptr, _) = workspace.device_ptr(stream);
        // SAFETY: the workspace outlives the handle (both live in `Self`).
        let status = unsafe { blas_sys::cublasSetWorkspace_v2(*blas.handle(), ptr as *mut _, WORKSPACE) };
        if status != blas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(format!("cublasSetWorkspace: {status:?}"));
        }
        Ok(Self { blas, _workspace: workspace })
    }

    /// Column-major `c[n, m] = alpha · aᵀ · b` with `a` [n x k] stored as k-major rows
    /// (op T, lda k), `b` [k x m] (ldb k) and f32 `c` (ldc n); f32 accumulate.
    ///
    /// # Safety
    /// The pointers must hold matrices of those shapes.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_tn(&self, n: usize, m: usize, k: usize, a: u64, b: u64, input: GemmIn, alpha: f32, c: u64) -> Result<(), String> {
        let ty = match input {
            GemmIn::F16 => blas_sys::cudaDataType::CUDA_R_16F,
            GemmIn::BF16 => blas_sys::cudaDataType::CUDA_R_16BF,
        };
        let beta = 0f32;
        unsafe {
            blas::gemm_ex(
                *self.blas.handle(),
                blas_sys::cublasOperation_t::CUBLAS_OP_T,
                blas_sys::cublasOperation_t::CUBLAS_OP_N,
                n as i32,
                m as i32,
                k as i32,
                &alpha as *const f32 as *const _,
                a as *const _,
                ty,
                k as i32,
                b as *const _,
                ty,
                k as i32,
                &beta as *const f32 as *const _,
                c as *mut _,
                blas_sys::cudaDataType::CUDA_R_32F,
                n as i32,
                blas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                blas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )
        }
        .map_err(|e| format!("{e:?}"))
    }
}
