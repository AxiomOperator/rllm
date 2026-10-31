//! The vendor API under the shared backend, AMD side: HIP, hiprtc and
//! hipBLAS through [`crate::ffi`]. Mirrors `rllm-cuda/src/gpu.rs` (names and
//! method signatures follow cudarc, which that module re-exports).

use std::ffi::{CStr, CString, c_int, c_void};
use std::marker::PhantomData;
use std::ops::Range;
use std::sync::Arc;

use crate::ffi::{self, HIP};

pub const VENDOR: &str = "hip";
pub const CODE_EXT: &str = "hsaco";

/// Targets the kernel source must compile for (tested): Strix Halo and RDNA3 discrete cards.
/// Whether hiprtc loads (it doesn't on machines without ROCm).
#[cfg(test)]
pub fn compiler_present() -> bool {
    ffi::RTC.is_ok()
}

#[cfg(test)]
pub const TEST_ARCHES: [&str; 2] = ["gfx1151", "gfx1100"];

// Enum values from ROCm 7's headers.
const ATTR_INTEGRATED: c_int = 16;
const ATTR_MULTIPROCESSOR_COUNT: c_int = 63;
const ATTR_SHARED_MEM_PER_BLOCK_OPTIN: c_int = 75;
const FUNC_ATTR_LOCAL_SIZE_BYTES: c_int = 3;
const FUNC_ATTR_NUM_REGS: c_int = 4;
const STREAM_NON_BLOCKING: u32 = 1;
const CAPTURE_MODE_RELAXED: c_int = 2;
const EVENT_DISABLE_TIMING: u32 = 2;
const HOST_MALLOC_PORTABLE: u32 = 1;
const HOST_MALLOC_MAPPED: u32 = 2;
const MEMPOOL_ATTR_RELEASE_THRESHOLD: c_int = 4;
const HIP_R_32F: c_int = 0;
const HIP_R_16F: c_int = 2;
const HIP_R_16BF: c_int = 14;
const HIPBLAS_OP_N: c_int = 111;
const HIPBLAS_OP_T: c_int = 112;
const HIPBLAS_COMPUTE_32F: c_int = 2;
const HIPBLAS_GEMM_DEFAULT: c_int = 160;
/// `hipDeviceProp_tR0600`: size and offset of `gcnArchName` (char[256]).
const PROPS_BYTES: usize = 1472;
const PROPS_ARCH_OFFSET: usize = 1160;

/// A failed HIP call: the function and its error name.
pub struct Error(String);

impl std::fmt::Debug for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

fn hip() -> Result<&'static ffi::Hip, Error> {
    HIP.as_ref().map_err(|e| Error(e.clone()))
}

/// Turn a HIP status into a Result, naming the call.
fn check(what: &str, code: ffi::HipError) -> Result<(), Error> {
    if code == 0 {
        return Ok(());
    }
    let name = hip()
        .ok()
        // SAFETY: hipGetErrorName returns a static string for any code.
        .map(|h| unsafe { CStr::from_ptr((h.hipGetErrorName)(code)) }.to_string_lossy().into_owned())
        .unwrap_or_default();
    Err(Error(format!("{what}: {name} ({code})")))
}

macro_rules! call {
    ($f:ident($($arg:expr),* $(,)?)) => {{
        let h = hip()?;
        check(stringify!($f), unsafe { (h.$f)($($arg),*) })
    }};
}

pub fn preload() {}

pub fn device_count() -> usize {
    let count = || -> Result<c_int, Error> {
        let mut n = 0;
        call!(hipInit(0))?;
        call!(hipGetDeviceCount(&mut n))?;
        Ok(n)
    };
    count().unwrap_or(0).max(0) as usize
}

/// One HIP device. HIP has no explicit contexts: calls that aren't tied to a
/// stream act on the thread's current device, so those select it first.
pub struct Context {
    ordinal: c_int,
}

pub fn new_context(ordinal: usize) -> Result<Arc<Context>, Error> {
    call!(hipInit(0))?;
    let ctx = Context { ordinal: ordinal as c_int };
    ctx.bind()?;
    Ok(Arc::new(ctx))
}

impl Context {
    fn bind(&self) -> Result<(), Error> {
        call!(hipSetDevice(self.ordinal))
    }

    fn attribute(&self, attr: c_int) -> Result<i32, Error> {
        let mut v = 0;
        call!(hipDeviceGetAttribute(&mut v, attr, self.ordinal))?;
        Ok(v)
    }

    pub fn name(&self) -> Result<String, Error> {
        let mut buf = [0 as std::ffi::c_char; 256];
        call!(hipDeviceGetName(buf.as_mut_ptr(), buf.len() as c_int, self.ordinal))?;
        // SAFETY: hipDeviceGetName writes a NUL-terminated string into `buf`.
        Ok(unsafe { CStr::from_ptr(buf.as_ptr()) }.to_string_lossy().into_owned())
    }

    /// (free, total) bytes.
    pub fn mem_get_info(&self) -> Result<(usize, usize), Error> {
        self.bind()?;
        let (mut free, mut total) = (0, 0);
        call!(hipMemGetInfo(&mut free, &mut total))?;
        Ok((free, total))
    }

    pub fn new_stream(self: &Arc<Self>) -> Result<Arc<Stream>, Error> {
        self.bind()?;
        let mut s = std::ptr::null_mut();
        call!(hipStreamCreateWithFlags(&mut s, STREAM_NON_BLOCKING))?;
        Ok(Arc::new(Stream { raw: s, ctx: self.clone() }))
    }

    /// Page-locked host memory for `n` values (contents uninitialized).
    ///
    /// # Safety
    /// The contents must be written before they are read.
    pub unsafe fn alloc_pinned<T: Copy>(&self, n: usize) -> Result<PinnedSlice<T>, Error> {
        self.alloc_host(n, 0)
    }

    fn alloc_host<T: Copy>(&self, n: usize, flags: u32) -> Result<PinnedSlice<T>, Error> {
        self.bind()?;
        let mut p = std::ptr::null_mut();
        call!(hipHostMalloc(&mut p, (n * size_of::<T>()).max(1), flags))?;
        Ok(PinnedSlice { ptr: p.cast(), len: n })
    }
}

pub struct Arch {
    pub compile: String,
    pub name: String,
    pub tensor_cores: bool,
    /// gfx11 (RDNA3/3.5): the WMMA kernels. gfx12's WMMA has other fragment
    /// layouts, so it isn't used there yet.
    pub wmma: bool,
}

pub fn arch(ctx: &Context) -> Result<Arch, Error> {
    let mut props = vec![0u8; PROPS_BYTES.next_multiple_of(4096)];
    call!(hipGetDevicePropertiesR0600(props.as_mut_ptr().cast(), ctx.ordinal))?;
    let field = &props[PROPS_ARCH_OFFSET..PROPS_ARCH_OFFSET + 256];
    let full = CStr::from_bytes_until_nul(field).map_err(|_| Error("gcnArchName not terminated".into()))?;
    // "gfx1151:sramecc-:xnack-" -> "gfx1151"; the target features don't change our code.
    let name = full.to_string_lossy().split(':').next().unwrap_or_default().to_string();
    if !name.starts_with("gfx") {
        return Err(Error(format!("unexpected device arch {name:?}")));
    }
    // The mma.sync kernels are PTX-only (stubs on HIP).
    let wmma = name.starts_with("gfx11");
    Ok(Arch { compile: name.clone(), name, tensor_cores: false, wmma })
}

/// Compile the kernel source with hiprtc for `arch` (`gfx1151`): a code object.
pub fn compile(src: &str, arch: &str) -> Result<Vec<u8>, String> {
    let rtc = ffi::RTC.as_ref().map_err(Clone::clone)?;
    let src = CString::new(src).map_err(|e| e.to_string())?;
    let opts: Vec<CString> = [format!("--offload-arch={arch}"), "-O3".into(), "-std=c++17".into(), "-DRLLM_HIP".into()]
        .into_iter()
        .map(|o| CString::new(o).unwrap())
        .collect();
    let opt_ptrs: Vec<_> = opts.iter().map(|o| o.as_ptr()).collect();
    let mut prog = std::ptr::null_mut();
    // SAFETY: valid NUL-terminated strings; the program is destroyed below.
    unsafe {
        if (rtc.hiprtcCreateProgram)(&mut prog, src.as_ptr(), c"kernels.cu".as_ptr(), 0, std::ptr::null(), std::ptr::null()) != 0 {
            return Err("hiprtcCreateProgram failed".into());
        }
        let status = (rtc.hiprtcCompileProgram)(prog, opt_ptrs.len() as c_int, opt_ptrs.as_ptr());
        let result = if status != 0 {
            let mut n = 0;
            (rtc.hiprtcGetProgramLogSize)(prog, &mut n);
            let mut log = vec![0u8; n.max(1)];
            (rtc.hiprtcGetProgramLog)(prog, log.as_mut_ptr().cast());
            Err(format!("hiprtc error {status}:\n{}", String::from_utf8_lossy(&log).trim_end_matches('\0')))
        } else {
            let mut n = 0;
            (rtc.hiprtcGetCodeSize)(prog, &mut n);
            let mut code = vec![0u8; n];
            (rtc.hiprtcGetCode)(prog, code.as_mut_ptr().cast());
            Ok(code)
        };
        (rtc.hiprtcDestroyProgram)(&mut prog);
        result
    }
}

pub struct Module {
    raw: ffi::Module,
}

// SAFETY: module and function handles are immutable once loaded and usable
// from any thread.
unsafe impl Send for Module {}
unsafe impl Sync for Module {}

pub fn load_module(ctx: &Arc<Context>, code: Vec<u8>) -> Result<Arc<Module>, Error> {
    ctx.bind()?;
    let mut m = std::ptr::null_mut();
    call!(hipModuleLoadData(&mut m, code.as_ptr().cast()))?;
    Ok(Arc::new(Module { raw: m }))
}

impl Drop for Module {
    fn drop(&mut self) {
        // Functions hold the module, so none is left to launch.
        if let Ok(h) = hip() {
            unsafe { (h.hipModuleUnload)(self.raw) };
        }
    }
}

impl Module {
    pub fn load_function(self: &Arc<Self>, name: &str) -> Result<Function, Error> {
        let c = CString::new(name).map_err(|e| Error(e.to_string()))?;
        let mut f = std::ptr::null_mut();
        call!(hipModuleGetFunction(&mut f, self.raw, c.as_ptr()))?;
        Ok(Function { raw: f, _module: self.clone() })
    }
}

pub struct Function {
    raw: ffi::Function,
    _module: Arc<Module>,
}

unsafe impl Send for Function {}
unsafe impl Sync for Function {}

pub fn sm_count(ctx: &Context) -> Result<usize, Error> {
    Ok(ctx.attribute(ATTR_MULTIPROCESSOR_COUNT)? as usize)
}

/// Whether the GPU shares system RAM with the CPU (APUs such as Strix Halo).
pub fn integrated(ctx: &Context) -> Result<bool, Error> {
    Ok(ctx.attribute(ATTR_INTEGRATED)? != 0)
}

pub fn max_smem_optin(ctx: &Context) -> Result<usize, Error> {
    Ok(ctx.attribute(ATTR_SHARED_MEM_PER_BLOCK_OPTIN)? as usize)
}

/// No-op: AMD needs no opt-in for dynamic LDS up to the device limit.
pub fn set_max_dynamic_smem(_f: &Function, _bytes: usize) -> Result<(), Error> {
    Ok(())
}

pub fn func_resources(f: &Function) -> Result<(i32, i32), Error> {
    let (mut regs, mut local) = (0, 0);
    call!(hipFuncGetAttribute(&mut regs, FUNC_ATTR_NUM_REGS, f.raw))?;
    call!(hipFuncGetAttribute(&mut local, FUNC_ATTR_LOCAL_SIZE_BYTES, f.raw))?;
    Ok((regs, local))
}

pub fn occupancy(f: &Function, block: u32, smem: usize) -> Result<u32, Error> {
    let mut n = 0;
    call!(hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(&mut n, f.raw, block as c_int, smem))?;
    Ok(n.max(0) as u32)
}

pub struct Stream {
    raw: ffi::Stream,
    ctx: Arc<Context>,
}

// SAFETY: HIP streams may be used from any thread.
unsafe impl Send for Stream {}
unsafe impl Sync for Stream {}

impl Drop for Stream {
    fn drop(&mut self) {
        if let Ok(h) = hip() {
            unsafe {
                (h.hipStreamSynchronize)(self.raw);
                (h.hipStreamDestroy)(self.raw);
            }
        }
    }
}

impl Stream {
    pub fn context(&self) -> &Arc<Context> {
        &self.ctx
    }

    pub fn synchronize(&self) -> Result<(), Error> {
        call!(hipStreamSynchronize(self.raw))
    }

    /// Stream-ordered allocation of `n` values (contents uninitialized).
    ///
    /// # Safety
    /// The contents must be written before they are read.
    pub unsafe fn alloc<T>(self: &Arc<Self>, n: usize) -> Result<Slice<T>, Error> {
        let mut p = std::ptr::null_mut();
        call!(hipMallocAsync(&mut p, (n * size_of::<T>()).max(1), self.raw))?;
        Ok(Slice { ptr: p as u64, len: n, stream: self.clone(), _t: PhantomData })
    }

    pub fn alloc_zeros<T>(self: &Arc<Self>, n: usize) -> Result<Slice<T>, Error> {
        // SAFETY: zeroed right away, in stream order.
        let s = unsafe { self.alloc::<T>(n) }?;
        call!(hipMemsetD8Async(s.ptr as *mut c_void, 0, n * size_of::<T>(), self.raw))?;
        Ok(s)
    }

    /// Copy `src` into the start of `dst`. Pageable sources are waited for
    /// (the caller may free them right after); pinned ones stay asynchronous.
    pub fn memcpy_htod<T: Copy, S: HostSrc<T> + ?Sized, D: DevDst<T>>(&self, src: &S, dst: &mut D) -> Result<(), Error> {
        let (p, n) = (src.host_ptr(), src.host_len());
        if n > dst.dev_len() {
            return Err(Error(format!("memcpy_htod: {n} values into {}", dst.dev_len())));
        }
        call!(hipMemcpyHtoDAsync(dst.dev_ptr() as *mut c_void, p.cast(), n * size_of::<T>(), self.raw))?;
        if !src.pinned() {
            self.synchronize()?;
        }
        Ok(())
    }

    pub fn clone_htod<T: Copy, S: HostSrc<T> + ?Sized>(self: &Arc<Self>, src: &S) -> Result<Slice<T>, Error> {
        // SAFETY: fully overwritten by the copy.
        let mut s = unsafe { self.alloc::<T>(src.host_len()) }?;
        self.memcpy_htod(src, &mut s)?;
        Ok(s)
    }

    pub fn launch_builder<'a>(&'a self, f: &'a Function) -> LaunchArgs<'a> {
        LaunchArgs { stream: self, f, args: Vec::with_capacity(20) }
    }
}

/// Device memory for `len` values of `T`, freed in stream order on drop.
pub struct Slice<T> {
    ptr: u64,
    len: usize,
    stream: Arc<Stream>,
    _t: PhantomData<T>,
}

impl<T> Drop for Slice<T> {
    fn drop(&mut self) {
        if let Ok(h) = hip() {
            unsafe { (h.hipFreeAsync)(self.ptr as *mut c_void, self.stream.raw) };
        }
    }
}

impl<T> Slice<T> {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn slice_mut(&mut self, r: Range<usize>) -> SliceMut<'_, T> {
        assert!(r.start <= r.end && r.end <= self.len, "slice_mut {r:?} of {}", self.len);
        SliceMut { ptr: self.ptr + (r.start * size_of::<T>()) as u64, len: r.end - r.start, _s: PhantomData }
    }
}

pub struct SliceMut<'a, T> {
    ptr: u64,
    len: usize,
    _s: PhantomData<&'a mut Slice<T>>,
}

/// A device address (cudarc's form: the address plus a guard, unit here).
pub trait DevicePtr {
    fn device_ptr(&self, stream: &Stream) -> (u64, ());
}

impl<T> DevicePtr for Slice<T> {
    fn device_ptr(&self, _stream: &Stream) -> (u64, ()) {
        (self.ptr, ())
    }
}

pub trait DevDst<T> {
    fn dev_ptr(&self) -> u64;
    fn dev_len(&self) -> usize;
}

impl<T> DevDst<T> for Slice<T> {
    fn dev_ptr(&self) -> u64 {
        self.ptr
    }
    fn dev_len(&self) -> usize {
        self.len
    }
}

impl<T> DevDst<T> for SliceMut<'_, T> {
    fn dev_ptr(&self) -> u64 {
        self.ptr
    }
    fn dev_len(&self) -> usize {
        self.len
    }
}

pub trait HostSrc<T> {
    fn host_ptr(&self) -> *const T;
    fn host_len(&self) -> usize;
    fn pinned(&self) -> bool {
        false
    }
}

impl<T> HostSrc<T> for [T] {
    fn host_ptr(&self) -> *const T {
        self.as_ptr()
    }
    fn host_len(&self) -> usize {
        self.len()
    }
}

impl<T> HostSrc<T> for Vec<T> {
    fn host_ptr(&self) -> *const T {
        self.as_ptr()
    }
    fn host_len(&self) -> usize {
        self.len()
    }
}

impl<T> HostSrc<T> for PinnedSlice<T> {
    fn host_ptr(&self) -> *const T {
        self.ptr
    }
    fn host_len(&self) -> usize {
        self.len
    }
    fn pinned(&self) -> bool {
        true
    }
}

/// Page-locked host memory.
pub struct PinnedSlice<T> {
    ptr: *mut T,
    len: usize,
}

// SAFETY: plain host memory, owned by this value.
unsafe impl<T: Send> Send for PinnedSlice<T> {}
unsafe impl<T: Sync> Sync for PinnedSlice<T> {}

impl<T> Drop for PinnedSlice<T> {
    fn drop(&mut self) {
        if let Ok(h) = hip() {
            unsafe { (h.hipHostFree)(self.ptr.cast()) };
        }
    }
}

impl<T> PinnedSlice<T> {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn as_mut_ptr(&mut self) -> Result<*mut T, Error> {
        Ok(self.ptr)
    }
}

/// Plain-old-data kernel arguments, passed by value.
///
/// # Safety
/// The type must have the C layout of the kernel parameter it is passed as.
pub unsafe trait DeviceRepr {}

unsafe impl DeviceRepr for i32 {}
unsafe impl DeviceRepr for u32 {}
unsafe impl DeviceRepr for i64 {}
unsafe impl DeviceRepr for u64 {}
unsafe impl DeviceRepr for f32 {}

/// Something that can be a kernel argument: a pointer to its value.
pub trait KernelArg {
    fn arg_ptr(&self) -> *mut c_void;
}

impl<T: DeviceRepr> KernelArg for T {
    fn arg_ptr(&self) -> *mut c_void {
        self as *const T as *mut c_void
    }
}

impl<T> KernelArg for Slice<T> {
    fn arg_ptr(&self) -> *mut c_void {
        &self.ptr as *const u64 as *mut c_void
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LaunchConfig {
    pub grid_dim: (u32, u32, u32),
    pub block_dim: (u32, u32, u32),
    pub shared_mem_bytes: u32,
}

pub struct LaunchArgs<'a> {
    stream: &'a Stream,
    f: &'a Function,
    args: Vec<*mut c_void>,
}

pub trait PushKernelArg<T> {
    fn arg(&mut self, arg: T) -> &mut Self;
}

impl<'a, A: KernelArg + ?Sized> PushKernelArg<&'a A> for LaunchArgs<'a> {
    fn arg(&mut self, arg: &'a A) -> &mut Self {
        self.args.push(arg.arg_ptr());
        self
    }
}

impl LaunchArgs<'_> {
    /// # Safety
    /// The arguments must match the kernel's parameters.
    pub unsafe fn launch(&mut self, cfg: LaunchConfig) -> Result<(), Error> {
        let (g, b) = (cfg.grid_dim, cfg.block_dim);
        call!(hipModuleLaunchKernel(
            self.f.raw,
            g.0,
            g.1,
            g.2,
            b.0,
            b.1,
            b.2,
            cfg.shared_mem_bytes,
            self.stream.raw,
            self.args.as_mut_ptr(),
            std::ptr::null_mut(),
        ))
    }
}

pub struct Event {
    raw: ffi::Event,
}

unsafe impl Send for Event {}
unsafe impl Sync for Event {}

impl Drop for Event {
    fn drop(&mut self) {
        if let Ok(h) = hip() {
            unsafe { (h.hipEventDestroy)(self.raw) };
        }
    }
}

pub fn new_event(ctx: &Arc<Context>, timing: bool) -> Result<Event, Error> {
    ctx.bind()?;
    let mut e = std::ptr::null_mut();
    call!(hipEventCreateWithFlags(&mut e, if timing { 0 } else { EVENT_DISABLE_TIMING }))?;
    Ok(Event { raw: e })
}

impl Event {
    pub fn record(&self, stream: &Stream) -> Result<(), Error> {
        call!(hipEventRecord(self.raw, stream.raw))
    }

    pub fn synchronize(&self) -> Result<(), Error> {
        call!(hipEventSynchronize(self.raw))
    }

    pub fn elapsed_ms(&self, end: &Event) -> Result<f32, Error> {
        let mut ms = 0f32;
        call!(hipEventElapsedTime(&mut ms, self.raw, end.raw))?;
        Ok(ms)
    }
}

/// An instantiated graph, launched on the stream it was captured from.
pub struct GraphExec {
    graph: ffi::Graph,
    exec: ffi::GraphExec,
    stream: Arc<Stream>,
}

unsafe impl Send for GraphExec {}

impl Drop for GraphExec {
    fn drop(&mut self) {
        if let Ok(h) = hip() {
            unsafe {
                (h.hipGraphExecDestroy)(self.exec);
                (h.hipGraphDestroy)(self.graph);
            }
        }
    }
}

impl GraphExec {
    pub fn launch(&self) -> Result<(), Error> {
        call!(hipGraphLaunch(self.exec, self.stream.raw))
    }
}

pub fn begin_capture(stream: &Stream) -> Result<(), Error> {
    call!(hipStreamBeginCapture(stream.raw, CAPTURE_MODE_RELAXED))
}

pub fn end_capture(stream: &Arc<Stream>) -> Result<Option<GraphExec>, Error> {
    let mut graph = std::ptr::null_mut();
    call!(hipStreamEndCapture(stream.raw, &mut graph))?;
    if graph.is_null() {
        return Ok(None);
    }
    let mut exec = std::ptr::null_mut();
    if let Err(e) = call!(hipGraphInstantiate(&mut exec, graph, std::ptr::null_mut(), std::ptr::null_mut(), 0)) {
        call!(hipGraphDestroy(graph))?;
        return Err(e);
    }
    Ok(Some(GraphExec { graph, exec, stream: stream.clone() }))
}

/// # Safety
/// `ptr..ptr + bytes` must be device memory this process owns.
pub unsafe fn memset_zero_async(ptr: u64, bytes: usize, stream: &Stream) -> Result<(), Error> {
    call!(hipMemsetD8Async(ptr as *mut c_void, 0, bytes, stream.raw))
}

/// # Safety
/// `ptr` must hold at least `out.len()` floats.
pub unsafe fn memcpy_dtoh_async(out: &mut [f32], ptr: u64, stream: &Stream) -> Result<(), Error> {
    call!(hipMemcpyDtoHAsync(out.as_mut_ptr().cast(), ptr as *mut c_void, size_of_val(out), stream.raw))
}

pub fn hold_freed_memory(ctx: &Context) -> Result<(), Error> {
    let mut pool = std::ptr::null_mut();
    let mut threshold = u64::MAX;
    call!(hipDeviceGetDefaultMemPool(&mut pool, ctx.ordinal))?;
    call!(hipMemPoolSetAttribute(pool, MEMPOOL_ATTR_RELEASE_THRESHOLD, (&mut threshold as *mut u64).cast()))
}

pub fn alloc_mapped_host(ctx: &Arc<Context>, bytes: usize) -> Result<(PinnedSlice<u8>, u64), Error> {
    let buf = ctx.alloc_host::<u8>(bytes, HOST_MALLOC_MAPPED | HOST_MALLOC_PORTABLE)?;
    let mut dev = std::ptr::null_mut();
    call!(hipHostGetDevicePointer(&mut dev, buf.ptr.cast(), 0))?;
    Ok((buf, dev as u64))
}

/// Decode matvecs split every row of 1024+ columns over 4 waves, as
/// llama.cpp's do: on gfx1151 a wave per row left the memory system
/// underused (BF16 0.6B decode 104 -> 125 tok/s).
pub const MATVEC_SPLIT_K: bool = true;

/// hipBLAS on gfx11 runs f16/bf16-output GEMMs about 4x faster than f32-output
/// ones (measured 20-29 vs 5-6 TFLOPS on gfx1151, f32 accumulate either way),
/// so GEMMs write the input's 16-bit type and a kernel widens it to f32, as
/// llama.cpp's hipBLAS path does (scaled; see HALF_OUT_SCALE in backend.rs).
pub const GEMM_HALF_OUT: bool = true;

/// BF16 weights are converted to f16 for prefill GEMMs: faster on hipBLAS
/// (+20% Nemotron prefill, whose per-layer embedding table is BF16) and the
/// f16 output keeps more precision than bf16 would (see GEMM_HALF_OUT).
pub const BF16_VIA_F16: bool = true;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum GemmIn {
    F16,
    BF16,
}

/// hipBLASLt's plan for one GEMM shape: descriptors and the chosen algorithm.
struct LtPlan {
    desc: ffi::LtDesc,
    layouts: [ffi::LtLayout; 3],
    /// `hipblasLtMatmulAlgo_t` (opaque, 24 bytes in ROCm 7).
    algo: [u64; 3],
}

/// hipBLASLt handle and plans by (n, m, k, input type).
struct Lt {
    handle: ffi::LtHandle,
    plans: std::sync::Mutex<std::collections::HashMap<(usize, usize, usize, GemmIn), Option<LtPlan>>>,
}

/// hipBLAS on the backend's stream, with a workspace when the library takes
/// one. Large f16/bf16-output GEMMs go through hipBLASLt when it loads: 1.5x
/// hipBLAS on gfx1151 from 4096 x 4096 weights up (30-33 vs 16-22 TFLOPS),
/// but no faster on small ones, where its higher per-call cost made
/// Qwen3-0.6B prefill 18% slower.
pub struct Blas {
    handle: ffi::BlasHandle,
    lt: Option<Lt>,
    stream: Arc<Stream>,
    workspace: Slice<u8>,
}

const WORKSPACE: usize = 32 << 20;
const LT_DESC_TRANSA: c_int = 0;
const LT_DESC_TRANSB: c_int = 1;
const LT_PREF_MAX_WORKSPACE_BYTES: c_int = 1;
/// `sizeof(hipblasLtMatmulHeuristicResult_t)`; the algorithm is its first 24 bytes.
const LT_HEURISTIC_BYTES: usize = 56;

unsafe impl Send for Blas {}
unsafe impl Sync for Blas {}

impl Drop for Blas {
    fn drop(&mut self) {
        if let Ok(b) = ffi::BLAS.as_ref() {
            unsafe { (b.hipblasDestroy)(self.handle) };
        }
        if let (Some(lt), Ok(f)) = (&self.lt, ffi::LT.as_ref()) {
            for plan in lt.plans.lock().unwrap().values().flatten() {
                unsafe {
                    (f.hipblasLtMatmulDescDestroy)(plan.desc);
                    plan.layouts.iter().for_each(|&l| _ = (f.hipblasLtMatrixLayoutDestroy)(l));
                }
            }
            unsafe { (f.hipblasLtDestroy)(lt.handle) };
        }
    }
}

impl Blas {
    pub fn new(stream: &Arc<Stream>) -> Result<Self, String> {
        let b = ffi::BLAS.as_ref().map_err(Clone::clone)?;
        stream.ctx.bind().map_err(|e| e.0)?;
        let mut handle = std::ptr::null_mut();
        let st = unsafe { (b.hipblasCreate)(&mut handle) };
        if st != 0 {
            return Err(format!("hipblasCreate: {st}"));
        }
        let workspace = match stream.alloc_zeros::<u8>(WORKSPACE) {
            Ok(w) => w,
            Err(e) => {
                unsafe { (b.hipblasDestroy)(handle) };
                return Err(e.0);
            }
        };
        unsafe {
            if (b.hipblasSetStream)(handle, stream.raw) != 0 {
                (b.hipblasDestroy)(handle);
                return Err("hipblasSetStream failed".into());
            }
            if let Some(set) = b.hipblasSetWorkspace {
                set(handle, workspace.ptr as *mut c_void, WORKSPACE);
            }
        }
        let lt = if std::env::var_os("RLLM_NO_HIPBLASLT").is_some() {
            None
        } else {
            ffi::LT.as_ref().ok().and_then(|f| {
                let mut h = std::ptr::null_mut();
                (unsafe { (f.hipblasLtCreate)(&mut h) } == 0).then(|| Lt { handle: h, plans: Default::default() })
            })
        };
        Ok(Self { handle, lt, stream: stream.clone(), workspace })
    }

    /// hipBLASLt's plan for this shape (made on first use; None if it has
    /// no algorithm for it).
    fn lt_plan<'a>(
        &self,
        lt: &'a Lt,
        n: usize,
        m: usize,
        k: usize,
        input: GemmIn,
        ty: c_int,
    ) -> std::sync::MutexGuard<'a, std::collections::HashMap<(usize, usize, usize, GemmIn), Option<LtPlan>>> {
        let mut plans = lt.plans.lock().unwrap();
        plans.entry((n, m, k, input)).or_insert_with(|| {
            let f = ffi::LT.as_ref().ok()?;
            // SAFETY: valid out-pointers and attribute buffers; everything
            // created is either kept in the plan or destroyed here.
            unsafe {
                let mut desc = std::ptr::null_mut();
                if (f.hipblasLtMatmulDescCreate)(&mut desc, HIPBLAS_COMPUTE_32F, HIP_R_32F) != 0 {
                    return None;
                }
                let (ta, tb) = (HIPBLAS_OP_T, HIPBLAS_OP_N);
                (f.hipblasLtMatmulDescSetAttribute)(desc, LT_DESC_TRANSA, (&ta as *const c_int).cast(), 4);
                (f.hipblasLtMatmulDescSetAttribute)(desc, LT_DESC_TRANSB, (&tb as *const c_int).cast(), 4);
                let mut layouts = [std::ptr::null_mut(); 3];
                // A stored [n][k] (k-major, transposed in the op), B [m][k], C [m][n].
                (f.hipblasLtMatrixLayoutCreate)(&mut layouts[0], ty, k as u64, n as u64, k as i64);
                (f.hipblasLtMatrixLayoutCreate)(&mut layouts[1], ty, k as u64, m as u64, k as i64);
                (f.hipblasLtMatrixLayoutCreate)(&mut layouts[2], ty, n as u64, m as u64, n as i64);
                let mut pref = std::ptr::null_mut();
                (f.hipblasLtMatmulPreferenceCreate)(&mut pref);
                let ws = WORKSPACE as u64;
                (f.hipblasLtMatmulPreferenceSetAttribute)(pref, LT_PREF_MAX_WORKSPACE_BYTES, (&ws as *const u64).cast(), 8);
                let mut result = [0u64; LT_HEURISTIC_BYTES / 8];
                let mut found = 0;
                let st = (f.hipblasLtMatmulAlgoGetHeuristic)(
                    lt.handle, desc, layouts[0], layouts[1], layouts[2], layouts[2], pref, 1, result.as_mut_ptr().cast(), &mut found,
                );
                (f.hipblasLtMatmulPreferenceDestroy)(pref);
                if st != 0 || found < 1 {
                    (f.hipblasLtMatmulDescDestroy)(desc);
                    layouts.iter().for_each(|&l| _ = (f.hipblasLtMatrixLayoutDestroy)(l));
                    return None;
                }
                Some(LtPlan { desc, layouts, algo: [result[0], result[1], result[2]] })
            }
        });
        plans
    }

    /// See `rllm-cuda`'s `Blas::gemm_tn`; `c` has the input's type when
    /// [`GEMM_HALF_OUT`].
    ///
    /// # Safety
    /// The pointers must hold matrices of those shapes.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_tn(&self, n: usize, m: usize, k: usize, a: u64, b: u64, input: GemmIn, alpha: f32, c: u64) -> Result<(), String> {
        let lib = ffi::BLAS.as_ref().map_err(Clone::clone)?;
        let ty = match input {
            GemmIn::F16 => HIP_R_16F,
            GemmIn::BF16 => HIP_R_16BF,
        };
        let beta = 0f32;
        const LT_MIN_WEIGHT: usize = 8 << 20;
        if let Some(lt) = self.lt.as_ref().filter(|_| GEMM_HALF_OUT && n * k >= LT_MIN_WEIGHT) {
            let plans = self.lt_plan(lt, n, m, k, input, ty);
            if let (Some(plan), Ok(f)) = (&plans[&(n, m, k, input)], ffi::LT.as_ref()) {
                let [la, lb, lc] = plan.layouts;
                // SAFETY: the plan describes exactly these shapes; the caller
                // guarantees the pointers.
                let st = unsafe {
                    (f.hipblasLtMatmul)(
                        lt.handle,
                        plan.desc,
                        (&alpha as *const f32).cast(),
                        a as *const c_void,
                        la,
                        b as *const c_void,
                        lb,
                        (&beta as *const f32).cast(),
                        c as *const c_void,
                        lc,
                        c as *mut c_void,
                        lc,
                        plan.algo.as_ptr().cast(),
                        self.workspace.ptr as *mut c_void,
                        WORKSPACE,
                        self.stream.raw,
                    )
                };
                return if st != 0 { Err(format!("hipblasLtMatmul: status {st}")) } else { Ok(()) };
            }
        }
        let st = unsafe {
            (lib.hipblasGemmEx)(
                self.handle,
                HIPBLAS_OP_T,
                HIPBLAS_OP_N,
                n as c_int,
                m as c_int,
                k as c_int,
                (&alpha as *const f32).cast(),
                a as *const c_void,
                ty,
                k as c_int,
                b as *const c_void,
                ty,
                k as c_int,
                (&beta as *const f32).cast(),
                c as *mut c_void,
                if GEMM_HALF_OUT { ty } else { HIP_R_32F },
                n as c_int,
                HIPBLAS_COMPUTE_32F,
                HIPBLAS_GEMM_DEFAULT,
            )
        };
        if st != 0 { Err(format!("hipblasGemmEx: status {st}")) } else { Ok(()) }
    }
}
