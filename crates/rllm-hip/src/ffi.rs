//! HIP runtime, hiprtc and hipBLAS entry points, loaded with dlopen on first
//! use so rllm builds and runs (on other backends) without ROCm installed.
//! Signatures follow ROCm 7's headers; enum values used by `gpu.rs` are
//! listed next to it.

use std::ffi::{c_char, c_int, c_uint, c_void};
use std::path::PathBuf;
use std::sync::LazyLock;

use libloading::Library;

pub type HipError = c_int;
pub type Stream = *mut c_void;
pub type Module = *mut c_void;
pub type Function = *mut c_void;
pub type Event = *mut c_void;
pub type Graph = *mut c_void;
pub type GraphExec = *mut c_void;
pub type MemPool = *mut c_void;
pub type Program = *mut c_void;
pub type BlasHandle = *mut c_void;
pub type LtHandle = *mut c_void;
pub type LtDesc = *mut c_void;
pub type LtLayout = *mut c_void;
pub type LtPref = *mut c_void;

macro_rules! api {
    ($name:ident { $( $(#[$opt:ident])? fn $f:ident($($ty:ty),* $(,)?) -> $ret:ty; )* }) => {
        #[allow(non_snake_case)]
        pub struct $name {
            _lib: Library,
            $( pub $f: api!(@ty $($opt)? unsafe extern "C" fn($($ty),*) -> $ret), )*
        }

        impl $name {
            /// # Safety
            /// `lib` must export these symbols with these signatures.
            #[allow(non_snake_case)]
            unsafe fn load(lib: Library) -> Result<Self, String> {
                $( let $f = api!(@get $($opt)? lib, $f); )*
                Ok(Self { _lib: lib, $( $f, )* })
            }
        }
    };
    (@ty optional $t:ty) => { Option<$t> };
    (@ty $t:ty) => { $t };
    (@get optional $lib:ident, $f:ident) => {
        unsafe { $lib.get(concat!(stringify!($f), "\0").as_bytes()) }.ok().map(|s| *s)
    };
    (@get $lib:ident, $f:ident) => {
        *unsafe { $lib.get(concat!(stringify!($f), "\0").as_bytes()) }.map_err(|e| format!("{}: {e}", stringify!($f)))?
    };
}

api!(Hip {
    fn hipInit(c_uint) -> HipError;
    fn hipGetDeviceCount(*mut c_int) -> HipError;
    fn hipSetDevice(c_int) -> HipError;
    fn hipDeviceGetName(*mut c_char, c_int, c_int) -> HipError;
    fn hipDeviceGetAttribute(*mut c_int, c_int, c_int) -> HipError;
    fn hipGetDevicePropertiesR0600(*mut c_void, c_int) -> HipError;
    fn hipMemGetInfo(*mut usize, *mut usize) -> HipError;
    fn hipGetErrorName(HipError) -> *const c_char;
    fn hipStreamCreateWithFlags(*mut Stream, c_uint) -> HipError;
    fn hipStreamDestroy(Stream) -> HipError;
    fn hipStreamSynchronize(Stream) -> HipError;
    fn hipMallocAsync(*mut *mut c_void, usize, Stream) -> HipError;
    fn hipFreeAsync(*mut c_void, Stream) -> HipError;
    fn hipMemsetD8Async(*mut c_void, u8, usize, Stream) -> HipError;
    fn hipMemcpyHtoDAsync(*mut c_void, *const c_void, usize, Stream) -> HipError;
    fn hipMemcpyDtoHAsync(*mut c_void, *mut c_void, usize, Stream) -> HipError;
    fn hipHostMalloc(*mut *mut c_void, usize, c_uint) -> HipError;
    fn hipHostFree(*mut c_void) -> HipError;
    fn hipHostGetDevicePointer(*mut *mut c_void, *mut c_void, c_uint) -> HipError;
    fn hipModuleLoadData(*mut Module, *const c_void) -> HipError;
    fn hipModuleGetFunction(*mut Function, Module, *const c_char) -> HipError;
    fn hipModuleLaunchKernel(Function, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, Stream, *mut *mut c_void, *mut *mut c_void) -> HipError;
    fn hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(*mut c_int, Function, c_int, usize) -> HipError;
    fn hipFuncGetAttribute(*mut c_int, c_int, Function) -> HipError;
    fn hipEventCreateWithFlags(*mut Event, c_uint) -> HipError;
    fn hipEventRecord(Event, Stream) -> HipError;
    fn hipEventSynchronize(Event) -> HipError;
    fn hipEventElapsedTime(*mut f32, Event, Event) -> HipError;
    fn hipEventDestroy(Event) -> HipError;
    fn hipStreamBeginCapture(Stream, c_int) -> HipError;
    fn hipStreamEndCapture(Stream, *mut Graph) -> HipError;
    fn hipGraphInstantiate(*mut GraphExec, Graph, *mut c_void, *mut c_char, usize) -> HipError;
    fn hipGraphLaunch(GraphExec, Stream) -> HipError;
    fn hipGraphExecDestroy(GraphExec) -> HipError;
    fn hipGraphDestroy(Graph) -> HipError;
    fn hipDeviceGetDefaultMemPool(*mut MemPool, c_int) -> HipError;
    fn hipMemPoolSetAttribute(MemPool, c_int, *mut c_void) -> HipError;
});

api!(Rtc {
    fn hiprtcCreateProgram(*mut Program, *const c_char, *const c_char, c_int, *const *const c_char, *const *const c_char) -> c_int;
    fn hiprtcCompileProgram(Program, c_int, *const *const c_char) -> c_int;
    fn hiprtcGetProgramLogSize(Program, *mut usize) -> c_int;
    fn hiprtcGetProgramLog(Program, *mut c_char) -> c_int;
    fn hiprtcGetCodeSize(Program, *mut usize) -> c_int;
    fn hiprtcGetCode(Program, *mut c_char) -> c_int;
    fn hiprtcDestroyProgram(*mut Program) -> c_int;
});

api!(Blas {
    fn hipblasCreate(*mut BlasHandle) -> c_int;
    fn hipblasDestroy(BlasHandle) -> c_int;
    fn hipblasSetStream(BlasHandle, Stream) -> c_int;
    #[optional]
    fn hipblasSetWorkspace(BlasHandle, *mut c_void, usize) -> c_int;
    fn hipblasGemmEx(
        BlasHandle, c_int, c_int, c_int, c_int, c_int, *const c_void, *const c_void, c_int, c_int, *const c_void,
        c_int, c_int, *const c_void, *mut c_void, c_int, c_int, c_int, c_int,
    ) -> c_int;
});

api!(Lt {
    fn hipblasLtCreate(*mut LtHandle) -> c_int;
    fn hipblasLtDestroy(LtHandle) -> c_int;
    fn hipblasLtMatmulDescCreate(*mut LtDesc, c_int, c_int) -> c_int;
    fn hipblasLtMatmulDescDestroy(LtDesc) -> c_int;
    fn hipblasLtMatmulDescSetAttribute(LtDesc, c_int, *const c_void, usize) -> c_int;
    fn hipblasLtMatrixLayoutCreate(*mut LtLayout, c_int, u64, u64, i64) -> c_int;
    fn hipblasLtMatrixLayoutDestroy(LtLayout) -> c_int;
    fn hipblasLtMatmulPreferenceCreate(*mut LtPref) -> c_int;
    fn hipblasLtMatmulPreferenceDestroy(LtPref) -> c_int;
    fn hipblasLtMatmulPreferenceSetAttribute(LtPref, c_int, *const c_void, usize) -> c_int;
    fn hipblasLtMatmulAlgoGetHeuristic(
        LtHandle, LtDesc, LtLayout, LtLayout, LtLayout, LtLayout, LtPref, c_int, *mut c_void, *mut c_int,
    ) -> c_int;
    fn hipblasLtMatmul(
        LtHandle, LtDesc, *const c_void, *const c_void, LtLayout, *const c_void, LtLayout, *const c_void,
        *const c_void, LtLayout, *mut c_void, LtLayout, *const c_void, *mut c_void, usize, Stream,
    ) -> c_int;
});

/// Directories to look in before the loader's default search:
/// `$RLLM_HIP_LIB_DIR`, `$ROCM_PATH/lib`, `/opt/rocm/lib`.
fn lib_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(d) = std::env::var_os("RLLM_HIP_LIB_DIR") {
        dirs.push(PathBuf::from(d));
    }
    if let Some(d) = std::env::var_os("ROCM_PATH") {
        dirs.push(PathBuf::from(d).join("lib"));
    }
    dirs.push(PathBuf::from("/opt/rocm/lib"));
    dirs
}

fn open(names: &[&str]) -> Result<Library, String> {
    let mut last = String::new();
    for dir in lib_dirs().into_iter().map(Some).chain([None]) {
        for name in names {
            let path = dir.as_ref().map_or_else(|| PathBuf::from(name), |d| d.join(name));
            if dir.is_some() && !path.exists() {
                continue;
            }
            // SAFETY: loading ROCm's libraries runs their initializers, nothing else.
            match unsafe { Library::new(&path) } {
                Ok(lib) => return Ok(lib),
                Err(e) => last = e.to_string(),
            }
        }
    }
    Err(format!("{} not found ({last})", names[0]))
}

pub static HIP: LazyLock<Result<Hip, String>> =
    LazyLock::new(|| unsafe { Hip::load(open(&["libamdhip64.so.7", "libamdhip64.so.6", "libamdhip64.so"])?) });
pub static RTC: LazyLock<Result<Rtc, String>> =
    LazyLock::new(|| unsafe { Rtc::load(open(&["libhiprtc.so.7", "libhiprtc.so.6", "libhiprtc.so"])?) });
pub static LT: LazyLock<Result<Lt, String>> =
    LazyLock::new(|| unsafe { Lt::load(open(&["libhipblaslt.so.1", "libhipblaslt.so.0", "libhipblaslt.so"])?) });
pub static BLAS: LazyLock<Result<Blas, String>> =
    LazyLock::new(|| unsafe { Blas::load(open(&["libhipblas.so.3", "libhipblas.so.2", "libhipblas.so"])?) });
