//! The GPU backend, shared by `rllm-cuda` and `rllm-hip`: each crate
//! includes this file next to its own `gpu` module (the vendor API). Type
//! names say Cuda; on AMD they are re-exported under Hip names.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::gpu::{self, DevicePtr, LaunchConfig, PushKernelArg};
use rllm_core::quant::dequantize_row;
use rllm_core::{Activation, AttnMeta, Backend, DType, Error, HostTensor, KvLayout, KvType, Result, RowSample};

const KERNELS: &str = include_str!("../kernels/kernels.cu");
/// Must match MV_MAX_M in kernels.cu.
const MATVEC_MAX_M: usize = 8;
const MATVEC_WARPS: u32 = 4;
/// Head dims with an `attn_fa_d{HD}` prefill kernel (INSTANTIATE_FA).
const FA_HEAD_DIMS: [usize; 4] = [64, 128, 256, 512];

/// Mirrors `FaCfg<HD>` in kernels.cu: (warps splitting head_dim, row groups, keys per chunk, smem bytes).
fn fa_cfg(hd: usize) -> (usize, usize, usize, usize) {
    let dw = if hd > 256 { hd / 256 } else { 1 };
    let warps = if hd > 256 { 8 } else { 4 };
    let rg = warps / dw;
    let bk = if hd <= 128 { 32 } else if hd <= 256 { 16 } else { 16 };
    let stage = 2 * bk * (hd + 8);
    let smem = 2 * stage * 2 + if dw > 1 { rg * dw * 16 * bk * 4 } else { 0 };
    (dw, rg, bk, smem)
}

/// Head dims with an `attn_wmma_d{HD}` prefill kernel (INSTANTIATE_WA, HIP only).
const WA_HEAD_DIMS: [usize; 4] = [64, 128, 256, 512];

/// Mirrors `WaCfg<HD>` in kernels.cu: (query rows per block, shared memory bytes).
fn wa_cfg(hd: usize) -> (usize, usize) {
    const WAVES: usize = 8;
    let dw = hd / hd.min(128);
    let bk = if hd <= 256 { 32 } else { 16 };
    let sx = if dw > 1 { WAVES * 16 * bk * 4 } else { 0 };
    (WAVES / dw * 16, (bk * (hd + 8) + hd * (bk + 8) + WAVES * 16 * (bk + 8)) * 2 + sx)
}

/// (group, head_dim) pairs with an `attn_decode_g{G}_d{HD}` kernel (INSTANTIATE_AD).
const ATTN_DECODE_SHAPES: [(usize, usize); 10] =
    [(1, 64), (1, 128), (1, 256), (2, 64), (2, 128), (2, 256), (4, 64), (4, 128), (4, 256), (4, 512)];

/// Threads per MMQ block (MMQ_THREADS in kernels.cu).
const MMQ_THREADS: u32 = 128;
/// Weight rows per MMQ block (MMQ_BM in kernels.cu).
const MMQ_BM: usize = 128;

/// Mirrors `struct QOut` in kernels.cu: where a producer kernel writes the int8
/// form of its output (raw device pointers), or `on == 0`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct QOut {
    q: u64,
    d: u64,
    s: u64,
    on: i32,
}

// SAFETY: plain-old-data with the same C layout as the kernel parameter.
unsafe impl gpu::DeviceRepr for QOut {}

/// Token tile of the MMQ kernel (MMQ_BN in kernels.cu).
const MMQ_BN: usize = 64;

/// Must match MV_TARGETS in kernels.cu.
const MATVEC_TARGETS: usize = 3;

/// Mirrors `struct MvTarget` / `MvTargets` in kernels.cu (raw device pointers).
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct MvTarget {
    qs: u64,
    qh: u64,
    sc: u64,
    out: u64,
    qs_row: i64,
    qh_row: i64,
    sc_row: i64,
    n: i32,
    block_start: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct MvTargets {
    t: [MvTarget; MATVEC_TARGETS],
    count: i32,
}

// SAFETY: plain-old-data with the same C layout as the kernel parameter.
unsafe impl gpu::DeviceRepr for MvTargets {}
unsafe impl gpu::DeviceRepr for MvTarget {}
/// SM count from which prefill attention also writes its output's int8 copy
/// (see `attention`).
const FA_FUSED_QUANT_MIN_SMS: usize = 80;

/// Must match ATTN_WARPS / ATTN_MAX_D in kernels.cu.
const ATTN_WARPS: u32 = 4;
const ATTN_MAX_D: usize = 512;
const WARP: u32 = 32;
/// With `gpu::GEMM_HALF_OUT`, GEMMs compute `x·wᵀ / HALF_OUT_SCALE` so f16
/// output only overflows past 65504 * 256 ≈ 1.7e7 (outlier activations reach
/// the thousands); values under 2^-14 * 256 ≈ 0.016 lose relative precision
/// as f16 subnormals, an absolute error under 2e-5.
const HALF_OUT_SCALE: f32 = 256.0;
/// Raw GGUF bytes staged per repack launch during upload.
const UPLOAD_CHUNK_BYTES: usize = 64 << 20;

fn err(e: impl std::fmt::Debug) -> Error {
    Error::Backend(format!("{e:?}"))
}

/// Device-side weight layout (see the format comment in kernels.cu).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Fmt {
    F32,
    F16,
    BF16,
    Q8,
    Q4,
    Q6,
    Q5,
    IQ4,
}

impl Fmt {
    const ALL: [Fmt; 8] = [Fmt::F32, Fmt::F16, Fmt::BF16, Fmt::Q8, Fmt::Q4, Fmt::Q6, Fmt::Q5, Fmt::IQ4];
    const QUANT: [Fmt; 5] = [Fmt::Q8, Fmt::Q4, Fmt::Q6, Fmt::Q5, Fmt::IQ4];

    fn of(dtype: DType) -> Self {
        match dtype {
            DType::F32 => Fmt::F32,
            DType::F16 => Fmt::F16,
            DType::BF16 => Fmt::BF16,
            DType::Q8_0 => Fmt::Q8,
            DType::Q4K => Fmt::Q4,
            DType::Q6K => Fmt::Q6,
            DType::Q5K => Fmt::Q5,
            DType::IQ4XS => Fmt::IQ4,
        }
    }

    fn suffix(self) -> &'static str {
        match self {
            Fmt::F32 => "f32",
            Fmt::F16 => "f16",
            Fmt::BF16 => "bf16",
            Fmt::Q8 => "q8",
            Fmt::Q4 => "q4",
            Fmt::Q6 => "q6",
            Fmt::Q5 => "q5",
            Fmt::IQ4 => "iq4",
        }
    }

    fn quantized(self) -> bool {
        matches!(self, Fmt::Q8 | Fmt::Q4 | Fmt::Q6 | Fmt::Q5 | Fmt::IQ4)
    }

    /// Bytes per row of (qs, qh, sc) for a row of `k` elements.
    fn strides(self, k: usize) -> (usize, usize, usize) {
        match self {
            Fmt::F32 => (4 * k, 0, 0),
            Fmt::F16 | Fmt::BF16 => (2 * k, 0, 0),
            Fmt::Q8 => (k, 0, k / 32 * 2),
            Fmt::Q4 => (k / 2, 0, k / 256 * 20),
            Fmt::Q5 => (k / 2, k / 8, k / 256 * 20),
            Fmt::IQ4 => (k / 2, 0, k / 32 * 2),
            Fmt::Q6 => (k / 2, k / 4, k / 16 * 2),
        }
    }
}

/// Compile kernels.cu for `arch` (NVRTC or hiprtc), caching the result on
/// disk (`$XDG_CACHE_HOME/rllm/ptx` or `~/.cache/rllm/ptx`) keyed by a hash of
/// the source and arch, since compiling takes several seconds.
fn compile_kernels(arch: &str) -> Result<Vec<u8>> {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    (KERNELS, arch, env!("CARGO_PKG_VERSION")).hash(&mut h);
    let cache_dir = std::env::var_os("XDG_CACHE_HOME")
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| std::path::Path::new(&home).join(".cache")))
        .map(|d| d.join("rllm").join("ptx"));
    let cache_file = cache_dir.as_ref().map(|d| d.join(format!("{:016x}-{arch}.{}", h.finish(), gpu::CODE_EXT)));
    if let Some(code) = cache_file.as_ref().and_then(|f| std::fs::read(f).ok()) {
        return Ok(code);
    }
    let code = gpu::compile(KERNELS, arch).map_err(|e| Error::Backend(format!("kernel compile failed: {e}")))?;
    if let (Some(dir), Some(file)) = (cache_dir, cache_file) {
        // Best effort: write to a temp name, then rename into place.
        let tmp = file.with_extension(format!("tmp{}", std::process::id()));
        if std::fs::create_dir_all(&dir).is_ok() && std::fs::write(&tmp, &code).is_ok() {
            let _ = std::fs::rename(&tmp, &file);
        }
    }
    Ok(code)
}

/// Smallest quantized matrix (in elements) that prefill sends to MMQ rather
/// than dequant + cuBLAS (RLLM_MMQ_MIN overrides).
fn mmq_min_elems() -> usize {
    FLAGS.mmq_min
}

fn grid_1d(n: usize, threads: u32) -> LaunchConfig {
    LaunchConfig { grid_dim: ((n as u32).div_ceil(threads), 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 }
}

/// Launch kernel `$name` on the backend's stream with the given arguments.
/// An optional `prof = <label>` names the entry in the RLLM_PROFILE report.
/// Debug/A-B switches, read once: per-op `std::env::var` calls (a global
/// lock plus an environment scan) showed up in the per-token host time.
struct Flags {
    /// RLLM_MMQ=0/1 forces prefill quantized matmuls off/onto MMQ.
    mmq: Option<bool>,
    mmq_min: usize,
    no_fused_quant: bool,
    no_glu_mmq: bool,
    bf16_via_f16: bool,
    no_fa: bool,
    no_tc_attn: bool,
    /// RLLM_FA_QUANT=0/1 forces prefill attention's fused int8 output off/on.
    fa_quant: Option<bool>,
}

static FLAGS: std::sync::LazyLock<Flags> = std::sync::LazyLock::new(|| {
    let set = |k: &str| std::env::var_os(k).is_some();
    Flags {
        mmq: match std::env::var("RLLM_MMQ").as_deref() {
            Ok("0") => Some(false),
            Ok("1") => Some(true),
            _ => None,
        },
        mmq_min: std::env::var("RLLM_MMQ_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(4 << 20),
        no_fused_quant: set("RLLM_NO_FUSED_QUANT"),
        no_glu_mmq: set("RLLM_NO_GLU_MMQ"),
        bf16_via_f16: set("RLLM_BF16_VIA_F16") || gpu::BF16_VIA_F16,
        no_fa: set("RLLM_NO_FA"),
        no_tc_attn: set("RLLM_NO_TC_ATTN"),
        fa_quant: match std::env::var("RLLM_FA_QUANT").as_deref() {
            Ok("0") => Some(false),
            Ok("1") => Some(true),
            _ => None,
        },
    }
});

macro_rules! launch {
    ($be:expr, $name:expr, $cfg:expr, [$($arg:expr),* $(,)?]) => {
        launch!($be, $name, $cfg, [$($arg),*], prof = "")
    };
    ($be:expr, $name:expr, $cfg:expr, [$($arg:expr),* $(,)?], prof = $prof:expr) => {{
        let name: &str = &$name;
        let prof: &str = &$prof;
        // Any launch may overwrite the buffer whose int8 copy is cached.
        *$be.prequant.lock().unwrap() = None;
        if $be.replaying() {
            // The captured graph already contains this launch.
            Ok(())
        } else {
            let t0 = $be.profile_start();
            let mut b = $be.stream.launch_builder($be.f(name));
            $( b.arg(&*$arg); )*
            let r = unsafe { b.launch($cfg) }.map(|_| ()).map_err(|e| Error::Backend(format!("{name}: {e:?}")));
            $be.profile_end(if prof.is_empty() { name } else { prof }, t0);
            r
        }
    }};
}

/// Opt-in per-kernel timing (`RLLM_PROFILE=1`). Records a CUDA event pair
/// around every launch (no host syncs, so it doesn't distort small kernels);
/// pairs are resolved in batches. Printed on drop, together with the wall time
/// between the first and last event so gaps between kernels are visible.
#[derive(Default)]
struct Profile {
    enabled: bool,
    pending: Mutex<Vec<(String, gpu::Event, gpu::Event)>>,
    totals: Mutex<HashMap<String, (u64, f64)>>,
    span: Mutex<Option<(gpu::Event, f64)>>,
}

pub struct CudaBackend {
    stream: Arc<gpu::Stream>,
    blas: gpu::Blas,
    funcs: HashMap<String, gpu::Function>,
    device_name: String,
    arch: String,
    /// 16-bit staging buffers for the GEMM path: (weight, activations,
    /// output; the last only with `gpu::GEMM_HALF_OUT`).
    scratch: Mutex<(gpu::Slice<u16>, gpu::Slice<u16>, gpu::Slice<u16>)>,
    /// int8 activations with their per-32 scales and scaled sums (matvec, mmq).
    qscratch: Mutex<(gpu::Slice<i8>, gpu::Slice<f32>, gpu::Slice<f32>)>,
    /// int8 output of the fused gate/up MMQ (it can't write `qscratch`, which
    /// holds its own input). Prefill only.
    glu_scratch: Mutex<(gpu::Slice<i8>, gpu::Slice<f32>, gpu::Slice<f32>)>,
    /// Split-K partial outputs of the MMQ matmuls, `[splits, m, n]`.
    split_part: Mutex<gpu::Slice<f32>>,
    /// f16 copies of the q8 KV blocks a prefill step reads (K, V).
    kv_stage: Mutex<(gpu::Slice<u8>, gpu::Slice<u8>)>,
    /// (buffer pointer, length, in `glu_scratch`) whose int8 form is held in
    /// scratch, written by the producing kernel. Cleared by every other launch.
    prequant: Mutex<Option<(u64, usize, bool)>>,
    /// Producers only emit int8 copies when a consumer will use them: in
    /// decode-sized steps (quantized weights then always take the int8 matvec)
    /// and only if any quantized matrix was uploaded.
    decode_step: std::sync::atomic::AtomicBool,
    has_quant_weights: std::sync::atomic::AtomicBool,
    /// Some quantized matrix is large enough for the MMQ prefill path, which
    /// also consumes producer-side int8 copies (so fuse in prefill too).
    has_mmq_weights: std::sync::atomic::AtomicBool,
    /// Max dynamic shared memory per block (opt-in limit), for attn_prefill.
    max_smem: usize,
    /// sm_80+: mma.sync f16 tensor cores (attn_prefill_tc).
    tensor_cores: bool,
    /// RDNA3 WMMA (attn_wmma).
    wmma: bool,
    /// A one-element f32 buffer for kernel parameters that go unread.
    ones_placeholder: gpu::Slice<f32>,
    /// The int8 MMQ kernels can run (mma.sync or WMMA).
    mmq: bool,
    /// Streaming multiprocessors, for sizing grids.
    n_sms: usize,
    /// Resident blocks per SM for each mmq kernel (from the occupancy API).
    mmq_blocks_per_sm: HashMap<Fmt, usize>,

    /// Per-step index arrays, re-uploaded only when their contents change
    /// (attention metadata is identical for every layer of a step).
    index_cache: Mutex<HashMap<&'static str, IndexBuf>>,
    graph: Mutex<GraphState>,
    /// CUDA graphs for decode steps (off with RLLM_NO_GRAPHS or when profiling).
    graphs_enabled: bool,
    profile: Profile,
}

pub struct CudaWeight {
    fmt: Fmt,
    rows: usize,
    cols: usize,
    /// Main data: raw values for float formats, quantized values otherwise.
    qs: gpu::Slice<u8>,
    /// High bits (Q6 only).
    qh: Option<gpu::Slice<u8>>,
    /// Scales (quantized formats only).
    sc: Option<gpu::Slice<u8>>,
    /// 1-D weights (norms) are dequantized to f32 on upload.
    dense: Option<gpu::Slice<f32>>,
    /// Small F32 matrices also keep an f16 copy for cuBLAS, so prefill doesn't
    /// re-convert them every pass.
    f16_copy: Option<gpu::Slice<u16>>,
}

impl CudaWeight {
    fn qh(&self) -> &gpu::Slice<u8> {
        self.qh.as_ref().unwrap_or(&self.qs)
    }

    fn sc(&self) -> &gpu::Slice<u8> {
        self.sc.as_ref().unwrap_or(&self.qs)
    }

    fn strides(&self) -> (i64, i64, i64) {
        let (a, b, c) = self.fmt.strides(self.cols);
        (a as i64, b as i64, c as i64)
    }
}

/// An f32 activation buffer: a raw device pointer, plus the allocation that
/// owns it unless it lives in the graph arena.
///
/// `#[repr(C)]` with the pointer first: passing `&CudaBuf` as a kernel argument
/// hands the kernel exactly that pointer.
#[repr(C)]
pub struct CudaBuf {
    ptr: u64,
    len: usize,
    _owner: Option<gpu::Slice<f32>>,
}

// SAFETY: only ever passed by reference; the driver reads the leading u64.
unsafe impl gpu::DeviceRepr for CudaBuf {}

impl CudaBuf {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// CUDA graph capture/replay for repeated step shapes (decode). See
/// `Backend::begin_step`: the first step with a key is measured, the second
/// is captured, later ones replay the graph.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum StepMode {
    Eager,
    Measure(u64),
    Capture(u64),
    Replay(u64),
}

struct GraphState {
    mode: StepMode,
    /// Stable-address memory for activation buffers in Capture/Replay.
    arena: Option<gpu::Slice<u8>>,
    arena_off: usize,
    /// Arena bytes each key needs (from its Measure step).
    measured: HashMap<u64, usize>,
    graphs: HashMap<u64, Graph>,
    /// Index arrays written in the current step (a second, different write
    /// would make the captured upload ambiguous).
    written: HashMap<&'static str, Vec<u32>>,
    /// How many index arrays each key's step writes (from its capture). In
    /// replay the graph launches right after the last one is written, so the
    /// rest of the host walk overlaps the GPU instead of preceding it.
    index_writes: HashMap<u64, usize>,
    /// The current replay step's graph has been launched.
    launched: bool,
}

const ARENA_ALIGN: usize = 256;

/// cudarc's graph handle isn't `Send`; ours is only used under the backend's
/// mutex, on the backend's own stream and context.
struct Graph(gpu::GraphExec);
unsafe impl Send for Graph {}

/// A small index array kept on the device at a stable address. Uploads go
/// through pinned host memory so they are truly asynchronous (pageable copies
/// can stall the host until the stream drains, leaving the GPU idle).
struct IndexBuf {
    host: Vec<u32>,
    pinned: gpu::PinnedSlice<u32>,
    dev: gpu::Slice<u32>,
    /// Recorded after each copy out of `pinned`; waited on before `pinned` is
    /// rewritten. (cudarc's own tracking for this is off: event tracking is
    /// disabled context-wide.)
    copied: Option<gpu::Event>,
}

/// Paged KV cache: per layer, K and V as raw bytes (f16 bits, or int8 for
/// q8 with f16 scales per 32 values in `ks`/`vs`; one-element placeholders
/// for f16, so kernels always get valid pointers).
pub struct CudaKvCache {
    layout: KvLayout,
    k: Vec<gpu::Slice<u8>>,
    v: Vec<gpu::Slice<u8>>,
    ks: Vec<gpu::Slice<u16>>,
    vs: Vec<gpu::Slice<u16>>,
}

impl CudaKvCache {
    /// Bytes of one block in a layer's K (or V) pool, and in its scales
    /// (0 for f16).
    fn block_parts(&self) -> (usize, usize) {
        let n = self.layout.block_size * self.layout.slot_width();
        match self.layout.dtype {
            KvType::F16 => (2 * n, 0),
            KvType::Q8 => (n, n / 32 * 2),
        }
    }
}

/// Host blocks for a [`CudaKvCache`]: pinned and mapped into the device's
/// address space, so a kernel copies blocks straight over PCIe. Host block
/// `h` holds, layer by layer, the K block, the V block and (q8) their scales.
pub struct CudaHostKv {
    buf: gpu::PinnedSlice<u8>,
    /// Device address of `buf`.
    dev: u64,
    block_bytes: usize,
}

/// What [`probe_devices`] reports about one GPU.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub ordinal: usize,
    pub name: String,
    /// `sm_XY` or `gfxNNNN`.
    pub arch: String,
    /// Multiprocessors as the driver counts them: SMs (NVIDIA), WGPs (RDNA)
    /// or CUs (CDNA).
    pub sms: usize,
    pub total_bytes: usize,
    /// Free right now (other processes' allocations excluded).
    pub free_bytes: usize,
    /// Shares system RAM with the CPU (an APU such as Strix Halo).
    pub integrated: bool,
}

/// How many devices of this kind the machine has (0 without a driver).
pub fn device_count() -> usize {
    gpu::preload();
    gpu::device_count()
}

/// The devices of this kind and their free memory, without loading any kernels.
/// Empty if there is no driver or no device.
pub fn probe_devices() -> Vec<DeviceInfo> {
    gpu::preload();
    (0..gpu::device_count())
        .filter_map(|i| {
            let ctx = gpu::new_context(i).ok()?;
            let arch = gpu::arch(&ctx).ok()?.name;
            let sms = gpu::sm_count(&ctx).ok()?;
            let (free_bytes, total_bytes) = ctx.mem_get_info().ok()?;
            let integrated = gpu::integrated(&ctx).ok()?;
            Some(DeviceInfo { ordinal: i, name: ctx.name().ok()?, arch, sms, total_bytes, free_bytes, integrated })
        })
        .collect()
}

impl CudaBackend {
    pub fn new(ordinal: usize) -> Result<Self> {
        gpu::preload();
        let ctx = gpu::new_context(ordinal).map_err(err)?;
        let arch = gpu::arch(&ctx).map_err(err)?;
        let device_name = ctx.name().map_err(err)?;
        let module = gpu::load_module(&ctx, compile_kernels(&arch.compile)?).map_err(err)?;

        let mut names: Vec<String> = [
            "f32_to_f16", "f32_to_bf16", "half_to_f32", "logit_adjust", "sample_rows", "rms_norm", "add_rms_norm", "rms_norm_add", "qkv_norm_rope_store", "rms_norm_rope", "rope_neox", "silu_mul", "gelu_mul", "gelu_mul_strided", "ple_gate_f32", "gelu", "mul_inplace", "scale_inplace", "softcap_inplace",
            "add_inplace", "gather_rows", "gather_strided", "kv_store", "kv_store_q8", "kv_copy_blocks", "kv_dequant_blocks", "kv_rotate_q", "attn_decode_combine", "paged_attention", "attn_prefill", "attn_prefill_tc", "quantize_q8", "repack_q8_0",
            "repack_q4_k", "repack_q6_k", "repack_q5_k", "repack_iq4_xs", "mmq_sum_splits",
            "matvec_glu_silu_f32", "matvec_glu_silu_f16", "matvec_glu_silu_bf16", "matvec_glu_gelu_f32", "matvec_glu_gelu_f16", "matvec_glu_gelu_bf16",
        ]
        .map(String::from)
        .to_vec();
        for fmt in Fmt::QUANT {
            for k in ["mmq", "mmq_glu_silu", "mmq_glu_gelu", "matvec_glu_silu", "matvec_glu_gelu", "matvec_glu_norm_silu", "matvec_glu_norm_gelu"] {
                names.push(format!("{k}_{}", fmt.suffix()));
            }
        }
        for fmt in Fmt::ALL {
            for k in ["matvec", "dequant_f16", "embed"] {
                names.push(format!("{k}_{}", fmt.suffix()));
            }
        }
        // Let the attention kernels use more than the default 48 KB of shared memory.
        let max_smem = gpu::max_smem_optin(&ctx).map_err(err)?;
        // Templated decode attention instantiations (see INSTANTIATE_AD in kernels.cu).
        for (g, hd) in ATTN_DECODE_SHAPES {
            names.push(format!("attn_decode_g{g}_d{hd}"));
            names.push(format!("attn_decode_q8_g{g}_d{hd}"));
        }
        // FlashAttention prefill instantiations (INSTANTIATE_FA, INSTANTIATE_WA).
        for hd in FA_HEAD_DIMS {
            names.push(format!("attn_fa_d{hd}"));
        }
        if arch.wmma {
            names.extend(WA_HEAD_DIMS.iter().map(|hd| format!("attn_wmma_d{hd}")));
        }
        let funcs = names
            .into_iter()
            .map(|n| module.load_function(&n).map(|f| (n.clone(), f)).map_err(|e| Error::Backend(format!("{n}: {e:?}"))))
            .collect::<Result<HashMap<_, _>>>()?;
        let attn_smem_fns: Vec<String> = ["attn_prefill".to_string(), "attn_prefill_tc".to_string()]
            .into_iter()
            .chain(ATTN_DECODE_SHAPES.iter().flat_map(|(g, hd)| [format!("attn_decode_g{g}_d{hd}"), format!("attn_decode_q8_g{g}_d{hd}")]))
            .chain(FA_HEAD_DIMS.iter().map(|hd| format!("attn_fa_d{hd}")))
            .collect();
        for f in &attn_smem_fns {
            gpu::set_max_dynamic_smem(&funcs[f.as_str()], max_smem).map_err(err)?;
        }

        let n_sms = gpu::sm_count(&ctx).map_err(err)?;
        let mut mmq_blocks_per_sm = HashMap::new();
        for fmt in Fmt::QUANT {
            let f = &funcs[format!("mmq_{}", fmt.suffix()).as_str()];
            let blocks = gpu::occupancy(f, MMQ_THREADS, 0).map_err(err)?;
            mmq_blocks_per_sm.insert(fmt, blocks.max(1) as usize);
        }
        // A real stream: graph capture is not allowed on the legacy default stream.
        let stream = ctx.new_stream().map_err(err)?;
        let blas = gpu::Blas::new(&stream).map_err(Error::Backend)?;
        let scratch =
            (stream.alloc_zeros::<u16>(1).map_err(err)?, stream.alloc_zeros::<u16>(1).map_err(err)?, stream.alloc_zeros::<u16>(1).map_err(err)?);
        let qscratch = (
            stream.alloc_zeros::<i8>(32).map_err(err)?,
            stream.alloc_zeros::<f32>(1).map_err(err)?,
            stream.alloc_zeros::<f32>(1).map_err(err)?,
        );
        let glu_scratch = (
            stream.alloc_zeros::<i8>(32).map_err(err)?,
            stream.alloc_zeros::<f32>(1).map_err(err)?,
            stream.alloc_zeros::<f32>(1).map_err(err)?,
        );
        let split_part = stream.alloc_zeros::<f32>(1).map_err(err)?;
        let kv_stage = (stream.alloc_zeros::<u8>(1).map_err(err)?, stream.alloc_zeros::<u8>(1).map_err(err)?);
        let ones_placeholder = stream.clone_htod(&[1f32][..]).map_err(err)?;
        Ok(Self {
            stream,
            blas,
            funcs,
            device_name,
            arch: arch.name,
            scratch: Mutex::new(scratch),
            qscratch: Mutex::new(qscratch),
            glu_scratch: Mutex::new(glu_scratch),
            split_part: Mutex::new(split_part),
            kv_stage: Mutex::new(kv_stage),
            prequant: Mutex::new(None),
            decode_step: std::sync::atomic::AtomicBool::new(false),
            has_quant_weights: std::sync::atomic::AtomicBool::new(false),
            has_mmq_weights: std::sync::atomic::AtomicBool::new(false),
            max_smem,
            tensor_cores: arch.tensor_cores,
            wmma: arch.wmma,
            ones_placeholder,
            mmq: arch.tensor_cores || arch.wmma,
            n_sms,
            mmq_blocks_per_sm,
            index_cache: Mutex::new(HashMap::new()),
            graph: Mutex::new(GraphState {
                mode: StepMode::Eager,
                arena: None,
                arena_off: 0,
                measured: HashMap::new(),
                graphs: HashMap::new(),
                written: HashMap::new(),
                index_writes: HashMap::new(),
                launched: false,
            }),
            graphs_enabled: std::env::var_os("RLLM_NO_GRAPHS").is_none() && std::env::var_os("RLLM_PROFILE").is_none(),
            profile: Profile { enabled: std::env::var_os("RLLM_PROFILE").is_some(), ..Default::default() },
        })
    }

    fn profile_start(&self) -> Option<gpu::Event> {
        if !self.profile.enabled {
            return None;
        }
        let ev = gpu::new_event(self.stream.context(), true).ok()?;
        ev.record(&self.stream).ok()?;
        Some(ev)
    }

    fn profile_end(&self, name: &str, start: Option<gpu::Event>) {
        let Some(start) = start else { return };
        let Ok(end) = gpu::new_event(self.stream.context(), true) else { return };
        if end.record(&self.stream).is_err() {
            return;
        }
        let mut pending = self.profile.pending.lock().unwrap();
        pending.push((name.to_string(), start, end));
        if pending.len() >= 4096 {
            self.profile_flush(&mut pending);
        }
    }

    fn profile_flush(&self, pending: &mut Vec<(String, gpu::Event, gpu::Event)>) {
        if pending.is_empty() {
            return;
        }
        let _ = self.stream.synchronize();
        let mut totals = self.profile.totals.lock().unwrap();
        let mut span = self.profile.span.lock().unwrap();
        for (name, a, b) in pending.drain(..) {
            let ms = a.elapsed_ms(&b).unwrap_or(0.0) as f64;
            let e = totals.entry(name).or_default();
            e.0 += 1;
            e.1 += ms;
            match span.as_mut() {
                None => *span = Some((a, 0.0)),
                Some((first, total)) => *total = first.elapsed_ms(&b).unwrap_or(0.0) as f64,
            }
        }
    }

    /// Registers per thread and resident blocks per SM for a kernel at a given
    /// block size and dynamic shared memory (for tuning; see examples/).
    pub fn sm_count(&self) -> usize {
        self.n_sms
    }

    /// (registers per thread, resident blocks per SM, local memory bytes per thread).
    pub fn kernel_info(&self, name: &str, block: u32, smem: usize) -> Result<(i32, u32, i32)> {
        let f = self.funcs.get(name).ok_or_else(|| Error::Backend(format!("no kernel {name}")))?;
        let (regs, local) = gpu::func_resources(f).map_err(err)?;
        let blocks = gpu::occupancy(f, block, smem).map_err(err)?;
        Ok((regs, blocks, local))
    }

    fn mode(&self) -> StepMode {
        self.graph.lock().unwrap().mode
    }

    /// True while replaying a captured graph: device work is skipped because
    /// the graph already contains it; only host-side index data is refreshed.
    fn replaying(&self) -> bool {
        matches!(self.mode(), StepMode::Replay(_))
    }

    fn f(&self, name: &str) -> &gpu::Function {
        &self.funcs[name]
    }

    /// Make sure `cache[key]` holds `data` on the device.
    ///
    /// Graph modes: while capturing, every call records a copy from the key's
    /// pinned buffer (so replays pick up new contents); while replaying, only
    /// the pinned buffer is updated and the graph's copy does the rest.
    /// Device memory (free, total) in bytes.
    pub fn mem_info(&self) -> Result<(usize, usize)> {
        self.stream.context().mem_get_info().map_err(err)
    }

    /// Keep memory freed by stream-ordered allocations in the device's pool
    /// instead of returning it at each synchronization, so [`Self::mem_info`]
    /// shows the peak of temporary buffers (for `rllm tune --measure`).
    pub fn hold_freed_memory(&self) -> Result<()> {
        gpu::hold_freed_memory(self.stream.context()).map_err(err)
    }

    /// Copy KV blocks between a cache and its host store, every layer:
    /// `from[i]` -> `to[i]`, device to host if `to_host`. One launch per layer
    /// and part (K, V, their scales).
    fn kv_copy(&self, kv: &CudaKvCache, store: &CudaHostKv, from: &[u32], to: &[u32], to_host: bool) -> Result<()> {
        if from.len() != to.len() {
            return Err(Error::Shape(format!("kv copy: {} source blocks, {} destinations", from.len(), to.len())));
        }
        if from.is_empty() {
            return Ok(());
        }
        let (dev_blocks, host_blocks) = if to_host { (from, to) } else { (to, from) };
        let n_host = store.buf.len() / store.block_bytes;
        if dev_blocks.iter().any(|&b| b as usize >= kv.layout.n_blocks) || host_blocks.iter().any(|&b| b as usize >= n_host) {
            return Err(Error::Shape("kv copy: block out of range".into()));
        }
        let (part, scales) = kv.block_parts();
        if part % 16 != 0 || scales % 16 != 0 || store.block_bytes != kv.layout.block_bytes() {
            return Err(Error::Shape(format!("kv copy: block parts {part}/{scales} bytes")));
        }
        let mut cache = self.index_cache.lock().unwrap();
        self.sync_index(&mut cache, "kv_copy_from", from)?;
        self.sync_index(&mut cache, "kv_copy_to", to)?;
        let (fi, ti) = (&cache["kv_copy_from"].dev, &cache["kv_copy_to"].dev);
        let cfg = LaunchConfig { grid_dim: (from.len() as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let host_stride = store.block_bytes as i64;
        let mut off = 0u64;
        for layer in 0..kv.layout.n_layers {
            let mut parts: Vec<(u64, usize)> = vec![(kv.k[layer].device_ptr(&self.stream).0, part), (kv.v[layer].device_ptr(&self.stream).0, part)];
            if scales > 0 {
                parts.push((kv.ks[layer].device_ptr(&self.stream).0, scales));
                parts.push((kv.vs[layer].device_ptr(&self.stream).0, scales));
            }
            for (pool, bytes) in parts {
                let host = store.dev + off;
                let (src, dst, src_stride, dst_stride) =
                    if to_host { (pool, host, bytes as i64, host_stride) } else { (host, pool, host_stride, bytes as i64) };
                let nb = bytes as i32;
                launch!(self, "kv_copy_blocks", cfg, [&src, &dst, fi, ti, &src_stride, &dst_stride, &nb])?;
                off += bytes as u64;
            }
        }
        Ok(())
    }

    /// End of a graph step (the model's last op before reading results):
    /// finish a capture and launch the graph, or launch a replayed one.
    fn end_graph_step(&self) -> Result<()> {
        let mut g = self.graph.lock().unwrap();
        // Eager from here, whatever happens below.
        let mode = std::mem::replace(&mut g.mode, StepMode::Eager);
        match mode {
            StepMode::Measure(key) => {
                let used = g.arena_off;
                g.measured.insert(key, used);
            }
            StepMode::Capture(key) => {
                let n = g.written.len();
                g.index_writes.insert(key, n);
                let graph = gpu::end_capture(&self.stream)
                    .map_err(err)?
                    .ok_or_else(|| Error::Backend("empty graph capture".into()))?;
                graph.launch().map_err(err)?;
                g.graphs.insert(key, Graph(graph));
            }
            StepMode::Replay(key) => {
                if !g.launched {
                    Self::launch_graph(&g, key)?;
                }
            }
            StepMode::Eager => {}
        }
        Ok(())
    }

    /// Launch the recorded graph for `key`; an error, not a panic, if it was
    /// dropped mid-step.
    fn launch_graph(g: &GraphState, key: u64) -> Result<()> {
        let graph = g.graphs.get(&key).ok_or_else(|| Error::Backend("graph invalidated during its replay step".into()))?;
        graph.0.launch().map_err(err)
    }

    fn sync_index(&self, cache: &mut HashMap<&'static str, IndexBuf>, key: &'static str, data: &[u32]) -> Result<()> {
        let mode = {
            let mut g = self.graph.lock().unwrap();
            if g.mode != StepMode::Eager {
                match g.written.get(key) {
                    Some(prev) if prev.as_slice() != data => {
                        return Err(Error::Backend(format!(
                            "index array {key:?} changed within a graph step; give it its own op"
                        )));
                    }
                    Some(_) => return Ok(()),
                    None => {
                        g.written.insert(key, data.to_vec());
                    }
                }
            }
            g.mode
        };
        if mode == StepMode::Eager
            && let Some(buf) = cache.get(key)
            && buf.host.as_slice() == data
        {
            return Ok(());
        }
        let need = data.len().max(1);
        if cache.get(key).is_none_or(|b| b.dev.len() < need) {
            if matches!(mode, StepMode::Capture(_) | StepMode::Replay(_)) {
                return Err(Error::Backend(format!("index array {key:?} grew inside a graph step")));
            }
            let cap = need.next_power_of_two().max(64);
            let pinned = unsafe { self.stream.context().alloc_pinned::<u32>(cap) }.map_err(err)?;
            let dev = self.stream.alloc_zeros::<u32>(cap).map_err(err)?;
            if cache.insert(key, IndexBuf { host: Vec::new(), pinned, dev, copied: None }).is_some() {
                self.invalidate_graphs();
            }
        }
        let buf = cache.get_mut(key).unwrap();
        // Don't overwrite the staging buffer while an earlier copy may still read it.
        if let Some(ev) = &buf.copied {
            ev.synchronize().map_err(err)?;
        }
        let p = buf.pinned.as_mut_ptr().map_err(err)?;
        // SAFETY: `pinned` holds at least `data.len()` u32s (capacity checked above).
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), p, data.len()) };
        if !matches!(mode, StepMode::Replay(_)) {
            self.stream.memcpy_htod(&buf.pinned, &mut buf.dev).map_err(err)?;
            if mode == StepMode::Eager {
                let ev = gpu::new_event(self.stream.context(), false).map_err(err)?;
                ev.record(&self.stream).map_err(err)?;
                buf.copied = Some(ev);
            }
        }
        buf.host = data.to_vec();
        if let StepMode::Replay(key) = mode {
            // All of this step's index data is in place: start the graph now.
            let mut g = self.graph.lock().unwrap();
            if !g.launched && g.index_writes.get(&key) == Some(&g.written.len()) {
                Self::launch_graph(&g, key)?;
                g.launched = true;
            }
        }
        Ok(())
    }

    fn ensure_len(&self, buf: &mut gpu::Slice<u16>, len: usize) -> Result<()> {
        if buf.len() < len {
            if matches!(self.graph.lock().unwrap().mode, StepMode::Capture(_) | StepMode::Replay(_)) {
                return Err(Error::Backend("GEMM scratch grew inside a graph step".into()));
            }
            *buf = self.stream.alloc_zeros::<u16>(len).map_err(err)?;
            // Recorded steps hold the old scratch's address.
            self.invalidate_graphs();
        }
        Ok(())
    }

    /// `out[m, n] = x[m, k] · w[n, k]ᵀ` via f16 weights/activations, f32 accumulate.
    /// Whether a quantized weight takes the MMQ path in prefill (vs dequant +
    /// cuBLAS). cuBLAS wins on small weights, MMQ on large ones, where
    /// re-expanding to f16 every pass costs real DRAM traffic. The threshold is
    /// in weight elements; RLLM_MMQ=0/1 forces a path.
    fn uses_mmq(&self, w: &CudaWeight) -> bool {
        self.mmq
            && w.fmt.quantized()
            && FLAGS.mmq.unwrap_or_else(|| w.rows * w.cols >= mmq_min_elems())
    }

    /// Whether a producer should emit an int8 copy of a `len`-long output.
    fn fuse_quant_useful(&self, len: usize) -> bool {
        use std::sync::atomic::Ordering::Relaxed;
        let useful = if self.decode_step.load(Relaxed) {
            self.has_quant_weights.load(Relaxed)
        } else {
            self.has_mmq_weights.load(Relaxed)
        };
        len % 32 == 0 && useful && !FLAGS.no_fused_quant
    }

    /// Where a producer kernel should also write the int8 form of its `len`-long
    /// output (the quantization scratch), or "off" if it can't be used here.
    fn qout(&self, len: usize) -> Result<QOut> {
        if !self.fuse_quant_useful(len) {
            return Ok(QOut::default());
        }
        let mut qs = self.qscratch.lock().unwrap();
        if qs.0.len() < len {
            // Buffers baked into a captured graph must not move.
            if matches!(self.mode(), StepMode::Capture(_) | StepMode::Replay(_)) {
                return Ok(QOut::default());
            }
            qs.0 = unsafe { self.stream.alloc::<i8>(len) }.map_err(err)?;
            qs.1 = unsafe { self.stream.alloc::<f32>(len / 32) }.map_err(err)?;
            qs.2 = unsafe { self.stream.alloc::<f32>(len / 32) }.map_err(err)?;
            self.invalidate_graphs();
        }
        Ok(QOut {
            q: qs.0.device_ptr(&self.stream).0,
            d: qs.1.device_ptr(&self.stream).0,
            s: qs.2.device_ptr(&self.stream).0,
            on: 1,
        })
    }

    /// Unfused `norm_add_gated_ffn`: rms_norm_add reading x_in and writing
    /// x_out (and the int8 normed input), then the gated FFN.
    #[allow(clippy::too_many_arguments)]
    fn norm_add_gated_ffn_split(
        &self,
        src: &CudaBuf,
        wd1: &gpu::Slice<f32>,
        x_in: &CudaBuf,
        x_out: &mut CudaBuf,
        rows: usize,
        width: usize,
        eps: f32,
        scale: f32,
        w2: &CudaWeight,
        ffn: (&CudaWeight, &CudaWeight, &CudaWeight, Activation),
        normed: &mut CudaBuf,
        out: &mut CudaBuf,
        gate_buf: &mut CudaBuf,
        up_buf: &mut CudaBuf,
        n_ffn: usize,
    ) -> Result<()> {
        let (gate, up, down, act) = ffn;
        let wd2 = w2.dense.as_ref().ok_or_else(|| Error::Shape("rms_norm weight must be 1-D".into()))?;
        const RNA_MAX: usize = 8; // kernels.cu
        if width > RNA_MAX * 1024 {
            return Err(Error::Shape(format!("rms_norm_add: width {width} too large")));
        }
        let min_threads = if rows < self.n_sms { 1024 } else { 128 };
        let threads = (width.div_ceil(RNA_MAX) as u32).next_multiple_of(WARP).max(min_threads).min(1024);
        let cfg = LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
        let qo = if width % 32 == 0 { self.qout(rows * width)? } else { QOut::default() };
        let int8_only = qo.on != 0
            && gate.fmt.quantized()
            && up.fmt.quantized()
            && (rows <= MATVEC_MAX_M || self.uses_mmq(gate) || self.uses_mmq(up));
        let (di, has, write_out) = (width as i32, 1i32, (!int8_only) as i32);
        launch!(self, "rms_norm_add", cfg, [src, wd1, x_in, x_out, wd2, &mut *normed, &di, &eps, &scale, &has, &qo, &write_out])?;
        self.set_prequant(&qo, normed.ptr, rows * width);
        self.gated_ffn(normed, gate, up, down, act, out, gate_buf, up_buf, rows, n_ffn)
    }

    /// Drop captured graphs: they bake in device addresses of buffers that were
    /// just reallocated (index arrays, quantization scratch). Keys re-capture.
    fn invalidate_graphs(&self) {
        let mut g = self.graph.lock().unwrap();
        g.graphs.clear();
    }

    /// Record that `buf`'s first `len` values now have an int8 copy in scratch.
    fn set_prequant(&self, qo: &QOut, buf: u64, len: usize) {
        if qo.on != 0 {
            *self.prequant.lock().unwrap() = Some((buf, len, false));
        }
    }

    /// Where `x[..len]` is already quantized, if anywhere: `Some(true)` for
    /// `glu_scratch`, `Some(false)` for `qscratch` (consumes the record).
    fn take_prequant_src(&self, x: &CudaBuf, len: usize) -> Option<bool> {
        match self.prequant.lock().unwrap().take() {
            Some((p, l, glu)) if p == x.ptr && l == len => Some(glu),
            _ => None,
        }
    }

    /// Prefill gate/up projections plus activation in one MMQ launch
    /// (`mmq_glu_*`): `gate_out = act(x·gateᵀ) * (x·upᵀ)`, with the int8 copy
    /// in `glu_scratch` for the down projection. The f32 product is written
    /// only if `write_out`. Returns false (nothing done) when not applicable.
    fn glu_mmq(
        &self,
        x: &CudaBuf,
        gate: &CudaWeight,
        up: &CudaWeight,
        act: Activation,
        gate_out: &mut CudaBuf,
        m: usize,
        write_out: bool,
    ) -> Result<bool> {
        let (n, k) = (gate.rows, gate.cols);
        let decode = m <= MATVEC_MAX_M;
        if !gate.fmt.quantized()
            || gate.fmt != up.fmt
            || (up.rows, up.cols) != (n, k)
            || n % 32 != 0
            || (!decode && !(self.uses_mmq(gate) || self.uses_mmq(up)))
            || FLAGS.no_glu_mmq
        {
            return Ok(false);
        }
        check_len("glu_mmq x", x.len(), m * k)?;
        check_len("glu_mmq out", gate_out.len(), m * n)?;
        let mut qscratch = self.qscratch.lock().unwrap();
        let mut gs = self.glu_scratch.lock().unwrap();
        let len = m * n;
        if gs.0.len() < len {
            // Buffers baked into a captured graph must not move: fall back.
            if matches!(self.mode(), StepMode::Capture(_) | StepMode::Replay(_)) {
                return Ok(false);
            }
            gs.0 = unsafe { self.stream.alloc::<i8>(len) }.map_err(err)?;
            gs.1 = unsafe { self.stream.alloc::<f32>(len / 32) }.map_err(err)?;
            gs.2 = unsafe { self.stream.alloc::<f32>(len / 32) }.map_err(err)?;
            self.invalidate_graphs();
        }
        let pre = self.take_prequant_src(x, m * k);
        if pre != Some(false) {
            if pre == Some(true) {
                return Err(Error::Backend("glu_mmq: input quantized into glu_scratch".into()));
            }
            self.quantize_activations(x, m * k, &mut qscratch)?;
        }
        let qo = QOut {
            q: gs.0.device_ptr(&self.stream).0,
            d: gs.1.device_ptr(&self.stream).0,
            s: gs.2.device_ptr(&self.stream).0,
            on: 1,
        };
        let (xq, xd, xs) = &*qscratch;
        let a = match act {
            Activation::Silu => "silu",
            Activation::GeluTanh => "gelu",
        };
        if decode {
            // matvec_glu: 32 rows (one warp each, gate and up) per block.
            let target = |w: &CudaWeight| {
                let (sq, sh, ss) = w.strides();
                MvTarget {
                    qs: w.qs.device_ptr(&self.stream).0,
                    qh: w.qh().device_ptr(&self.stream).0,
                    sc: w.sc().device_ptr(&self.stream).0,
                    out: 0,
                    qs_row: sq,
                    qh_row: sh,
                    sc_row: ss,
                    n: n as i32,
                    block_start: 0,
                }
            };
            let (tg, tu) = (target(gate), target(up));
            let name = format!("matvec_glu_{a}_{}", gate.fmt.suffix());
            let prof = if self.profile.enabled { format!("{name} {n}x{k}") } else { String::new() };
            let cfg = LaunchConfig { grid_dim: ((n / 32) as u32, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 };
            let (ki, mi, wo) = (k as i32, m as i32, write_out as i32);
            launch!(self, &name, cfg, [&tg, &tu, xq, xd, &ki, &mi, &qo, &mut *gate_out, &wo], prof = prof)?;
            *self.prequant.lock().unwrap() = Some((gate_out.ptr, len, true));
            return Ok(true);
        }
        let name = format!("mmq_glu_{a}_{}", gate.fmt.suffix());
        let prof = if self.profile.enabled { format!("{name} {n}x{k}") } else { String::new() };
        // Must match MMQ_BM / MMQ_BN / MMQ_THREADS in kernels.cu; each block
        // covers MMQ_BM / 2 gate rows and the matching up rows.
        let cfg = LaunchConfig {
            grid_dim: ((m as u32).div_ceil(MMQ_BN as u32), (n as u32).div_ceil(MMQ_BM as u32 / 2), 1),
            block_dim: (MMQ_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let (ni, ki, mi) = (n as i32, k as i32, m as i32);
        let (sq, sh, ss) = gate.strides();
        let wo = write_out as i32;
        launch!(
            self,
            &name,
            cfg,
            [&gate.qs, gate.qh(), gate.sc(), xq, xd, xs, &mut *gate_out, &ni, &ki, &mi, &sq, &sh, &ss, &up.qs, up.qh(), up.sc(), &qo, &wo],
            prof = prof
        )?;
        *self.prequant.lock().unwrap() = Some((gate_out.ptr, len, true));
        Ok(true)
    }

    /// Quantize `len` activations into `qscratch` (int8, per-32 scale and scaled sum).
    fn quantize_activations(
        &self,
        x: &CudaBuf,
        len: usize,
        qscratch: &mut (gpu::Slice<i8>, gpu::Slice<f32>, gpu::Slice<f32>),
    ) -> Result<()> {
        let (xq, xd, xs) = qscratch;
        let units = len / 32;
        if xq.len() < len {
            *xq = unsafe { self.stream.alloc::<i8>(len) }.map_err(err)?;
            *xd = unsafe { self.stream.alloc::<f32>(units) }.map_err(err)?;
            *xs = unsafe { self.stream.alloc::<f32>(units) }.map_err(err)?;
            self.invalidate_graphs();
        }
        let nu = units as i64;
        launch!(self, "quantize_q8", grid_1d(units * 8, 256), [x, &mut *xq, &mut *xd, &mut *xs, &nu])
    }

    /// Prefill matmuls (m > MATVEC_MAX_M): int8 tensor-core MMQ for quantized
    /// weights (falls back to dequant + cuBLAS without tensor cores), cuBLAS
    /// for float weights. Shared input conversions happen once.
    fn matmul_many_batched(
        &self,
        x: &CudaBuf,
        ws: &[&CudaWeight],
        outs: &mut [&mut CudaBuf],
        m: usize,
    ) -> Result<()> {
        let k = ws[0].cols;
        // Once one weight sharing this input takes MMQ, x is quantized anyway;
        // send the other quantized weights there too (skips their f16 dequant
        // and the activation conversion).
        let any_mmq = ws.iter().any(|w| self.uses_mmq(w));
        let use_mmq = |w: &CudaWeight| w.fmt.quantized() && self.mmq && (any_mmq || self.uses_mmq(w));
        let pre = self.take_prequant_src(x, m * k);
        if ws.iter().any(|w| use_mmq(w)) {
            let mut qscratch = self.qscratch.lock().unwrap();
            let glu_scratch = if pre == Some(true) { Some(self.glu_scratch.lock().unwrap()) } else { None };
            if pre.is_none() {
                self.quantize_activations(x, m * k, &mut qscratch)?;
            }
            let (xq, xd, xs) = match &glu_scratch {
                Some(g) => &**g,
                None => &*qscratch,
            };
            let (ki, mi) = (k as i32, m as i32);
            for (w, out) in ws.iter().zip(outs.iter_mut()).filter(|(w, _)| use_mmq(w)) {
                let name = format!("mmq_{}", w.fmt.suffix());
                let prof = if self.profile.enabled { format!("{name} {}x{k}", w.rows) } else { String::new() };
                // Must match MMQ_BM / MMQ_BN / MMQ_CU / MMQ_THREADS in kernels.cu.
                // Split k to minimize wave quantization: with `slots` resident
                // blocks, s splits take ceil(tiles*s/slots) waves of 1/s-sized
                // blocks. Extra splits cost an output memset and atomics (2) or
                // writing and summing partial outputs (more; see mmq in kernels.cu).
                let tiles = w.rows.div_ceil(MMQ_BM) * m.div_ceil(MMQ_BN);
                let slots = self.n_sms * self.mmq_blocks_per_sm[&w.fmt];
                let max_splits = (k / 32).div_ceil(4).min(8);
                // Per extra split: 5% at k = 2560 (measured best on Nemotron prefill;
                // 0.02 and 0.1 were both slower end to end), scaled by 2560 / k since
                // the partial-sum traffic is fixed per output while the work grows with k.
                let penalty = 0.05 * 2560.0 / k as f64;
                let cost = |s: usize| (tiles * s).div_ceil(slots) as f64 / s as f64 * (1.0 + penalty * (s - 1) as f64);
                let splits = (1..=max_splits).min_by(|&a, &b| cost(a).total_cmp(&cost(b))).unwrap_or(1);
                let mut part = self.split_part.lock().unwrap();
                let len = m * w.rows;
                if splits == 2 && !self.replaying() {
                    // SAFETY: `out` holds `len` floats (checked by matmul_many).
                    unsafe { gpu::memset_zero_async(out.ptr, len * 4, &self.stream) }.map_err(err)?;
                }
                if splits > 2 && part.len() < splits * len {
                    *part = unsafe { self.stream.alloc::<f32>(splits * len) }.map_err(err)?;
                    self.invalidate_graphs();
                }
                let cfg = LaunchConfig {
                    grid_dim: ((m as u32).div_ceil(MMQ_BN as u32), (w.rows as u32).div_ceil(MMQ_BM as u32), splits as u32),
                    block_dim: (MMQ_THREADS, 1, 1),
                    shared_mem_bytes: 0,
                };
                let ni = w.rows as i32;
                let (sq, sh, ss) = w.strides();
                launch!(self, name, cfg, [&w.qs, w.qh(), w.sc(), xq, xd, xs, &mut **out, &mut *part, &ni, &ki, &mi, &sq, &sh, &ss], prof = prof)?;
                if splits > 2 {
                    // Fixed summation order: the same output every run.
                    let (si, li) = (splits as i32, len as i64);
                    launch!(self, "mmq_sum_splits", grid_1d(len.div_ceil(4), 256), [&*part, &mut **out, &si, &li])?;
                }
            }
        }
        let mut x_conv = None;
        for (w, out) in ws.iter().zip(outs.iter_mut()) {
            if use_mmq(w) {
                continue;
            }
            self.gemm(x, w, out, m, &mut x_conv)?;
        }
        Ok(())
    }

    /// `out[m, n] = x[m, k] · w[n, k]ᵀ` via cuBLAS, f32 accumulate. F16 and BF16
    /// weights are used as stored (activations converted to match); others are
    /// expanded to f16 first. `x_conv` says which type the scratch activations
    /// already hold (set when several weights share the same input).
    fn gemm(&self, x: &CudaBuf, w: &CudaWeight, out: &mut CudaBuf, m: usize, x_conv: &mut Option<Fmt>) -> Result<()> {
        let (n, k) = (w.rows, w.cols);
        let mut scratch = self.scratch.lock().unwrap();
        let (wh, xh, oh) = &mut *scratch;
        self.ensure_len(xh, m * k)?;
        let bf16_direct = w.fmt == Fmt::BF16 && !FLAGS.bf16_via_f16;
        let act = if bf16_direct { Fmt::BF16 } else { Fmt::F16 };
        if *x_conv != Some(act) {
            let len = (m * k) as i64;
            let kern = if act == Fmt::BF16 { "f32_to_bf16" } else { "f32_to_f16" };
            launch!(self, kern, grid_1d(m * k, 256), [x, &mut *xh, &len])?;
            *x_conv = Some(act);
        }
        let n_i = n as i32;
        let k_i = k as i32;
        // Event tracking is off, so the SyncOnDrop guards from device_ptr are no-ops.
        let a: u64 = if w.fmt == Fmt::F16 || bf16_direct {
            w.qs.device_ptr(&self.stream).0
        } else if let Some(h) = &w.f16_copy {
            h.device_ptr(&self.stream).0
        } else {
            self.ensure_len(wh, n * k)?;
            let (sq, sh, ss) = w.strides();
            launch!(self, &format!("dequant_f16_{}", w.fmt.suffix()), grid_1d(n * k / 4, 256), [&w.qs, w.qh(), w.sc(), &mut *wh, &n_i, &k_i, &sq, &sh, &ss])?;
            wh.device_ptr(&self.stream).0
        };
        let input = if act == Fmt::BF16 { gpu::GemmIn::BF16 } else { gpu::GemmIn::F16 };

        // Row-major out[m, n] is column-major outᵀ[n, m] = Wᵀᵀ · xᵀ.
        let (bp, _rb) = xh.device_ptr(&self.stream);
        let c = if gpu::GEMM_HALF_OUT {
            self.ensure_len(oh, m * n)?;
            oh.device_ptr(&self.stream).0
        } else {
            out.ptr
        };
        if !self.replaying() {
            let t0 = self.profile_start();
            let alpha = if gpu::GEMM_HALF_OUT { 1.0 / HALF_OUT_SCALE } else { 1.0 };
            // SAFETY: `a` is n x k, the scratch activations m x k and `c` m x n.
            let r = unsafe { self.blas.gemm_tn(n, m, k, a, bp, input, alpha, c) };
            if self.profile.enabled {
                self.profile_end(&format!("{}blas {:?} {}x{} m={}", gpu::VENDOR, w.fmt, n, k, m), t0);
            }
            r.map_err(Error::Backend)?;
        }
        if gpu::GEMM_HALF_OUT {
            let (len, bf) = ((m * n) as i64, (input == gpu::GemmIn::BF16) as i32);
            launch!(self, "half_to_f32", grid_1d((m * n).div_ceil(4), 256), [&*oh, &mut *out, &len, &bf, &HALF_OUT_SCALE])?;
        }
        Ok(())
    }
}

impl Drop for CudaBackend {
    fn drop(&mut self) {
        if !self.profile.enabled {
            return;
        }
        let mut pending = std::mem::take(&mut *self.profile.pending.lock().unwrap());
        self.profile_flush(&mut pending);
        let totals = self.profile.totals.lock().unwrap();
        let mut rows: Vec<_> = totals.iter().collect();
        rows.sort_by(|a, b| b.1.1.total_cmp(&a.1.1));
        let all: f64 = rows.iter().map(|(_, (_, ms))| *ms).sum();
        let span = self.profile.span.lock().unwrap().as_ref().map_or(0.0, |s| s.1);
        eprintln!("[rllm] kernel profile: {all:.1} ms in kernels, {span:.1} ms GPU wall time from first to last kernel");
        for (name, (n, ms)) in rows {
            eprintln!("  {name:<28} {n:>7} calls {ms:>9.2} ms {:>8.1} us/call", ms * 1e3 / *n as f64);
        }
    }
}

fn check_len(what: &str, have: usize, need: usize) -> Result<()> {
    if have < need {
        return Err(Error::Shape(format!("{what}: buffer has {have} elements, need {need}")));
    }
    Ok(())
}

impl Backend for CudaBackend {
    type Buffer = CudaBuf;
    type Weight = CudaWeight;
    type KvCache = CudaKvCache;
    type HostKv = CudaHostKv;

    fn name(&self) -> String {
        format!("{} ({}, {})", gpu::VENDOR, self.device_name, self.arch)
    }

    fn upload(&self, t: &HostTensor) -> Result<CudaWeight> {
        let (rows, cols) = t.as_2d();
        if cols % 32 != 0 {
            return Err(Error::Shape(format!("{}: {cols} columns not a multiple of 32", t.name)));
        }
        let fmt = Fmt::of(t.dtype);
        let dense = if t.shape.len() == 1 {
            let mut v = vec![0f32; cols];
            dequantize_row(t.dtype, t.bytes.as_slice(), &mut v);
            Some(self.stream.clone_htod(&v).map_err(err)?)
        } else {
            None
        };
        if !fmt.quantized() {
            let qs = self.stream.clone_htod(t.bytes.as_slice()).map_err(err)?;
            const F16_COPY_MAX: usize = 16 << 20;
            let f16_copy = if fmt == Fmt::F32 && t.shape.len() == 2 && rows * cols <= F16_COPY_MAX {
                let mut h = self.stream.alloc_zeros::<u16>(rows * cols).map_err(err)?;
                let (n_i, k_i) = (rows as i32, cols as i32);
                let (sq, sh, ss) = Fmt::F32.strides(cols);
                let (sq, sh, ss) = (sq as i64, sh as i64, ss as i64);
                launch!(self, "dequant_f16_f32", grid_1d(rows * cols / 4, 256), [&qs, &qs, &qs, &mut h, &n_i, &k_i, &sq, &sh, &ss])?;
                Some(h)
            } else {
                None
            };
            return Ok(CudaWeight { fmt, rows, cols, qs, qh: None, sc: None, dense, f16_copy });
        }

        if rows > 1 {
            self.has_quant_weights.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        // Embedding/output tables only feed lookups and the logits matvec,
        // never a batched prefill matmul.
        let vocab_table = t.name.starts_with("token_embd") || t.name.starts_with("output.");
        if rows * cols >= mmq_min_elems() && !vocab_table {
            self.has_mmq_weights.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        // Repack on the GPU, staging the GGUF bytes in row chunks so a huge
        // tensor never needs two full copies in VRAM.
        let (sq, sh, ss) = fmt.strides(cols);
        let alloc = |n: usize| unsafe { self.stream.alloc::<u8>(n.max(1)) }.map_err(err);
        let mut qs = alloc(rows * sq)?;
        let mut qh = if sh > 0 { Some(alloc(rows * sh)?) } else { None };
        let mut sc = alloc(rows * ss)?;
        let raw = t.bytes.as_slice();
        let raw_row = t.dtype.row_bytes(cols);
        let chunk_rows = (UPLOAD_CHUNK_BYTES / raw_row).clamp(1, rows);
        let mut staging = alloc(chunk_rows * raw_row)?;
        let k = cols as i32;
        for row0 in (0..rows).step_by(chunk_rows) {
            let n = chunk_rows.min(rows - row0);
            let bytes = &raw[row0 * raw_row..(row0 + n) * raw_row];
            self.stream.memcpy_htod(bytes, &mut staging.slice_mut(0..bytes.len())).map_err(err)?;
            let (r0, ni) = (row0 as i32, n as i32);
            let cfg = grid_1d(n * cols / 32, 128);
            match fmt {
                Fmt::Q8 => launch!(self, "repack_q8_0", cfg, [&staging, &mut qs, &mut sc, &r0, &ni, &k])?,
                Fmt::Q4 => launch!(self, "repack_q4_k", cfg, [&staging, &mut qs, &mut sc, &r0, &ni, &k])?,
                Fmt::Q6 => {
                    let qh = qh.as_mut().unwrap();
                    launch!(self, "repack_q6_k", cfg, [&staging, &mut qs, qh, &mut sc, &r0, &ni, &k])?
                }
                Fmt::Q5 => {
                    let qh = qh.as_mut().unwrap();
                    launch!(self, "repack_q5_k", cfg, [&staging, &mut qs, qh, &mut sc, &r0, &ni, &k])?
                }
                Fmt::IQ4 => launch!(self, "repack_iq4_xs", cfg, [&staging, &mut qs, &mut sc, &r0, &ni, &k])?,
                _ => unreachable!(),
            }
        }
        // Keep `staging` alive until the repack kernels have read it.
        self.stream.synchronize().map_err(err)?;
        drop(staging);
        Ok(CudaWeight { fmt, rows, cols, qs, qh, sc: Some(sc), dense, f16_copy: None })
    }

    fn alloc(&self, len: usize) -> Result<CudaBuf> {
        let len = len.max(1);
        let mut g = self.graph.lock().unwrap();
        match g.mode {
            StepMode::Capture(_) | StepMode::Replay(_) => {
                // Same allocation order every step -> same addresses.
                let bytes = (len * 4).next_multiple_of(ARENA_ALIGN);
                let arena = g.arena.as_ref().ok_or_else(|| Error::Backend("graph arena missing".into()))?;
                if g.arena_off + bytes > arena.len() {
                    return Err(Error::Backend("graph arena too small for this step".into()));
                }
                let ptr = arena.device_ptr(&self.stream).0 + g.arena_off as u64;
                g.arena_off += bytes;
                Ok(CudaBuf { ptr, len, _owner: None })
            }
            mode => {
                if matches!(mode, StepMode::Measure(_)) {
                    g.arena_off += (len * 4).next_multiple_of(ARENA_ALIGN);
                }
                drop(g);
                // SAFETY: activation buffers are always written before they are
                // read; skipping the zero-fill saves a memset per buffer per step.
                let slice = unsafe { self.stream.alloc::<f32>(len) }.map_err(err)?;
                let ptr = slice.device_ptr(&self.stream).0;
                Ok(CudaBuf { ptr, len, _owner: Some(slice) })
            }
        }
    }

    fn begin_step(&self, graph_key: Option<u64>) -> Result<()> {
        // Models only pass a key for decode-sized steps.
        self.decode_step.store(graph_key.is_some(), std::sync::atomic::Ordering::Relaxed);
        let mut g = self.graph.lock().unwrap();
        // A step that failed mid-capture left the stream capturing: end
        // that capture and drop what it recorded.
        if let StepMode::Capture(_) = std::mem::replace(&mut g.mode, StepMode::Eager) {
            let _ = gpu::end_capture(&self.stream);
        }
        g.written.clear();
        g.arena_off = 0;
        g.launched = false;
        let Some(key) = graph_key.filter(|_| self.graphs_enabled) else {
            g.mode = StepMode::Eager;
            return Ok(());
        };
        if g.graphs.contains_key(&key) {
            g.mode = StepMode::Replay(key);
            if g.index_writes.get(&key) == Some(&0) {
                Self::launch_graph(&g, key)?;
                g.launched = true;
            }
        } else if let Some(&need) = g.measured.get(&key) {
            if g.arena.as_ref().is_none_or(|a| a.len() < need) {
                // New arena -> addresses baked into existing graphs are stale.
                g.graphs.clear();
                g.arena = None;
                let cap = need.next_power_of_two();
                g.arena = Some(unsafe { self.stream.alloc::<u8>(cap) }.map_err(err)?);
            }
            gpu::begin_capture(&self.stream).map_err(err)?;
            g.mode = StepMode::Capture(key);
        } else {
            g.mode = StepMode::Measure(key);
        }
        Ok(())
    }

    fn alloc_kv(&self, layout: KvLayout) -> Result<CudaKvCache> {
        if layout.head_dim % WARP as usize != 0 || layout.head_dim > ATTN_MAX_D {
            return Err(Error::Unsupported(format!("head_dim {} (need a multiple of 32, <= {ATTN_MAX_D})", layout.head_dim)));
        }
        let n = layout.n_blocks * layout.block_size * layout.slot_width();
        let (bytes, scales) = match layout.dtype {
            KvType::F16 => (2 * n, 1),
            KvType::Q8 => (n, n / 32),
        };
        let mk = || (0..layout.n_layers).map(|_| self.stream.alloc_zeros::<u8>(bytes).map_err(err)).collect::<Result<Vec<_>>>();
        let mk_s = || (0..layout.n_layers).map(|_| self.stream.alloc_zeros::<u16>(scales).map_err(err)).collect::<Result<Vec<_>>>();
        Ok(CudaKvCache { layout, k: mk()?, v: mk()?, ks: mk_s()?, vs: mk_s()? })
    }

    fn alloc_host_kv(&self, layout: KvLayout, n_blocks: usize) -> Result<CudaHostKv> {
        let block_bytes = layout.block_bytes();
        let (buf, dev) = gpu::alloc_mapped_host(self.stream.context(), n_blocks.max(1) * block_bytes).map_err(err)?;
        Ok(CudaHostKv { buf, dev, block_bytes })
    }

    fn kv_to_host(&self, kv: &CudaKvCache, store: &mut CudaHostKv, dev: &[u32], host: &[u32]) -> Result<()> {
        self.kv_copy(kv, store, dev, host, true)
    }

    fn kv_from_host(&self, kv: &mut CudaKvCache, store: &CudaHostKv, host: &[u32], dev: &[u32]) -> Result<()> {
        self.kv_copy(kv, store, host, dev, false)
    }

    fn to_host(&self, buf: &CudaBuf, out: &mut [f32]) -> Result<()> {
        check_len("to_host", buf.len(), out.len())?;
        self.end_graph_step()?;
        // SAFETY: `buf` holds at least `out.len()` floats (checked above).
        unsafe { gpu::memcpy_dtoh_async(out, buf.ptr, &self.stream) }.map_err(err)?;
        self.stream.synchronize().map_err(err)
    }

    fn sample_rows(&self, logits: &CudaBuf, n_rows: usize, vocab: usize, rows: &[(u32, &RowSample)], out: &mut [u32]) -> Result<()> {
        check_len("sample_rows", logits.len(), n_rows * vocab)?;
        if rows.len() != out.len() || rows.iter().any(|&(r, _)| r as usize >= n_rows) {
            return Err(Error::Shape("sample_rows: rows/out mismatch".into()));
        }
        if rows.is_empty() {
            return Ok(());
        }
        // Eager from here: the adjustment lists vary from step to step.
        self.end_graph_step()?;
        let mut cache = self.index_cache.lock().unwrap();
        // One entry per logit: the kernel writes each entry's sum without
        // atomics, so repeated ids (a bias and a penalty) are merged here.
        let mut adj: Vec<(u32, f32)> = rows
            .iter()
            .flat_map(|&(r, s)| s.adjust.iter().filter(|(id, _)| (*id as usize) < vocab).map(move |&(id, d)| (r * vocab as u32 + id, d)))
            .collect();
        adj.sort_unstable_by_key(|a| a.0);
        adj.dedup_by(|b, a| {
            let same = a.0 == b.0;
            if same {
                a.1 += b.1;
            }
            same
        });
        let (at, delta): (Vec<u32>, Vec<u32>) = adj.into_iter().map(|(a, d)| (a, d.to_bits())).unzip();
        if !at.is_empty() {
            self.sync_index(&mut cache, "sample_adj_at", &at)?;
            self.sync_index(&mut cache, "sample_adj_d", &delta)?;
            let (a, d, n) = (&cache["sample_adj_at"].dev, &cache["sample_adj_d"].dev, at.len() as i32);
            launch!(self, "logit_adjust", grid_1d(at.len(), 256), [logits, a, d, &n])?;
        }
        let params: Vec<u32> = rows
            .iter()
            .flat_map(|&(r, s)| {
                [s.temperature.to_bits(), s.top_k, s.top_p.to_bits(), s.seed as u32, (s.seed >> 32) as u32, s.counter as u32, (s.counter >> 32) as u32, r]
            })
            .collect();
        self.sync_index(&mut cache, "sample_params", &params)?;
        // The ids come back through an index buffer of the right size.
        let ids = vec![0u32; rows.len()];
        self.sync_index(&mut cache, "sample_out", &ids)?;
        let (pp, po) = (&cache["sample_params"].dev, &cache["sample_out"].dev);
        let cfg = LaunchConfig { grid_dim: (rows.len() as u32, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 };
        let v = vocab as i32;
        launch!(self, "sample_rows", cfg, [logits, &v, pp, po])?;
        let mut host = vec![0f32; rows.len()];
        // SAFETY: `sample_out` holds at least rows.len() u32s (synced above).
        unsafe { gpu::memcpy_dtoh_async(&mut host, po.device_ptr(&self.stream).0, &self.stream) }.map_err(err)?;
        self.stream.synchronize().map_err(err)?;
        // The cached host copy no longer matches the device contents.
        cache.get_mut("sample_out").unwrap().host.clear();
        for (o, h) in out.iter_mut().zip(host) {
            *o = h.to_bits();
        }
        Ok(())
    }

    fn embed(&self, table: &CudaWeight, tokens: &[u32], out: &mut CudaBuf) -> Result<()> {
        check_len("embed", out.len(), tokens.len() * table.cols)?;
        if let Some(&bad) = tokens.iter().find(|&&t| t as usize >= table.rows) {
            return Err(Error::Shape(format!("token id {bad} out of range ({})", table.rows)));
        }
        let mut cache = self.index_cache.lock().unwrap();
        self.sync_index(&mut cache, "tokens", tokens)?;
        let toks = &cache["tokens"].dev;
        let (n_tok, k) = (tokens.len() as i32, table.cols as i32);
        let (sq, sh, ss) = table.strides();
        launch!(self, &format!("embed_{}", table.fmt.suffix()), grid_1d(tokens.len() * table.cols / 4, 256), [&table.qs, table.qh(), table.sc(), toks, out, &n_tok, &k, &sq, &sh, &ss])
    }

    fn matmul(&self, x: &CudaBuf, w: &CudaWeight, out: &mut CudaBuf, m: usize) -> Result<()> {
        self.matmul_many(x, &[w], &mut [out], m)
    }

    fn matmul_many(&self, x: &CudaBuf, ws: &[&CudaWeight], outs: &mut [&mut CudaBuf], m: usize) -> Result<()> {
        if ws.len() != outs.len() || ws.is_empty() {
            return Err(Error::Shape(format!("matmul_many: {} weights, {} outputs", ws.len(), outs.len())));
        }
        let k = ws[0].cols;
        for (w, out) in ws.iter().zip(outs.iter()) {
            if w.cols != k {
                return Err(Error::Shape(format!("matmul_many: mixed input widths {k} and {}", w.cols)));
            }
            check_len("matmul out", out.len(), m * w.rows)?;
        }
        check_len("matmul x", x.len(), m * k)?;
        if m > MATVEC_MAX_M {
            return self.matmul_many_batched(x, ws, outs, m);
        }
        let prequantized = self.take_prequant_src(x, m * k);
        let (ki, mi) = (k as i32, m as i32);

        // Float weights: f32 activations; same-format runs share one launch.
        let float_idx: Vec<usize> = (0..ws.len()).filter(|&i| !ws[i].fmt.quantized()).collect();
        let mut i = 0;
        while i < float_idx.len() {
            let fmt = ws[float_idx[i]].fmt;
            let mut group = vec![float_idx[i]];
            while group.len() < MATVEC_TARGETS && i + group.len() < float_idx.len() && ws[float_idx[i + group.len()]].fmt == fmt {
                group.push(float_idx[i + group.len()]);
            }
            i += group.len();
            let mut targets = MvTargets { count: group.len() as i32, ..Default::default() };
            // Warps per row: split k when one warp per row would leave SMs idle
            // (e.g. Gemma's 256 x 2560 per-layer projections), or always for
            // long rows where the vendor prefers it (gpu::MATVEC_SPLIT_K).
            let rows: usize = group.iter().map(|&wi| ws[wi].rows).sum();
            // (kernels.cu instantiates kw = 1 and 4 = MATVEC_WARPS only.)
            let kw = if (gpu::MATVEC_SPLIT_K || rows < self.n_sms * 8) && k >= 1024 { MATVEC_WARPS as usize } else { 1 };
            let rows_per_block = MATVEC_WARPS as usize / kw;
            let mut blocks = 0usize;
            for (slot, &wi) in group.iter().enumerate() {
                let w = ws[wi];
                targets.t[slot] = MvTarget {
                    qs: w.qs.device_ptr(&self.stream).0,
                    out: outs[wi].ptr,
                    qs_row: w.strides().0,
                    n: w.rows as i32,
                    block_start: blocks as i32,
                    ..Default::default()
                };
                blocks += w.rows.div_ceil(rows_per_block);
            }
            let name = format!("matvec_{}", fmt.suffix());
            let prof = if self.profile.enabled {
                let rows: Vec<String> = group.iter().map(|&wi| ws[wi].rows.to_string()).collect();
                format!("{name} {}x{k}", rows.join("+"))
            } else {
                String::new()
            };
            let cfg = LaunchConfig { grid_dim: (blocks as u32, 1, 1), block_dim: (MATVEC_WARPS * WARP, 1, 1), shared_mem_bytes: 0 };
            let kwi = kw as i32;
            launch!(self, name, cfg, [&targets, x, &ki, &mi, &kwi], prof = prof)?;
        }
        if ws.iter().all(|w| !w.fmt.quantized()) {
            return Ok(());
        }

        // Quantized weights: quantize x once (int8 + scale per 32, like
        // llama.cpp's q8_1), then one launch per run of same-format weights.
        // (A fused gate/up kernel may have left x's int8 form in glu_scratch.)
        let mut qscratch = self.qscratch.lock().unwrap();
        let glu_scratch = if prequantized == Some(true) { Some(self.glu_scratch.lock().unwrap()) } else { None };
        if prequantized.is_none() {
            self.quantize_activations(x, m * k, &mut qscratch)?;
        }
        let (xq, xd, _) = match &glu_scratch {
            Some(g) => &**g,
            None => &*qscratch,
        };

        let quant: Vec<usize> = (0..ws.len()).filter(|&i| ws[i].fmt.quantized()).collect();
        let mut i = 0;
        while i < quant.len() {
            let fmt = ws[quant[i]].fmt;
            let mut group = vec![quant[i]];
            while group.len() < MATVEC_TARGETS && i + group.len() < quant.len() && ws[quant[i + group.len()]].fmt == fmt {
                group.push(quant[i + group.len()]);
            }
            i += group.len();

            let mut targets = MvTargets { count: group.len() as i32, ..Default::default() };
            // Split only if every lane of the 4 warps still gets a step (mv_dot
            // steps whole units for Q4, half units otherwise).
            let whole_units = matches!(fmt, Fmt::Q4 | Fmt::Q5 | Fmt::IQ4);
            let steps = if whole_units { k / 32 } else { k / 16 };
            // (kernels.cu instantiates kw = 1, 2 and 4 on HIP.) Two warps per
            // row only for unit-stepping formats: +2% Llama-3.2-1B decode
            // (2048-wide rows), but -1% on Qwen3-0.6B Q8 (1024-wide).
            let kw = match steps / WARP as usize {
                _ if !gpu::MATVEC_SPLIT_K => 1,
                s if s >= 4 => 4,
                s if s >= 2 && whole_units => 2,
                _ => 1,
            };
            let rows_per_block = MATVEC_WARPS as usize / kw;
            let mut blocks = 0usize;
            for (slot, &wi) in group.iter().enumerate() {
                let w = ws[wi];
                let (sq, sh, ss) = w.strides();
                targets.t[slot] = MvTarget {
                    qs: w.qs.device_ptr(&self.stream).0,
                    qh: w.qh().device_ptr(&self.stream).0,
                    sc: w.sc().device_ptr(&self.stream).0,
                    out: outs[wi].ptr,
                    qs_row: sq,
                    qh_row: sh,
                    sc_row: ss,
                    n: w.rows as i32,
                    block_start: blocks as i32,
                };
                blocks += w.rows.div_ceil(rows_per_block);
            }
            let name = format!("matvec_{}", fmt.suffix());
            let prof = if self.profile.enabled {
                let rows: Vec<String> = group.iter().map(|&wi| ws[wi].rows.to_string()).collect();
                format!("{name} {}x{k}", rows.join("+"))
            } else {
                String::new()
            };
            let cfg = LaunchConfig { grid_dim: (blocks as u32, 1, 1), block_dim: (MATVEC_WARPS * WARP, 1, 1), shared_mem_bytes: 0 };
            let kwi = kw as i32;
            launch!(self, name, cfg, [&targets, xq, xd, &ki, &mi, &kwi], prof = prof)?;
        }
        Ok(())
    }

    fn gated_ffn(
        &self,
        x: &CudaBuf,
        gate: &CudaWeight,
        up: &CudaWeight,
        down: &CudaWeight,
        act: Activation,
        out: &mut CudaBuf,
        gate_buf: &mut CudaBuf,
        up_buf: &mut CudaBuf,
        m: usize,
        n_ffn: usize,
    ) -> Result<()> {
        // The down projection reads only the int8 copy when it takes the int8
        // path (decode matvec or prefill MMQ); then the f32 activation is never
        // read and need not be written.
        let int8_down = down.fmt.quantized() && (m <= MATVEC_MAX_M || self.uses_mmq(down));
        if int8_down && self.glu_mmq(x, gate, up, act, gate_buf, m, false)? {
            return self.matmul(gate_buf, down, out, m);
        }
        // Decode with float gate/up: one launch for both matvecs and the activation.
        if m <= MATVEC_MAX_M
            && !gate.fmt.quantized()
            && gate.fmt == up.fmt
            && (gate.rows, gate.cols) == (up.rows, up.cols)
            && gate.cols % 8 == 0
            && !FLAGS.no_glu_mmq
        {
            let (n, k) = (gate.rows, gate.cols);
            check_len("gated_ffn x", x.len(), m * k)?;
            check_len("gated_ffn gate", gate_buf.len(), m * n)?;
            let target = |w: &CudaWeight| MvTarget {
                qs: w.qs.device_ptr(&self.stream).0,
                qs_row: w.strides().0,
                n: n as i32,
                ..Default::default()
            };
            let (tg, tu) = (target(gate), target(up));
            let a = match act {
                Activation::Silu => "silu",
                Activation::GeluTanh => "gelu",
            };
            let name = format!("matvec_glu_{a}_{}", gate.fmt.suffix());
            let prof = if self.profile.enabled { format!("{name} {n}x{k}") } else { String::new() };
            // Rows split over warps as in matvec_many's float path.
            let kw = if (gpu::MATVEC_SPLIT_K || n < self.n_sms * 8) && k >= 1024 { MATVEC_WARPS as usize } else { 1 };
            let cfg = LaunchConfig {
                grid_dim: (n.div_ceil(MATVEC_WARPS as usize / kw) as u32, 1, 1),
                block_dim: (MATVEC_WARPS * WARP, 1, 1),
                shared_mem_bytes: 0,
            };
            let (ki, mi, kwi) = (k as i32, m as i32, kw as i32);
            launch!(self, &name, cfg, [&tg, &tu, x, &ki, &mi, &mut *gate_buf, &kwi], prof = prof)?;
            return self.matmul(gate_buf, down, out, m);
        }
        self.matmul_many(x, &[gate, up], &mut [&mut *gate_buf, &mut *up_buf], m)?;
        let len = m * n_ffn;
        check_len("gated_ffn act", gate_buf.len().min(up_buf.len()), len)?;
        let qo = self.qout(len)?;
        let write_out = (qo.on == 0 || !int8_down) as i32;
        let n = len as i64;
        let kern = match act {
            Activation::Silu => "silu_mul",
            Activation::GeluTanh => "gelu_mul",
        };
        launch!(self, kern, grid_1d(len, 256), [&*gate_buf, &*up_buf, &n, &qo, &write_out])?;
        self.set_prequant(&qo, gate_buf.ptr, len);
        self.matmul(gate_buf, down, out, m)
    }

    fn rms_norm(&self, x: &CudaBuf, w: &CudaWeight, out: &mut CudaBuf, rows: usize, eps: f32) -> Result<()> {
        let wd = w.dense.as_ref().ok_or_else(|| Error::Shape("rms_norm weight must be 1-D".into()))?;
        let d = w.cols;
        check_len("rms_norm x", x.len(), rows * d)?;
        check_len("rms_norm out", out.len(), rows * d)?;
        let threads = (d as u32).next_power_of_two().clamp(WARP, 1024);
        let cfg = LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
        let di = d as i32;
        let qo = self.qout(rows * d)?;
        launch!(self, "rms_norm", cfg, [x, wd, out, &di, &eps, &qo])?;
        self.set_prequant(&qo, out.ptr, rows * d);
        Ok(())
    }

    fn rope(
        &self,
        x: &mut CudaBuf,
        positions: &[u32],
        n_heads: usize,
        head_dim: usize,
        theta: f32,
        freq_factors: Option<&CudaWeight>,
    ) -> Result<()> {
        check_len("rope", x.len(), positions.len() * n_heads * head_dim)?;
        let factors = match freq_factors {
            Some(w) => {
                let f = w.dense.as_ref().ok_or_else(|| Error::Shape("rope freq factors must be 1-D".into()))?;
                check_len("rope freq factors", f.len(), head_dim / 2)?;
                Some(f)
            }
            None => None,
        };
        let mut cache = self.index_cache.lock().unwrap();
        self.sync_index(&mut cache, "positions", positions)?;
        let pos = &cache["positions"].dev;
        let (nt, nh, hd) = (positions.len() as i32, n_heads as i32, head_dim as i32);
        let has = factors.is_some() as i32;
        // Raw pointer so "no factors" can be null (the kernel never reads it then).
        // Event tracking is off, so dropping the SyncOnDrop guard right away is fine.
        let fptr: u64 = factors.map_or(0, |f| f.device_ptr(&self.stream).0);
        launch!(self, "rope_neox", grid_1d(positions.len() * n_heads * head_dim / 2, 256), [x, pos, &nt, &nh, &hd, &theta, &fptr, &has])
    }

    fn add_rms_norm(
        &self,
        x: &mut CudaBuf,
        y: &CudaBuf,
        w: &CudaWeight,
        out: &mut CudaBuf,
        rows: usize,
        width: usize,
        eps: f32,
    ) -> Result<()> {
        let wd = w.dense.as_ref().ok_or_else(|| Error::Shape("rms_norm weight must be 1-D".into()))?;
        check_len("add_rms_norm weight", wd.len(), width)?;
        check_len("add_rms_norm x", x.len().min(y.len()), rows * width)?;
        check_len("add_rms_norm out", out.len(), rows * width)?;
        let threads = (width as u32).next_power_of_two().clamp(WARP, 1024);
        let cfg = LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
        let di = width as i32;
        let qo = self.qout(rows * width)?;
        launch!(self, "add_rms_norm", cfg, [x, y, wd, out, &di, &eps, &qo])?;
        self.set_prequant(&qo, out.ptr, rows * width);
        Ok(())
    }

    fn rms_norm_add(
        &self,
        src: &CudaBuf,
        w: &CudaWeight,
        x: &mut CudaBuf,
        rows: usize,
        width: usize,
        eps: f32,
        scale: f32,
        next: Option<(&CudaWeight, &mut CudaBuf, &[&CudaWeight])>,
    ) -> Result<()> {
        let wd = w.dense.as_ref().ok_or_else(|| Error::Shape("rms_norm weight must be 1-D".into()))?;
        check_len("rms_norm_add weight", wd.len(), width)?;
        check_len("rms_norm_add src", src.len().min(x.len()), rows * width)?;
        // Each thread keeps up to RNA_MAX (kernels.cu) elements in registers.
        const RNA_MAX: usize = 8;
        if width > RNA_MAX * 1024 {
            return Err(Error::Shape(format!("rms_norm_add: width {width} too large")));
        }
        // Few rows (decode): latency-bound, so spread each row over more threads.
        let min_threads = if rows < self.n_sms { 1024 } else { 128 };
        let threads = (width.div_ceil(RNA_MAX) as u32).next_multiple_of(WARP).max(min_threads).min(1024);
        let cfg = LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
        let di = width as i32;
        match next {
            Some((w2, out, consumers)) => {
                let wd2 = w2.dense.as_ref().ok_or_else(|| Error::Shape("rms_norm weight must be 1-D".into()))?;
                check_len("rms_norm_add next", out.len(), rows * width)?;
                let has = 1i32;
                let qo = if width % 32 == 0 { self.qout(rows * width)? } else { QOut::default() };
                // Skip the f32 output when every consumer reads the int8 copy:
                // decode matvec / prefill MMQ on quantized weights.
                let int8_only = qo.on != 0
                    && !consumers.is_empty()
                    && consumers.iter().all(|c| c.fmt.quantized())
                    && (rows <= MATVEC_MAX_M || consumers.iter().any(|c| self.uses_mmq(c)));
                let write_out = (!int8_only) as i32;
                let xin = x.ptr;
                launch!(self, "rms_norm_add", cfg, [src, wd, &xin, x, wd2, out, &di, &eps, &scale, &has, &qo, &write_out])?;
                self.set_prequant(&qo, out.ptr, rows * width);
                Ok(())
            }
            None => {
                // w2/out unused; pass valid placeholders.
                let has = 0i32;
                let xp = x.ptr;
                let qo = QOut::default();
                let write_out = 1i32;
                launch!(self, "rms_norm_add", cfg, [src, wd, &xp, x, wd, &xp, &di, &eps, &scale, &has, &qo, &write_out])
            }
        }
    }

    fn rms_norm_rope(
        &self,
        x: &CudaBuf,
        w: &CudaWeight,
        out: &mut CudaBuf,
        positions: &[u32],
        n_heads: usize,
        head_dim: usize,
        eps: f32,
        theta: f32,
        freq_factors: Option<&CudaWeight>,
    ) -> Result<()> {
        let wd = w.dense.as_ref().ok_or_else(|| Error::Shape("rms_norm weight must be 1-D".into()))?;
        check_len("rms_norm_rope weight", wd.len(), head_dim)?;
        let rows = positions.len() * n_heads;
        check_len("rms_norm_rope x", x.len(), rows * head_dim)?;
        check_len("rms_norm_rope out", out.len(), rows * head_dim)?;
        let factors = match freq_factors {
            Some(f) => {
                let f = f.dense.as_ref().ok_or_else(|| Error::Shape("rope freq factors must be 1-D".into()))?;
                check_len("rope freq factors", f.len(), head_dim / 2)?;
                Some(f)
            }
            None => None,
        };
        let mut cache = self.index_cache.lock().unwrap();
        self.sync_index(&mut cache, "positions", positions)?;
        let pos = &cache["positions"].dev;
        let threads = ((head_dim / 2) as u32).next_multiple_of(WARP);
        let cfg = LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
        let (nh, hd, has) = (n_heads as i32, head_dim as i32, factors.is_some() as i32);
        let fptr: u64 = factors.map_or(0, |f| f.device_ptr(&self.stream).0);
        launch!(self, "rms_norm_rope", cfg, [x, wd, out, pos, &nh, &hd, &eps, &theta, &fptr, &has])
    }

    fn qkv_norm_rope_store(
        &self,
        q: &CudaBuf,
        k: &CudaBuf,
        v: &CudaBuf,
        qk_norm: Option<(&CudaWeight, &CudaWeight)>,
        v_norm: Option<&CudaWeight>,
        qn: &mut CudaBuf,
        kv: &mut CudaKvCache,
        layer: usize,
        positions: &[u32],
        slots: &[u32],
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        eps: f32,
        theta: f32,
        freq_factors: Option<&CudaWeight>,
    ) -> Result<()> {
        fn dense(w: &CudaWeight) -> Result<&gpu::Slice<f32>> {
            w.dense.as_ref().ok_or_else(|| Error::Shape("norm weight must be 1-D".into()))
        }
        // Without norms the kernel never reads qw/kw: any valid pointer will do.
        let (qw, kw) = match qk_norm {
            Some((q_norm, k_norm)) => (dense(q_norm)?, dense(k_norm)?),
            None => (&self.ones_placeholder, &self.ones_placeholder),
        };
        if qk_norm.is_some() {
            check_len("qkv q_norm", qw.len(), head_dim)?;
            check_len("qkv k_norm", kw.len(), head_dim)?;
        }
        let n = positions.len();
        check_len("qkv q", q.len().min(qn.len()), n * n_heads * head_dim)?;
        check_len("qkv k", k.len().min(v.len()), n * n_kv_heads * head_dim)?;
        if slots.len() != n || kv.layout.n_kv_heads != n_kv_heads || kv.layout.head_dim != head_dim {
            return Err(Error::Shape("qkv_norm_rope_store: slots/cache layout mismatch".into()));
        }
        let n_slots = kv.layout.n_blocks * kv.layout.block_size;
        if let Some(&bad) = slots.iter().find(|&&s| s as usize >= n_slots) {
            return Err(Error::Shape(format!("kv slot {bad} out of range ({n_slots})")));
        }
        let vw = match v_norm {
            Some(w) => Some(dense(w)?),
            None => None,
        };
        let factors = match freq_factors {
            Some(f) => Some(dense(f)?),
            None => None,
        };
        let mut cache = self.index_cache.lock().unwrap();
        self.sync_index(&mut cache, "positions", positions)?;
        self.sync_index(&mut cache, "slots", slots)?;
        let (pos, sl) = (&cache["positions"].dev, &cache["slots"].dev);
        let threads = ((head_dim / 2) as u32).next_multiple_of(WARP);
        let cfg = LaunchConfig {
            grid_dim: ((n * (n_heads + n_kv_heads)) as u32, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        // Unused pointers get valid placeholders.
        let vwp: u64 = vw.map_or(qw.device_ptr(&self.stream).0, |w| w.device_ptr(&self.stream).0);
        let fptr: u64 = factors.map_or(0, |f| f.device_ptr(&self.stream).0);
        let (hv, hf, hn) = (vw.is_some() as i32, factors.is_some() as i32, qk_norm.is_some() as i32);
        let (nt, nh, nkv, hd) = (n as i32, n_heads as i32, n_kv_heads as i32, head_dim as i32);
        let q8 = (kv.layout.dtype == KvType::Q8) as i32;
        let (kc, vc, ks, vs) = (&mut kv.k[layer], &mut kv.v[layer], &mut kv.ks[layer], &mut kv.vs[layer]);
        launch!(self, "qkv_norm_rope_store", cfg, [q, k, v, qw, kw, &hn, &vwp, &hv, qn, kc, vc, pos, sl, &nt, &nh, &nkv, &hd, &eps, &theta, &fptr, &hf, ks, vs, &q8])
    }

    fn silu_mul(&self, gate: &mut CudaBuf, up: &CudaBuf, len: usize) -> Result<()> {
        check_len("silu_mul", gate.len().min(up.len()), len)?;
        let n = len as i64;
        let qo = self.qout(len)?;
        let write_out = 1i32;
        launch!(self, "silu_mul", grid_1d(len, 256), [gate, up, &n, &qo, &write_out])?;
        self.set_prequant(&qo, gate.ptr, len);
        Ok(())
    }

    fn gelu_mul(&self, gate: &mut CudaBuf, up: &CudaBuf, len: usize) -> Result<()> {
        check_len("gelu_mul", gate.len().min(up.len()), len)?;
        let n = len as i64;
        let qo = self.qout(len)?;
        let write_out = 1i32;
        launch!(self, "gelu_mul", grid_1d(len, 256), [gate, up, &n, &qo, &write_out])?;
        self.set_prequant(&qo, gate.ptr, len);
        Ok(())
    }

    fn gelu_mul_strided(
        &self,
        gate: &mut CudaBuf,
        x: &CudaBuf,
        first: usize,
        stride: usize,
        count: usize,
        width: usize,
        _scratch: &mut CudaBuf,
    ) -> Result<()> {
        let len = count * width;
        check_len("gelu_mul_strided gate", gate.len(), len)?;
        if count > 0 {
            check_len("gelu_mul_strided x", x.len(), (first + (count - 1) * stride + 1) * width)?;
        }
        let (f, st, c, w) = (first as i32, stride as i32, count as i32, width as i32);
        launch!(self, "gelu_mul_strided", grid_1d(len, 256), [gate, x, &f, &st, &c, &w])
    }

    #[allow(clippy::too_many_arguments)]
    fn norm_add_gated_ffn(
        &self,
        src: &CudaBuf,
        w1: &CudaWeight,
        x_in: &CudaBuf,
        x_out: &mut CudaBuf,
        rows: usize,
        width: usize,
        eps: f32,
        scale: f32,
        w2: &CudaWeight,
        ffn: (&CudaWeight, &CudaWeight, &CudaWeight, Activation),
        normed: &mut CudaBuf,
        out: &mut CudaBuf,
        gate_buf: &mut CudaBuf,
        up_buf: &mut CudaBuf,
        n_ffn: usize,
    ) -> Result<()> {
        let (gate, up, down, act) = ffn;
        let (wd1, wd2) = match (w1.dense.as_ref(), w2.dense.as_ref()) {
            (Some(a), Some(b)) => (a, b),
            _ => return Err(Error::Shape("rms_norm weight must be 1-D".into())),
        };
        check_len("norm_add_gated_ffn", src.len().min(x_in.len()).min(x_out.len()).min(normed.len()), rows * width)?;
        // One decode row with quantized gate/up and an int8-reading down: one
        // fused launch for the norms and gate/up/activation (matvec_glu_norm).
        const GLU_NORM_MAX_D: usize = 4096; // kernels.cu
        let fused = rows == 1
            && width % 32 == 0
            && width <= GLU_NORM_MAX_D
            && gate.fmt.quantized()
            && gate.fmt == up.fmt
            && (gate.rows, gate.cols) == (up.rows, up.cols)
            && (gate.rows, gate.cols) == (n_ffn, width)
            && n_ffn % 32 == 0
            && down.fmt.quantized()
            && !FLAGS.no_glu_mmq;
        if fused {
            // The int8 product goes to glu_scratch; it must not move under a graph.
            let mut gs = self.glu_scratch.lock().unwrap();
            if gs.0.len() < n_ffn {
                if matches!(self.mode(), StepMode::Capture(_) | StepMode::Replay(_)) {
                    drop(gs);
                    return self.norm_add_gated_ffn_split(src, wd1, x_in, x_out, rows, width, eps, scale, w2, (gate, up, down, act), normed, out, gate_buf, up_buf, n_ffn);
                }
                gs.0 = unsafe { self.stream.alloc::<i8>(n_ffn) }.map_err(err)?;
                gs.1 = unsafe { self.stream.alloc::<f32>(n_ffn / 32) }.map_err(err)?;
                gs.2 = unsafe { self.stream.alloc::<f32>(n_ffn / 32) }.map_err(err)?;
                self.invalidate_graphs();
            }
            let qo = QOut {
                q: gs.0.device_ptr(&self.stream).0,
                d: gs.1.device_ptr(&self.stream).0,
                s: gs.2.device_ptr(&self.stream).0,
                on: 1,
            };
            drop(gs);
            let target = |w: &CudaWeight| {
                let (sq, sh, ss) = w.strides();
                MvTarget {
                    qs: w.qs.device_ptr(&self.stream).0,
                    qh: w.qh().device_ptr(&self.stream).0,
                    sc: w.sc().device_ptr(&self.stream).0,
                    out: 0,
                    qs_row: sq,
                    qh_row: sh,
                    sc_row: ss,
                    n: n_ffn as i32,
                    block_start: 0,
                }
            };
            let (tg, tu) = (target(gate), target(up));
            let a = match act {
                Activation::Silu => "silu",
                Activation::GeluTanh => "gelu",
            };
            let name = format!("matvec_glu_norm_{a}_{}", gate.fmt.suffix());
            let prof = if self.profile.enabled { format!("{name} {n_ffn}x{width}") } else { String::new() };
            let cfg = LaunchConfig { grid_dim: ((n_ffn / 32) as u32, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 };
            let di = width as i32;
            launch!(self, &name, cfg, [&tg, &tu, src, wd1, x_in, x_out, wd2, &di, &eps, &scale, &qo], prof = prof)?;
            *self.prequant.lock().unwrap() = Some((gate_buf.ptr, n_ffn, true));
            return self.matmul(gate_buf, down, out, rows);
        }
        self.norm_add_gated_ffn_split(src, wd1, x_in, x_out, rows, width, eps, scale, w2, (gate, up, down, act), normed, out, gate_buf, up_buf, n_ffn)
    }

    #[allow(clippy::too_many_arguments)]
    fn ple_gate(
        &self,
        src: &CudaBuf,
        w: &CudaWeight,
        x_in: &CudaBuf,
        x_out: &mut CudaBuf,
        rows: usize,
        width: usize,
        eps: f32,
        gate_w: &CudaWeight,
        inputs: &CudaBuf,
        layer: usize,
        n_layers: usize,
        p_dim: usize,
        gate: &mut CudaBuf,
        scratch: &mut CudaBuf,
    ) -> Result<()> {
        // One decode row with an F32 gate: one fused kernel (see ple_gate_mv;
        // it launches 128 threads and loads with float4).
        const PLE_MAX_D: usize = 4096; // kernels.cu
        if rows != 1 || gate_w.fmt != Fmt::F32 || width > PLE_MAX_D || width % 8 != 0 || gate_w.cols != width || gate_w.rows != p_dim {
            self.gather_strided(x_in, 0, 1, rows, width, x_out)?;
            self.rms_norm_add(src, w, x_out, rows, width, eps, 1.0, None)?;
            self.matmul(x_out, gate_w, gate, rows)?;
            return self.gelu_mul_strided(gate, inputs, layer, n_layers, rows, p_dim, scratch);
        }
        let wd = w.dense.as_ref().ok_or_else(|| Error::Shape("rms_norm weight must be 1-D".into()))?;
        check_len("ple_gate src", src.len().min(x_in.len()).min(x_out.len()), width)?;
        check_len("ple_gate inputs", inputs.len(), n_layers * p_dim)?;
        check_len("ple_gate gate", gate.len(), p_dim)?;
        let cfg = LaunchConfig { grid_dim: (p_dim as u32, 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
        let (rb, di, li, pi) = (gate_w.strides().0, width as i32, layer as i32, p_dim as i32);
        launch!(self, "ple_gate_f32", cfg, [src, wd, x_in, x_out, &gate_w.qs, &rb, inputs, gate, &di, &eps, &li, &pi])
    }

    fn gelu(&self, x: &mut CudaBuf, len: usize) -> Result<()> {
        check_len("gelu", x.len(), len)?;
        let n = len as i64;
        launch!(self, "gelu", grid_1d(len, 256), [x, &n])
    }

    fn mul(&self, x: &mut CudaBuf, y: &CudaBuf, len: usize) -> Result<()> {
        check_len("mul", x.len().min(y.len()), len)?;
        let n = len as i64;
        launch!(self, "mul_inplace", grid_1d(len, 256), [x, y, &n])
    }

    fn scale(&self, x: &mut CudaBuf, s: f32, len: usize) -> Result<()> {
        check_len("scale", x.len(), len)?;
        let n = len as i64;
        launch!(self, "scale_inplace", grid_1d(len, 256), [x, &s, &n])
    }

    fn softcap(&self, x: &mut CudaBuf, cap: f32, len: usize) -> Result<()> {
        check_len("softcap", x.len(), len)?;
        let n = len as i64;
        launch!(self, "softcap_inplace", grid_1d(len, 256), [x, &cap, &n])
    }

    fn add(&self, x: &mut CudaBuf, y: &CudaBuf, len: usize) -> Result<()> {
        check_len("add", x.len().min(y.len()), len)?;
        let n = len as i64;
        launch!(self, "add_inplace", grid_1d(len, 256), [x, y, &n])
    }

    fn gather_rows(&self, x: &CudaBuf, rows: &[u32], width: usize, out: &mut CudaBuf) -> Result<()> {
        check_len("gather_rows out", out.len(), rows.len() * width)?;
        if let Some(&max) = rows.iter().max() {
            check_len("gather_rows x", x.len(), (max as usize + 1) * width)?;
        }
        let mut cache = self.index_cache.lock().unwrap();
        self.sync_index(&mut cache, "gather_rows", rows)?;
        let r = &cache["gather_rows"].dev;
        let (nr, w) = (rows.len() as i32, width as i32);
        launch!(self, "gather_rows", grid_1d(rows.len() * width, 256), [x, r, out, &nr, &w])
    }

    fn gather_strided(
        &self,
        x: &CudaBuf,
        first: usize,
        stride: usize,
        count: usize,
        width: usize,
        out: &mut CudaBuf,
    ) -> Result<()> {
        check_len("gather_strided out", out.len(), count * width)?;
        if count > 0 {
            check_len("gather_strided x", x.len(), (first + (count - 1) * stride + 1) * width)?;
        }
        let (f, st, c, w) = (first as i32, stride as i32, count as i32, width as i32);
        launch!(self, "gather_strided", grid_1d(count * width, 256), [x, out, &f, &st, &c, &w])
    }

    fn kv_store(&self, kv: &mut CudaKvCache, layer: usize, k: &CudaBuf, v: &CudaBuf, slots: &[u32]) -> Result<()> {
        let w = kv.layout.slot_width();
        let n_slots = kv.layout.n_blocks * kv.layout.block_size;
        check_len("kv_store k", k.len(), slots.len() * w)?;
        check_len("kv_store v", v.len(), slots.len() * w)?;
        if let Some(&bad) = slots.iter().find(|&&s| s as usize >= n_slots) {
            return Err(Error::Shape(format!("kv slot {bad} out of range ({n_slots})")));
        }
        let mut cache = self.index_cache.lock().unwrap();
        self.sync_index(&mut cache, "slots", slots)?;
        let s = &cache["slots"].dev;
        let (nt, wi) = (slots.len() as i32, w as i32);
        let (kc, vc) = (&mut kv.k[layer], &mut kv.v[layer]);
        match kv.layout.dtype {
            KvType::F16 => launch!(self, "kv_store", grid_1d(slots.len() * w, 256), [k, v, s, kc, vc, &nt, &wi]),
            KvType::Q8 => {
                let (ks, vs) = (&mut kv.ks[layer], &mut kv.vs[layer]);
                let (nkv, hd) = (kv.layout.n_kv_heads as i32, kv.layout.head_dim as i32);
                let cfg = LaunchConfig {
                    grid_dim: ((slots.len() * kv.layout.n_kv_heads) as u32, 1, 1),
                    block_dim: ((kv.layout.head_dim / 2).next_multiple_of(WARP as usize) as u32, 1, 1),
                    shared_mem_bytes: 0,
                };
                launch!(self, "kv_store_q8", cfg, [k, v, s, kc, ks, vc, vs, &nkv, &hd])
            }
        }
    }

    fn attention(&self, q: &CudaBuf, kv: &CudaKvCache, layer: usize, meta: &AttnMeta, out: &mut CudaBuf) -> Result<()> {
        let n_tok = meta.positions.len();
        let w = meta.n_heads * meta.head_dim;
        check_len("attention q", q.len(), n_tok * w)?;
        check_len("attention out", out.len(), n_tok * w)?;
        if meta.n_heads % meta.n_kv_heads != 0 {
            return Err(Error::Shape(format!("{} heads not divisible by {} kv heads", meta.n_heads, meta.n_kv_heads)));
        }
        // Block tables padded to a [n_seq, max_blocks] matrix.
        // Padded to a power of two so decode steps keep the same shape (graph key)
        // while the context grows.
        let max_blocks = meta.block_tables.iter().map(Vec::len).max().unwrap_or(0).max(1).next_power_of_two();
        let mut tables = vec![0u32; meta.block_tables.len() * max_blocks];
        for (row, t) in tables.chunks_exact_mut(max_blocks).zip(meta.block_tables) {
            row[..t.len()].copy_from_slice(t);
        }
        let mut cache = self.index_cache.lock().unwrap();
        self.sync_index(&mut cache, "positions", meta.positions)?;
        self.sync_index(&mut cache, "token_seq", meta.token_seq)?;
        self.sync_index(&mut cache, "block_tables", &tables)?;
        let (mb, nh, nkv, hd, bs) =
            (max_blocks as i32, meta.n_heads as i32, meta.n_kv_heads as i32, meta.head_dim as i32, meta.block_size as i32);
        let window = meta.window as i32;

        // Prefill: tile runs of consecutive tokens from the same sequence and
        // share K/V loads across them (and across the GQA group).
        let group = meta.n_heads / meta.n_kv_heads;
        let hdim = meta.head_dim;
        let build_tiles = |tile_tokens: usize| {
            let mut tiles: Vec<u32> = Vec::new();
            let mut t = 0;
            while t < n_tok {
                let mut c = 1;
                while c < tile_tokens
                    && t + c < n_tok
                    && meta.token_seq[t + c] == meta.token_seq[t]
                    && meta.positions[t + c] == meta.positions[t + c - 1] + 1
                {
                    c += 1;
                }
                tiles.extend([t as u32, c as u32]);
                t += c;
            }
            tiles
        };
        let is_prefill = (1..n_tok).any(|i| meta.token_seq[i] == meta.token_seq[i - 1]);

        // A q8 cache: the prefill kernels read f16. Small steps (decode,
        // speculative verification; CUDA-graph sized) read the int8 cache
        // with the decode kernels, which handle any mix of tokens. Larger
        // prefill steps expand the blocks they attend to into f16 staging
        // buffers and run the f16 kernels on a remapped block table.
        let q8 = kv.layout.dtype == KvType::Q8;
        let staged = q8 && is_prefill && n_tok > MATVEC_MAX_M;
        // Decode attention warps: 8, or fewer where that needs more shared
        // memory than the GPU has (512-wide heads on AMD's 64 KB: 8 warps
        // take 74 KB, and the generic fallback ran Nemotron decode at 16k
        // context at 5 tok/s). The kernel takes its warp count from the launch.
        let ad_smem_for = |warps: usize| (group * hdim + warps * group * 2 + warps * group * hdim) * 4;
        let ad_warps = [8, 4, 2].into_iter().find(|&w| ad_smem_for(w) <= self.max_smem).unwrap_or(8);
        let ad_smem = ad_smem_for(ad_warps);
        let decode_kernel = (!is_prefill || n_tok <= MATVEC_MAX_M) && ATTN_DECODE_SHAPES.contains(&(group, hdim)) && ad_smem <= self.max_smem;
        // Keys in a q8 cache are rotated (see kv_rotate_q in kernels.cu):
        // queries get the same rotation (the decode kernel does its own).
        let q_rot = if q8 && !decode_kernel {
            drop(cache);
            let mut r = self.alloc(n_tok * w)?;
            let hd = meta.head_dim as i32;
            let cfg = LaunchConfig {
                grid_dim: ((n_tok * meta.n_heads) as u32, 1, 1),
                block_dim: ((meta.head_dim / 2).next_multiple_of(WARP as usize) as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            launch!(self, "kv_rotate_q", cfg, [q, &mut r, &hd])?;
            cache = self.index_cache.lock().unwrap();
            Some(r)
        } else {
            None
        };
        let q = q_rot.as_ref().unwrap_or(q);
        let mut stage = if staged { Some(self.kv_stage.lock().unwrap()) } else { None };
        if let Some(stage) = &mut stage {
            let bs = meta.block_size;
            // Positions each sequence attends to in this step.
            let mut range = vec![(u32::MAX, 0u32); meta.block_tables.len()];
            for (&sq, &pos) in meta.token_seq.iter().zip(meta.positions) {
                let lo = if meta.window > 0 { (pos + 1).saturating_sub(meta.window as u32) } else { 0 };
                let r = &mut range[sq as usize];
                *r = (r.0.min(lo), r.1.max(pos));
            }
            let mut list: Vec<u32> = Vec::new();
            let mut index: HashMap<u32, u32> = HashMap::new();
            let mut staged_tables = vec![0u32; tables.len()];
            for (sq, &(lo, hi)) in range.iter().enumerate() {
                if lo > hi {
                    continue;
                }
                for b in lo as usize / bs..=hi as usize / bs {
                    let blk = tables[sq * max_blocks + b];
                    staged_tables[sq * max_blocks + b] = *index.entry(blk).or_insert_with(|| {
                        list.push(blk);
                        list.len() as u32 - 1
                    });
                }
            }
            self.sync_index(&mut cache, "stage_blocks", &list)?;
            self.sync_index(&mut cache, "stage_tables", &staged_tables)?;
            let per_block = bs * kv.layout.slot_width();
            let need = list.len() * per_block * 2;
            if stage.0.len() < need {
                stage.0 = unsafe { self.stream.alloc::<u8>(need) }.map_err(err)?;
                stage.1 = unsafe { self.stream.alloc::<u8>(need) }.map_err(err)?;
            }
            let (nl, pb) = (list.len() as i32, per_block as i64);
            let (ko, vo) = { let st = &mut **stage; (&mut st.0, &mut st.1) };
            let blocks = &cache["stage_blocks"].dev;
            let (kc, vc, ks, vs) = (&kv.k[layer], &kv.v[layer], &kv.ks[layer], &kv.vs[layer]);
            launch!(self, "kv_dequant_blocks", grid_1d(list.len() * per_block / 8, 256), [kc, ks, vc, vs, ko, vo, blocks, &nl, &pb])?;
        }
        // The cache the prefill kernels read, and its block tables.
        let (pkc, pvc, tab_key) = match &stage {
            Some(st) => (&st.0, &st.1, "stage_tables"),
            None => (&kv.k[layer], &kv.v[layer], "block_tables"),
        };
        let is_prefill = is_prefill && !decode_kernel && (!q8 || staged);

        // FlashAttention-2 style kernel (cp.async ring, ldmatrix), 4 warps.
        if is_prefill && self.tensor_cores && FA_HEAD_DIMS.contains(&hdim) && !FLAGS.no_fa {
            let (dw, rg, _bk, smem) = fa_cfg(hdim);
            let rows = rg * 16;
            if group <= rows && smem <= self.max_smem {
                let tile_tokens = rows / group;
                let tiles = build_tiles(tile_tokens);
                self.sync_index(&mut cache, "attn_tiles", &tiles)?;
                let cfg = LaunchConfig {
                    grid_dim: ((tiles.len() / 2) as u32, meta.n_kv_heads as u32, 1),
                    block_dim: ((rg * dw) as u32 * WARP, 1, 1),
                    shared_mem_bytes: smem as u32,
                };
                let tt = tile_tokens as i32;
                let (kc, vc) = (pkc, pvc);
                let name = format!("attn_fa_d{hdim}");
                let (pos, seq, tab, til) =
                    (&cache["positions"].dev, &cache["token_seq"].dev, &cache[tab_key].dev, &cache["attn_tiles"].dev);
                // int8 copy of the output for the output projection (when a
                // quantized consumer will use it; see fuse_quant_useful). The
                // epilogue work only pays off on big GPUs: measured +2% Nemotron
                // prefill on the RTX 6000 Ada (142 SMs), -1.2% on the 46-SM laptop
                // GPU, where the separate quantize launch stays cheaper.
                let fuse = w % 32 == 0 && FLAGS.fa_quant.unwrap_or(self.n_sms >= FA_FUSED_QUANT_MIN_SMS);
                let qo = if fuse { self.qout(n_tok * w)? } else { QOut::default() };
                launch!(self, name, cfg, [q, kc, vc, out, til, seq, pos, tab, &mb, &nh, &nkv, &bs, &meta.scale, &window, &tt, &qo])?;
                self.set_prequant(&qo, out.ptr, n_tok * w);
                return Ok(());
            }
        }

        // RDNA3 matrix cores: 4 waves of 16 query rows.
        let (wa_rows, smem) = wa_cfg(hdim);
        if is_prefill && self.wmma && WA_HEAD_DIMS.contains(&hdim) && group <= wa_rows && !FLAGS.no_fa {
            if smem <= self.max_smem {
                let tile_tokens = wa_rows / group;
                let tiles = build_tiles(tile_tokens);
                self.sync_index(&mut cache, "attn_tiles", &tiles)?;
                let (pos, seq, tab, til) =
                    (&cache["positions"].dev, &cache["token_seq"].dev, &cache[tab_key].dev, &cache["attn_tiles"].dev);
                let cfg = LaunchConfig {
                    grid_dim: ((tiles.len() / 2) as u32, meta.n_kv_heads as u32, 1),
                    block_dim: (8 * WARP, 1, 1),
                    shared_mem_bytes: smem as u32,
                };
                let tt = tile_tokens as i32;
                let (kc, vc) = (pkc, pvc);
                let name = format!("attn_wmma_d{hdim}");
                return launch!(self, name, cfg, [q, kc, vc, out, til, seq, pos, tab, &mb, &nh, &nkv, &bs, &meta.scale, &window, &tt]);
            }
        }

        // Tensor-core path: 4 warps; head_dim split evenly over `dw` warps,
        // each a whole number of 16-dim MMA steps, <= 128 dims (320-dim heads
        // take 4 x 80; 352, 416 and 480 have no such split up to 4 warps).
        let tc_dw = (hdim.div_ceil(128).max(1)..=4).find(|&dw| hdim % dw == 0 && (hdim / dw) % 16 == 0);
        if is_prefill && self.tensor_cores && !FLAGS.no_tc_attn && let Some(dw) = tc_dw {
            // 4 row groups (64 query rows) per block so each K/V tile is reused
            // widely; big heads split dims across `dw` warps (up to 16 warps).
            let smem_for = |rg: usize| 32 * (hdim + 8) * 2 + hdim * 40 * 2 + if dw > 1 { rg * dw * 16 * 32 * 4 } else { 0 };
            let mut row_groups = (16 / dw).clamp(1, 4).max(group.div_ceil(16));
            while row_groups > group.div_ceil(16).max(1) && smem_for(row_groups) > self.max_smem {
                row_groups -= 1;
            }
            let rows = row_groups * 16;
            let tile_tokens = (rows / group).max(1);
            let smem = smem_for(row_groups);
            if smem <= self.max_smem {
                let tiles = build_tiles(tile_tokens);
                self.sync_index(&mut cache, "attn_tiles", &tiles)?;
                let (pos, seq, tab, til) =
                    (&cache["positions"].dev, &cache["token_seq"].dev, &cache[tab_key].dev, &cache["attn_tiles"].dev);
                let cfg = LaunchConfig {
                    grid_dim: ((tiles.len() / 2) as u32, meta.n_kv_heads as u32, 1),
                    block_dim: ((row_groups * dw) as u32 * WARP, 1, 1),
                    shared_mem_bytes: smem as u32,
                };
                let (tt, dwi) = (tile_tokens as i32, dw as i32);
                let (kc, vc) = (pkc, pvc);
                return launch!(self, "attn_prefill_tc", cfg, [q, kc, vc, out, til, seq, pos, tab, &mb, &nh, &nkv, &hd, &bs, &meta.scale, &window, &tt, &dwi]);
            }
        }

        // CUDA-core prefill path.
        let target_rows = if hdim > 256 { 8 } else { 16 };
        let tile_tokens = (target_rows / group).max(1);
        let rows = tile_tokens * group;
        if is_prefill && rows <= 32 {
            let smem_for = |bk: usize| (rows * hdim + bk * (hdim + 1) + bk * hdim + rows * bk) * 4;
            let mut bk = 32;
            while bk > 8 && smem_for(bk) > self.max_smem {
                bk /= 2;
            }
            if smem_for(bk) <= self.max_smem {
                let tiles = build_tiles(tile_tokens);
                self.sync_index(&mut cache, "attn_tiles", &tiles)?;
                let (pos, seq, tab, til) =
                    (&cache["positions"].dev, &cache["token_seq"].dev, &cache[tab_key].dev, &cache["attn_tiles"].dev);
                let cfg = LaunchConfig {
                    grid_dim: ((tiles.len() / 2) as u32, meta.n_kv_heads as u32, 1),
                    block_dim: (rows as u32 * WARP, 1, 1),
                    shared_mem_bytes: smem_for(bk) as u32,
                };
                let (tt, bki) = (tile_tokens as i32, bk as i32);
                let (kc, vc) = (pkc, pvc);
                return launch!(self, "attn_prefill", cfg, [q, kc, vc, out, til, seq, pos, tab, &mb, &nh, &nkv, &hd, &bs, &meta.scale, &window, &tt, &bki]);
            }
        }

        let (pos, seq, tab) = (&cache["positions"].dev, &cache["token_seq"].dev, &cache["block_tables"].dev);

        // Decode kernel: one block per (token, KV head) serving the whole GQA group.
        if decode_kernel {
            // Split long contexts over several blocks per (token, KV head)
            // until one token's grid covers the GPU about twice (at least
            // 256 positions per split). Sized from the padded block-table
            // width, not the exact context, so a CUDA graph's shape stays
            // valid while the context grows; and per token, not from this
            // step's token count, so a token's attention doesn't depend on
            // how many tokens share the step (a verification step matches
            // one-token steps bit for bit at any depth). Batched decode at
            // depth gains too: 8 Qwen3-8B requests at 32k on the RTX 6000 Ada,
            // 489 -> 568 tok/s.
            let pairs = n_tok * meta.n_kv_heads;
            let capacity = max_blocks * meta.block_size;
            let splits = (2 * self.n_sms).div_ceil(meta.n_kv_heads).min(capacity / 256).clamp(1, 32);
            let cfg = LaunchConfig {
                grid_dim: (n_tok as u32, meta.n_kv_heads as u32, splits as u32),
                block_dim: (ad_warps as u32 * WARP, 1, 1),
                shared_mem_bytes: ad_smem as u32,
            };
            let (kc, vc, ks, vs) = (&kv.k[layer], &kv.v[layer], &kv.ks[layer], &kv.vs[layer]);
            let len = n_tok * w;
            drop(cache);
            let qo = self.qout(len)?;
            // Split states (unused with one split: the kernel writes `out`).
            let mut part = if splits > 1 { Some(self.alloc(pairs * splits * group * (hdim + 2))?) } else { None };
            let part_ptr: u64 = part.as_ref().map_or(out.ptr, |p| p.ptr);
            let cache = self.index_cache.lock().unwrap();
            let (pos, seq, tab) = (&cache["positions"].dev, &cache["token_seq"].dev, &cache["block_tables"].dev);
            // The fused int8 output comes from whichever kernel writes `out`.
            let qo_attn = if splits > 1 { QOut::default() } else { qo };
            if q8 {
                let name = format!("attn_decode_q8_g{group}_d{hdim}");
                launch!(self, name, cfg, [q, kc, vc, out, seq, pos, tab, &mb, &nh, &nkv, &bs, &meta.scale, &window, &qo_attn, ks, vs, &part_ptr])?;
            } else {
                let name = format!("attn_decode_g{group}_d{hdim}");
                launch!(self, name, cfg, [q, kc, vc, out, seq, pos, tab, &mb, &nh, &nkv, &bs, &meta.scale, &window, &qo_attn, &part_ptr])?;
            }
            if let Some(part) = &mut part {
                let (si, gi) = (splits as i32, group as i32);
                let cfg = LaunchConfig { grid_dim: (n_tok as u32, meta.n_kv_heads as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
                launch!(self, "attn_decode_combine", cfg, [part, out, &si, &gi, &hd, &nh, &nkv, &qo])?;
            }
            self.set_prequant(&qo, out.ptr, len);
            return Ok(());
        }

        let cfg = LaunchConfig {
            grid_dim: (n_tok as u32, meta.n_heads as u32, 1),
            block_dim: (ATTN_WARPS * WARP, 1, 1),
            shared_mem_bytes: 0,
        };
        let (kc, vc, ks, vs) = (&kv.k[layer], &kv.v[layer], &kv.ks[layer], &kv.vs[layer]);
        let q8i = q8 as i32;
        launch!(self, "paged_attention", cfg, [q, kc, vc, out, seq, pos, tab, &mb, &nh, &nkv, &hd, &bs, &meta.scale, &window, ks, vs, &q8i])
    }

    fn synchronize(&self) -> Result<()> {
        self.stream.synchronize().map_err(err)
    }
}

#[cfg(test)]
mod arch_tests {
    /// The kernels must build for every target in docs/plan.md, not just the
    /// dev machine (see `gpu::TEST_ARCHES`). Needs only the runtime compiler,
    /// no GPU of that kind.
    #[test]
    fn kernels_compile_for_target_arches() {
        if !crate::gpu::compiler_present() {
            eprintln!("skipping: no {} kernel compiler on this machine", crate::gpu::VENDOR);
            return;
        }
        for arch in crate::gpu::TEST_ARCHES {
            if let Err(e) = super::compile_kernels(arch) {
                panic!("{arch}: {e}");
            }
        }
    }
}


