//! `rllm tune`: recommend `rllm serve` settings for this machine and model.
//!
//! The estimate reads the model's metadata (weights, KV bytes per token,
//! trained context), the GPU's free memory and the system's available RAM,
//! and sizes the KV pool from what is left. `--measure` then loads the model
//! and runs a short workload at the planned settings: it replaces estimated
//! memory with measured memory, shrinks the pool if the workload runs out,
//! and checks that a draft model is fast enough to pay off.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Args;
use rllm_core::{Backend, KvType};
use rllm_cuda::CudaBackend;
use rllm_hip::HipBackend;
use rllm_gguf::Gguf;
use rllm_kv::SeqKv;
use rllm_model::{AnyModel, KvPool, Model, ModelProfile, StepItem, Want};
use rllm_server::{DRAFT_MAX_CONTEXT, draft_pool_tokens};
use rllm_tokenizer::Tokenizer;

const GIB: f64 = (1u64 << 30) as f64;
const MIB: usize = 1 << 20;
const BLOCK: usize = 16;
/// The server's step size for CUDA graphs and drafts; more parallel
/// requests than this still work but gain less.
const MAX_PARALLEL: usize = 8;
/// Default context when the model supports more: long enough for most chat
/// and RAG use, without spending the whole pool on one request.
const DEFAULT_CTX: usize = 32768;
/// Prompt tokens per forward pass (the server's default).
const PREFILL_CHUNK: usize = 512;

#[derive(Args)]
pub struct TuneArgs {
    model: PathBuf,
    /// `cpu`, `gpu[:N]`, `cuda[:N]` or `hip[:N]` (default: the GPU with the most free memory).
    #[arg(short, long)]
    device: Option<String>,
    /// Context to plan for, in tokens (default: the model's, up to 32768).
    #[arg(long)]
    ctx: Option<usize>,
    /// Requests you expect to run at once (default: plan for up to 8, as
    /// memory allows, and a draft model for when few are running).
    #[arg(long)]
    parallel: Option<usize>,
    /// Draft model to consider (default: look for one next to the model).
    #[arg(long)]
    draft: Option<PathBuf>,
    /// Don't suggest a draft model.
    #[arg(long)]
    no_draft: bool,
    /// Load the model and run a short workload (under a minute, saturates
    /// the GPU) to confirm memory and draft speed.
    #[arg(long)]
    measure: bool,
    /// Print the plan as JSON.
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GpuKind {
    Cuda,
    Hip,
}

/// A GPU of either vendor (rllm_cuda's or rllm_hip's `DeviceInfo`).
#[derive(Debug, Clone)]
struct Gpu {
    kind: GpuKind,
    ordinal: usize,
    name: String,
    arch: String,
    sms: usize,
    total_bytes: usize,
    free_bytes: usize,
    integrated: bool,
}

macro_rules! gpus {
    ($kind:expr, $devices:expr) => {
        $devices.into_iter().map(|d| Gpu {
            kind: $kind,
            ordinal: d.ordinal,
            name: d.name,
            arch: d.arch,
            sms: d.sms,
            total_bytes: d.total_bytes,
            free_bytes: d.free_bytes,
            integrated: d.integrated,
        })
    };
}

impl Gpu {
    /// Its `-d` value.
    fn device(&self) -> String {
        let kind = match self.kind {
            GpuKind::Cuda => "cuda",
            GpuKind::Hip => "hip",
        };
        if self.ordinal == 0 { kind.into() } else { format!("{kind}:{}", self.ordinal) }
    }

    /// Memory rllm may plan to use. An integrated GPU's free memory is also
    /// the system's, so leave 8 GiB of RAM for everything else.
    fn usable_bytes(&self, ram_available: usize) -> usize {
        if self.integrated { self.free_bytes.min(ram_available.saturating_sub(8 << 30)) } else { self.free_bytes }
    }
}

/// The machine, as far as sizing is concerned.
struct Machine {
    gpu: Option<Gpu>,
    ram_total: usize,
    ram_available: usize,
    cpu: String,
    hybrid: bool,
}

/// A draft model that could serve the target.
struct Draft {
    path: PathBuf,
    profile: ModelProfile,
}

/// Recommended `rllm serve` settings, with the reasons.
#[derive(Debug, Clone)]
struct Plan {
    device: String,
    ctx: usize,
    parallel: usize,
    kv_tokens: usize,
    kv_type: KvType,
    prefill_chunk: usize,
    host_cache_gb: usize,
    draft: Option<PathBuf>,
    prompt_lookup: bool,
    /// GPU memory: (free, weights, draft weights, runtime overhead, margin), bytes.
    memory: Option<[usize; 5]>,
    reasons: Vec<String>,
    warnings: Vec<String>,
}

impl Plan {
    fn command(&self, model: &Path) -> String {
        let mut c = format!("rllm serve {}", model.display());
        if self.device != "cuda" {
            c += &format!(" -d {}", self.device);
        }
        c += &format!(" --ctx {} --parallel {} --kv-tokens {}", self.ctx, self.parallel, self.kv_tokens);
        if self.kv_type != KvType::F16 {
            c += &format!(" --kv-type {}", self.kv_type);
        }
        if self.prefill_chunk != PREFILL_CHUNK {
            c += &format!(" --prefill-chunk {}", self.prefill_chunk);
        }
        if let Some(d) = &self.draft {
            c += &format!(" --draft {}", d.display());
        }
        if self.prompt_lookup {
            c += " --prompt-lookup";
        }
        if self.host_cache_gb > 0 {
            c += &format!(" --host-cache-gb {}", self.host_cache_gb);
        }
        c
    }
}

/// GPU memory the engine needs besides weights and KV cache: cuBLAS
/// workspace, activation buffers for a prefill chunk, logits, decode graphs,
/// and for q8 the f16 staging of one layer's attended blocks (old and new
/// buffer while it grows). Calibrated with `rllm tune --measure` on the RTX
/// 6000 Ada: Qwen3-0.6B Q8_0/BF16, Qwen3-8B Q4_K_M and Nemotron E4B Q8_K_XL
/// measured 0.19-0.61 GiB at 32k context.
fn runtime_overhead(m: &ModelProfile, kv: KvType, ctx: usize) -> usize {
    let activations = PREFILL_CHUNK * (12 * m.hidden + 4 * m.ffn) * 4;
    let logits = (MAX_PARALLEL + 8) * m.vocab * 4;
    let stage = if kv == KvType::Q8 { 2 * ctx * m.max_layer_f16_bytes_per_token() } else { 0 };
    // Gemma 4's per-layer embeddings and wider vocabulary add buffers.
    let arch = if m.arch == "gemma4" { 128 * MIB } else { 0 };
    128 * MIB + activations + logits + stage + arch
}

/// Device memory weights take once uploaded: the file's size, K-quants
/// repacked about 11% larger (measured 5.19 GiB for Qwen3-8B Q4_K_M's 4.68).
fn device_weights(m: &ModelProfile) -> usize {
    m.weight_bytes + m.kquant_bytes / 9 + 32 * MIB
}

fn plan(m: &ModelProfile, draft: Option<&Draft>, mach: &Machine, want_ctx: Option<usize>, want_parallel: Option<usize>) -> Plan {
    let mut p = Plan {
        device: "cuda".into(),
        ctx: want_ctx.unwrap_or(m.max_context.min(DEFAULT_CTX)),
        parallel: want_parallel.unwrap_or(MAX_PARALLEL).clamp(1, 64),
        kv_tokens: 0,
        kv_type: KvType::F16,
        prefill_chunk: PREFILL_CHUNK,
        host_cache_gb: 0,
        draft: None,
        prompt_lookup: true,
        memory: None,
        reasons: Vec::new(),
        warnings: Vec::new(),
    };
    if p.ctx > m.max_context {
        p.warnings.push(format!("--ctx {} is beyond the {} tokens the model was trained for; quality may drop past it.", p.ctx, m.max_context));
    }
    let Some(gpu) = &mach.gpu else {
        p.device = "cpu".into();
        p.ctx = p.ctx.min(8192);
        p.parallel = p.parallel.min(2);
        p.kv_tokens = p.ctx * p.parallel;
        p.prompt_lookup = false;
        p.warnings.push("No GPU found: the CPU backend is rllm's correctness reference, not tuned for speed (a few tokens/s on 7B models).".into());
        return p;
    };
    p.device = gpu.device();
    let free = gpu.usable_bytes(mach.ram_available);

    // Draft model: only when it is much smaller than the target, and not for
    // busy servers (speculation pauses once 5 requests decode together).
    let busy = want_parallel.is_some_and(|n| n > 4);
    let draft = draft.filter(|d| d.profile.weight_bytes * 4 <= m.weight_bytes && !busy);
    let margin = (512 * MIB).max(free * 4 / 100);
    // The draft only drafts for sequences up to DRAFT_MAX_CONTEXT tokens, so
    // its runtime buffers and (capped) KV pool don't grow with the context.
    let draft_bytes = |kv| {
        draft.map_or(0, |d| {
            device_weights(&d.profile)
                + runtime_overhead(&d.profile, kv, DRAFT_MAX_CONTEXT)
                + draft_pool_tokens(usize::MAX) * d.profile.kv_bytes_per_token(kv)
        })
    };
    let fixed = |kv, ctx, with_draft: bool| {
        device_weights(m) + runtime_overhead(m, kv, ctx) + margin + if with_draft { draft_bytes(kv) } else { 0 }
    };
    let per_token = |kv| m.kv_bytes_per_token(kv);
    let tokens = |kv, ctx, with_draft| free.saturating_sub(fixed(kv, ctx, with_draft)) / per_token(kv) / 1024 * 1024;

    // q8 halves the cache and decodes faster from about 8k tokens of
    // context (measured on Qwen3-8B and Nemotron); f16 stays for short
    // contexts with memory to spare, where it is ~1.5% faster.
    let f16_fits = tokens(KvType::F16, p.ctx, draft.is_some()) >= 2 * p.ctx;
    p.kv_type = if p.ctx >= 8192 || !f16_fits { KvType::Q8 } else { KvType::F16 };
    if p.kv_type == KvType::Q8 {
        p.reasons.push(if p.ctx >= 8192 {
            format!("q8 KV cache: half the memory of f16, and faster decoding at contexts like {} (perplexity within noise).", p.ctx)
        } else {
            "q8 KV cache: f16 would leave room for less than two full contexts.".into()
        });
    }

    let mut with_draft = draft.is_some();
    let mut kv_tokens = tokens(p.kv_type, p.ctx, with_draft);
    if with_draft && kv_tokens < p.ctx {
        if want_ctx.is_none() && kv_tokens >= p.ctx * 3 / 4 {
            // A slightly shorter context is worth the draft's speedup.
            p.reasons.push(format!("--ctx {kv_tokens} (not {}): leaves room for the draft model's weights and KV cache.", p.ctx));
            p.ctx = kv_tokens;
        } else {
            // The draft's weights and cache cost more context than it's worth.
            with_draft = false;
            kv_tokens = tokens(p.kv_type, p.ctx, false);
            p.reasons.push("No draft model: its weights and KV cache would not leave room for the context.".into());
        }
    }
    if kv_tokens < 2048 {
        p.kv_tokens = kv_tokens;
        p.warnings.push(format!(
            "Only {} tokens of KV cache fit: the model needs about {:.1} GiB of weights and runtime memory, and {:.1} GiB is free. Use a smaller quantization or free GPU memory.",
            kv_tokens,
            fixed(p.kv_type, p.ctx, false) as f64 / GIB,
            free as f64 / GIB
        ));
        p.ctx = kv_tokens.max(BLOCK);
        p.parallel = 1;
        return p;
    }
    if kv_tokens < p.ctx {
        p.warnings.push(format!("Context limited to {kv_tokens} tokens by GPU memory (wanted {}).", p.ctx));
        p.ctx = kv_tokens;
    }
    p.kv_tokens = kv_tokens;
    p.memory = Some([
        free,
        device_weights(m),
        if with_draft { draft_bytes(p.kv_type) } else { 0 },
        runtime_overhead(m, p.kv_type, p.ctx),
        margin,
    ]);
    p.reasons.push(format!(
        "KV pool of {kv_tokens} tokens: all GPU memory left after weights, {}runtime buffers and a {:.1} GiB margin ({:.0} KiB per token).",
        if with_draft { "the draft model, " } else { "" },
        margin as f64 / GIB,
        per_token(p.kv_type) as f64 / 1024.0,
    ));

    // Parallel requests share the pool; assume a typical request holds
    // about 2k tokens (prompt and output) at a time.
    if want_parallel.is_none() {
        p.parallel = (kv_tokens / 2048).clamp(1, MAX_PARALLEL);
        p.reasons.push(if p.parallel == MAX_PARALLEL {
            format!("--parallel {MAX_PARALLEL}: batching gains up to 8 requests per step; idle slots cost nothing.")
        } else {
            format!("--parallel {}: about 2k tokens of cache per running request.", p.parallel)
        });
    }

    if with_draft && let Some(d) = draft {
        p.draft = Some(d.path.clone());
        p.reasons.push(format!(
            "Draft model {}: same tokenizer, {:.1}x smaller; output is unchanged (exact speculative sampling). It drafts for requests up to {DRAFT_MAX_CONTEXT} tokens (beyond that it would slow decoding). If 5 or more requests will usually run at once, rerun `rllm tune --parallel 8`: speculation pauses then, so the draft's memory is better spent on the KV pool.",
            d.path.file_name().unwrap_or_default().to_string_lossy(),
            m.weight_bytes as f64 / d.profile.weight_bytes as f64
        ));
    }
    p.reasons.push(if m.arch == "gemma4" {
        "--prompt-lookup: free speedup when output repeats its input (code edits, quoting). On Gemma 4, greedy output matches plain decoding as closely as batched requests do.".into()
    } else {
        "--prompt-lookup: free speedup when output repeats its input (code edits, quoting); output is unchanged.".into()
    });

    // Host tier: a share of available RAM, enough to hold several pools'
    // worth of evicted prefixes, leaving plenty for the rest of the system.
    let pool_bytes = kv_tokens * m.kv_bytes_per_token(p.kv_type);
    let share = (mach.ram_available / 4).min(4 * pool_bytes).min(64 << 30);
    let gb = (share as f64 / GIB).floor() as usize;
    if gpu.integrated {
        p.reasons.push("No --host-cache-gb: the GPU shares system RAM, so a host tier would only move KV within the same memory.".into());
    } else if gb >= 1 && mach.ram_available.saturating_sub(gb << 30) >= 8 << 30 {
        p.host_cache_gb = gb;
        p.reasons.push(format!(
            "--host-cache-gb {gb}: evicted prompt prefixes ({} tokens' worth) wait in system RAM instead of being recomputed; {:.0} GiB stays available.",
            (gb << 30) / m.kv_bytes_per_token(p.kv_type),
            (mach.ram_available - (gb << 30)) as f64 / GIB
        ));
    }
    if mach.hybrid {
        p.reasons.push("Hybrid CPU: the engine thread pins itself to the performance cores (automatic).".into());
    }
    p
}

fn read_machine(device: Option<&str>) -> Result<Machine> {
    let probe = |kind| -> Vec<Gpu> {
        match kind {
            GpuKind::Cuda => gpus!(kind, rllm_cuda::probe_devices()).collect(),
            GpuKind::Hip => gpus!(kind, rllm_hip::probe_devices()).collect(),
        }
    };
    let find = |kind, ord: usize| -> Result<Gpu> {
        probe(kind).into_iter().find(|g| g.ordinal == ord).with_context(|| format!("no {kind:?} device {ord}"))
    };
    let gpu = match device.map(crate::Device::parse).transpose()? {
        Some(crate::Device::Cpu) => None,
        Some(crate::Device::Cuda(i)) => Some(find(GpuKind::Cuda, i)?),
        Some(crate::Device::Hip(i)) => Some(find(GpuKind::Hip, i)?),
        None => probe(GpuKind::Cuda).into_iter().chain(probe(GpuKind::Hip)).max_by_key(|g| g.free_bytes),
    };
    let meminfo = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    let kib = |key: &str| {
        meminfo.lines().find(|l| l.starts_with(key)).and_then(|l| l.split_whitespace().nth(1)).and_then(|v| v.parse::<usize>().ok()).unwrap_or(0) * 1024
    };
    let cpuinfo = std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
    let cpu = cpuinfo.lines().find(|l| l.starts_with("model name")).and_then(|l| l.split(':').nth(1)).unwrap_or("unknown").trim().to_owned();
    Ok(Machine {
        gpu,
        ram_total: kib("MemTotal:"),
        ram_available: kib("MemAvailable:"),
        cpu,
        hybrid: Path::new("/sys/devices/cpu_core/cpus").exists(),
    })
}

/// A draft model: `explicit`, or the smallest GGUF next to the model with
/// the same tokenizer (and a supported architecture).
fn find_draft(model: &Path, g: &Gguf, profile: &ModelProfile, explicit: Option<&Path>) -> Option<Draft> {
    let tok = Tokenizer::from_gguf(g).ok()?;
    let same_vocab = |d: &Gguf| -> bool {
        let Ok(dt) = Tokenizer::from_gguf(d) else { return false };
        dt.vocab_size() == tok.vocab_size() && (0..tok.vocab_size() as u32).all(|i| dt.token_text(i) == tok.token_text(i))
    };
    let candidates: Vec<PathBuf> = match explicit {
        Some(p) => vec![p.to_path_buf()],
        None => {
            let dir = model.parent().filter(|d| !d.as_os_str().is_empty()).unwrap_or(Path::new("."));
            let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
                .ok()?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|e| e == "gguf") && p.canonicalize().ok() != model.canonicalize().ok())
                .collect();
            v.sort_by_key(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(u64::MAX));
            v
        }
    };
    candidates.into_iter().find_map(|path| {
        let d = Gguf::open(&path).ok()?;
        let dp = ModelProfile::read(&d).ok()?;
        (dp.vocab == profile.vocab && (explicit.is_some() || dp.weight_bytes * 4 <= profile.weight_bytes) && same_vocab(&d))
            .then_some(Draft { path, profile: dp })
    })
}

/// What `--measure` found.
struct Measured {
    weights: usize,
    draft_weights: usize,
    /// Runtime buffers at the busiest point: memory in use beyond weights and KV pools.
    runtime: usize,
    /// Smallest free memory seen during the workload.
    min_free: usize,
    prefill_tok_s: f64,
    depth: usize,
    decode_one_tok_s: f64,
    decode_batch_tok_s: f64,
    /// Draft step time / target step time (one sequence, short context).
    draft_ratio: Option<f64>,
}

/// Pseudo-random token ids below `vocab`.
fn tokens(n: usize, vocab: usize, seed: u64) -> Vec<u32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s % vocab as u64) as u32
        })
        .collect()
}

/// The device-memory queries `measure` needs (inherent on both GPU backends).
trait MemInfo: Backend {
    fn mem_info(&self) -> Result<(usize, usize)>;
    fn hold_freed_memory(&self) -> Result<()>;
}

macro_rules! impl_mem_info {
    ($b:ty) => {
        impl MemInfo for $b {
            fn mem_info(&self) -> Result<(usize, usize)> {
                Ok(<$b>::mem_info(self)?)
            }
            fn hold_freed_memory(&self) -> Result<()> {
                Ok(<$b>::hold_freed_memory(self)?)
            }
        }
    };
}
impl_mem_info!(CudaBackend);
impl_mem_info!(HipBackend);

/// [`measure`] on `gpu`, with a fresh backend.
fn measure_on(gpu: &Gpu, model_path: &Path, p: &Plan) -> Result<Measured> {
    match gpu.kind {
        GpuKind::Cuda => measure(&CudaBackend::new(gpu.ordinal)?, model_path, p),
        GpuKind::Hip => measure(&HipBackend::new(gpu.ordinal)?, model_path, p),
    }
}

/// Load the model and run the planned configuration: one request prefilled
/// to near the full context, then 8 requests decoding together, then one
/// alone; the draft model's decode step is timed against the target's.
fn measure<B: MemInfo>(b: &B, model_path: &Path, p: &Plan) -> Result<Measured> {
    b.hold_freed_memory()?;
    let (free0, _) = b.mem_info()?;
    let g = Gguf::open(model_path)?;
    let model = AnyModel::load(b, &g)?;
    let (free1, _) = b.mem_info()?;
    let dg = p.draft.as_ref().map(Gguf::open).transpose()?;
    let draft = dg.as_ref().map(|g| AnyModel::load(b, g)).transpose()?;
    let (free2, _) = b.mem_info()?;
    let vocab = model.vocab_size();

    let mut pool = KvPool::new(b, &model, p.kv_tokens, BLOCK, p.kv_type)?;
    let mut dpool = draft.as_ref().map(|d| KvPool::new(b, d, draft_pool_tokens(p.kv_tokens), BLOCK, p.kv_type)).transpose()?;
    let pools = free2.saturating_sub(b.mem_info()?.0);
    let mut min_free = b.mem_info()?.0;
    let mut note = |b: &B| -> Result<()> {
        min_free = min_free.min(b.mem_info()?.0);
        Ok(())
    };

    // One long request, then 7 short ones: the pool's worst case is a full
    // context plus the others' prompts.
    let short = 128;
    let depth = p.ctx.min(p.kv_tokens.saturating_sub(MAX_PARALLEL * (short + 64))).saturating_sub(64).max(short);
    let long = tokens(depth, vocab, 1);
    let mut seqs: Vec<SeqKv> = (0..MAX_PARALLEL).map(|_| SeqKv::default()).collect();
    let t = Instant::now();
    for part in long.chunks(p.prefill_chunk) {
        pool.step(&mut [StepItem { seq: &mut seqs[0], tokens: part, want: Want::Last }], false)?;
    }
    let prefill_tok_s = depth as f64 / t.elapsed().as_secs_f64();
    note(b)?;
    for (i, s) in seqs.iter_mut().enumerate().skip(1) {
        pool.step(&mut [StepItem { seq: s, tokens: &tokens(short, vocab, i as u64 + 2), want: Want::Last }], false)?;
    }
    // Decode: all sequences together, then the long one alone.
    let steps = 24;
    let next: Vec<[u32; 1]> = (0..MAX_PARALLEL).map(|i| [i as u32 + 7]).collect();
    let timed = |seqs: &mut [SeqKv], pool: &mut KvPool<_, _>| -> Result<f64> {
        let mut t0 = Instant::now();
        for step in 0..steps + 4 {
            if step == 4 {
                t0 = Instant::now(); // after graph capture
            }
            let mut items: Vec<StepItem> = seqs.iter_mut().zip(&next).map(|(s, t)| StepItem { seq: s, tokens: t, want: Want::Last }).collect();
            pool.step(&mut items, false)?;
        }
        Ok((steps * seqs.len()) as f64 / t0.elapsed().as_secs_f64())
    };
    let decode_batch_tok_s = timed(&mut seqs, &mut pool)?;
    note(b)?;
    let decode_one_tok_s = timed(&mut seqs[..1], &mut pool)?;
    // Draft vs target, one short sequence each.
    let draft_ratio = match (&mut dpool, draft.as_ref()) {
        (Some(dp), Some(_)) => {
            let mut ds = [SeqKv::default()];
            dp.step(&mut [StepItem { seq: &mut ds[0], tokens: &tokens(short, vocab, 9), want: Want::Last }], false)?;
            let d_tok_s = timed(&mut ds, dp)?;
            let t_tok_s = timed(&mut seqs[1..2], &mut pool)?;
            note(b)?;
            Some(t_tok_s / d_tok_s)
        }
        _ => None,
    };
    Ok(Measured {
        weights: free0.saturating_sub(free1),
        draft_weights: free1.saturating_sub(free2),
        runtime: free2.saturating_sub(pools).saturating_sub(min_free),
        min_free,
        prefill_tok_s,
        depth,
        decode_one_tok_s,
        decode_batch_tok_s,
        draft_ratio,
    })
}

pub fn tune(a: TuneArgs) -> Result<()> {
    let g = Gguf::open(&a.model).with_context(|| format!("open {}", a.model.display()))?;
    let profile = ModelProfile::read(&g)?;
    let mach = read_machine(a.device.as_deref())?;
    let draft = if a.no_draft { None } else { find_draft(&a.model, &g, &profile, a.draft.as_deref()) };
    let mut p = plan(&profile, draft.as_ref(), &mach, a.ctx, a.parallel);

    let mut measured = None;
    let estimated = p.memory;
    if a.measure && let Some(gpu) = &mach.gpu {
        if !a.json {
            eprintln!("[rllm] measuring: loading the model and running {} at --kv-tokens {} ...", p.ctx, p.kv_tokens);
        }
        // Shrink the pool until the workload fits (rarely needed).
        let mut tries = 0;
        let m = loop {
            match measure_on(gpu, &a.model, &p) {
                Ok(m) => break m,
                // CUDA_ERROR_OUT_OF_MEMORY, hipErrorOutOfMemory
                Err(e) if tries < 3 && ["OUT_OF_MEMORY", "OutOfMemory"].iter().any(|s| format!("{e:#}").contains(s)) => {
                    tries += 1;
                    p.kv_tokens = p.kv_tokens * 85 / 100 / 1024 * 1024;
                    p.ctx = p.ctx.min(p.kv_tokens);
                    p.warnings.push(format!("Out of memory at the estimated pool size; retried with --kv-tokens {}.", p.kv_tokens));
                }
                Err(e) => return Err(e.context("measuring")),
            }
        };
        // Free memory left at the busiest point: grow or shrink the pool to
        // leave exactly the margin.
        if let Some(mem) = &mut p.memory {
            let margin = mem[4];
            let per_token = profile.kv_bytes_per_token(p.kv_type);
            let adjust = (m.min_free as i64 - margin as i64) / per_token as i64 / 1024 * 1024;
            let before = p.kv_tokens;
            p.kv_tokens = (p.kv_tokens as i64 + adjust).max(p.ctx as i64) as usize;
            if p.kv_tokens != before {
                p.reasons.push(format!("Measured memory: KV pool adjusted from {before} to {} tokens.", p.kv_tokens));
            }
            // The draft's share stays as estimated (its pool is in `runtime`'s
            // complement: measured free memory already reflects it).
            mem[1] = m.weights;
            mem[3] = m.runtime;
        }
        if let Some(r) = m.draft_ratio
            && r > 0.3
        {
            p.reasons.push(format!("No draft model: its decode step takes {:.0}% of the target's here (it pays off below about 30%).", r * 100.0));
            p.reasons.retain(|s| !s.starts_with("Draft model "));
            p.draft = None;
        }
        measured = Some(m);
    }

    if a.json {
        let v = serde_json::json!({
            "command": p.command(&a.model),
            "device": p.device, "ctx": p.ctx, "parallel": p.parallel, "kv_tokens": p.kv_tokens,
            "kv_type": p.kv_type.to_string(), "prefill_chunk": p.prefill_chunk, "host_cache_gb": p.host_cache_gb,
            "draft": p.draft, "prompt_lookup": p.prompt_lookup,
            "reasons": p.reasons, "warnings": p.warnings,
            "measured": measured.as_ref().map(|m| serde_json::json!({
                "weights_bytes": m.weights, "min_free_bytes": m.min_free, "prefill_tok_s": m.prefill_tok_s, "prefill_depth": m.depth,
                "decode_one_tok_s": m.decode_one_tok_s, "decode_batch_tok_s": m.decode_batch_tok_s, "draft_ratio": m.draft_ratio,
            })),
        });
        println!("{}", serde_json::to_string_pretty(&v)?);
        return Ok(());
    }

    println!("Machine");
    match &mach.gpu {
        Some(gpu) => println!(
            "  GPU {}: {} ({}, {} {}), {:.1} GiB free of {:.1} GiB{}",
            gpu.device(),
            gpu.name,
            gpu.arch,
            gpu.sms,
            // What the driver counts: SMs; WGPs (2 CUs each) on RDNA, CUs on CDNA.
            match gpu.kind {
                GpuKind::Cuda => "SMs",
                GpuKind::Hip if gpu.arch.starts_with("gfx1") => "WGPs",
                GpuKind::Hip => "CUs",
            },
            gpu.free_bytes as f64 / GIB,
            gpu.total_bytes as f64 / GIB,
            if gpu.integrated { " (integrated: shares system RAM)" } else { "" }
        ),
        None => println!("  GPU: none"),
    }
    println!("  RAM: {:.0} GiB available of {:.0} GiB", mach.ram_available as f64 / GIB, mach.ram_total as f64 / GIB);
    println!("  CPU: {}{}", mach.cpu, if mach.hybrid { " (hybrid)" } else { "" });
    println!("Model");
    println!(
        "  {} ({}, {} layers, trained context {}), weights {:.2} GiB",
        a.model.file_name().unwrap_or_default().to_string_lossy(),
        profile.arch,
        profile.n_layers,
        profile.max_context,
        profile.weight_bytes as f64 / GIB
    );
    println!(
        "  KV cache per token: {:.0} KiB f16, {:.0} KiB q8",
        profile.kv_bytes_per_token(KvType::F16) as f64 / 1024.0,
        profile.kv_bytes_per_token(KvType::Q8) as f64 / 1024.0
    );
    if let Some(d) = &draft {
        println!("  Draft candidate: {} ({:.2} GiB, same tokenizer)", d.path.display(), d.profile.weight_bytes as f64 / GIB);
    }
    if let Some([free, w, dw, over, margin]) = p.memory {
        let kv = free.saturating_sub(w + dw + over + margin);
        println!("GPU memory plan");
        println!(
            "  {:.1} GiB free = weights {:.1} + draft {:.1} + runtime {:.1} + margin {:.1} + KV cache {:.1}",
            free as f64 / GIB,
            w as f64 / GIB,
            dw as f64 / GIB,
            over as f64 / GIB,
            margin as f64 / GIB,
            kv as f64 / GIB
        );
    }
    if let Some(m) = &measured {
        println!("Measured (at the recommended settings)");
        println!("  Prefill at {} tokens of context: {:.0} tok/s", m.depth, m.prefill_tok_s);
        println!("  Decode, one request at that depth: {:.1} tok/s", m.decode_one_tok_s);
        println!("  Decode, {MAX_PARALLEL} requests together: {:.0} tok/s total", m.decode_batch_tok_s);
        if let Some(r) = m.draft_ratio {
            println!("  Draft step / model step: {:.0}%", r * 100.0);
        }
        if let Some([_, w, dw, over, _]) = estimated {
            println!(
                "  Memory: weights {:.2} GiB (estimated {:.2}), runtime buffers {:.2} ({:.2}){}",
                m.weights as f64 / GIB,
                w as f64 / GIB,
                m.runtime as f64 / GIB,
                over as f64 / GIB,
                if dw > 0 { format!(", draft weights {:.2}", m.draft_weights as f64 / GIB) } else { String::new() }
            );
        }
        println!("  Lowest free GPU memory during the run: {:.2} GiB", m.min_free as f64 / GIB);
    }
    println!("Recommended");
    println!("  {}", p.command(&a.model));
    println!("Why");
    for r in &p.reasons {
        println!("  - {r}");
    }
    for w in &p.warnings {
        println!("  ! {w}");
    }
    if !a.measure && mach.gpu.is_some() {
        println!("Memory figures are estimates; `rllm tune --measure` loads the model and checks them (under a minute, uses the GPU).");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu(free_gib: usize, total_gib: usize) -> Gpu {
        Gpu {
            kind: GpuKind::Cuda,
            ordinal: 0,
            name: "test".into(),
            arch: "sm_89".into(),
            sms: 142,
            total_bytes: total_gib << 30,
            free_bytes: free_gib << 30,
            integrated: false,
        }
    }

    fn machine(free_gib: usize, ram_gib: usize) -> Machine {
        Machine { gpu: Some(gpu(free_gib, 48)), ram_total: ram_gib << 30, ram_available: ram_gib << 30, cpu: "test".into(), hybrid: false }
    }

    /// Qwen3-8B-like: 36 layers, 8 KV heads of 128, 5 GB of weights.
    fn qwen8b() -> ModelProfile {
        profile(36, 8, 128, 5 << 30, 40960)
    }

    fn profile(layers: usize, kv_heads: usize, head_dim: usize, weights: usize, ctx: usize) -> ModelProfile {
        ModelProfile::for_tests(layers, kv_heads, head_dim, weights, ctx)
    }

    #[test]
    fn big_gpu_gets_q8_long_context_and_full_batching() {
        let p = plan(&qwen8b(), None, &machine(40, 128), None, None);
        assert_eq!((p.ctx, p.parallel, p.kv_type), (32768, 8, KvType::Q8));
        // About 40 GiB - weights - overhead - margin, at 72 KiB per token.
        assert!(p.kv_tokens > 400_000 && p.kv_tokens < 520_000, "{}", p.kv_tokens);
        assert!(p.host_cache_gb > 0 && p.host_cache_gb <= 32);
        assert!(p.warnings.is_empty(), "{:?}", p.warnings);
        assert!(p.command(Path::new("m.gguf")).contains("--kv-type q8 --prompt-lookup"));
    }

    #[test]
    fn short_context_with_room_stays_f16() {
        let p = plan(&qwen8b(), None, &machine(40, 128), Some(4096), None);
        assert_eq!(p.kv_type, KvType::F16);
    }

    #[test]
    fn small_gpu_limits_context_and_parallel() {
        let p = plan(&qwen8b(), None, &machine(7, 16), None, None);
        assert_eq!(p.kv_type, KvType::Q8);
        assert!(p.ctx < 32768 && p.ctx == p.kv_tokens, "{} {}", p.ctx, p.kv_tokens);
        assert!(p.parallel < 8);
        assert!(p.warnings.iter().any(|w| w.contains("limited")));
        // 16 GiB of RAM: a small host cache at most, most RAM stays free.
        assert!(p.host_cache_gb <= 4, "{}", p.host_cache_gb);
        let p = plan(&qwen8b(), None, &machine(7, 8), None, None);
        assert_eq!(p.host_cache_gb, 0);
    }

    #[test]
    fn too_little_memory_warns() {
        let p = plan(&qwen8b(), None, &machine(5, 16), None, None);
        assert!(p.warnings.iter().any(|w| w.contains("smaller quantization")), "{:?}", p.warnings);
    }

    #[test]
    fn draft_is_used_when_it_fits_and_dropped_when_it_does_not() {
        let d = Draft { path: "d.gguf".into(), profile: profile(28, 8, 128, 600 << 20, 40960) };
        let p = plan(&qwen8b(), Some(&d), &machine(40, 128), None, None);
        assert_eq!(p.draft.as_deref(), Some(Path::new("d.gguf")));
        // Busy servers skip it.
        let p = plan(&qwen8b(), Some(&d), &machine(40, 128), None, Some(8));
        assert!(p.draft.is_none());
        // A draft as large as a quarter of the model and no room left: dropped.
        let p = plan(&qwen8b(), Some(&d), &machine(6, 16), Some(32768), None);
        assert!(p.draft.is_none());
    }

    #[test]
    fn integrated_gpu_leaves_ram_and_skips_the_host_tier() {
        // Strix Halo: 44 GiB free to the GPU, but only 30 GiB of RAM available.
        let apu = Gpu { kind: GpuKind::Hip, integrated: true, ..gpu(44, 100) };
        let mach = Machine { gpu: Some(apu), ram_total: 128 << 30, ram_available: 30 << 30, cpu: "test".into(), hybrid: false };
        let p = plan(&qwen8b(), None, &mach, None, None);
        assert_eq!((p.device.as_str(), p.host_cache_gb), ("hip", 0));
        // Planned from 30 - 8 = 22 GiB, not 44.
        let mem = p.memory.unwrap();
        assert_eq!(mem[0], 22 << 30);
        assert!(p.kv_tokens * qwen8b().kv_bytes_per_token(p.kv_type) < 22 << 30);
    }

    #[test]
    fn no_gpu_falls_back_to_cpu() {
        let mach = Machine { gpu: None, ram_total: 32 << 30, ram_available: 24 << 30, cpu: "test".into(), hybrid: false };
        let p = plan(&qwen8b(), None, &mach, None, None);
        assert_eq!(p.device, "cpu");
        assert!(p.command(Path::new("m.gguf")).contains("-d cpu"));
    }
}
