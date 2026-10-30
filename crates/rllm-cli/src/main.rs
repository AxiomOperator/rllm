use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use clap::{Args, Parser, Subcommand};
use rllm_core::{Backend, KvType};
use rllm_cpu::CpuBackend;
use rllm_cuda::CudaBackend;
use rllm_gguf::Gguf;
use rllm_model::{AnyModel, Generator, Model, Sampler, SamplerConfig};
use rllm_tokenizer::{ChatMessage, Tokenizer};

mod tune;

#[derive(Parser)]
#[command(name = "rllm", about = "Rust LLM inference engine")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Print GGUF metadata and tensor list.
    Inspect {
        model: PathBuf,
        /// Also list every tensor.
        #[arg(long)]
        tensors: bool,
    },
    /// Show how text tokenizes.
    Tokenize { model: PathBuf, text: String },
    /// Generate text.
    Run(RunArgs),
    /// Serve an OpenAI-compatible API (chat completions, completions,
    /// responses, embeddings, models) over HTTP.
    Serve {
        model: PathBuf,
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = 8080)]
        port: u16,
        #[arg(short, long, default_value = "cuda")]
        device: String,
        /// Context size in tokens (prompt + output) for every request.
        #[arg(long, default_value_t = 8192)]
        ctx: usize,
        /// Require `Authorization: Bearer <key>` (default: $RLLM_API_KEY, else none).
        #[arg(long, env = "RLLM_API_KEY", hide_env_values = true)]
        api_key: Option<String>,
        /// Model id reported to clients (default: the file name).
        #[arg(long)]
        model_name: Option<String>,
        /// Reject requests that name a different model (404), like OpenAI.
        /// By default any model name is served by the loaded model.
        #[arg(long)]
        strict_model: bool,
        /// Prompt tokens per forward pass.
        #[arg(long, default_value_t = 512)]
        prefill_chunk: usize,
        /// Most requests decoded together in one batch.
        #[arg(long, default_value_t = 4)]
        parallel: usize,
        /// KV cache pool shared by all requests, in tokens (default: ctx * parallel).
        #[arg(long)]
        kv_tokens: Option<usize>,
        /// Don't keep finished requests' KV for reuse by later prompts with the same prefix.
        #[arg(long)]
        no_prefix_cache: bool,
        /// KV cache storage: f16, or q8 (8-bit, about half the memory).
        #[arg(long, default_value = "f16")]
        kv_type: KvType,
        /// Speculative decoding: a small model with the same tokenizer that
        /// proposes tokens for this one to check (e.g. Qwen3-0.6B for Qwen3-8B).
        #[arg(long)]
        draft: Option<PathBuf>,
        /// Most tokens the draft model proposes per step.
        #[arg(long, default_value_t = 5)]
        draft_tokens: usize,
        /// Speculative decoding without a draft model: propose the tokens that
        /// followed the latest few tokens' earlier occurrence in the prompt or
        /// output. Helps when output copies input (code edits, quoting, RAG).
        #[arg(long)]
        prompt_lookup: bool,
        /// Keep evicted prefix-cache blocks in this much pinned system memory
        /// (GiB) and copy them back on a hit instead of recomputing.
        #[arg(long, default_value_t = 0.0)]
        host_cache_gb: f64,
        /// Save stored chat completions and responses to this JSON file (loaded at start).
        #[arg(long)]
        store_file: Option<PathBuf>,
        /// Don't log a line per finished generation.
        #[arg(long, short)]
        quiet: bool,
    },
    /// Recommend `rllm serve` settings for this machine and model: GPU and
    /// system memory, KV cache size and type, parallel requests, draft
    /// model, host cache. `--measure` loads the model to confirm them.
    Tune(tune::TuneArgs),
    /// Perplexity over a text file, same method as llama.cpp's llama-perplexity:
    /// the text is tokenized once and cut into `ctx`-token chunks; each chunk
    /// starts with BOS (if the model uses one) and is scored on its second half.
    Perplexity {
        model: PathBuf,
        file: PathBuf,
        #[arg(short, long, default_value = "cuda")]
        device: String,
        #[arg(short, long, default_value_t = 512)]
        ctx: usize,
        /// Stop after this many chunks (default: all).
        #[arg(long)]
        chunks: Option<usize>,
        /// KV cache storage: f16 or q8.
        #[arg(long, default_value = "f16")]
        kv_type: KvType,
    },
    /// Throughput benchmark, same method as llama.cpp's llama-bench:
    /// pp = one N-token prompt in a single batch, tg = N tokens decoded one at
    /// a time from an empty context. Token ids are fixed pseudo-random.
    Bench {
        model: PathBuf,
        #[arg(short, long, default_value = "cuda")]
        device: String,
        /// Prompt-processing size (0 to skip).
        #[arg(short = 'p', long, default_value_t = 512)]
        n_prompt: usize,
        /// Generation size (0 to skip).
        #[arg(short = 'n', long, default_value_t = 128)]
        n_gen: usize,
        /// Measured repetitions (after one warmup).
        #[arg(short, long, default_value_t = 3)]
        reps: usize,
        /// Context already in the KV cache before each test (llama-bench's -d).
        #[arg(long, default_value_t = 0)]
        depth: usize,
        /// KV cache storage: f16 or q8.
        #[arg(long, default_value = "f16")]
        kv_type: KvType,
    },
    /// Compare against a llama.cpp reference dump (tools/llama-ref): same token
    /// ids in, logits compared at every position.
    CompareRef {
        model: PathBuf,
        reference: PathBuf,
        /// The prompt text the reference was made from; also compares tokenization.
        #[arg(long)]
        prompt_file: Option<PathBuf>,
        #[arg(short, long, default_value = "cpu")]
        device: String,
        #[arg(long, default_value_t = 512)]
        prefill_chunk: usize,
    },
    /// Check the CUDA backend against the CPU reference: same prompt and
    /// tokens on both, logits compared at every step.
    Compare {
        model: PathBuf,
        prompt: String,
        /// Decode steps to compare after the prompt.
        #[arg(short = 'n', long, default_value_t = 32)]
        steps: usize,
        #[arg(long, default_value_t = 0)]
        gpu: usize,
        #[arg(long, default_value_t = 16)]
        block_size: usize,
        /// Max prompt tokens per forward pass (small values exercise the decode kernels).
        #[arg(long, default_value_t = 512)]
        prefill_chunk: usize,
        /// KV cache storage on both backends (the CPU emulates q8's rounding).
        #[arg(long, default_value = "f16")]
        kv_type: KvType,
    },
}

#[derive(Args)]
struct RunArgs {
    model: PathBuf,
    /// `cpu`, `cuda` or `cuda:N`.
    #[arg(short, long, default_value = "cuda")]
    device: String,
    prompt: String,
    /// Send the prompt as raw text instead of through the chat template.
    #[arg(long)]
    raw: bool,
    #[arg(long)]
    system: Option<String>,
    /// Disable thinking mode (passes enable_thinking=false to the chat template).
    #[arg(long)]
    no_think: bool,
    /// Print the rendered prompt and its token count, then exit without loading weights.
    #[arg(long)]
    print_prompt: bool,
    #[arg(short = 'n', long, default_value_t = 256)]
    max_tokens: usize,
    #[arg(short, long, default_value_t = 0.0)]
    temperature: f32,
    #[arg(long, default_value_t = 0)]
    top_k: usize,
    #[arg(long, default_value_t = 1.0)]
    top_p: f32,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// KV cache capacity in tokens.
    #[arg(long, default_value_t = 4096)]
    ctx: usize,
    #[arg(long, default_value_t = 16)]
    block_size: usize,
    /// Max prompt tokens per forward pass.
    #[arg(long, default_value_t = 512)]
    prefill_chunk: usize,
    /// Print token ids alongside text.
    #[arg(long)]
    show_ids: bool,
    /// KV cache storage: f16 or q8.
    #[arg(long, default_value = "f16")]
    kv_type: KvType,
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Inspect { model, tensors } => inspect(&model, tensors),
        Cmd::Tune(args) => tune::tune(args),
        Cmd::Tokenize { model, text } => tokenize(&model, &text),
        Cmd::Run(args) => match args.device.as_str() {
            "cpu" => run(&CpuBackend::new(), args),
            d => run(&CudaBackend::new(parse_cuda(d)?)?, args),
        },
        Cmd::Serve {
            model,
            host,
            port,
            device,
            ctx,
            api_key,
            model_name,
            strict_model,
            prefill_chunk,
            parallel,
            kv_tokens,
            no_prefix_cache,
            kv_type,
            draft,
            draft_tokens,
            prompt_lookup,
            host_cache_gb,
            store_file,
            quiet,
        } => {
            let cfg = rllm_server::ServeConfig {
                model,
                model_name,
                host,
                port,
                api_key,
                ctx,
                prefill_chunk,
                strict_model,
                parallel,
                kv_tokens,
                prefix_cache: !no_prefix_cache,
                kv_type,
                draft,
                draft_tokens,
                prompt_lookup,
                host_cache_bytes: (host_cache_gb.max(0.0) * (1u64 << 30) as f64) as usize,
                store_file,
                log: !quiet,
            };
            match device.as_str() {
                "cpu" => rllm_server::serve(cfg, || Ok(CpuBackend::new())),
                d => {
                    let ordinal = parse_cuda(d)?;
                    rllm_server::serve(cfg, move || Ok(CudaBackend::new(ordinal)?))
                }
            }
        }
        Cmd::Perplexity { model, file, device, ctx, chunks, kv_type } => match device.as_str() {
            "cpu" => perplexity(&CpuBackend::new(), &model, &file, ctx, chunks, kv_type),
            d => perplexity(&CudaBackend::new(parse_cuda(d)?)?, &model, &file, ctx, chunks, kv_type),
        },
        Cmd::Bench { model, device, n_prompt, n_gen, reps, depth, kv_type } => match device.as_str() {
            "cpu" => bench(&CpuBackend::new(), &model, n_prompt, n_gen, reps, depth, kv_type),
            d => bench(&CudaBackend::new(parse_cuda(d)?)?, &model, n_prompt, n_gen, reps, depth, kv_type),
        },
        Cmd::CompareRef { model, reference, prompt_file, device, prefill_chunk } => match device.as_str() {
            "cpu" => compare_ref(&CpuBackend::new(), &model, &reference, prompt_file.as_deref(), prefill_chunk),
            d => compare_ref(&CudaBackend::new(parse_cuda(d)?)?, &model, &reference, prompt_file.as_deref(), prefill_chunk),
        },
        Cmd::Compare { model, prompt, steps, gpu, block_size, prefill_chunk, kv_type } => {
            compare(&model, &prompt, steps, gpu, block_size, prefill_chunk, kv_type)
        }
    }
}

fn inspect(path: &PathBuf, list_tensors: bool) -> Result<()> {
    let g = Gguf::open(path)?;
    println!("GGUF v{}, {} metadata keys, {} tensors", g.version, g.metadata.len(), g.tensors.len());
    for (k, v) in &g.metadata {
        println!("  {k} = {}", v.summary());
    }
    let mut by_type = std::collections::BTreeMap::<&str, (usize, usize)>::new();
    for t in &g.tensors {
        let e = by_type.entry(t.dtype.name()).or_default();
        e.0 += 1;
        e.1 += t.size;
        if list_tensors {
            println!("  {:<32} {:>6} {:?}", t.name, t.dtype.name(), t.dims);
        }
    }
    for (ty, (n, bytes)) in by_type {
        println!("  {ty}: {n} tensors, {:.1} MiB", bytes as f64 / (1 << 20) as f64);
    }
    Ok(())
}

fn tokenize(path: &PathBuf, text: &str) -> Result<()> {
    let g = Gguf::open(path)?;
    let tok = Tokenizer::from_gguf(&g)?;
    let ids = tok.encode(text)?;
    for id in &ids {
        println!("{id:>8}  {:?}", tok.id_to_token(*id).unwrap_or_default());
    }
    println!("{} tokens", ids.len());
    Ok(())
}

fn run<B: Backend>(backend: &B, a: RunArgs) -> Result<()> {
    let t0 = Instant::now();
    let g = Gguf::open(&a.model)?;
    let tok = Tokenizer::from_gguf(&g)?;

    let text = if a.raw {
        a.prompt.clone()
    } else {
        let mut msgs = Vec::new();
        if let Some(s) = &a.system {
            msgs.push(ChatMessage { role: "system".into(), content: s.clone() });
        }
        msgs.push(ChatMessage { role: "user".into(), content: a.prompt.clone() });
        tok.apply_chat_template(&msgs, true, !a.no_think)?
    };
    let prompt = tok.encode(&text)?;
    if a.print_prompt {
        println!("{text}");
        eprintln!("[rllm] {} tokens", prompt.len());
        return Ok(());
    }
    anyhow::ensure!(prompt.len() + a.max_tokens <= a.ctx, "prompt + max_tokens exceeds --ctx {}", a.ctx);

    let model = AnyModel::load(backend, &g)?;
    eprintln!(
        "[rllm] {} on {}: {}, loaded in {:.2}s",
        g.get_str("general.name").unwrap_or("model"),
        backend.name(),
        model.summary(),
        t0.elapsed().as_secs_f64()
    );

    let mut generator = Generator::with_kv(backend, &model, a.ctx, a.block_size, a.kv_type)?;
    let mut sampler = Sampler::new(SamplerConfig { temperature: a.temperature, top_k: a.top_k, top_p: a.top_p, seed: a.seed });
    let mut stream = tok.stream(false);
    let mut out = std::io::stdout().lock();

    let t_prefill = Instant::now();
    let mut logits = generator.feed(&prompt, a.prefill_chunk)?;
    let prefill_s = t_prefill.elapsed().as_secs_f64();

    let t_decode = Instant::now();
    let mut n_gen = 0;
    for _ in 0..a.max_tokens {
        let id = sampler.sample(&logits);
        n_gen += 1;
        if tok.is_stop(id) {
            break;
        }
        if a.show_ids {
            write!(out, "[{id}]")?;
        }
        if let Some(s) = stream.push(id)? {
            write!(out, "{s}")?;
            out.flush()?;
        }
        logits = generator.feed(&[id], 1)?;
    }
    let decode_s = t_decode.elapsed().as_secs_f64();
    writeln!(out)?;
    eprintln!(
        "[rllm] prompt {} tok in {:.2}s ({:.1} tok/s) | generated {} tok in {:.2}s ({:.1} tok/s)",
        prompt.len(),
        prefill_s,
        prompt.len() as f64 / prefill_s,
        n_gen,
        decode_s,
        n_gen as f64 / decode_s
    );
    Ok(())
}

fn compare(path: &PathBuf, prompt: &str, steps: usize, gpu: usize, block_size: usize, prefill_chunk: usize, kv: KvType) -> Result<()> {
    let g = Gguf::open(path)?;
    let tok = Tokenizer::from_gguf(&g)?;
    let cpu = CpuBackend::new();
    let cuda = CudaBackend::new(gpu)?;
    eprintln!("[rllm] comparing {} against {}", cuda.name(), cpu.name());
    let (m_cpu, m_gpu) = (AnyModel::load(&cpu, &g)?, AnyModel::load(&cuda, &g)?);
    let ids = tok.encode(prompt)?;
    let ctx = ids.len() + steps + 1;
    let mut a = Generator::with_kv(&cpu, &m_cpu, ctx, block_size, kv)?;
    let mut b = Generator::with_kv(&cuda, &m_gpu, ctx, block_size, kv)?;

    let mut la = a.feed(&ids, prefill_chunk)?;
    let mut lb = b.feed(&ids, prefill_chunk)?;
    let mut worst = 0f32;
    let mut mismatches = 0;
    println!("{:>5} {:>10} {:>10} {:>8} {:>8}  token", "step", "max|diff|", "rel", "cpu", "cuda");
    for step in 0..=steps {
        let (diff, rel) = logit_diff(&la, &lb);
        worst = worst.max(rel);
        let (ta, tb) = (rllm_model::sampler::argmax(&la), rllm_model::sampler::argmax(&lb));
        if ta != tb {
            mismatches += 1;
        }
        let label = if step == 0 { "prompt".to_string() } else { format!("decode {step}") };
        println!(
            "{label:>5} {diff:>10.4} {rel:>10.2e} {ta:>8} {tb:>8}  {:?}{}",
            tok.id_to_token(ta).unwrap_or_default(),
            if ta != tb {
                // How close the reference itself was between the two choices.
                format!("  <-- argmax differs (CPU margin {:.3})", la[ta as usize] - la[tb as usize])
            } else {
                String::new()
            }
        );
        if step == steps {
            break;
        }
        // Both backends continue from the CPU's choice so errors don't compound.
        la = a.feed(&[ta], 1)?;
        lb = b.feed(&[ta], 1)?;
    }
    println!("worst relative diff {worst:.2e}, argmax mismatches {mismatches}/{}", steps + 1);
    Ok(())
}

/// Max absolute difference, and that divided by the reference's max |logit|.
fn logit_diff(reference: &[f32], other: &[f32]) -> (f32, f32) {
    let diff = reference.iter().zip(other).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    let scale = reference.iter().map(|x| x.abs()).fold(0f32, f32::max).max(1e-6);
    (diff, diff / scale)
}

fn parse_cuda(d: &str) -> Result<usize> {
    Ok(match d.strip_prefix("cuda") {
        Some("") => 0,
        Some(n) => n.trim_start_matches(':').parse()?,
        None => anyhow::bail!("unknown device {d:?} (expected cpu, cuda or cuda:N)"),
    })
}

struct RefDump {
    n_vocab: usize,
    n_prompt: usize,
    tokens: Vec<u32>,
    logits: Vec<f32>,
}

fn read_ref(path: &std::path::Path) -> Result<RefDump> {
    let bytes = std::fs::read(path)?;
    anyhow::ensure!(bytes.len() >= 20 && &bytes[..8] == b"RLLMREF1", "{} is not a llama_ref dump", path.display());
    let u32_at = |i: usize| u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap()) as usize;
    let (n_vocab, n_prompt, n_total) = (u32_at(8), u32_at(12), u32_at(16));
    let tok_end = 20 + 4 * n_total;
    anyhow::ensure!(bytes.len() == tok_end + 4 * n_total * n_vocab, "truncated reference dump");
    let tokens = (0..n_total).map(|i| u32_at(20 + 4 * i) as u32).collect();
    let logits = bytes[tok_end..].chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
    Ok(RefDump { n_vocab, n_prompt, tokens, logits })
}

/// KL(p || q) for softmax(p_logits), softmax(q_logits), in nats.
fn kl_divergence(p_logits: &[f32], q_logits: &[f32]) -> f64 {
    let log_softmax = |x: &[f32]| {
        let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let lse = x.iter().map(|&v| (v as f64 - max).exp()).sum::<f64>().ln() + max;
        x.iter().map(move |&v| v as f64 - lse).collect::<Vec<f64>>()
    };
    let (lp, lq) = (log_softmax(p_logits), log_softmax(q_logits));
    lp.iter().zip(&lq).map(|(a, b)| a.exp() * (a - b)).sum::<f64>().max(0.0)
}

fn compare_ref<B: Backend>(
    backend: &B,
    model_path: &PathBuf,
    ref_path: &std::path::Path,
    prompt_file: Option<&std::path::Path>,
    prefill_chunk: usize,
) -> Result<()> {
    let r = read_ref(ref_path)?;
    let g = Gguf::open(model_path)?;
    let model = AnyModel::load(backend, &g)?;
    anyhow::ensure!(model.vocab_size() == r.n_vocab, "vocab mismatch: model {} vs reference {}", model.vocab_size(), r.n_vocab);
    eprintln!("[rllm] {} on {} vs llama.cpp: {} prompt + {} generated tokens", model.summary(), backend.name(), r.n_prompt, r.tokens.len() - r.n_prompt);

    if let Some(pf) = prompt_file {
        let tok = Tokenizer::from_gguf(&g)?;
        let ours = tok.encode(&std::fs::read_to_string(pf)?)?;
        let theirs = &r.tokens[..r.n_prompt];
        match ours.iter().zip(theirs).position(|(a, b)| a != b) {
            None if ours.len() == theirs.len() => println!("tokenization: identical ({} tokens)", ours.len()),
            first => {
                let i = first.unwrap_or(ours.len().min(theirs.len()));
                println!("tokenization: DIFFERS at token {i} (ours {} tokens, llama.cpp {})", ours.len(), theirs.len());
                let show = |ids: &[u32]| ids.iter().map(|&t| format!("{t}:{:?}", tok.id_to_token(t).unwrap_or_default())).collect::<Vec<_>>().join(" ");
                println!("  ours:      {}", show(&ours[i.saturating_sub(2)..(i + 4).min(ours.len())]));
                println!("  llama.cpp: {}", show(&theirs[i.saturating_sub(2)..(i + 4).min(theirs.len())]));
            }
        }
    }

    let ctx = r.tokens.len() + 1;
    let mut generator = Generator::new(backend, &model, ctx, 16)?;
    let ours = generator.feed_all(&r.tokens, prefill_chunk)?;
    let v = r.n_vocab;

    let (mut kl_sum, mut kl_max, mut top1_agree, mut worst_rel) = (0f64, 0f64, 0usize, 0f32);
    println!("{:>6} {:>10} {:>10} {:>10}  {:>8} {:>8}  next token", "pos", "max|diff|", "rel", "KL", "llama", "rllm");
    for i in 0..r.tokens.len() {
        let (a, b) = (&r.logits[i * v..(i + 1) * v], &ours[i * v..(i + 1) * v]);
        let (diff, rel) = logit_diff(a, b);
        let kl = kl_divergence(a, b);
        let (ta, tb) = (rllm_model::sampler::argmax(a), rllm_model::sampler::argmax(b));
        kl_sum += kl;
        kl_max = kl_max.max(kl);
        worst_rel = worst_rel.max(rel);
        top1_agree += (ta == tb) as usize;
        // Print the last few prompt rows and every generated row.
        if i + 3 >= r.n_prompt || ta != tb {
            println!("{i:>6} {diff:>10.4} {rel:>10.2e} {kl:>10.2e}  {ta:>8} {tb:>8}{}", if ta != tb { "  <-- top-1 differs" } else { "" });
        }
    }
    let n = r.tokens.len();
    println!(
        "summary: {n} positions | top-1 agreement {top1_agree}/{n} | KL mean {:.2e} max {:.2e} | worst relative logit diff {:.2e}",
        kl_sum / n as f64,
        kl_max,
        worst_rel
    );
    Ok(())
}

fn perplexity<B: Backend>(
    backend: &B,
    model_path: &PathBuf,
    file: &PathBuf,
    n_ctx: usize,
    max_chunks: Option<usize>,
    kv_type: KvType,
) -> Result<()> {
    let g = Gguf::open(model_path)?;
    let tok = Tokenizer::from_gguf(&g)?;
    let model = AnyModel::load(backend, &g)?;
    let text = std::fs::read_to_string(file)?;
    let tokens = tok.encode(&text)?;
    anyhow::ensure!(n_ctx >= 4 && tokens.len() >= 2 * n_ctx, "need at least {} tokens for --ctx {n_ctx}", 2 * n_ctx);
    let n_chunk = (tokens.len() / n_ctx).min(max_chunks.unwrap_or(usize::MAX));
    let vocab = model.vocab_size();
    // Score positions first..n_ctx-2, each predicting the next token.
    let first = n_ctx / 2;
    let rows: Vec<u32> = (first as u32..n_ctx as u32 - 1).collect();
    eprintln!("[rllm] {} tokens, {n_chunk} chunks of {n_ctx}, {}", tokens.len(), backend.name());
    let (mut nll, mut nll2, mut count) = (0f64, 0f64, 0usize);
    let t0 = Instant::now();
    for c in 0..n_chunk {
        let mut chunk = tokens[c * n_ctx..(c + 1) * n_ctx].to_vec();
        if tok.add_bos
            && let Some(b) = tok.bos
        {
            chunk[0] = b;
        }
        let mut generator = Generator::with_kv(backend, &model, n_ctx, 16, kv_type)?;
        let logits = generator.feed_rows(&chunk, &rows)?;
        // Per-row negative log-likelihood of the next token; rows split
        // across threads (a full-vocab log-softmax per row is the slow part).
        let row_nll = |i: usize, row: &[f32]| {
            let target = chunk[first + i + 1] as usize;
            let max = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64;
            let lse = row.iter().map(|&v| (v as f64 - max).exp()).sum::<f64>().ln() + max;
            lse - row[target] as f64
        };
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
        let per = rows.len().div_ceil(threads);
        let parts: Vec<Vec<f64>> = std::thread::scope(|sc| {
            let handles: Vec<_> = logits
                .chunks(per * vocab)
                .enumerate()
                .map(|(t, block)| {
                    let row_nll = &row_nll;
                    sc.spawn(move || block.chunks_exact(vocab).enumerate().map(|(j, row)| row_nll(t * per + j, row)).collect())
                })
                .collect();
            handles.into_iter().map(|h| h.join().expect("scoring thread")).collect()
        });
        for v in parts.into_iter().flatten() {
            nll += v;
            nll2 += v * v;
            count += 1;
        }
        eprint!("[{}]{:.4} ", c + 1, (nll / count as f64).exp());
        let _ = std::io::stderr().flush();
    }
    eprintln!();
    let mean = nll / count as f64;
    let ppl = mean.exp();
    // Same uncertainty estimate as llama-perplexity.
    let var = (nll2 / count as f64 - mean * mean).max(0.0);
    let err = ppl * (var / (count - 1) as f64).sqrt();
    println!(
        "Final estimate: PPL = {ppl:.4} +/- {err:.5}  ({n_chunk} chunks, {count} scored tokens, {:.1} s)",
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

fn bench<B: Backend>(
    backend: &B,
    model_path: &PathBuf,
    n_prompt: usize,
    n_gen: usize,
    reps: usize,
    depth: usize,
    kv_type: KvType,
) -> Result<()> {
    let g = Gguf::open(model_path)?;
    let model = AnyModel::load(backend, &g)?;
    let name = g.get_str("general.name").unwrap_or("model").to_string();
    eprintln!("[rllm] {name} on {}: {}", backend.name(), model.summary());

    // Fixed pseudo-random token ids (content doesn't affect speed).
    let vocab = model.vocab_size() as u64;
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next_tok = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state % vocab) as u32
    };
    let ctx = depth + n_prompt.max(n_gen) + 1;
    let context: Vec<u32> = (0..depth).map(|_| next_tok()).collect();

    let mut measure = |label: String, n: usize, batched: bool| -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let tokens: Vec<u32> = (0..n).map(|_| next_tok()).collect();
        let mut rates = Vec::with_capacity(reps);
        // One cache for all iterations (CUDA graphs recorded in the warmup
        // are tied to it); the first iteration is warmup.
        let mut generator = Generator::with_kv(backend, &model, ctx, 16, kv_type)?;
        for i in 0..=reps {
            generator.reset();
            if depth > 0 {
                generator.feed(&context, 512)?;
            }
            backend.synchronize()?;
            let t0 = Instant::now();
            if batched {
                generator.feed(&tokens, n)?;
            } else {
                for t in &tokens {
                    generator.feed(std::slice::from_ref(t), 1)?;
                }
            }
            backend.synchronize()?;
            if i > 0 {
                rates.push(n as f64 / t0.elapsed().as_secs_f64());
            }
        }
        let mean = rates.iter().sum::<f64>() / rates.len() as f64;
        let sd = (rates.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / rates.len() as f64).sqrt();
        println!("| {name} | {} | {label} | {mean:.2} ± {sd:.2} |", backend.name());
        Ok(())
    };
    println!("| model | backend | test | t/s |");
    println!("|---|---|---|---:|");
    let at = match (depth, kv_type) {
        (0, KvType::F16) => String::new(),
        (0, kv) => format!(" (KV {kv})"),
        (d, kv) => format!(" @ d{d} (KV {kv})"),
    };
    measure(format!("pp{n_prompt}{at}"), n_prompt, true)?;
    measure(format!("tg{n_gen}{at}"), n_gen, false)?;
    Ok(())
}
