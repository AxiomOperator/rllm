//! The inference engine: one thread that owns the backend, the model and a
//! paged KV cache shared by all requests.
//!
//! It runs a continuous-batching loop: up to `parallel` generations share
//! each forward pass (one decode token each, plus a chunk of one waiting
//! prompt), new requests join as soon as there is room, and finished ones
//! leave without waiting for the rest.
//!
//! Finished sequences stay in a prefix cache: a request whose prompt starts
//! with the same tokens (the next turn of a chat, a repeated system prompt)
//! reuses their KV blocks instead of recomputing them. When blocks run out,
//! the cache is evicted first, then the newest generation is paused
//! (preempted) and resumed later by recomputing its tokens.
//!
//! With a draft model (speculative decoding), a small model with the same
//! vocabulary proposes a few tokens per decoding sequence, and one forward
//! pass of the main model checks them all: every accepted token saves a
//! main-model step. Verification keeps the output distributed exactly as
//! without a draft (identical text when greedy).
//!
//! The KV pools live as long as the engine. CUDA decode graphs capture their
//! device addresses, so they must not be reallocated.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::time::Instant;

use anyhow::{Context, Result};
use rllm_core::{Backend, KvType};
use rllm_gguf::Gguf;
use rllm_kv::{PrefixCache, SeqKv};
use rllm_model::{AnyModel, Dist, KvPool, Model, RowSample, Sampler, SamplerConfig, StepItem, Want};
use rllm_tokenizer::Tokenizer;
use tokio::sync::{mpsc as tmpsc, oneshot};

use crate::json_constraint::JsonState;
use crate::metrics::Metrics;

/// Sampling and stopping parameters for one generation.
#[derive(Debug, Clone)]
pub struct GenParams {
    /// Upper bound on generated tokens (further capped by the context size).
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub seed: u64,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub logit_bias: Vec<(u32, f32)>,
    /// Return the sampled token's logprob plus this many alternatives.
    pub logprobs: Option<usize>,
    /// Constrain output to JSON (optionally matching a schema), starting
    /// from this state; stop when the value is complete.
    pub json: Option<JsonState>,
}

impl Default for GenParams {
    fn default() -> Self {
        Self {
            max_tokens: usize::MAX,
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            seed: 0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            logit_bias: Vec::new(),
            logprobs: None,
            json: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TokenLogprobs {
    pub logprob: f32,
    /// Most likely tokens at this position: (id, logprob), descending.
    pub top: Vec<(u32, f32)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Finish {
    /// End of sequence (or the JSON constraint completed).
    Stop,
    /// Token budget or context exhausted.
    Length,
    /// Cancelled by the caller (stop sequence found, client gone, cancel).
    Cancelled,
}

#[derive(Debug)]
pub enum GenEvent {
    Token { id: u32, logprobs: Option<TokenLogprobs> },
    /// `cached_tokens`: prompt tokens whose KV came from the prefix cache.
    Done { finish: Finish, prompt_tokens: usize, cached_tokens: usize },
    Error(String),
}

pub enum Job {
    Generate { prompt: Vec<u32>, params: GenParams, events: tmpsc::UnboundedSender<GenEvent>, cancel: Arc<AtomicBool> },
    Embed { inputs: Vec<Vec<u32>>, reply: oneshot::Sender<Result<Vec<Vec<f32>>, String>> },
}

/// Facts about the loaded model, reported once the engine is ready.
#[derive(Debug, Clone)]
pub struct EngineInfo {
    pub backend: String,
    pub summary: String,
    pub vocab: usize,
    /// The draft model, when speculative decoding is on.
    pub draft: Option<String>,
    /// Prefix cache host tier size, in blocks.
    pub host_blocks: Option<usize>,
}

#[derive(Clone)]
pub struct EngineHandle {
    jobs: mpsc::Sender<Job>,
    /// Generations submitted but not yet taken by the engine thread; with
    /// the engine's waiting queue (`metrics.waiting`), capped by `max_queue`.
    queued: Arc<AtomicUsize>,
    metrics: Arc<Metrics>,
    max_queue: usize,
    /// False once the engine thread has exited (including by a panic).
    alive: Arc<AtomicBool>,
}

impl EngineHandle {
    pub fn submit(&self, job: Job) -> Result<()> {
        if !self.is_alive() {
            anyhow::bail!("engine stopped");
        }
        if let Job::Generate { .. } = job {
            let n = self.queued.fetch_add(1, Ordering::Relaxed) + self.metrics.waiting.load(Ordering::Relaxed) as usize;
            if self.max_queue > 0 && n >= self.max_queue {
                self.queued.fetch_sub(1, Ordering::Relaxed);
                anyhow::bail!("server busy: {n} requests already queued (--max-queue {})", self.max_queue);
            }
        }
        self.jobs.send(job).map_err(|_| anyhow::anyhow!("engine stopped"))
    }

    pub fn is_alive(&self) -> bool {
        self.alive.load(Ordering::Relaxed)
    }
}

/// Clears the engine's alive flag when its thread ends, however it ends.
struct AliveGuard(Arc<AtomicBool>);

impl Drop for AliveGuard {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Relaxed);
    }
}

/// Token texts for constrained decoding: decoded text per id, and whether
/// the id is a special/control token (never allowed inside JSON).
pub struct Vocab {
    pub text: Vec<String>,
    pub special: Vec<bool>,
}

impl Vocab {
    pub fn build(tok: &Tokenizer) -> Self {
        let n = tok.vocab_size();
        let mut text = Vec::with_capacity(n);
        let mut special = Vec::with_capacity(n);
        // SentencePiece decoders drop a leading space at the start of the
        // text, so a token decoded alone loses it: decode each after an
        // anchor token and keep what follows the anchor's text.
        let anchor = tok.encode_with("a", false).ok().and_then(|ids| ids.last().copied());
        let anchor_text = anchor.map(|a| tok.token_text(a)).unwrap_or_default();
        for id in 0..n as u32 {
            let raw = tok.token_text(id);
            let plain = tok.decode(&[id], true).unwrap_or_default();
            special.push(raw != plain || tok.is_stop(id));
            let in_context = anchor
                .and_then(|a| tok.decode(&[a, id], false).ok())
                .and_then(|t| t.strip_prefix(anchor_text.as_str()).map(str::to_owned));
            text.push(in_context.unwrap_or(raw));
        }
        Self { text, special }
    }
}

pub struct EngineConfig {
    /// Most tokens (prompt + output) one request may use.
    pub ctx: usize,
    /// Prompt tokens per forward pass.
    pub prefill_chunk: usize,
    /// Most generations batched together.
    pub parallel: usize,
    /// Size of the shared KV pool, in tokens (at least `ctx`).
    pub kv_tokens: usize,
    /// Keep finished sequences for prefix reuse.
    pub prefix_cache: bool,
    /// KV cache storage (both models).
    pub kv_type: KvType,
    /// Draft model for speculative decoding (same vocabulary as the model).
    pub draft: Option<Arc<Gguf>>,
    /// Most tokens the draft proposes per sequence and step.
    pub draft_tokens: usize,
    /// Prompt lookup: propose drafts by finding the latest tokens earlier in
    /// the sequence (no draft model needed; with one, lookup goes first).
    pub prompt_lookup: bool,
    /// Pinned system memory for the prefix cache's host tier, in bytes (0: none).
    pub host_cache_bytes: usize,
    /// Sample every token on the host from full logits rows, instead of on
    /// the device where possible (see `device_sample`).
    pub host_sampling: bool,
    /// Most generations waiting for a batch slot before new ones get 503
    /// (0: no limit).
    pub max_queue: usize,
}

pub const BLOCK_SIZE: usize = 16;

/// Most tokens in a step that verifies drafts. Up to 8 tokens a step stays
/// on the matvec kernels and in a CUDA graph; beyond that the main model's
/// pass costs much more, so drafts are shortened to fit (and switched off
/// once 5 sequences decode together).
const SPEC_STEP_TOKENS: usize = 8;
/// After this many verifications in a row that accept no draft token, a
/// sequence decodes without drafts for SPEC_PAUSE_STEPS steps.
const SPEC_MAX_MISSES: u32 = 3;
const SPEC_PAUSE_STEPS: u32 = 8;
/// The draft model drafts only for sequences up to this many tokens. Beyond
/// it the draft's own attention and its catch-up over the prompt (it has no
/// prefix cache) cost more than it saves: Qwen3-8B with a Qwen3-0.6B draft
/// on the RTX 6000 Ada decoded +30% at 900 tokens, +22% at 4k, -2% at 8k
/// and -24% at 17k. Prompt lookup has no such cost and keeps going.
pub const DRAFT_MAX_CONTEXT: usize = 6144;

/// Size of the draft model's KV pool for a main pool of `kv_tokens`: enough
/// for every sequence that can draft at once (at most 4 decode together
/// while speculating). If it runs out, that step goes without drafts.
pub fn draft_pool_tokens(kv_tokens: usize) -> usize {
    kv_tokens.min(4 * DRAFT_MAX_CONTEXT)
}

/// Prompt lookup matches the last LOOKUP_MAX_NGRAM tokens earlier in the
/// sequence, then shorter tails down to LOOKUP_MIN_NGRAM.
const LOOKUP_MAX_NGRAM: usize = 4;
const LOOKUP_MIN_NGRAM: usize = 2;

/// Start the engine thread. `make_backend` runs on that thread (CUDA
/// contexts are tied to it). Returns once the model is loaded.
pub fn start<B, F>(
    make_backend: F,
    gguf: Arc<Gguf>,
    tok: Arc<Tokenizer>,
    vocab: Arc<Vocab>,
    cfg: EngineConfig,
    metrics: Arc<Metrics>,
) -> Result<(EngineHandle, EngineInfo)>
where
    B: Backend + 'static,
    F: FnOnce() -> Result<B> + Send + 'static,
{
    let (jobs_tx, jobs_rx) = mpsc::channel::<Job>();
    let (ready_tx, ready_rx) = mpsc::channel::<Result<EngineInfo, String>>();
    let alive = Arc::new(AtomicBool::new(true));
    let queued = Arc::new(AtomicUsize::new(0));
    let (alive_engine, queued_engine, metrics_handle) = (alive.clone(), queued.clone(), metrics.clone());
    let max_queue = cfg.max_queue;
    std::thread::Builder::new()
        .name("rllm-engine".into())
        .spawn(move || {
            let _alive = AliveGuard(alive_engine);
            pin_to_performance_cores();
            let setup = (|| -> Result<_> {
                let backend = make_backend()?;
                let model = AnyModel::load(&backend, &gguf)?;
                let draft = match &cfg.draft {
                    Some(g) => {
                        let d = AnyModel::load(&backend, g).context("loading the draft model")?;
                        check_draft_vocab(&tok, &model, g, &d)?;
                        Some(d)
                    }
                    None => None,
                };
                Ok((backend, model, draft))
            })();
            let (backend, model, draft_model) = match setup {
                Ok(v) => v,
                Err(e) => {
                    let _ = ready_tx.send(Err(format!("{e:#}")));
                    return;
                }
            };
            let kv_tokens = cfg.kv_tokens.max(cfg.ctx);
            let pools = (|| -> Result<_> {
                let mut pool = KvPool::new(&backend, &model, kv_tokens, BLOCK_SIZE, cfg.kv_type)?;
                if cfg.host_cache_bytes > 0 {
                    pool.add_host_tier(cfg.host_cache_bytes, cfg.kv_type).context("allocating the host cache")?;
                }
                let draft = match &draft_model {
                    Some(d) => Some(KvPool::new(&backend, d, draft_pool_tokens(kv_tokens), BLOCK_SIZE, cfg.kv_type).context("draft model KV cache")?),
                    None => None,
                };
                Ok((pool, draft))
            })();
            let (pool, draft) = match pools {
                Ok(p) => p,
                Err(e) => {
                    let _ = ready_tx.send(Err(format!("{e:#}")));
                    return;
                }
            };
            let info = EngineInfo {
                backend: backend.name(),
                summary: model.summary(),
                vocab: model.vocab_size(),
                draft: draft_model.as_ref().map(|d| d.summary()),
                host_blocks: pool.host_blocks().map(|(n, _)| n),
            };
            let _ = ready_tx.send(Ok(info));
            Scheduler::new(pool, draft, &tok, &vocab, &cfg, &metrics, queued_engine).run(jobs_rx);
        })
        .context("spawn engine thread")?;
    let info = ready_rx.recv().context("engine thread exited")?.map_err(|e| anyhow::anyhow!(e))?;
    Ok((EngineHandle { jobs: jobs_tx, queued, metrics: metrics_handle, max_queue, alive }, info))
}

/// The draft model must tokenize like the main model: same logits width and
/// the same token strings.
fn check_draft_vocab<B: Backend>(tok: &Tokenizer, model: &AnyModel<B>, draft_gguf: &Gguf, draft: &AnyModel<B>) -> Result<()> {
    anyhow::ensure!(
        model.vocab_size() == draft.vocab_size(),
        "draft model vocabulary ({}) differs from the model's ({})",
        draft.vocab_size(),
        model.vocab_size()
    );
    let dt = Tokenizer::from_gguf(draft_gguf).context("draft model tokenizer")?;
    let n = tok.vocab_size().min(dt.vocab_size());
    if let Some(id) = (0..n as u32).find(|&i| tok.token_text(i) != dt.token_text(i)) {
        anyhow::bail!(
            "draft model tokenizer differs from the model's (token {id}: {:?} vs {:?})",
            dt.token_text(id),
            tok.token_text(id)
        );
    }
    Ok(())
}

/// On hybrid CPUs (Intel P- and E-cores), keep the engine thread on the
/// performance cores. Each decode step runs host work between GPU waits
/// (launches, sampling a 262k-entry vocabulary); on an E-core it is several
/// times slower: measured 7.92 → 7.64 ms per Nemotron token on an
/// i9-14900K. Sampler threads inherit the mask. `RLLM_NO_PIN=1` disables it.
fn pin_to_performance_cores() {
    #[cfg(target_os = "linux")]
    {
        if std::env::var_os("RLLM_NO_PIN").is_some() {
            return;
        }
        // Only hybrid Intel CPUs have this; elsewhere all cores are alike.
        let Ok(list) = std::fs::read_to_string("/sys/devices/cpu_core/cpus") else { return };
        let mut cpus = Vec::new();
        for part in list.trim().split(',').filter(|p| !p.is_empty()) {
            let (a, b) = part.split_once('-').unwrap_or((part, part));
            let (Ok(a), Ok(b)) = (a.parse::<usize>(), b.parse::<usize>()) else { return };
            cpus.extend(a..=b);
        }
        if cpus.is_empty() {
            return;
        }
        // SAFETY: cpu_set_t is plain data; the set is initialized before use.
        unsafe {
            let mut set: libc::cpu_set_t = std::mem::zeroed();
            for &c in &cpus {
                if c < libc::CPU_SETSIZE as usize {
                    libc::CPU_SET(c, &mut set);
                }
            }
            // pid 0: the calling thread.
            libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
        }
    }
}

/// One generation, waiting or running.
struct Seq {
    params: GenParams,
    events: tmpsc::UnboundedSender<GenEvent>,
    cancel: Arc<AtomicBool>,
    prompt_len: usize,
    /// Prompt plus generated tokens. `kv.len` of them are in the cache; the
    /// rest still have to be fed (the prompt during prefill, the last
    /// sampled token while decoding, everything after a preemption).
    tokens: Vec<u32>,
    kv: SeqKv,
    /// Prompt tokens served from the prefix cache (first admission).
    cached: Option<usize>,
    budget: usize,
    generated: usize,
    sampler: Sampler,
    counts: HashMap<u32, u32>,
    json: Option<JsonState>,
    /// Admission order; the newest running sequence is preempted first.
    admitted: u64,
    /// The draft model's cache: `draft_kv.len` tokens of `tokens`.
    draft_kv: SeqKv,
    /// Draws the draft's proposals (its own random stream).
    draft_sampler: Sampler,
    /// Draft tokens to propose next: one more than were last accepted, so
    /// it follows how predictable the text currently is.
    spec_k: usize,
    /// Steps without drafts left, after drafts kept failing outright.
    spec_pause: u32,
    /// Verifications in a row that accepted nothing.
    spec_misses: u32,
}

impl Seq {
    fn todo(&self) -> &[u32] {
        &self.tokens[self.kv.len..]
    }

    fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed) || self.events.is_closed()
    }
}

/// What sampling decided for a sequence.
enum Next {
    Continue,
    Done(Finish),
}

/// Draft tokens proposed for one sequence, with the distribution each was
/// drawn from.
#[derive(Default)]
struct Proposal {
    tokens: Vec<u32>,
    dists: Vec<Dist>,
    /// Found by prompt lookup rather than the draft model.
    lookup: bool,
}

impl Proposal {
    /// Prompt lookup's proposal for a sequence: up to `k` tokens that
    /// followed the most recent earlier occurrence of its last few tokens.
    /// Copying is deterministic, so each token's draft distribution is a
    /// point mass, which keeps verification exact.
    fn lookup(tokens: &[u32], k: usize) -> Self {
        let len = tokens.len();
        for n in (LOOKUP_MIN_NGRAM..=LOOKUP_MAX_NGRAM).rev() {
            if len <= n {
                continue;
            }
            let tail = &tokens[len - n..];
            if let Some(start) = (0..len - n).rev().find(|&i| tokens[i..i + n] == *tail) {
                let next = &tokens[start + n..(start + n + k).min(len)];
                return Self { tokens: next.to_vec(), dists: next.iter().map(|&t| Dist::Point(t)).collect(), lookup: true };
            }
        }
        Self::default()
    }
}

struct Scheduler<'a, 'm, B: Backend> {
    /// Shared with `EngineHandle`: jobs in the channel (decremented on accept).
    queued: Arc<AtomicUsize>,
    pool: KvPool<'m, B, AnyModel<B>>,
    /// The draft model's pool (speculative decoding).
    draft: Option<KvPool<'m, B, AnyModel<B>>>,
    cache: PrefixCache,
    tok: &'a Tokenizer,
    vocab: &'a Vocab,
    cfg: &'a EngineConfig,
    metrics: &'a Metrics,
    waiting: VecDeque<Seq>,
    embeds: VecDeque<Job>,
    running: Vec<Seq>,
    admissions: u64,
    timing: StepTiming,
}

/// Where step time goes (`RLLM_ENGINE_TIMING=1` prints it every 256 steps).
#[derive(Default)]
struct StepTiming {
    enabled: bool,
    steps: u32,
    forward: std::time::Duration,
    sample: std::time::Duration,
    /// Draft model (speculative decoding), forward passes and sampling.
    draft: std::time::Duration,
    total: std::time::Duration,
}

impl StepTiming {
    fn report(&mut self) {
        if !self.enabled {
            return;
        }
        self.steps += 1;
        if self.steps == 256 {
            let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3 / 256.0;
            eprintln!(
                "[rllm] engine step: {:.3} ms (draft {:.3}, forward {:.3}, sampling {:.3}, other {:.3})",
                ms(self.total),
                ms(self.draft),
                ms(self.forward),
                ms(self.sample),
                ms(self.total - self.draft - self.forward - self.sample)
            );
            *self = Self { enabled: true, ..Self::default() };
        }
    }
}

impl<'a, 'm, B: Backend> Scheduler<'a, 'm, B> {
    fn new(
        pool: KvPool<'m, B, AnyModel<B>>,
        draft: Option<KvPool<'m, B, AnyModel<B>>>,
        tok: &'a Tokenizer,
        vocab: &'a Vocab,
        cfg: &'a EngineConfig,
        metrics: &'a Metrics,
        queued: Arc<AtomicUsize>,
    ) -> Self {
        Metrics::set(&metrics.kv_blocks_total, pool.alloc.n_blocks());
        Self {
            queued,
            pool,
            draft,
            cache: PrefixCache::default(),
            tok,
            vocab,
            cfg,
            metrics,
            waiting: VecDeque::new(),
            embeds: VecDeque::new(),
            running: Vec::new(),
            admissions: 0,
            timing: StepTiming { enabled: std::env::var_os("RLLM_ENGINE_TIMING").is_some(), ..StepTiming::default() },
        }
    }

    fn run(mut self, jobs: mpsc::Receiver<Job>) {
        loop {
            if self.running.is_empty() && self.waiting.is_empty() && self.embeds.is_empty() {
                match jobs.recv() {
                    Ok(j) => self.accept(j),
                    Err(_) => return,
                }
            }
            while let Ok(j) = jobs.try_recv() {
                self.accept(j);
            }
            self.drop_cancelled();
            self.run_embeds();
            self.admit();
            self.publish();
            if self.running.is_empty() {
                if let Some(s) = self.waiting.pop_front() {
                    // Nothing running and it still doesn't fit: it never will.
                    let msg = format!("request needs more KV cache than the pool has ({} tokens)", self.pool.alloc.n_blocks() * BLOCK_SIZE);
                    let _ = s.events.send(GenEvent::Error(msg));
                }
                continue;
            }
            let t0 = Instant::now();
            if let Err(e) = self.step() {
                // A failed forward pass leaves the batch in an unknown state:
                // fail its requests, keep serving.
                let msg = format!("{e:#}");
                for mut s in std::mem::take(&mut self.running) {
                    let _ = s.events.send(GenEvent::Error(msg.clone()));
                    s.kv.free(&mut self.pool.alloc);
                    self.free_draft(&mut s);
                }
            }
            Metrics::add(&self.metrics.busy_us, t0.elapsed().as_micros() as usize);
        }
    }

    fn accept(&mut self, job: Job) {
        match job {
            Job::Generate { prompt, params, events, cancel } => {
                self.queued.fetch_sub(1, Ordering::Relaxed);
                if prompt.is_empty() || prompt.len() >= self.cfg.ctx {
                    let msg = format!("prompt ({} tokens) does not fit the context ({} tokens)", prompt.len(), self.cfg.ctx);
                    let _ = events.send(GenEvent::Error(msg));
                    return;
                }
                let budget = params.max_tokens.min(self.cfg.ctx - prompt.len());
                if budget == 0 {
                    let _ = events.send(GenEvent::Done { finish: Finish::Length, prompt_tokens: prompt.len(), cached_tokens: 0 });
                    return;
                }
                let sampler_cfg =
                    SamplerConfig { temperature: params.temperature, top_k: params.top_k, top_p: params.top_p, seed: params.seed };
                let sampler = Sampler::new(sampler_cfg.clone());
                let draft_sampler = Sampler::new(SamplerConfig { seed: params.seed ^ 0xD1B5_4A32_D192_ED03, ..sampler_cfg });
                let json = params.json.clone();
                self.waiting.push_back(Seq {
                    prompt_len: prompt.len(),
                    tokens: prompt,
                    kv: SeqKv::default(),
                    cached: None,
                    budget,
                    generated: 0,
                    sampler,
                    counts: HashMap::new(),
                    json,
                    admitted: 0,
                    draft_kv: SeqKv::default(),
                    draft_sampler,
                    spec_k: self.cfg.draft_tokens,
                    spec_pause: 0,
                    spec_misses: 0,
                    params,
                    events,
                    cancel,
                });
            }
            embed @ Job::Embed { .. } => self.embeds.push_back(embed),
        }
    }

    /// Finish cancelled generations (client gone, stop sequence, cancel).
    fn drop_cancelled(&mut self) {
        let mut i = 0;
        while i < self.running.len() {
            if self.running[i].cancelled() {
                let s = self.running.swap_remove(i);
                self.finish(s, Finish::Cancelled);
            } else {
                i += 1;
            }
        }
        let (keep, gone): (VecDeque<Seq>, VecDeque<Seq>) = self.waiting.drain(..).partition(|s| !s.cancelled());
        self.waiting = keep;
        for s in gone {
            self.finish(s, Finish::Cancelled);
        }
    }

    /// Report the end of a generation and retire its KV into the prefix cache.
    fn finish(&mut self, mut s: Seq, finish: Finish) {
        let cached = s.cached.unwrap_or(0);
        let _ = s.events.send(GenEvent::Done { finish, prompt_tokens: s.prompt_len, cached_tokens: cached });
        Metrics::add(&self.metrics.requests_finished, 1);
        Metrics::add(&self.metrics.prompt_tokens, s.prompt_len);
        Metrics::add(&self.metrics.cached_prompt_tokens, cached);
        Metrics::add(&self.metrics.generated_tokens, s.generated);
        self.retire(&mut s);
    }

    fn free_draft(&mut self, s: &mut Seq) {
        if let Some(d) = &mut self.draft {
            s.draft_kv.free(&mut d.alloc);
        }
    }

    fn retire(&mut self, s: &mut Seq) {
        self.free_draft(s);
        let kv = std::mem::take(&mut s.kv);
        if self.cfg.prefix_cache {
            self.cache.insert(&mut self.pool, &s.tokens[..kv.len], kv);
        } else {
            let mut kv = kv;
            kv.free(&mut self.pool.alloc);
        }
    }

    /// Move waiting generations into the batch while there is room: a free
    /// slot and KV blocks for their whole prompt (evicting cached prefixes
    /// as needed). Strictly in arrival order.
    fn admit(&mut self) {
        while self.running.len() < self.cfg.parallel.max(1) {
            let Some(mut s) = self.waiting.pop_front() else { return };
            if self.cfg.prefix_cache && s.kv.len == 0 {
                let (n, blocks) = self.cache.lookup(&mut self.pool, &s.tokens);
                s.kv = SeqKv::with_prefix(blocks, n);
            }
            // The prompt's blocks, plus one block of headroom per running
            // sequence so they can keep decoding.
            let need = s.kv.blocks_needed(BLOCK_SIZE, s.todo().len()) + self.running.len();
            if !self.cache.make_room(&mut self.pool, need) {
                s.kv.free(&mut self.pool.alloc);
                self.waiting.push_front(s);
                return;
            }
            if s.cached.is_none() {
                s.cached = Some(s.kv.len);
            }
            self.admissions += 1;
            s.admitted = self.admissions;
            self.running.push(s);
        }
    }

    fn publish(&self) {
        let m = self.metrics;
        Metrics::set(&m.running, self.running.len());
        Metrics::set(&m.waiting, self.waiting.len());
        Metrics::set(&m.kv_blocks_free, self.pool.alloc.n_free());
        Metrics::set(&m.prefix_cache_entries, self.cache.len());
        Metrics::set(&m.prefix_cache_blocks, self.cache.n_blocks());
        if let Some((total, free)) = self.pool.host_blocks() {
            let stats = self.cache.stats();
            Metrics::set(&m.host_blocks_total, total);
            Metrics::set(&m.host_blocks_free, free);
            Metrics::set(&m.host_cache_entries, self.cache.host_len());
            Metrics::set(&m.host_offloaded_blocks, stats.offloaded);
            Metrics::set(&m.host_restored_blocks, stats.restored);
        }
    }

    /// Draft tokens to propose for a sequence this step (0: none). Only
    /// for decoding sequences without a JSON constraint, within the
    /// remaining budget, and so that a step of `decoding` sequences stays
    /// within SPEC_STEP_TOKENS.
    fn spec_len(&self, s: &Seq, decoding: usize) -> usize {
        let any = self.draft.is_some() || self.cfg.prompt_lookup;
        if !any || s.todo().len() != 1 || s.json.is_some() || s.spec_pause > 0 {
            return 0;
        }
        let fit = (SPEC_STEP_TOKENS / decoding.max(1)).saturating_sub(1);
        s.spec_k.min(fit).min((s.budget - s.generated).saturating_sub(1))
    }

    /// Plan, run and sample one batched forward pass.
    fn step(&mut self) -> Result<()> {
        let t_start = Instant::now();
        // Decoding sequences first (one token each, plus drafts), then
        // prompt chunks in admission order within the prefill budget.
        self.running.sort_by_key(|s| (s.todo().len() > 1, s.admitted));
        let mut takes: Vec<usize>;
        let mut spec: Vec<usize>;
        let mut drafts = true;
        loop {
            let decoding = self.running.iter().filter(|s| s.todo().len() == 1).count();
            spec = self.running.iter().map(|s| if drafts { self.spec_len(s, decoding) } else { 0 }).collect();
            let mut budget = self.cfg.prefill_chunk.max(1);
            takes = self
                .running
                .iter()
                .zip(&spec)
                .map(|(s, &k)| {
                    let n = s.todo().len();
                    if n == 1 {
                        1 + k
                    } else {
                        let t = n.min(budget);
                        budget -= t;
                        t
                    }
                })
                .collect();
            let need: usize = self.running.iter().zip(&takes).map(|(s, &t)| s.kv.blocks_needed(BLOCK_SIZE, t)).sum();
            if self.cache.make_room(&mut self.pool, need) {
                break;
            }
            // Out of blocks: drafts only save work, so go without them first.
            if spec.iter().any(|&k| k > 0) {
                drafts = false;
                continue;
            }
            // Then pause the newest generation and recompute it later.
            let newest = (0..self.running.len()).max_by_key(|&i| self.running[i].admitted).unwrap();
            let mut s = self.running.remove(newest);
            self.retire(&mut s);
            self.waiting.push_front(s);
            Metrics::add(&self.metrics.preemptions, 1);
            anyhow::ensure!(!self.running.is_empty(), "KV cache too small for one sequence");
        }
        for s in &mut self.running {
            s.spec_pause = s.spec_pause.saturating_sub(1);
        }

        // Draft proposals: prompt lookup first (free), then the draft model
        // for the rest. If the draft can't run (its pool is full), this step
        // goes without: drafts only ever save work.
        let mut proposals: Vec<Proposal> = (0..self.running.len()).map(|_| Proposal::default()).collect();
        if spec.iter().any(|&k| k > 0) {
            let t = Instant::now();
            let mut model_spec = spec.clone();
            if self.cfg.prompt_lookup {
                for (i, s) in self.running.iter().enumerate() {
                    if spec[i] > 0 {
                        let p = Proposal::lookup(&s.tokens, spec[i]);
                        if !p.tokens.is_empty() {
                            model_spec[i] = 0;
                            proposals[i] = p;
                        }
                    }
                }
            }
            // Long sequences don't use the draft model (see DRAFT_MAX_CONTEXT);
            // their draft cache is released once they grow past it.
            for (i, s) in self.running.iter_mut().enumerate() {
                if s.tokens.len() > DRAFT_MAX_CONTEXT {
                    model_spec[i] = 0;
                    if let Some(d) = &mut self.draft {
                        s.draft_kv.free(&mut d.alloc);
                    }
                }
            }
            if self.draft.is_some() && model_spec.iter().any(|&k| k > 0) {
                match self.propose(&model_spec) {
                    Ok(p) => {
                        for (i, p) in p.into_iter().enumerate() {
                            if model_spec[i] > 0 {
                                proposals[i] = p;
                            }
                        }
                    }
                    Err(_) => {
                        for i in 0..self.running.len() {
                            let mut s = std::mem::take(&mut self.running[i].draft_kv);
                            if let Some(d) = &mut self.draft {
                                s.free(&mut d.alloc);
                            }
                        }
                    }
                }
            }
            for (i, p) in proposals.iter().enumerate() {
                if spec[i] > 0 {
                    takes[i] = 1 + p.tokens.len();
                    let c = if p.lookup { &self.metrics.lookup_tokens } else { &self.metrics.draft_tokens };
                    Metrics::add(c, p.tokens.len());
                }
            }
            self.timing.draft += t.elapsed();
        }

        let active: Vec<usize> = (0..self.running.len()).filter(|&i| takes[i] > 0).collect();
        // Logits rows per active sequence, and where its KV stood before.
        let mut rows_of = Vec::with_capacity(active.len());
        // Sequences whose next token is sampled on the device (see
        // `device_sample`); one spec per logits row, in row order.
        let mut on_device = vec![false; self.running.len()];
        let mut sample: Vec<Option<RowSample>> = Vec::new();
        let mut start_of = vec![0usize; self.running.len()];
        let out = {
            // Verified tokens: the pending one, then the drafts.
            let spec_tokens: Vec<Vec<u32>> = self
                .running
                .iter()
                .zip(&proposals)
                .map(|(s, p)| if p.tokens.is_empty() { Vec::new() } else { [s.todo(), &p.tokens].concat() })
                .collect();
            let mut items: Vec<StepItem> = Vec::with_capacity(active.len());
            let mut n_tok = 0;
            for (i, s) in self.running.iter_mut().enumerate() {
                if takes[i] == 0 {
                    continue;
                }
                let start = s.kv.len;
                start_of[i] = start;
                n_tok += takes[i];
                // `tokens` and `kv` are disjoint fields.
                let Seq { tokens, kv, .. } = s;
                if !spec_tokens[i].is_empty() {
                    rows_of.push(takes[i]);
                    sample.extend(std::iter::repeat_n(None, takes[i]));
                    items.push(StepItem { seq: kv, tokens: &spec_tokens[i], want: Want::All });
                } else {
                    let all = start + takes[i] == tokens.len();
                    rows_of.push(all as usize);
                    if all {
                        let spec = device_sample(&s.params, &s.json, &s.counts, tokens.len(), self.cfg.host_sampling);
                        on_device[i] = spec.is_some();
                        sample.push(spec);
                    }
                    items.push(StepItem { seq: kv, tokens: &tokens[start..start + takes[i]], want: if all { Want::Last } else { Want::None } });
                }
            }
            Metrics::add(&self.metrics.steps, 1);
            Metrics::add(&self.metrics.step_tokens, n_tok);
            let t = Instant::now();
            if sample.iter().all(Option::is_none) {
                sample.clear();
            }
            let out = self.pool.step_sampled(&mut items, false, &sample)?;
            self.timing.forward += t.elapsed();
            out
        };
        let t_sample = Instant::now();
        // Sample every sequence whose pending tokens are all in the cache now.
        let vocab_n = self.pool.model().vocab_size();
        let rllm_model::StepOut { rows: mut out, ids } = out;
        // Tokens already sampled on the device: send them.
        let mut ids = ids.into_iter();
        let mut device_results: Vec<(usize, Next, usize)> = Vec::new();
        for &i in active.iter().filter(|&&i| on_device[i]) {
            let id = ids.next().context("missing device-sampled id")?;
            device_results.push((i, emit(&mut self.running[i], id, None, &[], self.tok), 0));
        }
        // Logits rows are processed in place (no copies: a fresh 1 MB
        // buffer per token costs more in page faults than the argmax).
        let mut rest: &mut [f32] = &mut out;
        let lookup: Vec<bool> = proposals.iter().map(|p| p.lookup).collect();
        let mut jobs: Vec<(usize, &mut [f32], Proposal)> = Vec::new();
        for (&i, &n) in active.iter().zip(&rows_of) {
            if n > 0 && !on_device[i] {
                anyhow::ensure!(rest.len() >= n * vocab_n, "missing logits rows");
                let (rows, tail) = rest.split_at_mut(n * vocab_n);
                rest = tail;
                jobs.push((i, rows, std::mem::take(&mut proposals[i])));
            }
        }
        let (tok, vocab) = (self.tok, self.vocab);
        let run = move |s: &mut Seq, rows: &mut [f32], p: Proposal| -> (Next, usize) {
            if p.tokens.is_empty() { (sample_next(s, rows, tok, vocab), 0) } else { verify_drafts(s, rows, vocab_n, &p, tok) }
        };
        let results: Vec<(usize, Next, usize)> = if jobs.len() <= 1 {
            jobs.into_iter()
                .map(|(i, l, p)| {
                    let (n, kept) = run(&mut self.running[i], l, p);
                    (i, n, kept)
                })
                .collect()
        } else {
            // Sampling a large vocabulary takes a while: one thread per sequence.
            let mut refs: Vec<Option<&mut Seq>> = self.running.iter_mut().map(Some).collect();
            std::thread::scope(|sc| {
                let handles: Vec<_> = jobs
                    .into_iter()
                    .map(|(i, l, p)| {
                        let s = refs[i].take().unwrap();
                        sc.spawn(move || {
                            let (n, kept) = run(s, l, p);
                            (i, n, kept)
                        })
                    })
                    .collect();
                handles.into_iter().map(|h| h.join().expect("sampler thread panicked")).collect()
            })
        };
        let mut done: Vec<(usize, Finish)> = Vec::new();
        for (i, next, kept) in results.into_iter().chain(device_results) {
            if spec[i] > 0 && takes[i] > 1 {
                // Keep the KV of the pending token and the accepted drafts;
                // the draft cache keeps what it shares with them.
                let c = if lookup[i] { &self.metrics.lookup_accepted } else { &self.metrics.draft_accepted };
                Metrics::add(c, kept);
                let keep = start_of[i] + 1 + kept;
                let s = &mut self.running[i];
                s.spec_k = (kept + 1).min(self.cfg.draft_tokens);
                s.spec_misses = if kept == 0 { s.spec_misses + 1 } else { 0 };
                if s.spec_misses >= SPEC_MAX_MISSES {
                    // The draft keeps guessing wrong here: decode plainly
                    // for a while (drafts that all fail only cost time).
                    s.spec_pause = SPEC_PAUSE_STEPS;
                    s.spec_misses = 0;
                }
                s.kv.truncate(&mut self.pool.alloc, keep);
                if let Some(d) = &mut self.draft {
                    s.draft_kv.truncate(&mut d.alloc, keep);
                }
            }
            if let Next::Done(f) = next {
                done.push((i, f));
            }
        }
        self.timing.sample += t_sample.elapsed();
        self.timing.total += t_start.elapsed();
        self.timing.report();
        // Remove from the back so indices stay valid.
        done.sort_by_key(|d| std::cmp::Reverse(d.0));
        for (i, f) in done {
            let s = self.running.remove(i);
            self.finish(s, f);
        }
        Ok(())
    }

    /// Run the draft model for the sequences with `spec[i] > 0`: catch its
    /// cache up with their tokens, then propose up to `spec[i]` tokens each
    /// (one batched draft step per token). Fails only if the draft pool is
    /// out of blocks.
    fn propose(&mut self, spec: &[usize]) -> Result<Vec<Proposal>> {
        let draft = self.draft.as_mut().context("no draft model")?;
        let vocab_n = draft.model().vocab_size();
        let mut props: Vec<Proposal> = (0..self.running.len()).map(|_| Proposal::default()).collect();
        let idx: Vec<usize> = (0..self.running.len()).filter(|&i| spec[i] > 0).collect();
        // Logits after each sequence's latest token, as rows of `logits`.
        let mut logits: Vec<f32> = vec![0.0; self.running.len() * vocab_n];
        // Catch up: everything not yet in the draft cache (the last token,
        // or the whole prompt the first time), in prefill-sized chunks.
        loop {
            let mut items = Vec::new();
            let mut owners = Vec::new();
            let mut refs: Vec<Option<&mut Seq>> = self.running.iter_mut().map(Some).collect();
            for &i in &idx {
                let s = refs[i].take().unwrap();
                let todo = s.tokens.len() - s.draft_kv.len;
                if todo == 0 {
                    continue;
                }
                let take = todo.min(self.cfg.prefill_chunk.max(1));
                let start = s.draft_kv.len;
                let last = take == todo;
                let Seq { tokens, draft_kv, .. } = s;
                items.push(StepItem { seq: draft_kv, tokens: &tokens[start..start + take], want: if last { Want::Last } else { Want::None } });
                if last {
                    owners.push(i);
                }
            }
            if items.is_empty() {
                break;
            }
            let out = draft.step(&mut items, false)?;
            for (row, &i) in out.chunks_exact(vocab_n).zip(&owners) {
                logits[i * vocab_n..(i + 1) * vocab_n].copy_from_slice(row);
            }
        }
        // Propose: draw a token from each row, then feed it to get the next.
        let max_k = idx.iter().map(|&i| spec[i]).max().unwrap_or(0);
        for round in 0..max_k {
            let mut feed: Vec<(usize, u32)> = Vec::new();
            for &i in &idx {
                let s = &mut self.running[i];
                let p = &mut props[i];
                if p.tokens.len() != round || round >= spec[i] {
                    continue; // stopped early, or has enough
                }
                let row = &mut logits[i * vocab_n..(i + 1) * vocab_n];
                draft_logits(s, &p.tokens, row);
                let q = s.draft_sampler.dist(row);
                let t = s.draft_sampler.draw(&q);
                p.tokens.push(t);
                p.dists.push(q);
                // Nothing to propose after the end of the output.
                if round + 1 < spec[i] && !self.tok.is_stop(t) {
                    feed.push((i, t));
                }
            }
            if feed.is_empty() {
                break;
            }
            let toks: Vec<[u32; 1]> = feed.iter().map(|&(_, t)| [t]).collect();
            let mut items = Vec::with_capacity(feed.len());
            let mut refs: Vec<Option<&mut Seq>> = self.running.iter_mut().map(Some).collect();
            for (k, &(i, _)) in feed.iter().enumerate() {
                let s = refs[i].take().unwrap();
                items.push(StepItem { seq: &mut s.draft_kv, tokens: &toks[k], want: Want::Last });
            }
            let out = draft.step(&mut items, false)?;
            for (row, &(i, _)) in out.chunks_exact(vocab_n).zip(&feed) {
                logits[i * vocab_n..(i + 1) * vocab_n].copy_from_slice(row);
            }
        }
        Ok(props)
    }

    /// Embedding jobs run between steps, one input at a time, in their own
    /// temporary sequences (not cached).
    fn run_embeds(&mut self) {
        while let Some(Job::Embed { inputs, reply }) = self.embeds.pop_front() {
            let longest = inputs.iter().map(Vec::len).max().unwrap_or(0);
            if !self.cache.make_room(&mut self.pool, longest.div_ceil(BLOCK_SIZE)) {
                if self.running.is_empty() {
                    let _ = reply.send(Err(format!("input needs more KV cache than the pool has ({longest} tokens)")));
                    continue;
                }
                // Wait for running generations to free blocks.
                self.embeds.push_front(Job::Embed { inputs, reply });
                return;
            }
            let t0 = Instant::now();
            let r = self.embed(&inputs).map_err(|e| format!("{e:#}"));
            Metrics::add(&self.metrics.busy_us, t0.elapsed().as_micros() as usize);
            let _ = reply.send(r);
        }
    }

    /// Mean-pooled, L2-normalized final hidden states, one vector per input.
    fn embed(&mut self, inputs: &[Vec<u32>]) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(inputs.len());
        for ids in inputs {
            anyhow::ensure!(!ids.is_empty(), "empty input");
            anyhow::ensure!(ids.len() <= self.cfg.ctx, "input ({} tokens) exceeds the context ({} tokens)", ids.len(), self.cfg.ctx);
            let mut seq = SeqKv::default();
            let mut v: Vec<f32> = Vec::new();
            let r = (|| -> Result<()> {
                for part in ids.chunks(self.cfg.prefill_chunk.max(1)) {
                    let h = self.pool.step(&mut [StepItem { seq: &mut seq, tokens: part, want: Want::All }], true)?;
                    let d = h.len() / part.len();
                    v.resize(d, 0.0);
                    for row in h.chunks_exact(d) {
                        for (a, &b) in v.iter_mut().zip(row) {
                            *a += b;
                        }
                    }
                }
                Ok(())
            })();
            seq.free(&mut self.pool.alloc);
            r?;
            Metrics::add(&self.metrics.embedded_tokens, ids.len());
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            v.iter_mut().for_each(|x| *x /= norm);
            out.push(v);
        }
        Ok(out)
    }
}

/// How to sample a sequence's next token on the device, or None when it
/// needs its logits row on the host: logprobs, a JSON constraint (its
/// verification steps never get here), or `--host-sampling`.
fn device_sample(params: &GenParams, json: &Option<JsonState>, counts: &HashMap<u32, u32>, position: usize, host_only: bool) -> Option<RowSample> {
    if host_only || params.logprobs.is_some() || json.is_some() {
        return None;
    }
    let mut adjust = params.logit_bias.clone();
    if params.frequency_penalty != 0.0 || params.presence_penalty != 0.0 {
        adjust.extend(counts.iter().map(|(&id, &c)| (id, -(c as f32 * params.frequency_penalty + params.presence_penalty))));
    }
    Some(RowSample {
        temperature: params.temperature,
        top_k: params.top_k.min(u32::MAX as usize) as u32,
        top_p: params.top_p,
        seed: params.seed,
        counter: position as u64,
        adjust,
    })
}

/// Process one sequence's logits: logprobs, logit bias, penalties, the JSON
/// constraint, sampling; then send the token and decide whether to go on.
fn sample_next(s: &mut Seq, logits: &mut [f32], tok: &Tokenizer, vocab: &Vocab) -> Next {
    let lp = process_logits(s, logits);
    let id = match &mut s.json {
        Some(state) => match pick_json(logits, state, vocab, &s.params, &mut s.sampler) {
            Some(id) => id,
            None => {
                // The output so far is not valid JSON: don't report success.
                let _ = s.events.send(GenEvent::Error("the JSON constraint reached a dead end: no token continues the output as valid JSON".into()));
                return Next::Done(Finish::Cancelled);
            }
        },
        None => s.sampler.sample(logits),
    };
    emit(s, id, lp, logits, tok)
}

/// Check a sequence's draft tokens against the main model's logits rows
/// (one per verified position: the pending token, then each draft) and
/// emit the accepted ones plus one more from the main model: a replacement
/// for the first rejected draft, or the token after the last one.
/// Returns what happens next and how many drafts went into the sequence.
fn verify_drafts(s: &mut Seq, rows: &mut [f32], vocab_n: usize, p: &Proposal, tok: &Tokenizer) -> (Next, usize) {
    let mut kept = 0;
    for (j, row) in rows.chunks_exact_mut(vocab_n).enumerate() {
        let lp = process_logits(s, row);
        let (id, accepted) = match p.tokens.get(j) {
            Some(&d) => {
                let target = s.sampler.dist(row);
                s.sampler.verify(&target, &p.dists[j], d)
            }
            None => (s.sampler.sample(row), false),
        };
        match emit(s, id, lp, row, tok) {
            Next::Continue if accepted => kept += 1,
            next => return (next, kept),
        }
    }
    (Next::Continue, kept)
}

/// Logprobs of the raw logits (if requested), then the request's logit
/// bias and frequency/presence penalties, in place.
fn process_logits(s: &Seq, logits: &mut [f32]) -> Option<LogSoftmax> {
    let params = &s.params;
    let mut lp = params.logprobs.map(|k| LogSoftmax::new(logits, k));
    // Request-level logit processing (before sampling, after logprobs).
    for &(id, b) in &params.logit_bias {
        if let Some(l) = logits.get_mut(id as usize) {
            if let Some(lp) = &mut lp {
                lp.note(id, *l);
            }
            *l += b;
        }
    }
    if params.frequency_penalty != 0.0 || params.presence_penalty != 0.0 {
        for (&id, &c) in &s.counts {
            if let Some(l) = logits.get_mut(id as usize) {
                if let Some(lp) = &mut lp {
                    lp.note(id, *l);
                }
                *l -= c as f32 * params.frequency_penalty + params.presence_penalty;
            }
        }
    }
    lp
}

/// The draft model's view of the request's logit processing: bias and
/// penalties, counting the tokens drafted so far as generated.
fn draft_logits(s: &Seq, drafted: &[u32], logits: &mut [f32]) {
    let params = &s.params;
    for &(id, b) in &params.logit_bias {
        if let Some(l) = logits.get_mut(id as usize) {
            *l += b;
        }
    }
    if params.frequency_penalty != 0.0 || params.presence_penalty != 0.0 {
        let mut counts = s.counts.clone();
        for &t in drafted {
            *counts.entry(t).or_default() += 1;
        }
        for (&id, &c) in &counts {
            if let Some(l) = logits.get_mut(id as usize) {
                *l -= c as f32 * params.frequency_penalty + params.presence_penalty;
            }
        }
    }
}

/// Send a chosen token and decide whether the sequence goes on.
fn emit(s: &mut Seq, id: u32, lp: Option<LogSoftmax>, logits: &[f32], tok: &Tokenizer) -> Next {
    if tok.is_stop(id) {
        return Next::Done(Finish::Stop);
    }
    let logprobs = lp.map(|l| l.for_token(id, logits));
    if s.events.send(GenEvent::Token { id, logprobs }).is_err() {
        return Next::Done(Finish::Cancelled);
    }
    s.generated += 1;
    *s.counts.entry(id).or_default() += 1;
    if s.json.as_ref().is_some_and(JsonState::is_done) {
        return Next::Done(Finish::Stop);
    }
    if s.generated >= s.budget {
        return Next::Done(Finish::Length);
    }
    s.tokens.push(id);
    Next::Continue
}

/// Pick the next token under the JSON constraint: the best candidates (by
/// processed logit) that keep the output a valid JSON prefix. Greedy when
/// temperature is 0, otherwise sampled among the valid candidates.
/// None if nothing valid is left (the caller stops).
fn pick_json(logits: &[f32], state: &mut JsonState, vocab: &Vocab, params: &GenParams, sampler: &mut Sampler) -> Option<u32> {
    let valid = |id: u32| -> Option<JsonState> {
        let i = id as usize;
        // The model's vocabulary can be padded past the tokenizer's.
        if i >= vocab.text.len() || (vocab.special[i] && !state.allows_special(&vocab.text[i])) || vocab.text[i].is_empty() {
            return None;
        }
        let mut s = state.clone();
        s.feed(&vocab.text[i]).then_some(s)
    };
    let mut order: Vec<u32> = (0..logits.len() as u32).collect();
    let head = 256.min(order.len());
    order.select_nth_unstable_by(head - 1, |&a, &b| logits[b as usize].total_cmp(&logits[a as usize]));
    order[..head].sort_unstable_by(|&a, &b| logits[b as usize].total_cmp(&logits[a as usize]));
    let want = if params.temperature <= 0.0 { 1 } else { 32 };
    let mut found: Vec<(u32, JsonState)> = Vec::new();
    for &id in &order[..head] {
        if let Some(s) = valid(id) {
            found.push((id, s));
            if found.len() >= want {
                break;
            }
        }
    }
    if found.is_empty() {
        // Rare: nothing valid in the head; scan the rest in logit order.
        order[head..].sort_unstable_by(|&a, &b| logits[b as usize].total_cmp(&logits[a as usize]));
        found.extend(order[head..].iter().find_map(|&id| valid(id).map(|s| (id, s))));
    }
    if found.is_empty() {
        return None;
    }
    let pick = if found.len() == 1 {
        0
    } else {
        // Sample among the valid candidates with the request's sampler.
        let mut sub = vec![f32::NEG_INFINITY; found.len()];
        for (j, (id, _)) in found.iter().enumerate() {
            sub[j] = logits[*id as usize];
        }
        sampler.sample(&sub) as usize
    };
    let (id, s) = found.swap_remove(pick);
    *state = s;
    Some(id)
}

/// Log-softmax of the model's raw logits at one position, for logprobs.
/// The chosen token's logprob is read later from the (by then processed)
/// logits, so the raw values of any ids changed in between are recorded.
struct LogSoftmax {
    lse: f32,
    top: Vec<(u32, f32)>,
    /// Raw logits of ids that logit bias or penalties changed.
    changed: Vec<(u32, f32)>,
}

impl LogSoftmax {
    fn new(logits: &[f32], k: usize) -> Self {
        let lse = rllm_model::log_sum_exp(logits);
        let top = if k > 0 { rllm_model::top_candidates(logits, k.min(logits.len())).into_iter().map(|(id, l)| (id, l - lse)).collect() } else { Vec::new() };
        Self { lse, top, changed: Vec::new() }
    }

    fn note(&mut self, id: u32, raw: f32) {
        if !self.changed.iter().any(|c| c.0 == id) {
            self.changed.push((id, raw));
        }
    }

    fn for_token(self, id: u32, logits: &[f32]) -> TokenLogprobs {
        let raw = self.changed.iter().find(|c| c.0 == id).map_or(logits[id as usize], |c| c.1);
        TokenLogprobs { logprob: raw - self.lse, top: self.top }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_proposes_what_followed_the_latest_match() {
        // "1 2 3" occurs twice; the latest earlier occurrence wins.
        let t = [1, 2, 3, 7, 8, 9, 1, 2, 3, 4, 5, 6, 1, 2, 3];
        let p = Proposal::lookup(&t, 3);
        assert_eq!(p.tokens, vec![4, 5, 6]);
        assert!(p.lookup);
        assert!(p.dists.iter().zip(&p.tokens).all(|(d, &t)| d.prob(t) == 1.0));
        // Proposals stop at the end of the sequence.
        assert_eq!(Proposal::lookup(&[5, 6, 5, 6], 4).tokens, vec![5, 6]);
    }

    #[test]
    fn lookup_falls_back_to_shorter_tails() {
        // The 4- and 3-token tails never occurred before; the 2-token tail did.
        let t = [9, 4, 5, 0, 1, 2, 3, 4, 5];
        assert_eq!(Proposal::lookup(&t, 2).tokens, vec![0, 1]);
        // A single repeated token is not enough.
        assert!(Proposal::lookup(&[1, 2, 3, 9, 4, 9], 2).tokens.is_empty());
        assert!(Proposal::lookup(&[1], 2).tokens.is_empty());
    }
}
