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
//! The KV pool lives as long as the engine. CUDA decode graphs capture its
//! device addresses, so it must not be reallocated.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::Instant;

use anyhow::{Context, Result};
use rllm_core::Backend;
use rllm_gguf::Gguf;
use rllm_kv::{PrefixCache, SeqKv};
use rllm_model::{AnyModel, KvPool, Model, Sampler, SamplerConfig, StepItem, Want};
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
}

#[derive(Clone)]
pub struct EngineHandle {
    jobs: mpsc::Sender<Job>,
}

impl EngineHandle {
    pub fn submit(&self, job: Job) -> Result<()> {
        self.jobs.send(job).map_err(|_| anyhow::anyhow!("engine stopped"))
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
        for id in 0..n as u32 {
            let raw = tok.token_text(id);
            let plain = tok.decode(&[id], true).unwrap_or_default();
            special.push(raw != plain || tok.is_stop(id));
            text.push(raw);
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
}

const BLOCK_SIZE: usize = 16;

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
    std::thread::Builder::new()
        .name("rllm-engine".into())
        .spawn(move || {
            pin_to_performance_cores();
            let setup = (|| -> Result<_> {
                let backend = make_backend()?;
                let model = AnyModel::load(&backend, &gguf)?;
                Ok((backend, model))
            })();
            let (backend, model) = match setup {
                Ok(v) => v,
                Err(e) => {
                    let _ = ready_tx.send(Err(format!("{e:#}")));
                    return;
                }
            };
            let pool = match KvPool::new(&backend, &model, cfg.kv_tokens.max(cfg.ctx), BLOCK_SIZE) {
                Ok(p) => p,
                Err(e) => {
                    let _ = ready_tx.send(Err(format!("{e:#}")));
                    return;
                }
            };
            let info = EngineInfo { backend: backend.name(), summary: model.summary(), vocab: model.vocab_size() };
            let _ = ready_tx.send(Ok(info));
            Scheduler::new(pool, &tok, &vocab, &cfg, &metrics).run(jobs_rx);
        })
        .context("spawn engine thread")?;
    let info = ready_rx.recv().context("engine thread exited")?.map_err(|e| anyhow::anyhow!(e))?;
    Ok((EngineHandle { jobs: jobs_tx }, info))
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

struct Scheduler<'a, 'm, B: Backend> {
    pool: KvPool<'m, B, AnyModel<B>>,
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
                "[rllm] engine step: {:.3} ms (forward {:.3}, sampling {:.3}, other {:.3})",
                ms(self.total),
                ms(self.forward),
                ms(self.sample),
                ms(self.total - self.forward - self.sample)
            );
            *self = Self { enabled: true, ..Self::default() };
        }
    }
}

impl<'a, 'm, B: Backend> Scheduler<'a, 'm, B> {
    fn new(pool: KvPool<'m, B, AnyModel<B>>, tok: &'a Tokenizer, vocab: &'a Vocab, cfg: &'a EngineConfig, metrics: &'a Metrics) -> Self {
        Metrics::set(&metrics.kv_blocks_total, pool.alloc.n_blocks());
        Self {
            pool,
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
                for mut s in self.running.drain(..) {
                    let _ = s.events.send(GenEvent::Error(msg.clone()));
                    s.kv.free(&mut self.pool.alloc);
                }
            }
            Metrics::add(&self.metrics.busy_us, t0.elapsed().as_micros() as usize);
        }
    }

    fn accept(&mut self, job: Job) {
        match job {
            Job::Generate { prompt, params, events, cancel } => {
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
                let sampler = Sampler::new(SamplerConfig {
                    temperature: params.temperature,
                    top_k: params.top_k,
                    top_p: params.top_p,
                    seed: params.seed,
                });
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

    fn retire(&mut self, s: &mut Seq) {
        let kv = std::mem::take(&mut s.kv);
        if self.cfg.prefix_cache {
            self.cache.insert(&mut self.pool.alloc, &s.tokens[..kv.len], kv);
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
                let (n, blocks) = self.cache.lookup(&mut self.pool.alloc, &s.tokens);
                s.kv = SeqKv::with_prefix(blocks, n);
            }
            // The prompt's blocks, plus one block of headroom per running
            // sequence so they can keep decoding.
            let need = s.kv.blocks_needed(BLOCK_SIZE, s.todo().len()) + self.running.len();
            if !self.cache.make_room(&mut self.pool.alloc, need) {
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
    }

    /// Plan, run and sample one batched forward pass.
    fn step(&mut self) -> Result<()> {
        let t_start = Instant::now();
        // Decoding sequences first (one token each), then prompt chunks
        // in admission order within the prefill budget.
        self.running.sort_by_key(|s| (s.todo().len() > 1, s.admitted));
        let mut takes: Vec<usize>;
        loop {
            let mut budget = self.cfg.prefill_chunk.max(1);
            takes = self
                .running
                .iter()
                .map(|s| {
                    let n = s.todo().len();
                    if n == 1 {
                        1
                    } else {
                        let t = n.min(budget);
                        budget -= t;
                        t
                    }
                })
                .collect();
            let need: usize = self.running.iter().zip(&takes).map(|(s, &t)| s.kv.blocks_needed(BLOCK_SIZE, t)).sum();
            if self.cache.make_room(&mut self.pool.alloc, need) {
                break;
            }
            // Out of blocks: pause the newest generation and recompute it later.
            let newest = (0..self.running.len()).max_by_key(|&i| self.running[i].admitted).unwrap();
            let mut s = self.running.remove(newest);
            self.retire(&mut s);
            self.waiting.push_front(s);
            Metrics::add(&self.metrics.preemptions, 1);
            anyhow::ensure!(!self.running.is_empty(), "KV cache too small for one sequence");
        }
        let active: Vec<usize> = (0..self.running.len()).filter(|&i| takes[i] > 0).collect();
        let mut rows_of = Vec::with_capacity(active.len());
        let out = {
            let mut items: Vec<StepItem> = Vec::with_capacity(active.len());
            let mut n_tok = 0;
            for (i, s) in self.running.iter_mut().enumerate() {
                if takes[i] == 0 {
                    continue;
                }
                let start = s.kv.len;
                let all = start + takes[i] == s.tokens.len();
                rows_of.push(all);
                n_tok += takes[i];
                // `tokens` and `kv` are disjoint fields.
                let Seq { tokens, kv, .. } = s;
                items.push(StepItem { seq: kv, tokens: &tokens[start..start + takes[i]], want: if all { Want::Last } else { Want::None } });
            }
            Metrics::add(&self.metrics.steps, 1);
            Metrics::add(&self.metrics.step_tokens, n_tok);
            let t = Instant::now();
            let out = self.pool.step(&mut items, false)?;
            self.timing.forward += t.elapsed();
            out
        };
        let t_sample = Instant::now();
        // Sample every sequence whose pending tokens are all in the cache now.
        let vocab_n = self.pool.model().vocab_size();
        let mut out = out;
        // Logits rows are processed in place (no copies: a fresh 1 MB
        // buffer per token costs more in page faults than the argmax).
        let mut rows = out.chunks_exact_mut(vocab_n);
        let mut jobs: Vec<(usize, &mut [f32])> = Vec::new();
        for (&i, &want) in active.iter().zip(&rows_of) {
            if want {
                jobs.push((i, rows.next().context("missing logits row")?));
            }
        }
        let (tok, vocab) = (self.tok, self.vocab);
        let results: Vec<(usize, Next)> = if jobs.len() <= 1 {
            jobs.into_iter().map(|(i, l)| (i, sample_next(&mut self.running[i], l, tok, vocab))).collect()
        } else {
            // Sampling a large vocabulary takes a while: one thread per sequence.
            let mut refs: Vec<Option<&mut Seq>> = self.running.iter_mut().map(Some).collect();
            std::thread::scope(|sc| {
                let handles: Vec<_> = jobs
                    .into_iter()
                    .map(|(i, l)| {
                        let s = refs[i].take().unwrap();
                        sc.spawn(move || (i, sample_next(s, l, tok, vocab)))
                    })
                    .collect();
                handles.into_iter().map(|h| h.join().expect("sampler thread panicked")).collect()
            })
        };
        let mut done: Vec<(usize, Finish)> = results.into_iter().filter_map(|(i, n)| match n {
            Next::Continue => None,
            Next::Done(f) => Some((i, f)),
        }).collect();
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

    /// Embedding jobs run between steps, one input at a time, in their own
    /// temporary sequences (not cached).
    fn run_embeds(&mut self) {
        while let Some(Job::Embed { inputs, reply }) = self.embeds.pop_front() {
            let longest = inputs.iter().map(Vec::len).max().unwrap_or(0);
            if !self.cache.make_room(&mut self.pool.alloc, longest.div_ceil(BLOCK_SIZE)) {
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

/// Process one sequence's logits: logprobs, logit bias, penalties, the JSON
/// constraint, sampling; then send the token and decide whether to go on.
fn sample_next(s: &mut Seq, logits: &mut [f32], tok: &Tokenizer, vocab: &Vocab) -> Next {
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
    let id = match &mut s.json {
        Some(state) => match pick_json(logits, state, vocab, params, &mut s.sampler) {
            Some(id) => id,
            None => return Next::Done(Finish::Stop),
        },
        None => s.sampler.sample(logits),
    };
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
        if (vocab.special[i] && !state.allows_special(&vocab.text[i])) || vocab.text[i].is_empty() {
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

