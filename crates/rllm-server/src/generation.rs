//! Running one generation for a handler: submit the job, decode tokens,
//! apply stop sequences and the output parser, and report parsed pieces.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use tokio::sync::mpsc;

use crate::engine::{Finish, GenEvent, GenParams, Job};
use crate::error::ApiError;
use crate::output::{OutputParser, Piece, StopMatcher};
use crate::state::AppState;

pub struct GenSpec {
    pub prompt: Vec<u32>,
    pub params: GenParams,
    pub stops: Vec<String>,
    pub parser: OutputParser,
    /// Text already "generated" by the prompt (forced tool-call opening).
    pub forced: String,
    /// Text that closes a constrained output once it completes (the end of
    /// a forced tool call); the parser sees it after the generated text.
    pub suffix: String,
    /// Drop special tokens from the text (plain completions).
    pub skip_special: bool,
    /// Queue places the handler reserved before responding (one is used).
    pub slots: Option<Arc<crate::engine::Slots>>,
}

/// One generated token with its logprobs, as text.
#[derive(Debug, Clone)]
pub struct TokenInfo {
    pub token: String,
    /// The token's exact bytes (a byte-fallback token can hold part of a
    /// UTF-8 character, which `token` can only show as U+FFFD).
    pub bytes: Vec<u8>,
    pub logprob: f32,
    /// (text, bytes, logprob) per alternative.
    pub top: Vec<(String, Vec<u8>, f32)>,
    /// Part of the reasoning (including its open/close markers), not of
    /// the message content.
    pub reasoning: bool,
}

fn token_info(tok: &rllm_tokenizer::Tokenizer, id: u32) -> (String, Vec<u8>) {
    let text = tok.token_text(id);
    let bytes = tok.token_bytes(id).unwrap_or_else(|| text.as_bytes().to_vec());
    (text, bytes)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishKind {
    Stop,
    Length,
    /// Ended by a cancel request (background `cancel`).
    Cancelled,
}

impl FinishKind {
    pub fn as_str(self) -> &'static str {
        match self {
            FinishKind::Stop | FinishKind::Cancelled => "stop",
            FinishKind::Length => "length",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Summary {
    pub finish: FinishKind,
    pub prompt_tokens: usize,
    /// Prompt tokens served from the prefix cache.
    pub cached_tokens: usize,
    pub completion_tokens: usize,
    pub reasoning_tokens: usize,
}

#[derive(Debug)]
pub enum Out {
    Piece(Piece),
    Token(TokenInfo),
    Done(Summary),
}

/// Start a generation. Events arrive on the returned channel; an `Err`
/// ends it. Dropping the receiver (client gone) cancels the job.
pub fn spawn(state: &Arc<AppState>, spec: GenSpec, cancel: Arc<AtomicBool>) -> mpsc::UnboundedReceiver<Result<Out, ApiError>> {
    let (out_tx, out_rx) = mpsc::unbounded_channel();
    let state = state.clone();
    tokio::spawn(async move {
        let GenSpec { prompt, params, stops, mut parser, forced, suffix, skip_special, slots } = spec;
        let want_logprobs = params.logprobs.is_some();
        let (etx, mut erx) = mpsc::unbounded_channel();
        let user_cancel = cancel.clone();
        if let Err(e) = state.engine.submit_with(Job::Generate { prompt, params, events: etx, cancel: cancel.clone() }, slots.as_deref()) {
            let _ = out_tx.send(Err(ApiError::unavailable(format!("{e:#}"))));
            return;
        }
        let t0 = std::time::Instant::now();
        let mut first_token: Option<std::time::Duration> = None;
        let tok = state.tok.clone();
        let mut dec = tok.stream(skip_special);
        let mut stop = StopMatcher::new(stops);
        let mut stopped = false;
        let (mut completion, mut reasoning) = (0usize, 0usize);
        let send = |o: Out| out_tx.send(Ok(o)).is_ok();
        let mut alive = true;
        for p in parser.push(&forced) {
            alive &= send(Out::Piece(p));
        }
        // The server's time limit: past it the job is cancelled and its
        // end reported as `length` (cut short by a limit, like max_tokens).
        let deadline = state.request_timeout.map(|d| tokio::time::Instant::now() + d);
        let mut timed_out = false;
        loop {
            // A client that leaves while its request is still queued sends
            // nothing to fail on: watch the output channel too.
            let ev = tokio::select! {
                ev = erx.recv() => match ev {
                    Some(ev) => ev,
                    None => break,
                },
                _ = out_tx.closed() => {
                    cancel.store(true, Ordering::Relaxed);
                    return;
                }
                _ = async { tokio::time::sleep_until(deadline.unwrap()).await }, if deadline.is_some() && !timed_out => {
                    timed_out = true;
                    cancel.store(true, Ordering::Relaxed);
                    continue;
                }
            };
            if !alive {
                cancel.store(true, Ordering::Relaxed);
            }
            match ev {
                GenEvent::Token { id, logprobs } => {
                    if stopped {
                        continue;
                    }
                    completion += 1;
                    first_token.get_or_insert_with(|| t0.elapsed());
                    let was_reasoning = parser.in_reasoning();
                    if was_reasoning {
                        reasoning += 1;
                    }
                    let pieces = match dec.push(id) {
                        Ok(Some(text)) => parser.push(&stop.push(&text)),
                        Ok(None) => Vec::new(),
                        Err(e) => {
                            let _ = out_tx.send(Err(ApiError::server(format!("{e:#}"))));
                            cancel.store(true, Ordering::Relaxed);
                            return;
                        }
                    };
                    // Sent before the pieces its text produced. A token that
                    // opens, continues or closes reasoning (`<think>`,
                    // `</think>`) is reasoning even when it yields no piece.
                    if want_logprobs && let Some(lp) = logprobs {
                        let top = lp
                            .top
                            .iter()
                            .map(|&(t, l)| {
                                let (text, bytes) = token_info(&tok, t);
                                (text, bytes, l)
                            })
                            .collect();
                        let (token, bytes) = token_info(&tok, id);
                        let in_reasoning = was_reasoning || parser.in_reasoning();
                        alive &= send(Out::Token(TokenInfo { token, bytes, logprob: lp.logprob, top, reasoning: in_reasoning }));
                    }
                    for p in pieces {
                        alive &= send(Out::Piece(p));
                    }
                    if stop.stopped() {
                        stopped = true;
                        cancel.store(true, Ordering::Relaxed);
                    }
                }
                GenEvent::Done { finish, prompt_tokens, cached_tokens } => {
                    let mut pieces = Vec::new();
                    if !stopped {
                        let held = dec.finish().unwrap_or_default();
                        let mut passed = stop.push(&held);
                        if !stop.stopped() {
                            passed.push_str(&stop.finish());
                        }
                        pieces.extend(parser.push(&passed));
                        stopped = stop.stopped();
                        if finish == Finish::Stop && !stopped {
                            pieces.extend(parser.push(&suffix));
                        }
                    }
                    pieces.extend(parser.finish());
                    for p in pieces {
                        send(Out::Piece(p));
                    }
                    let finish = match finish {
                        Finish::Stop => FinishKind::Stop,
                        Finish::Length => FinishKind::Length,
                        Finish::Cancelled if stopped => FinishKind::Stop,
                        Finish::Cancelled if timed_out => FinishKind::Length,
                        Finish::Cancelled if user_cancel.load(Ordering::Relaxed) => FinishKind::Cancelled,
                        Finish::Cancelled => FinishKind::Stop,
                    };
                    if state.log {
                        let total = t0.elapsed().as_secs_f64();
                        let ttft = first_token.map_or(total, |d| d.as_secs_f64());
                        let rate = if completion > 1 && total > ttft { (completion - 1) as f64 / (total - ttft) } else { 0.0 };
                        eprintln!(
                            "[rllm] {prompt_tokens} prompt tokens ({cached_tokens} cached) + {completion} generated, first token {:.0} ms, {rate:.1} tok/s, {:.2} s, {}",
                            ttft * 1e3,
                            total,
                            match finish {
                                FinishKind::Stop => "stop",
                                FinishKind::Length => "length",
                                FinishKind::Cancelled => "cancelled",
                            }
                        );
                    }
                    send(Out::Done(Summary { finish, prompt_tokens, cached_tokens, completion_tokens: completion, reasoning_tokens: reasoning }));
                    return;
                }
                GenEvent::Error(e) => {
                    let _ = out_tx.send(Err(ApiError::server(e)));
                    return;
                }
            }
        }
        let _ = out_tx.send(Err(ApiError::server("the engine stopped unexpectedly")));
    });
    out_rx
}

/// Run a generation to completion and collect everything.
pub async fn collect(state: &Arc<AppState>, spec: GenSpec, cancel: Arc<AtomicBool>) -> Result<Collected, ApiError> {
    let mut rx = spawn(state, spec, cancel);
    let mut c = Collected::default();
    while let Some(ev) = rx.recv().await {
        match ev? {
            Out::Piece(Piece::Content(t)) => c.content.push_str(&t),
            Out::Piece(Piece::Reasoning(t)) => c.reasoning.push_str(&t),
            Out::Piece(Piece::ToolCall(call)) => c.tool_calls.push(call),
            Out::Token(t) => c.tokens.push(t),
            Out::Done(s) => {
                c.summary = Some(s);
                return Ok(c);
            }
        }
    }
    Err(ApiError::server("generation ended without a result"))
}

/// Run several generations concurrently (the engine batches them) and
/// collect them in order. The first error cancels the rest.
pub async fn collect_all(state: &Arc<AppState>, specs: Vec<GenSpec>) -> Result<Vec<Collected>, ApiError> {
    let n = specs.len();
    let mut set = tokio::task::JoinSet::new();
    for (i, spec) in specs.into_iter().enumerate() {
        let state = state.clone();
        set.spawn(async move { (i, collect(&state, spec, Arc::new(AtomicBool::new(false))).await) });
    }
    let mut out: Vec<Option<Collected>> = (0..n).map(|_| None).collect();
    while let Some(r) = set.join_next().await {
        let (i, c) = r.map_err(|e| ApiError::server(format!("generation task failed: {e}")))?;
        out[i] = Some(c?);
    }
    Ok(out.into_iter().map(|c| c.expect("every task reported")).collect())
}

#[derive(Debug, Default)]
pub struct Collected {
    pub content: String,
    pub reasoning: String,
    pub tool_calls: Vec<crate::output::ToolCall>,
    pub tokens: Vec<TokenInfo>,
    pub summary: Option<Summary>,
}

impl Collected {
    pub fn summary(&self) -> &Summary {
        self.summary.as_ref().expect("collect returns only completed generations")
    }
}
