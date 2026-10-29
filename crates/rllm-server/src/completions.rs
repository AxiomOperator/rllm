//! `/v1/completions` (legacy): raw text in, raw text out.

use std::convert::Infallible;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use axum::Json;
use axum::body::Bytes;
use axum::extract::State;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};
use tokio_stream::wrappers::UnboundedReceiverStream;

use crate::error::{ApiError, ApiResult};
use crate::generation::{self, GenSpec, Out, TokenInfo};
use crate::output::{OutputParser, Piece};
use crate::params;
use crate::prompt;
use crate::state::AppState;
use crate::util;

/// One prompt: its token ids and its text (for `echo`).
struct Prompt {
    ids: Vec<u32>,
    text: String,
}

fn prompts(state: &AppState, v: Option<&Value>) -> ApiResult<Vec<Prompt>> {
    let tok = &state.tok;
    let from_text = |s: &str| -> ApiResult<Prompt> { Ok(Prompt { ids: tok.encode(s).map_err(ApiError::from)?, text: s.to_owned() }) };
    let from_ids = |a: &[Value]| -> ApiResult<Prompt> {
        let ids: Vec<u32> = a
            .iter()
            .map(|t| t.as_u64().filter(|&t| (t as usize) < state.info.vocab).map(|t| t as u32))
            .collect::<Option<_>>()
            .ok_or_else(|| ApiError::invalid("Invalid token in 'prompt'.", Some("prompt")))?;
        let text = tok.decode(&ids, false).map_err(ApiError::from)?;
        Ok(Prompt { ids, text })
    };
    let out = match v {
        None | Some(Value::Null) => vec![from_text("")?],
        Some(Value::String(s)) => vec![from_text(s)?],
        Some(Value::Array(a)) if a.is_empty() => return Err(ApiError::invalid("[] is too short - 'prompt'", Some("prompt"))),
        Some(Value::Array(a)) if a.iter().all(Value::is_string) => a.iter().map(|s| from_text(s.as_str().unwrap())).collect::<ApiResult<_>>()?,
        Some(Value::Array(a)) if a.iter().all(Value::is_number) => vec![from_ids(a)?],
        Some(Value::Array(a)) if a.iter().all(Value::is_array) => a.iter().map(|x| from_ids(x.as_array().unwrap())).collect::<ApiResult<_>>()?,
        Some(_) => return Err(ApiError::invalid("Invalid type for 'prompt'.", Some("prompt"))),
    };
    for p in &out {
        if p.ids.is_empty() {
            return Err(ApiError::invalid("The prompt is empty (this model adds no beginning-of-text token).", Some("prompt")));
        }
        prompt::check_context(state, p.ids.len())?;
    }
    Ok(out)
}

struct Request {
    model: String,
    prompts: Vec<Prompt>,
    params: crate::engine::GenParams,
    stops: Vec<String>,
    n: usize,
    best_of: usize,
    echo: bool,
    logprobs: Option<usize>,
    stream: bool,
    include_usage: bool,
}

fn parse(state: &AppState, body: &[u8]) -> ApiResult<Request> {
    let m = params::object(body)?;
    let model = state.resolve_model(m.get("model"))?;
    if m.get("suffix").is_some_and(|v| !v.is_null()) {
        return Err(ApiError::invalid("The 'suffix' parameter (insertion) is not supported by this model.", Some("suffix")));
    }
    let mut gp = params::sampling(&m, state.info.vocab)?;
    gp.max_tokens = params::opt_u64(&m, "max_tokens", 0, u32::MAX as u64)?.unwrap_or(16) as usize;
    let logprobs = params::opt_u64(&m, "logprobs", 0, 5)?.map(|v| v as usize);
    let n = params::opt_u64(&m, "n", 1, 128)?.unwrap_or(1) as usize;
    let best_of = params::opt_u64(&m, "best_of", 0, 20)?.map_or(n, |b| b as usize);
    if best_of < n {
        return Err(ApiError::invalid("best_of must be greater than or equal to n.", Some("best_of")));
    }
    let stream = params::opt_bool(&m, "stream")?.unwrap_or(false);
    if stream && best_of > n {
        return Err(ApiError::invalid("Cannot stream results when best_of is greater than n.", Some("best_of")));
    }
    // best_of ranks candidates by logprob, so it needs them.
    gp.logprobs = logprobs.or((best_of > n).then_some(0));
    Ok(Request {
        model,
        prompts: prompts(state, m.get("prompt"))?,
        params: gp,
        stops: params::stops(&m)?,
        n,
        best_of,
        echo: params::opt_bool(&m, "echo")?.unwrap_or(false),
        logprobs,
        stream,
        include_usage: m.get("stream_options").and_then(|o| o.get("include_usage")).and_then(Value::as_bool).unwrap_or(false),
    })
}

fn spec(req: &Request, p: &Prompt, seed_offset: usize) -> GenSpec {
    let mut params = req.params.clone();
    params.seed = params.seed.wrapping_add(seed_offset as u64);
    GenSpec { prompt: p.ids.clone(), params, stops: req.stops.clone(), parser: OutputParser::plain(), forced: String::new(), suffix: String::new(), skip_special: true }
}

/// Legacy logprobs object; offsets count characters into `text` (which
/// starts with `prefix_len` characters of echoed prompt).
fn logprobs_obj(tokens: &[TokenInfo], prefix_len: usize) -> Value {
    let mut offset = prefix_len;
    let mut offsets = Vec::with_capacity(tokens.len());
    for t in tokens {
        offsets.push(offset);
        offset += t.token.chars().count();
    }
    json!({
        "tokens": tokens.iter().map(|t| t.token.clone()).collect::<Vec<_>>(),
        "token_logprobs": tokens.iter().map(|t| t.logprob).collect::<Vec<_>>(),
        "top_logprobs": tokens.iter().map(|t| t.top.iter().map(|(k, v)| (k.clone(), json!(v))).collect::<serde_json::Map<_, _>>()).collect::<Vec<_>>(),
        "text_offset": offsets,
    })
}

fn usage(prompt: usize, completion: usize) -> Value {
    json!({ "prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion })
}

pub async fn create(State(state): State<Arc<AppState>>, body: Bytes) -> ApiResult<Response> {
    let req = parse(&state, &body)?;
    let id = util::alnum_id("cmpl-", 29);
    let created = util::now();
    if req.stream {
        return Ok(stream(state, req, id, created).into_response());
    }
    let mut choices = Vec::new();
    let (mut prompt_tokens, mut completion) = (0, 0);
    // All prompts and candidates at once; the engine batches them.
    let specs = req.prompts.iter().flat_map(|p| (0..req.best_of).map(|j| spec(&req, p, j))).collect();
    let mut all = generation::collect_all(&state, specs).await?.into_iter();
    for (pi, p) in req.prompts.iter().enumerate() {
        prompt_tokens += p.ids.len();
        let mut cands: Vec<generation::Collected> = all.by_ref().take(req.best_of).collect();
        completion += cands.iter().map(|c| c.summary().completion_tokens).sum::<usize>();
        if req.best_of > req.n {
            // Highest mean token logprob first.
            let score = |c: &generation::Collected| c.tokens.iter().map(|t| t.logprob).sum::<f32>() / c.tokens.len().max(1) as f32;
            cands.sort_by(|a, b| score(b).total_cmp(&score(a)));
            cands.truncate(req.n);
        }
        for (j, c) in cands.into_iter().enumerate() {
            let prefix = if req.echo { p.text.clone() } else { String::new() };
            let logprobs = if req.logprobs.is_some() { logprobs_obj(&c.tokens, prefix.chars().count()) } else { Value::Null };
            choices.push(json!({
                "text": format!("{prefix}{}", c.content),
                "index": pi * req.n + j,
                "logprobs": logprobs,
                "finish_reason": c.summary().finish.as_str(),
            }));
        }
    }
    Ok(Json(json!({
        "id": id,
        "object": "text_completion",
        "created": created,
        "model": req.model,
        "system_fingerprint": state.fingerprint,
        "choices": choices,
        "usage": usage(prompt_tokens, completion),
    }))
    .into_response())
}

fn stream(state: Arc<AppState>, req: Request, id: String, created: u64) -> Sse<UnboundedReceiverStream<Result<Event, Infallible>>> {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Result<Event, Infallible>>();
    tokio::spawn(async move {
        let chunk = |choices: Value, usage: Value| {
            json!({ "id": id, "object": "text_completion", "created": created, "model": req.model, "system_fingerprint": state.fingerprint, "choices": choices, "usage": usage })
        };
        let send = |v: Value| tx.send(Ok(Event::default().data(v.to_string()))).is_ok();
        let (mut prompt_tokens, mut completion) = (0, 0);
        for (pi, p) in req.prompts.iter().enumerate() {
            prompt_tokens += p.ids.len();
            for j in 0..req.n {
                let index = pi * req.n + j;
                let mut offset = 0;
                if req.echo {
                    offset = p.text.chars().count();
                    if !send(chunk(json!([{ "text": p.text, "index": index, "logprobs": null, "finish_reason": null }]), Value::Null)) {
                        return;
                    }
                }
                let mut rx = generation::spawn(&state, spec(&req, p, j), Arc::new(AtomicBool::new(false)));
                let mut pending: Vec<TokenInfo> = Vec::new();
                while let Some(ev) = rx.recv().await {
                    let (text, finish) = match ev {
                        Err(e) => {
                            send(e.body());
                            let _ = tx.send(Ok(Event::default().data("[DONE]")));
                            return;
                        }
                        Ok(Out::Token(t)) => {
                            pending.push(t);
                            continue;
                        }
                        Ok(Out::Piece(Piece::Content(t))) => (t, Value::Null),
                        Ok(Out::Piece(_)) => continue,
                        Ok(Out::Done(s)) => {
                            completion += s.completion_tokens;
                            (String::new(), json!(s.finish.as_str()))
                        }
                    };
                    let lp = if req.logprobs.is_some() {
                        let o = logprobs_obj(&pending, offset);
                        offset += pending.iter().map(|t| t.token.chars().count()).sum::<usize>();
                        pending.clear();
                        o
                    } else {
                        Value::Null
                    };
                    let done = !finish.is_null();
                    if !send(chunk(json!([{ "text": text, "index": index, "logprobs": lp, "finish_reason": finish }]), Value::Null)) {
                        return;
                    }
                    if done {
                        break;
                    }
                }
            }
        }
        if req.include_usage {
            send(chunk(json!([]), usage(prompt_tokens, completion)));
        }
        let _ = tx.send(Ok(Event::default().data("[DONE]")));
    });
    Sse::new(UnboundedReceiverStream::new(rx))
}
