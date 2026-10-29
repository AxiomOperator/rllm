//! `/v1/chat/completions`: create (streaming and not) and the stored
//! completion endpoints.

use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use axum::body::Bytes;
use axum::extract::{Path, RawQuery, State};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{Value, json};
use tokio_stream::wrappers::UnboundedReceiverStream;

use crate::error::{ApiError, ApiResult};
use crate::generation::{self, GenSpec, Out, TokenInfo};
use crate::output::{OutputParser, Piece, ToolCall};
use crate::params;
use crate::prompt::{self, JsonFormat, ToolChoice};
use crate::state::{AppState, StoredChat, list_object, paginate};
use crate::util;

/// Everything a chat request asks for, validated.
struct ChatRequest {
    model: String,
    messages_raw: Vec<Value>,
    prompt: prompt::ChatPrompt,
    params: crate::engine::GenParams,
    stops: Vec<String>,
    n: usize,
    stream: bool,
    include_usage: bool,
    logprobs: bool,
    parallel_tool_calls: bool,
    store: bool,
    metadata: Value,
}

fn parse(state: &AppState, body: &[u8]) -> ApiResult<ChatRequest> {
    let m = params::object(body)?;
    let model = state.resolve_model(m.get("model"))?;
    let messages_raw = m
        .get("messages")
        .and_then(Value::as_array)
        .cloned()
        .ok_or_else(|| ApiError::invalid("Missing required parameter: 'messages'.", Some("messages")))?;
    if m.get("modalities").and_then(Value::as_array).is_some_and(|a| a.iter().any(|v| v == "audio")) || m.get("audio").is_some_and(|v| !v.is_null()) {
        return Err(ApiError::invalid("Audio output is not supported by this model.", Some("modalities")));
    }
    if m.get("web_search_options").is_some_and(|v| !v.is_null()) {
        return Err(ApiError::invalid("Web search is not supported by this server.", Some("web_search_options")));
    }
    let messages = prompt::chat_messages(&messages_raw)?;
    // Legacy `functions` / `function_call` map onto tools.
    let tools_value = match (m.get("tools"), m.get("functions").and_then(Value::as_array)) {
        (Some(t), _) if !t.is_null() => Some(t.clone()),
        (_, Some(fs)) => Some(Value::Array(fs.iter().map(|f| json!({ "type": "function", "function": f })).collect())),
        _ => None,
    };
    let tools = prompt::chat_tools(tools_value.as_ref())?;
    let choice_value = m.get("tool_choice").or_else(|| m.get("function_call")).cloned().map(|v| match v {
        Value::Object(o) if o.contains_key("name") && !o.contains_key("type") => json!({ "type": "function", "function": o }),
        other => other,
    });
    let choice = ToolChoice::parse(choice_value.as_ref(), !tools.is_empty())?;
    let json_format = JsonFormat::parse(m.get("response_format"), "response_format")?;
    let reasoning_effort = params::opt_str(&m, "reasoning_effort")?;
    let prompt = prompt::build(state, messages, &tools, &choice, reasoning_effort, json_format.as_ref())?;

    let mut gp = params::sampling(&m, state.info.vocab)?;
    gp.json = prompt.constraint.clone();
    let max = params::opt_u64(&m, "max_completion_tokens", 1, u32::MAX as u64)?.or(params::opt_u64(&m, "max_tokens", 1, u32::MAX as u64)?);
    gp.max_tokens = max.map_or(usize::MAX, |v| v as usize);
    let logprobs = params::opt_bool(&m, "logprobs")?.unwrap_or(false);
    let top = params::opt_u64(&m, "top_logprobs", 0, 20)?;
    if top.is_some() && !logprobs {
        return Err(ApiError::invalid("Invalid value for 'top_logprobs': 'logprobs' must be set to true.", Some("top_logprobs")));
    }
    if logprobs {
        gp.logprobs = Some(top.unwrap_or(0) as usize);
    }
    let stream = params::opt_bool(&m, "stream")?.unwrap_or(false);
    let include_usage = m.get("stream_options").and_then(|o| o.get("include_usage")).and_then(Value::as_bool).unwrap_or(false);
    if m.get("stream_options").is_some_and(|v| !v.is_null()) && !stream {
        return Err(ApiError::invalid("The 'stream_options' parameter is only allowed when 'stream' is enabled.", Some("stream_options")));
    }
    Ok(ChatRequest {
        model,
        messages_raw,
        prompt,
        params: gp,
        stops: params::stops(&m)?,
        n: params::opt_u64(&m, "n", 1, 128)?.unwrap_or(1) as usize,
        stream,
        include_usage,
        logprobs,
        parallel_tool_calls: params::opt_bool(&m, "parallel_tool_calls")?.unwrap_or(true),
        store: params::opt_bool(&m, "store")?.unwrap_or(false),
        metadata: params::metadata(&m)?,
    })
}

fn spec(state: &AppState, req: &ChatRequest, choice: usize) -> GenSpec {
    let mut params = req.params.clone();
    // Distinct seeds per choice, still reproducible for a given seed.
    params.seed = params.seed.wrapping_add(choice as u64);
    GenSpec {
        prompt: req.prompt.ids.clone(),
        params,
        stops: req.stops.clone(),
        parser: OutputParser::new(state.format, req.prompt.tools, req.prompt.reasoning),
        forced: req.prompt.forced.clone(),
        suffix: req.prompt.suffix.clone(),
        skip_special: false,
    }
}

fn logprob_entry(t: &TokenInfo) -> Value {
    let top: Vec<Value> = t.top.iter().map(|(tok, lp)| json!({ "token": tok, "bytes": tok.as_bytes(), "logprob": lp })).collect();
    json!({ "token": t.token, "bytes": t.token.as_bytes(), "logprob": t.logprob, "top_logprobs": top })
}

fn tool_call_json(call: &ToolCall) -> Value {
    json!({ "id": util::alnum_id("call_", 24), "type": "function", "function": { "name": call.name, "arguments": call.arguments } })
}

fn usage(prompt: usize, cached: usize, completion: usize, reasoning: usize) -> Value {
    json!({
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "prompt_tokens_details": { "cached_tokens": cached, "audio_tokens": 0 },
        "completion_tokens_details": { "reasoning_tokens": reasoning, "audio_tokens": 0, "accepted_prediction_tokens": 0, "rejected_prediction_tokens": 0 },
    })
}

pub async fn create(State(state): State<Arc<AppState>>, body: Bytes) -> ApiResult<Response> {
    let req = parse(&state, &body)?;
    let id = util::alnum_id("chatcmpl-", 29);
    let created = util::now();
    if req.stream {
        return Ok(stream(state, req, id, created).into_response());
    }
    let mut choices = Vec::with_capacity(req.n);
    let (mut completion, mut reasoning, mut prompt_tokens, mut cached) = (0, 0, req.prompt.ids.len(), 0);
    let specs = (0..req.n).map(|i| spec(&state, &req, i)).collect();
    for (i, c) in generation::collect_all(&state, specs).await?.into_iter().enumerate() {
        let s = c.summary().clone();
        prompt_tokens = s.prompt_tokens;
        cached = cached.max(s.cached_tokens);
        completion += s.completion_tokens;
        reasoning += s.reasoning_tokens;
        let mut calls: Vec<Value> = c.tool_calls.iter().map(tool_call_json).collect();
        if !req.parallel_tool_calls {
            calls.truncate(1);
        }
        let finish = if calls.is_empty() { s.finish.as_str() } else { "tool_calls" };
        let mut message = json!({
            "role": "assistant",
            "content": if c.content.is_empty() && !calls.is_empty() { Value::Null } else { json!(c.content) },
            "refusal": null,
            "annotations": [],
        });
        if !calls.is_empty() {
            message["tool_calls"] = Value::Array(calls);
        }
        if !c.reasoning.is_empty() {
            // Not in the OpenAI schema; the de facto field for reasoning text.
            message["reasoning_content"] = json!(c.reasoning);
        }
        let logprobs = if req.logprobs { json!({ "content": c.tokens.iter().map(logprob_entry).collect::<Vec<_>>(), "refusal": null }) } else { Value::Null };
        choices.push(json!({ "index": i, "message": message, "logprobs": logprobs, "finish_reason": finish }));
    }
    let completion_obj = json!({
        "id": id,
        "object": "chat.completion",
        "created": created,
        "model": req.model,
        "choices": choices,
        "usage": usage(prompt_tokens, cached, completion, reasoning),
        "service_tier": "default",
        "system_fingerprint": state.fingerprint,
        "metadata": req.metadata,
    });
    if req.store {
        let messages = req.messages_raw.iter().enumerate().map(|(i, m)| with_id(m, &format!("{id}-{i}"))).collect();
        state.store.lock().unwrap().chats.insert(id.clone(), StoredChat { completion: completion_obj.clone(), messages });
    }
    Ok(Json(completion_obj).into_response())
}

fn with_id(m: &Value, id: &str) -> Value {
    let mut v = m.clone();
    v["id"] = json!(id);
    v
}

fn stream(state: Arc<AppState>, req: ChatRequest, id: String, created: u64) -> Sse<UnboundedReceiverStream<Result<Event, Infallible>>> {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Result<Event, Infallible>>();
    tokio::spawn(async move {
        let chunk = |choices: Value, usage: Value| {
            json!({
                "id": id, "object": "chat.completion.chunk", "created": created, "model": req.model,
                "service_tier": "default", "system_fingerprint": state.fingerprint, "choices": choices, "usage": usage,
            })
        };
        let send = |v: Value| tx.send(Ok(Event::default().data(v.to_string()))).is_ok();
        let (mut completion, mut reasoning, mut prompt_tokens, mut cached) = (0, 0, req.prompt.ids.len(), 0);
        for i in 0..req.n {
            let first = json!([{ "index": i, "delta": { "role": "assistant", "content": "", "refusal": null }, "logprobs": null, "finish_reason": null }]);
            if !send(chunk(first, Value::Null)) {
                return;
            }
            let mut rx = generation::spawn(&state, spec(&state, &req, i), Arc::new(AtomicBool::new(false)));
            let mut pending_lp: Vec<Value> = Vec::new();
            let mut n_calls = 0usize;
            let lp_value = |pending: &mut Vec<Value>| {
                if req.logprobs { json!({ "content": std::mem::take(pending), "refusal": null }) } else { Value::Null }
            };
            while let Some(ev) = rx.recv().await {
                let delta = match ev {
                    Err(e) => {
                        send(e.body());
                        let _ = tx.send(Ok(Event::default().data("[DONE]")));
                        return;
                    }
                    Ok(Out::Token(t)) => {
                        pending_lp.push(logprob_entry(&t));
                        continue;
                    }
                    Ok(Out::Piece(Piece::Content(t))) => json!({ "content": t }),
                    Ok(Out::Piece(Piece::Reasoning(t))) => json!({ "reasoning_content": t }),
                    Ok(Out::Piece(Piece::ToolCall(call))) => {
                        if !req.parallel_tool_calls && n_calls > 0 {
                            continue;
                        }
                        let mut tc = tool_call_json(&call);
                        tc["index"] = json!(n_calls);
                        n_calls += 1;
                        json!({ "tool_calls": [tc] })
                    }
                    Ok(Out::Done(s)) => {
                        prompt_tokens = s.prompt_tokens;
                        cached = cached.max(s.cached_tokens);
                        completion += s.completion_tokens;
                        reasoning += s.reasoning_tokens;
                        let finish = if n_calls > 0 { "tool_calls" } else { s.finish.as_str() };
                        let lp = lp_value(&mut pending_lp);
                        if !send(chunk(json!([{ "index": i, "delta": {}, "logprobs": lp, "finish_reason": finish }]), Value::Null)) {
                            return;
                        }
                        break;
                    }
                };
                let lp = lp_value(&mut pending_lp);
                if !send(chunk(json!([{ "index": i, "delta": delta, "logprobs": lp, "finish_reason": null }]), Value::Null)) {
                    return;
                }
            }
        }
        if req.include_usage {
            send(chunk(json!([]), usage(prompt_tokens, cached, completion, reasoning)));
        }
        let _ = tx.send(Ok(Event::default().data("[DONE]")));
    });
    Sse::new(UnboundedReceiverStream::new(rx))
}

/// Parse `?a=b&metadata[k]=v` style query strings.
pub fn query_pairs(q: Option<&str>) -> Vec<(String, String)> {
    let decode = |s: &str| {
        let s = s.replace('+', " ");
        let bytes = s.as_bytes();
        let mut out = Vec::with_capacity(bytes.len());
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == b'%'
                && i + 2 < bytes.len()
                && let Ok(b) = u8::from_str_radix(std::str::from_utf8(&bytes[i + 1..i + 3]).unwrap_or(""), 16)
            {
                out.push(b);
                i += 3;
                continue;
            }
            out.push(bytes[i]);
            i += 1;
        }
        String::from_utf8_lossy(&out).into_owned()
    };
    q.unwrap_or("")
        .split('&')
        .filter(|kv| !kv.is_empty())
        .map(|kv| {
            let (k, v) = kv.split_once('=').unwrap_or((kv, ""));
            (decode(k), decode(v))
        })
        .collect()
}

/// `limit` (1..=100, default 20), `order` (`asc` default), `after`.
pub fn list_params(q: &[(String, String)], default_desc: bool) -> ApiResult<(Option<String>, usize, bool)> {
    let get = |k: &str| q.iter().find(|(a, _)| a == k).map(|(_, v)| v.clone());
    let limit = match get("limit") {
        Some(l) => l.parse::<usize>().ok().filter(|l| (1..=100).contains(l)).ok_or_else(|| ApiError::invalid("Invalid 'limit': must be an integer between 1 and 100.", Some("limit")))?,
        None => 20,
    };
    let desc = match get("order").as_deref() {
        Some("desc") => true,
        Some("asc") => false,
        None => default_desc,
        Some(o) => return Err(ApiError::invalid(format!("Invalid 'order': '{o}'. Expected 'asc' or 'desc'."), Some("order"))),
    };
    Ok((get("after"), limit, desc))
}

pub async fn list(State(state): State<Arc<AppState>>, RawQuery(q): RawQuery) -> ApiResult<Json<Value>> {
    let q = query_pairs(q.as_deref());
    let (after, limit, desc) = list_params(&q, false)?;
    let model = q.iter().find(|(k, _)| k == "model").map(|(_, v)| v.clone());
    let meta: HashMap<String, String> =
        q.iter().filter_map(|(k, v)| k.strip_prefix("metadata[").and_then(|k| k.strip_suffix(']')).map(|k| (k.to_owned(), v.clone()))).collect();
    let store = state.store.lock().unwrap();
    let items: Vec<(String, Value)> = store
        .chats
        .iter()
        .filter(|(_, c)| model.as_ref().is_none_or(|m| c.completion["model"] == m.as_str()))
        .filter(|(_, c)| meta.iter().all(|(k, v)| c.completion["metadata"][k] == v.as_str()))
        .map(|(id, c)| (id.clone(), c.completion.clone()))
        .collect();
    let (page, has_more) = paginate(items, after.as_deref(), limit, desc);
    Ok(Json(list_object(page.into_iter().map(|(_, v)| v).collect(), has_more)))
}

fn not_found(id: &str) -> ApiError {
    ApiError::not_found(format!("No chat completion found with id '{id}'."))
}

pub async fn retrieve(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> ApiResult<Json<Value>> {
    let store = state.store.lock().unwrap();
    store.chats.get(&id).map(|c| Json(c.completion.clone())).ok_or_else(|| not_found(&id))
}

pub async fn update(State(state): State<Arc<AppState>>, Path(id): Path<String>, body: Bytes) -> ApiResult<Json<Value>> {
    let m = params::object(&body)?;
    if !m.contains_key("metadata") {
        return Err(ApiError::invalid("Missing required parameter: 'metadata'.", Some("metadata")));
    }
    let metadata = params::metadata(&m)?;
    let mut store = state.store.lock().unwrap();
    let c = store.chats.get_mut(&id).ok_or_else(|| not_found(&id))?;
    c.completion["metadata"] = metadata;
    Ok(Json(c.completion.clone()))
}

pub async fn delete(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> ApiResult<Json<Value>> {
    let mut store = state.store.lock().unwrap();
    store.chats.remove(&id).ok_or_else(|| not_found(&id))?;
    Ok(Json(json!({ "object": "chat.completion.deleted", "id": id, "deleted": true })))
}

pub async fn messages(State(state): State<Arc<AppState>>, Path(id): Path<String>, RawQuery(q): RawQuery) -> ApiResult<Json<Value>> {
    let q = query_pairs(q.as_deref());
    let (after, limit, desc) = list_params(&q, false)?;
    let store = state.store.lock().unwrap();
    let c = store.chats.get(&id).ok_or_else(|| not_found(&id))?;
    let items: Vec<(String, Value)> = c
        .messages
        .iter()
        .map(|m| {
            let mid = m["id"].as_str().unwrap_or("").to_owned();
            let content = match &m["content"] {
                Value::Array(parts) => json!(parts.iter().filter_map(|p| p["text"].as_str()).collect::<Vec<_>>().join("")),
                other => other.clone(),
            };
            let parts = if m["content"].is_array() { m["content"].clone() } else { Value::Null };
            (mid.clone(), json!({ "id": mid, "role": m["role"], "content": content, "content_parts": parts, "name": m.get("name") }))
        })
        .collect();
    let (page, has_more) = paginate(items, after.as_deref(), limit, desc);
    Ok(Json(list_object(page.into_iter().map(|(_, v)| v).collect(), has_more)))
}
