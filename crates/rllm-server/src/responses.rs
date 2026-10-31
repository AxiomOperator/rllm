//! `/v1/responses`: create (streaming, background), retrieve, delete,
//! cancel, input items and input token counts.

use std::convert::Infallible;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use axum::Json;
use axum::body::Bytes;
use axum::extract::{Path, RawQuery, State};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use serde_json::{Map, Value, json};
use tokio_stream::wrappers::UnboundedReceiverStream;

use crate::chat::{list_params, query_pairs};
use crate::error::{ApiError, ApiResult};
use crate::generation::{self, FinishKind, GenSpec, Out, Summary, TokenInfo};
use crate::output::{OutputParser, Piece, ToolCall};
use crate::params;
use crate::prompt::{self, ChatPrompt, JsonFormat, ToolChoice};
use crate::state::{AppState, History, StoredResponse, list_object, paginate};
use crate::util;

/// A validated create request.
struct Request {
    id: String,
    created: u64,
    model: String,
    prompt: ChatPrompt,
    /// Conversation (template messages) up to this response's output.
    history: Arc<History>,
    input_items: Vec<Value>,
    params: crate::engine::GenParams,
    stream: bool,
    store: bool,
    background: bool,
    parallel_tool_calls: bool,
    max_tool_calls: Option<usize>,
    logprobs: bool,
    /// Request fields echoed on the Response object.
    echo: Map<String, Value>,
}

fn input_messages(input: Option<&Value>) -> ApiResult<(Vec<Value>, Vec<Value>)> {
    let mut messages: Vec<Value> = Vec::new();
    let mut items: Vec<Value> = Vec::new();
    let text_item = |role: &str, text: &str| {
        let part = if role == "assistant" { json!({ "type": "output_text", "text": text, "annotations": [] }) } else { json!({ "type": "input_text", "text": text }) };
        json!({ "id": util::hex_id("msg_", 48), "type": "message", "role": role, "status": "completed", "content": [part] })
    };
    match input {
        None | Some(Value::Null) => return Err(ApiError::invalid("Missing required parameter: 'input'.", Some("input"))),
        Some(Value::String(s)) => {
            messages.push(json!({ "role": "user", "content": s }));
            items.push(text_item("user", s));
        }
        Some(Value::Array(arr)) => {
            for (i, it) in arr.iter().enumerate() {
                let p = format!("input[{i}]");
                let ty = it.get("type").and_then(Value::as_str).unwrap_or("message");
                match ty {
                    "message" => {
                        let role = it.get("role").and_then(Value::as_str).ok_or_else(|| ApiError::invalid(format!("Missing required parameter: '{p}.role'."), Some(&format!("{p}.role"))))?;
                        let text = prompt::content_text(it.get("content"), &format!("{p}.content"))?;
                        let role = match role {
                            "user" | "assistant" => role,
                            "system" | "developer" => "system",
                            other => return Err(ApiError::invalid(format!("Invalid value: '{other}'. Supported values are: 'user', 'assistant', 'system', and 'developer'."), Some(&format!("{p}.role")))),
                        };
                        messages.push(json!({ "role": role, "content": text }));
                        let mut item = text_item(if role == "system" { it["role"].as_str().unwrap_or("system") } else { role }, &text);
                        if let Some(id) = it.get("id").filter(|v| v.is_string()) {
                            item["id"] = id.clone();
                        }
                        items.push(item);
                    }
                    "function_call" => {
                        let call_id = it.get("call_id").and_then(Value::as_str).unwrap_or("").to_owned();
                        let name = it.get("name").and_then(Value::as_str).unwrap_or("").to_owned();
                        let args_str = it.get("arguments").and_then(Value::as_str).unwrap_or("{}");
                        let args: Value = serde_json::from_str(args_str)
                            .map_err(|_| ApiError::invalid("Function call arguments must be a JSON object encoded as a string.", Some(&format!("{p}.arguments"))))?;
                        let call = json!({ "id": call_id, "type": "function", "function": { "name": name, "arguments": args } });
                        // Consecutive calls belong to one assistant turn.
                        match messages.last_mut() {
                            Some(last) if last["role"] == "assistant" && last.get("tool_calls").is_some() => {
                                last["tool_calls"].as_array_mut().unwrap().push(call);
                            }
                            _ => messages.push(json!({ "role": "assistant", "content": "", "tool_calls": [call] })),
                        }
                        let mut item = it.clone();
                        if item.get("id").is_none() {
                            item["id"] = json!(util::hex_id("fc_", 48));
                        }
                        item["status"] = json!("completed");
                        items.push(item);
                    }
                    "function_call_output" => {
                        let call_id = it.get("call_id").cloned().unwrap_or(Value::Null);
                        let output = match it.get("output") {
                            Some(Value::String(s)) => s.clone(),
                            other => prompt::content_text(other, &format!("{p}.output"))?,
                        };
                        messages.push(json!({ "role": "tool", "content": output, "tool_call_id": call_id }));
                        let mut item = it.clone();
                        if item.get("id").is_none() {
                            item["id"] = json!(util::hex_id("fco_", 48));
                        }
                        item["status"] = json!("completed");
                        items.push(item);
                    }
                    // Reasoning from earlier turns is not replayed into the prompt.
                    "reasoning" => items.push(it.clone()),
                    other => {
                        return Err(ApiError::invalid(format!("Input item type '{other}' is not supported by this server."), Some(&format!("{p}.type"))));
                    }
                }
            }
        }
        Some(_) => return Err(ApiError::invalid("Invalid type for 'input'.", Some("input"))),
    }
    Ok((messages, items))
}

/// Responses `tools` (flat function definitions) to the template shape.
fn tools(v: Option<&Value>) -> ApiResult<Vec<Value>> {
    let Some(arr) = v.and_then(Value::as_array) else { return Ok(Vec::new()) };
    arr.iter()
        .enumerate()
        .map(|(i, t)| match t.get("type").and_then(Value::as_str) {
            Some("function") => {
                let name = t.get("name").and_then(Value::as_str).ok_or_else(|| ApiError::invalid("Missing required parameter: 'name'.", Some(&format!("tools[{i}].name"))))?;
                Ok(json!({ "type": "function", "function": { "name": name, "description": t.get("description").cloned().unwrap_or(json!("")), "parameters": t.get("parameters").cloned().unwrap_or(json!({"type": "object", "properties": {}})) } }))
            }
            Some(other) => Err(ApiError::invalid(format!("Tool type '{other}' is not supported by this server."), Some(&format!("tools[{i}].type")))),
            None => Err(ApiError::invalid("Missing required parameter: 'type'.", Some(&format!("tools[{i}].type")))),
        })
        .collect()
}

/// Build the prompt for a create or input_tokens body.
fn build(state: &AppState, m: &Map<String, Value>) -> ApiResult<(ChatPrompt, Arc<History>, Vec<Value>, Map<String, Value>)> {
    for (key, what) in [("conversation", "Conversations"), ("prompt", "Prompt templates"), ("context_management", "Context management")] {
        if m.get(key).is_some_and(|v| !v.is_null()) {
            return Err(ApiError::invalid(format!("{what} are not supported by this server."), Some(key)));
        }
    }
    let (new_messages, input_items) = input_messages(m.get("input"))?;
    let previous = params::opt_str(m, "previous_response_id")?;
    let prev = match previous {
        Some(prev) => {
            let store = state.store.lock().unwrap();
            let r = store.responses.get(prev).ok_or_else(|| ApiError::not_found(format!("Previous response with id '{prev}' not found.")))?;
            Some(r.history.clone())
        }
        None => None,
    };
    let history = History::new(prev, new_messages);
    let instructions = params::opt_str(m, "instructions")?;
    let mut messages = history.messages();
    if let Some(ins) = instructions {
        messages.insert(0, json!({ "role": "system", "content": ins }));
    }
    let tool_defs = tools(m.get("tools"))?;
    let choice = ToolChoice::parse(m.get("tool_choice"), !tool_defs.is_empty())?;
    let text = m.get("text").cloned().unwrap_or(json!({ "format": { "type": "text" } }));
    let json_format = JsonFormat::parse(text.get("format"), "text.format")?;
    let effort = m.get("reasoning").and_then(|r| r.get("effort")).and_then(Value::as_str);
    let truncation = params::opt_str(m, "truncation")?.unwrap_or("disabled");
    // Turns start at user messages: dropping whole turns keeps the roles
    // alternating as templates require. `cuts[d]`: where the kept messages
    // start with `d` turns dropped.
    let first = messages.iter().position(|m| m["role"] != "system").unwrap_or(messages.len());
    let cuts: Vec<usize> = std::iter::once(first).chain((first + 1..messages.len()).filter(|&i| messages[i]["role"] == "user")).collect();
    let render = |drop: usize| {
        let mut msgs = messages.clone();
        msgs.drain(first..cuts[drop]);
        prompt::build(state, msgs, &tool_defs, &choice, effort, json_format.as_ref())
    };
    let too_long = |e: &ApiError| e.code.as_deref() == Some("context_length_exceeded");
    // truncation "auto": drop the fewest oldest turns that make the prompt
    // fit, found by bisection (keeping at least the newest turn).
    let prompt = match render(0) {
        Err(e) if truncation == "auto" && too_long(&e) => {
            let (mut lo, mut hi) = (0, cuts.len() - 1);
            // render(lo) is too long; render(hi) fits.
            let mut best = match render(hi) {
                Err(e2) if too_long(&e2) => return Err(e),
                r => r?,
            };
            while hi - lo > 1 {
                let mid = (lo + hi) / 2;
                match render(mid) {
                    Err(e) if too_long(&e) => lo = mid,
                    r => {
                        best = r?;
                        hi = mid;
                    }
                }
            }
            best
        }
        other => other?,
    };
    let mut echo = Map::new();
    echo.insert("instructions".into(), instructions.map_or(Value::Null, |s| json!(s)));
    echo.insert("previous_response_id".into(), previous.map_or(Value::Null, |s| json!(s)));
    echo.insert("tools".into(), m.get("tools").cloned().unwrap_or(json!([])));
    echo.insert("tool_choice".into(), m.get("tool_choice").cloned().unwrap_or(json!("auto")));
    echo.insert("text".into(), text);
    echo.insert("truncation".into(), json!(truncation));
    echo.insert(
        "reasoning".into(),
        json!({ "effort": effort, "summary": m.get("reasoning").and_then(|r| r.get("summary")).cloned().unwrap_or(Value::Null) }),
    );
    Ok((prompt, history, input_items, echo))
}

fn parse(state: &AppState, body: &[u8]) -> ApiResult<Request> {
    let m = params::object(body)?;
    let model = state.resolve_model(m.get("model"))?;
    let (prompt, history, input_items, mut echo) = build(state, &m)?;
    let mut gp = params::sampling(&m, state.info.vocab)?;
    gp.json = prompt.constraint.clone();
    let max_output = params::opt_u64(&m, "max_output_tokens", 1, u32::MAX as u64)?;
    gp.max_tokens = max_output.map_or(usize::MAX, |v| v as usize);
    let top_logprobs = params::opt_u64(&m, "top_logprobs", 0, 20)?;
    let include: Vec<String> = m.get("include").and_then(Value::as_array).map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_owned)).collect()).unwrap_or_default();
    let logprobs = top_logprobs.is_some() || include.iter().any(|i| i == "message.output_text.logprobs");
    if logprobs {
        gp.logprobs = Some(top_logprobs.unwrap_or(0) as usize);
    }
    let store = params::opt_bool(&m, "store")?.unwrap_or(true);
    let background = params::opt_bool(&m, "background")?.unwrap_or(false);
    if background && !store {
        return Err(ApiError::invalid("Background responses must be stored: 'store' cannot be false with 'background'.", Some("store")));
    }
    let parallel_tool_calls = params::opt_bool(&m, "parallel_tool_calls")?.unwrap_or(true);
    let max_tool_calls = params::opt_u64(&m, "max_tool_calls", 1, u32::MAX as u64)?.map(|v| v as usize);
    echo.insert("max_output_tokens".into(), max_output.map_or(Value::Null, |v| json!(v)));
    echo.insert("max_tool_calls".into(), max_tool_calls.map_or(Value::Null, |v| json!(v)));
    echo.insert("parallel_tool_calls".into(), json!(parallel_tool_calls));
    echo.insert("temperature".into(), json!(params::opt_f32(&m, "temperature", 0.0, 2.0)?.unwrap_or(1.0)));
    echo.insert("top_p".into(), json!(params::opt_f32(&m, "top_p", 0.0, 1.0)?.unwrap_or(1.0)));
    echo.insert("top_logprobs".into(), top_logprobs.map_or(Value::Null, |v| json!(v)));
    echo.insert("metadata".into(), params::metadata(&m)?);
    echo.insert("user".into(), m.get("user").cloned().unwrap_or(Value::Null));
    echo.insert("safety_identifier".into(), m.get("safety_identifier").cloned().unwrap_or(Value::Null));
    echo.insert("prompt_cache_key".into(), m.get("prompt_cache_key").cloned().unwrap_or(Value::Null));
    echo.insert("store".into(), json!(store));
    echo.insert("background".into(), json!(background));
    Ok(Request {
        id: util::hex_id("resp_", 48),
        created: util::now(),
        model,
        prompt,
        history,
        input_items,
        params: gp,
        stream: params::opt_bool(&m, "stream")?.unwrap_or(false),
        store,
        background,
        parallel_tool_calls,
        max_tool_calls,
        logprobs,
        echo,
    })
}

fn response_object(state: &AppState, req: &Request, status: &str, output: Vec<Value>, usage: Value, extra: Option<(&str, Value)>) -> Value {
    let mut r = json!({
        "id": req.id,
        "object": "response",
        "created_at": req.created,
        "status": status,
        "completed_at": if matches!(status, "completed" | "incomplete" | "failed" | "cancelled") { json!(util::now()) } else { Value::Null },
        "error": null,
        "incomplete_details": null,
        "model": req.model,
        "output": output,
        "service_tier": "default",
        "usage": usage,
    });
    for (k, v) in &req.echo {
        r[k] = v.clone();
    }
    let text: String = r["output"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|i| i["type"] == "message")
        .flat_map(|i| i["content"].as_array().cloned().unwrap_or_default())
        .filter_map(|c| c["text"].as_str().map(str::to_owned))
        .collect();
    r["output_text"] = json!(text);
    if let Some((k, v)) = extra {
        r[k] = v;
    }
    let _ = state;
    r
}

fn usage(s: &Summary) -> Value {
    json!({
        "input_tokens": s.prompt_tokens,
        "input_tokens_details": { "cached_tokens": s.cached_tokens },
        "output_tokens": s.completion_tokens,
        "output_tokens_details": { "reasoning_tokens": s.reasoning_tokens },
        "total_tokens": s.prompt_tokens + s.completion_tokens,
    })
}

fn logprobs_json(tokens: &[TokenInfo]) -> Vec<Value> {
    tokens
        .iter()
        .map(|t| {
            let top: Vec<Value> = t.top.iter().map(|(tok, lp)| json!({ "token": tok, "bytes": tok.as_bytes(), "logprob": lp })).collect();
            json!({ "token": t.token, "bytes": t.token.as_bytes(), "logprob": t.logprob, "top_logprobs": top })
        })
        .collect()
}

/// Builds output items as pieces arrive, and the matching stream events.
struct OutputBuilder {
    items: Vec<Value>,
    /// Index into `items` of the open reasoning / message item.
    open: Option<usize>,
    pending_lp: Vec<TokenInfo>,
    logprobs: bool,
    tool_calls: usize,
    max_tool_calls: usize,
    /// Assistant turn for the stored conversation.
    content: String,
    reasoning: String,
    calls: Vec<Value>,
}

impl OutputBuilder {
    fn new(req: &Request) -> Self {
        let max = if req.parallel_tool_calls { req.max_tool_calls.unwrap_or(usize::MAX) } else { 1 };
        Self {
            items: Vec::new(),
            open: None,
            pending_lp: Vec::new(),
            logprobs: req.logprobs,
            tool_calls: 0,
            max_tool_calls: max,
            content: String::new(),
            reasoning: String::new(),
            calls: Vec::new(),
        }
    }

    /// Close the open item; returns its done events.
    fn close(&mut self, events: &mut Vec<Value>, status: &str) {
        let Some(i) = self.open.take() else { return };
        let item = &mut self.items[i];
        item["status"] = json!(status);
        let (id, kind) = (item["id"].clone(), item["type"].as_str().unwrap_or("").to_owned());
        let part = item["content"][0].clone();
        if kind == "message" {
            events.push(json!({ "type": "response.output_text.done", "item_id": id, "output_index": i, "content_index": 0, "text": part["text"], "logprobs": part["logprobs"] }));
        } else {
            events.push(json!({ "type": "response.reasoning_text.done", "item_id": id, "output_index": i, "content_index": 0, "text": part["text"] }));
        }
        events.push(json!({ "type": "response.content_part.done", "item_id": id, "output_index": i, "content_index": 0, "part": part }));
        events.push(json!({ "type": "response.output_item.done", "output_index": i, "item": self.items[i].clone() }));
    }

    fn open_item(&mut self, kind: &str, events: &mut Vec<Value>) -> usize {
        if let Some(i) = self.open
            && self.items[i]["type"] == kind
        {
            return i;
        }
        self.close(events, "completed");
        let i = self.items.len();
        let (item, part) = if kind == "message" {
            (
                json!({ "id": util::hex_id("msg_", 48), "type": "message", "status": "in_progress", "role": "assistant", "content": [] }),
                json!({ "type": "output_text", "text": "", "annotations": [], "logprobs": [] }),
            )
        } else {
            (
                json!({ "id": util::hex_id("rs_", 48), "type": "reasoning", "status": "in_progress", "summary": [], "content": [] }),
                json!({ "type": "reasoning_text", "text": "" }),
            )
        };
        events.push(json!({ "type": "response.output_item.added", "output_index": i, "item": item }));
        let mut item = item;
        item["content"] = json!([part.clone()]);
        events.push(json!({ "type": "response.content_part.added", "item_id": item["id"], "output_index": i, "content_index": 0, "part": part }));
        self.items.push(item);
        self.open = Some(i);
        i
    }

    fn piece(&mut self, p: Piece, events: &mut Vec<Value>) {
        match p {
            Piece::Reasoning(t) => {
                let i = self.open_item("reasoning", events);
                self.reasoning.push_str(&t);
                let cur = self.items[i]["content"][0]["text"].as_str().unwrap_or("").to_owned();
                self.items[i]["content"][0]["text"] = json!(cur + &t);
                events.push(json!({ "type": "response.reasoning_text.delta", "item_id": self.items[i]["id"], "output_index": i, "content_index": 0, "delta": t }));
            }
            Piece::Content(t) => {
                let i = self.open_item("message", events);
                self.content.push_str(&t);
                let cur = self.items[i]["content"][0]["text"].as_str().unwrap_or("").to_owned();
                self.items[i]["content"][0]["text"] = json!(cur + &t);
                let lp = if self.logprobs { logprobs_json(&std::mem::take(&mut self.pending_lp)) } else { Vec::new() };
                if self.logprobs {
                    self.items[i]["content"][0]["logprobs"].as_array_mut().unwrap().extend(lp.iter().cloned());
                }
                events.push(json!({ "type": "response.output_text.delta", "item_id": self.items[i]["id"], "output_index": i, "content_index": 0, "delta": t, "logprobs": lp }));
            }
            Piece::ToolCall(ToolCall { name, arguments }) => {
                if self.tool_calls >= self.max_tool_calls {
                    return;
                }
                self.tool_calls += 1;
                self.close(events, "completed");
                let i = self.items.len();
                let call_id = util::alnum_id("call_", 24);
                let id = util::hex_id("fc_", 48);
                let item = json!({ "id": id, "type": "function_call", "status": "in_progress", "call_id": call_id, "name": name, "arguments": "" });
                events.push(json!({ "type": "response.output_item.added", "output_index": i, "item": item }));
                events.push(json!({ "type": "response.function_call_arguments.delta", "item_id": id, "output_index": i, "delta": arguments }));
                events.push(json!({ "type": "response.function_call_arguments.done", "item_id": id, "output_index": i, "name": name, "arguments": arguments }));
                let mut done = item;
                done["arguments"] = json!(arguments);
                done["status"] = json!("completed");
                events.push(json!({ "type": "response.output_item.done", "output_index": i, "item": done }));
                self.items.push(done);
                let args: Value = serde_json::from_str(&arguments).unwrap_or(json!({}));
                self.calls.push(json!({ "id": call_id, "type": "function", "function": { "name": name, "arguments": args } }));
            }
        }
    }

    /// The assistant turn, for the stored conversation.
    fn assistant_message(&self) -> Value {
        let mut m = json!({ "role": "assistant", "content": self.content });
        if !self.calls.is_empty() {
            m["tool_calls"] = json!(self.calls);
        }
        if !self.reasoning.is_empty() {
            m["reasoning_content"] = json!(self.reasoning);
        }
        m
    }
}

/// Final status for a summary: completed, incomplete (token limit), cancelled.
fn final_status(s: &Summary) -> (&'static str, Value) {
    match s.finish {
        FinishKind::Length => ("incomplete", json!({ "reason": "max_output_tokens" })),
        FinishKind::Cancelled => ("cancelled", Value::Null),
        FinishKind::Stop => ("completed", Value::Null),
    }
}

/// Run the generation, feeding stream events to `emit`; returns the final
/// Response object (also stored if requested).
async fn run(state: Arc<AppState>, req: Arc<Request>, cancel: Arc<AtomicBool>, mut emit: impl FnMut(Value) -> bool) -> Value {
    let spec = GenSpec {
        prompt: req.prompt.ids.clone(),
        params: req.params.clone(),
        stops: Vec::new(),
        parser: OutputParser::new(state.format, req.prompt.tools, req.prompt.reasoning),
        forced: req.prompt.forced.clone(),
        suffix: req.prompt.suffix.clone(),
        skip_special: false,
    };
    let mut rx = generation::spawn(&state, spec, cancel);
    let mut b = OutputBuilder::new(&req);
    let mut events = Vec::new();
    let mut alive = true;
    let fin = loop {
        let Some(ev) = rx.recv().await else { break Err(ApiError::server("generation ended unexpectedly")) };
        match ev {
            Err(e) => break Err(e),
            Ok(Out::Token(t)) => b.pending_lp.push(t),
            Ok(Out::Piece(p)) => b.piece(p, &mut events),
            Ok(Out::Done(s)) => break Ok(s),
        }
        for e in events.drain(..) {
            alive &= emit(e);
        }
        if !alive {
            // The client went away: dropping `rx` cancels the generation.
            break Err(ApiError::server("client disconnected"));
        }
    };
    let response = match fin {
        Ok(s) => {
            let (status, incomplete) = final_status(&s);
            b.close(&mut events, if status == "incomplete" { "incomplete" } else { "completed" });
            for e in events.drain(..) {
                emit(e);
            }
            let mut r = response_object(&state, &req, status, b.items.clone(), usage(&s), None);
            r["incomplete_details"] = incomplete;
            r
        }
        Err(e) => response_object(
            &state,
            &req,
            "failed",
            b.items.clone(),
            Value::Null,
            Some(("error", json!({ "code": e.code.clone().unwrap_or_else(|| "server_error".into()), "message": e.message }))),
        ),
    };
    if req.store {
        let mut store = state.store.lock().unwrap();
        // Deleted (or evicted) while running: stays gone.
        if let Some(stored) = store.responses.get_mut(&req.id) {
            // A cancel that arrived first wins.
            let cancelled = stored.response["status"] == "cancelled";
            stored.response = response.clone();
            if cancelled {
                stored.response["status"] = json!("cancelled");
            }
            stored.history = History::new(Some(req.history.clone()), vec![b.assistant_message()]);
            stored.cancel = None;
        }
    }
    response
}

/// Marks a stored response failed if its request is dropped (the client
/// disconnected) before it finishes.
struct Unfinished {
    state: Arc<AppState>,
    id: Option<String>,
}

impl Drop for Unfinished {
    fn drop(&mut self) {
        let Some(id) = self.id.take() else { return };
        let mut store = self.state.store.lock().unwrap();
        if let Some(r) = store.responses.get_mut(&id)
            && matches!(r.response["status"].as_str(), Some("queued" | "in_progress"))
        {
            r.response["status"] = json!("failed");
            r.response["error"] = json!({ "code": "server_error", "message": "The client disconnected before the response finished." });
            r.cancel = None;
        }
    }
}

pub async fn create(State(state): State<Arc<AppState>>, body: Bytes) -> ApiResult<Response> {
    // Rendering and tokenizing (several times for truncation "auto") is CPU
    // work: keep it off the async workers.
    let s = state.clone();
    let req = tokio::task::spawn_blocking(move || parse(&s, &body)).await.map_err(|e| ApiError::server(format!("{e}")))??;
    let req = Arc::new(req);
    let cancel = Arc::new(AtomicBool::new(false));
    if req.store {
        let initial = response_object(&state, &req, if req.background { "queued" } else { "in_progress" }, Vec::new(), Value::Null, None);
        state.store.lock().unwrap().responses.insert(
            req.id.clone(),
            StoredResponse { response: initial, input_items: req.input_items.clone(), history: req.history.clone(), cancel: Some(cancel.clone()) },
        );
    }
    if req.stream {
        return Ok(stream(state, req, cancel).into_response());
    }
    if req.background {
        let initial = response_object(&state, &req, "queued", Vec::new(), Value::Null, None);
        let (s, r) = (state.clone(), req.clone());
        tokio::spawn(async move {
            if let Some(stored) = s.store.lock().unwrap().responses.get_mut(&r.id) {
                stored.response["status"] = json!("in_progress");
            }
            run(s, r, cancel, |_| true).await;
        });
        return Ok(Json(initial).into_response());
    }
    let mut guard = Unfinished { state: state.clone(), id: req.store.then(|| req.id.clone()) };
    let response = run(state, req, cancel, |_| true).await;
    guard.id = None;
    if response["status"] == "failed" {
        let message = response["error"]["message"].as_str().unwrap_or("The server had an error while processing your request.");
        return Err(ApiError::server(message));
    }
    Ok(Json(response).into_response())
}

fn stream(state: Arc<AppState>, req: Arc<Request>, cancel: Arc<AtomicBool>) -> Sse<UnboundedReceiverStream<Result<Event, Infallible>>> {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Result<Event, Infallible>>();
    tokio::spawn(async move {
        let mut seq = 0u64;
        let mut send = move |mut v: Value| {
            v["sequence_number"] = json!(seq);
            seq += 1;
            let kind = v["type"].as_str().unwrap_or("message").to_owned();
            tx.send(Ok(Event::default().event(kind).data(v.to_string()))).is_ok()
        };
        let initial = response_object(&state, &req, "in_progress", Vec::new(), Value::Null, None);
        send(json!({ "type": "response.created", "response": initial }));
        send(json!({ "type": "response.in_progress", "response": initial }));
        let final_resp = run(state, req, cancel, &mut send).await;
        let kind = match final_resp["status"].as_str() {
            Some("incomplete") => "response.incomplete",
            Some("failed") => {
                send(json!({ "type": "error", "code": final_resp["error"]["code"], "message": final_resp["error"]["message"], "param": null }));
                "response.failed"
            }
            _ => "response.completed",
        };
        send(json!({ "type": kind, "response": final_resp }));
    });
    Sse::new(UnboundedReceiverStream::new(rx))
}

fn not_found(id: &str) -> ApiError {
    ApiError::not_found(format!("Response with id '{id}' not found."))
}

pub async fn retrieve(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> ApiResult<Json<Value>> {
    let store = state.store.lock().unwrap();
    store.responses.get(&id).map(|r| Json(r.response.clone())).ok_or_else(|| not_found(&id))
}

pub async fn delete(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> ApiResult<Json<Value>> {
    let mut store = state.store.lock().unwrap();
    let r = store.responses.remove(&id).ok_or_else(|| not_found(&id))?;
    if let Some(c) = &r.cancel {
        c.store(true, Ordering::Relaxed);
    }
    Ok(Json(json!({ "id": id, "object": "response.deleted", "deleted": true })))
}

pub async fn cancel(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> ApiResult<Json<Value>> {
    let mut store = state.store.lock().unwrap();
    let r = store.responses.get_mut(&id).ok_or_else(|| not_found(&id))?;
    if r.response["background"] != true {
        return Err(ApiError::invalid("Only responses created with background=true can be cancelled.", None));
    }
    if matches!(r.response["status"].as_str(), Some("queued" | "in_progress")) {
        if let Some(c) = &r.cancel {
            c.store(true, Ordering::Relaxed);
        }
        r.response["status"] = json!("cancelled");
    }
    Ok(Json(r.response.clone()))
}

pub async fn input_items(State(state): State<Arc<AppState>>, Path(id): Path<String>, RawQuery(q): RawQuery) -> ApiResult<Json<Value>> {
    let q = query_pairs(q.as_deref());
    let (after, limit, desc) = list_params(&q, true)?;
    let store = state.store.lock().unwrap();
    let r = store.responses.get(&id).ok_or_else(|| not_found(&id))?;
    let items: Vec<(String, Value)> = r.input_items.iter().map(|i| (i["id"].as_str().unwrap_or("").to_owned(), i.clone())).collect();
    let (page, has_more) = paginate(items, after.as_deref(), limit, desc);
    Ok(Json(list_object(page.into_iter().map(|(_, v)| v).collect(), has_more)))
}

pub async fn input_tokens(State(state): State<Arc<AppState>>, body: Bytes) -> ApiResult<Json<Value>> {
    let m = params::object(&body)?;
    if m.contains_key("model") {
        state.resolve_model(m.get("model"))?;
    }
    let (prompt, ..) = tokio::task::spawn_blocking(move || build(&state, &m)).await.map_err(|e| ApiError::server(format!("{e}")))??;
    Ok(Json(json!({ "object": "response.input_tokens", "input_tokens": prompt.ids.len() })))
}
