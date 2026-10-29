//! Endpoints outside the OpenAI API, in the shape llama.cpp's server uses
//! (so tools written for it work): `POST /tokenize`, `POST /detokenize`,
//! `POST /apply-template`, and `GET /metrics` (Prometheus).

use std::sync::Arc;

use axum::Json;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::header;
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};

use crate::error::{ApiError, ApiResult};
use crate::params;
use crate::prompt;
use crate::state::AppState;

/// `{"content": "...", "add_special": false, "with_pieces": false}` →
/// `{"tokens": [ids]}` or, with pieces, `{"tokens": [{"id", "piece"}]}`.
pub async fn tokenize(State(state): State<Arc<AppState>>, body: Bytes) -> ApiResult<Json<Value>> {
    let m = params::object(&body)?;
    let content = params::opt_str(&m, "content")?.unwrap_or("");
    let add_special = params::opt_bool(&m, "add_special")?.unwrap_or(false);
    let ids = state.tok.encode_with(content, add_special).map_err(ApiError::from)?;
    if params::opt_bool(&m, "with_pieces")?.unwrap_or(false) {
        let pieces: Vec<Value> = ids.iter().map(|&id| json!({ "id": id, "piece": state.tok.token_text(id) })).collect();
        return Ok(Json(json!({ "tokens": pieces })));
    }
    Ok(Json(json!({ "tokens": ids })))
}

/// `{"tokens": [ids]}` → `{"content": "..."}`.
pub async fn detokenize(State(state): State<Arc<AppState>>, body: Bytes) -> ApiResult<Json<Value>> {
    let m = params::object(&body)?;
    let ids: Vec<u32> = match m.get("tokens") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Array(a)) => a
            .iter()
            .map(|t| t.as_u64().filter(|&t| (t as usize) < state.info.vocab).map(|t| t as u32))
            .collect::<Option<_>>()
            .ok_or_else(|| ApiError::invalid("Invalid token in 'tokens'.", Some("tokens")))?,
        Some(_) => return Err(ApiError::invalid("Invalid type for 'tokens': expected an array of token ids.", Some("tokens"))),
    };
    let content = state.tok.decode(&ids, false).map_err(ApiError::from)?;
    Ok(Json(json!({ "content": content })))
}

/// `{"messages": [...], "tools"?: [...]}` → `{"prompt": "..."}`: the chat
/// template applied exactly as `/v1/chat/completions` would.
pub async fn apply_template(State(state): State<Arc<AppState>>, body: Bytes) -> ApiResult<Json<Value>> {
    let m = params::object(&body)?;
    let raw = m.get("messages").and_then(Value::as_array).ok_or_else(|| ApiError::invalid("Missing required parameter: 'messages'.", Some("messages")))?;
    let messages = prompt::chat_messages(raw)?;
    let tools = prompt::chat_tools(m.get("tools"))?;
    let text = state
        .tok
        .apply_chat_template_json(&messages, (!tools.is_empty()).then_some(&tools[..]), true, None)
        .map_err(|e| ApiError::invalid(format!("Could not apply the chat template: {e:#}"), Some("messages")))?;
    Ok(Json(json!({ "prompt": text })))
}

pub async fn metrics(State(state): State<Arc<AppState>>) -> Response {
    ([(header::CONTENT_TYPE, "text/plain; version=0.0.4")], state.metrics.render(&state.model_id)).into_response()
}
