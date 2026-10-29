//! `/v1/embeddings`: mean-pooled, L2-normalized final hidden states of the
//! served model. `dimensions` truncates and renormalizes (as with
//! OpenAI's `text-embedding-3` models).

use std::sync::Arc;

use axum::Json;
use axum::body::Bytes;
use axum::extract::State;
use base64::Engine as _;
use serde_json::{Value, json};
use tokio::sync::oneshot;

use crate::engine::Job;
use crate::error::{ApiError, ApiResult};
use crate::params;
use crate::state::AppState;

fn inputs(state: &AppState, v: Option<&Value>) -> ApiResult<Vec<Vec<u32>>> {
    let tok = &state.tok;
    let text = |s: &str| -> ApiResult<Vec<u32>> {
        if s.is_empty() {
            return Err(ApiError::invalid("'$.input' is invalid. Please check the API reference.", Some("input")));
        }
        tok.encode(s).map_err(ApiError::from)
    };
    let ids = |a: &[Value]| -> ApiResult<Vec<u32>> {
        a.iter()
            .map(|t| t.as_u64().filter(|&t| (t as usize) < state.info.vocab).map(|t| t as u32))
            .collect::<Option<Vec<u32>>>()
            .filter(|v| !v.is_empty())
            .ok_or_else(|| ApiError::invalid("Invalid token array in 'input'.", Some("input")))
    };
    let out: Vec<Vec<u32>> = match v {
        Some(Value::String(s)) => vec![text(s)?],
        Some(Value::Array(a)) if a.is_empty() => return Err(ApiError::invalid("[] is too short - 'input'", Some("input"))),
        Some(Value::Array(a)) if a.len() > 2048 => return Err(ApiError::invalid("Invalid 'input': at most 2048 inputs per request.", Some("input"))),
        Some(Value::Array(a)) if a.iter().all(Value::is_string) => a.iter().map(|s| text(s.as_str().unwrap())).collect::<ApiResult<_>>()?,
        Some(Value::Array(a)) if a.iter().all(Value::is_number) => vec![ids(a)?],
        Some(Value::Array(a)) if a.iter().all(Value::is_array) => a.iter().map(|x| ids(x.as_array().unwrap())).collect::<ApiResult<_>>()?,
        None | Some(Value::Null) => return Err(ApiError::invalid("Missing required parameter: 'input'.", Some("input"))),
        Some(_) => return Err(ApiError::invalid("Invalid type for 'input'.", Some("input"))),
    };
    for i in &out {
        if i.len() > state.ctx {
            return Err(ApiError::invalid_code(
                format!("This model's maximum context length is {} tokens, however you requested {} tokens in your input.", state.ctx, i.len()),
                Some("input"),
                "context_length_exceeded",
            ));
        }
    }
    Ok(out)
}

pub async fn create(State(state): State<Arc<AppState>>, body: Bytes) -> ApiResult<Json<Value>> {
    let m = params::object(&body)?;
    let model = state.resolve_model(m.get("model"))?;
    let inputs = inputs(&state, m.get("input"))?;
    let dims = params::opt_u64(&m, "dimensions", 1, u32::MAX as u64)?.map(|d| d as usize);
    let base64 = match params::opt_str(&m, "encoding_format")? {
        None | Some("float") => false,
        Some("base64") => true,
        Some(f) => return Err(ApiError::invalid(format!("Invalid value for 'encoding_format': '{f}'. Expected 'float' or 'base64'."), Some("encoding_format"))),
    };
    let prompt_tokens: usize = inputs.iter().map(Vec::len).sum();
    let (reply, rx) = oneshot::channel();
    state.engine.submit(Job::Embed { inputs, reply }).map_err(|e| ApiError::unavailable(format!("{e:#}")))?;
    let vectors = rx.await.map_err(|_| ApiError::server("the engine stopped unexpectedly"))?.map_err(ApiError::server)?;
    let mut data = Vec::with_capacity(vectors.len());
    for (i, mut v) in vectors.into_iter().enumerate() {
        if let Some(d) = dims {
            if d > v.len() {
                return Err(ApiError::invalid(format!("Invalid 'dimensions': this model's embeddings have {} dimensions.", v.len()), Some("dimensions")));
            }
            v.truncate(d);
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            v.iter_mut().for_each(|x| *x /= norm);
        }
        let embedding = if base64 {
            let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
            json!(base64::engine::general_purpose::STANDARD.encode(bytes))
        } else {
            json!(v)
        };
        data.push(json!({ "object": "embedding", "index": i, "embedding": embedding }));
    }
    Ok(Json(json!({
        "object": "list",
        "data": data,
        "model": model,
        "usage": { "prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens },
    })))
}
