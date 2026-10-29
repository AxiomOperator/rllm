//! Parsing and validating request parameters shared by the endpoints.

use serde_json::Value;

use crate::engine::GenParams;
use crate::error::{ApiError, ApiResult};
use crate::util;

/// Request body as a JSON object (400 if it isn't one).
pub fn object(body: &[u8]) -> ApiResult<serde_json::Map<String, Value>> {
    match serde_json::from_slice::<Value>(body) {
        Ok(Value::Object(m)) => Ok(m),
        Ok(_) => Err(ApiError::invalid("The request body must be a JSON object.", None)),
        Err(e) => Err(ApiError::invalid(
            format!("We could not parse the JSON body of your request. (HINT: This likely means you aren't using your HTTP library correctly. The OpenAI API expects a JSON payload, but what was sent was not valid JSON.) Details: {e}"),
            None,
        )),
    }
}

fn type_error(name: &str, want: &str) -> ApiError {
    ApiError::invalid(format!("Invalid type for '{name}': expected {want}."), Some(name))
}

pub fn opt_f32(m: &serde_json::Map<String, Value>, name: &str, lo: f32, hi: f32) -> ApiResult<Option<f32>> {
    match m.get(name) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(n)) => {
            let v = n.as_f64().unwrap_or(f64::NAN) as f32;
            if !(lo..=hi).contains(&v) {
                return Err(ApiError::invalid(format!("Invalid '{name}': {v} is out of range [{lo}, {hi}]."), Some(name)));
            }
            Ok(Some(v))
        }
        Some(_) => Err(type_error(name, "a number")),
    }
}

pub fn opt_u64(m: &serde_json::Map<String, Value>, name: &str, lo: u64, hi: u64) -> ApiResult<Option<u64>> {
    match m.get(name) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(n)) => {
            let v = n.as_u64().ok_or_else(|| type_error(name, "an integer"))?;
            if !(lo..=hi).contains(&v) {
                return Err(ApiError::invalid(format!("Invalid '{name}': {v} is out of range [{lo}, {hi}]."), Some(name)));
            }
            Ok(Some(v))
        }
        Some(_) => Err(type_error(name, "an integer")),
    }
}

pub fn opt_bool(m: &serde_json::Map<String, Value>, name: &str) -> ApiResult<Option<bool>> {
    match m.get(name) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Bool(b)) => Ok(Some(*b)),
        Some(_) => Err(type_error(name, "a boolean")),
    }
}

pub fn opt_str<'a>(m: &'a serde_json::Map<String, Value>, name: &str) -> ApiResult<Option<&'a str>> {
    match m.get(name) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(s)),
        Some(_) => Err(type_error(name, "a string")),
    }
}

/// `stop`: a string or up to 4 strings.
pub fn stops(m: &serde_json::Map<String, Value>) -> ApiResult<Vec<String>> {
    match m.get("stop") {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::String(s)) => Ok(vec![s.clone()]),
        Some(Value::Array(a)) => {
            if a.len() > 4 {
                return Err(ApiError::invalid(format!("Invalid 'stop': array too long. Expected an array with maximum length 4, but got an array with length {}.", a.len()), Some("stop")));
            }
            a.iter().map(|v| v.as_str().map(str::to_owned).ok_or_else(|| type_error("stop", "a string or array of strings"))).collect()
        }
        Some(_) => Err(type_error("stop", "a string or array of strings")),
    }
}

/// `metadata`: up to 16 string pairs (keys ≤ 64 chars, values ≤ 512).
pub fn metadata(m: &serde_json::Map<String, Value>) -> ApiResult<Value> {
    match m.get("metadata") {
        None | Some(Value::Null) => Ok(Value::Object(Default::default())),
        Some(Value::Object(o)) => {
            if o.len() > 16 {
                return Err(ApiError::invalid("Invalid 'metadata': too many properties (maximum 16).", Some("metadata")));
            }
            for (k, v) in o {
                let Some(s) = v.as_str() else { return Err(type_error(&format!("metadata.{k}"), "a string")) };
                if k.len() > 64 || s.len() > 512 {
                    return Err(ApiError::invalid("Invalid 'metadata': keys are limited to 64 and values to 512 characters.", Some("metadata")));
                }
            }
            Ok(Value::Object(o.clone()))
        }
        Some(_) => Err(type_error("metadata", "an object")),
    }
}

/// Sampling parameters common to the generation endpoints.
pub fn sampling(m: &serde_json::Map<String, Value>, vocab: usize) -> ApiResult<GenParams> {
    let mut p = GenParams {
        temperature: opt_f32(m, "temperature", 0.0, 2.0)?.unwrap_or(1.0),
        top_p: opt_f32(m, "top_p", 0.0, 1.0)?.unwrap_or(1.0),
        // Not in the OpenAI API, but common in compatible servers.
        top_k: opt_u64(m, "top_k", 0, 1 << 20)?.unwrap_or(0) as usize,
        frequency_penalty: opt_f32(m, "frequency_penalty", -2.0, 2.0)?.unwrap_or(0.0),
        presence_penalty: opt_f32(m, "presence_penalty", -2.0, 2.0)?.unwrap_or(0.0),
        seed: match m.get("seed") {
            None | Some(Value::Null) => util::random_u64(),
            Some(Value::Number(n)) => n.as_i64().map(|v| v as u64).or(n.as_u64()).ok_or_else(|| type_error("seed", "an integer"))?,
            Some(_) => return Err(type_error("seed", "an integer")),
        },
        ..Default::default()
    };
    if p.top_p == 0.0 {
        p.temperature = 0.0;
    }
    if let Some(bias) = m.get("logit_bias").filter(|v| !v.is_null()) {
        let o = bias.as_object().ok_or_else(|| type_error("logit_bias", "an object"))?;
        for (k, v) in o {
            let id: u32 = k.parse().map_err(|_| ApiError::invalid(format!("Invalid key in 'logit_bias': {k}"), Some("logit_bias")))?;
            if id as usize >= vocab {
                return Err(ApiError::invalid(format!("Invalid key in 'logit_bias': token {id} is out of vocabulary."), Some("logit_bias")));
            }
            let b = v.as_f64().ok_or_else(|| type_error("logit_bias", "numbers"))? as f32;
            if !(-100.0..=100.0).contains(&b) {
                return Err(ApiError::invalid(format!("Invalid 'logit_bias': {b} is out of range [-100, 100]."), Some("logit_bias")));
            }
            p.logit_bias.push((id, b));
        }
    }
    Ok(p)
}
