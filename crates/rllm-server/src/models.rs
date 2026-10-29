//! `/v1/models`: the one served model.

use std::sync::Arc;

use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use serde_json::{Value, json};

use crate::error::{ApiError, ApiResult};
use crate::state::AppState;

fn model_object(state: &AppState) -> Value {
    json!({ "id": state.model_id, "object": "model", "created": state.created, "owned_by": "rllm" })
}

pub async fn list(State(state): State<Arc<AppState>>) -> Json<Value> {
    Json(json!({ "object": "list", "data": [model_object(&state)] }))
}

fn not_found(id: &str) -> ApiError {
    ApiError { code: Some("model_not_found".into()), param: Some("model".into()), ..ApiError::not_found(format!("The model '{id}' does not exist")) }
}

pub async fn retrieve(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> ApiResult<Json<Value>> {
    if id != state.model_id {
        return Err(not_found(&id));
    }
    Ok(Json(model_object(&state)))
}

/// Only fine-tuned models can be deleted; the served model can't.
pub async fn delete(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> ApiResult<Json<Value>> {
    if id != state.model_id {
        return Err(not_found(&id));
    }
    Err(ApiError { status: StatusCode::FORBIDDEN, ..ApiError::invalid(format!("You do not have permission to delete the model '{id}'."), Some("model")) })
}
