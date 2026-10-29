//! OpenAI-style errors: `{"error": {"message", "type", "param", "code"}}`.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::json;

#[derive(Debug, Clone)]
pub struct ApiError {
    pub status: StatusCode,
    pub message: String,
    pub kind: &'static str,
    pub param: Option<String>,
    pub code: Option<String>,
}

pub type ApiResult<T> = Result<T, ApiError>;

impl ApiError {
    /// 400 `invalid_request_error`, optionally naming the offending parameter.
    pub fn invalid(message: impl Into<String>, param: Option<&str>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
            kind: "invalid_request_error",
            param: param.map(str::to_owned),
            code: None,
        }
    }

    pub fn invalid_code(message: impl Into<String>, param: Option<&str>, code: &str) -> Self {
        Self { code: Some(code.to_owned()), ..Self::invalid(message, param) }
    }

    pub fn not_found(message: impl Into<String>) -> Self {
        Self { status: StatusCode::NOT_FOUND, ..Self::invalid(message, None) }
    }

    pub fn unauthorized(message: impl Into<String>, code: Option<&str>) -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            message: message.into(),
            kind: "invalid_request_error",
            param: None,
            code: code.map(str::to_owned),
        }
    }

    pub fn server(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.into(),
            kind: "server_error",
            param: None,
            code: None,
        }
    }

    pub fn unavailable(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: message.into(),
            kind: "service_unavailable_error",
            param: None,
            code: None,
        }
    }

    pub fn body(&self) -> serde_json::Value {
        json!({ "error": { "message": self.message, "type": self.kind, "param": self.param, "code": self.code } })
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status, axum::Json(self.body())).into_response()
    }
}

impl From<anyhow::Error> for ApiError {
    fn from(e: anyhow::Error) -> Self {
        Self::server(format!("{e:#}"))
    }
}
