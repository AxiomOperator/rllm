//! Shared server state and the in-memory stores behind `store: true`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};

use rllm_tokenizer::Tokenizer;
use serde_json::{Value, json};

use crate::engine::{EngineHandle, EngineInfo};
use crate::metrics::Metrics;
use crate::error::{ApiError, ApiResult};
use crate::output::Format;

pub struct AppState {
    pub engine: EngineHandle,
    pub info: EngineInfo,
    pub tok: Arc<Tokenizer>,
    pub format: Format,
    /// The id clients use for the served model (`/v1/models`).
    pub model_id: String,
    pub created: u64,
    pub ctx: usize,
    pub fingerprint: String,
    pub api_key: Option<String>,
    /// Reject requests naming another model (404) instead of serving them.
    pub strict_model: bool,
    pub store: Mutex<Store>,
    /// Where the store is saved (`--store-file`), if anywhere.
    pub store_file: Option<PathBuf>,
    pub metrics: Arc<Metrics>,
    /// Log one line per finished generation.
    pub log: bool,
}

impl AppState {
    /// Check a request's `model` and return the id to report.
    pub fn resolve_model(&self, requested: Option<&Value>) -> ApiResult<String> {
        let Some(name) = requested.and_then(Value::as_str) else {
            return Err(ApiError::invalid("you must provide a model parameter", Some("model")));
        };
        if self.strict_model && name != self.model_id {
            return Err(ApiError {
                code: Some("model_not_found".into()),
                ..ApiError::not_found(format!("The model `{name}` does not exist or you do not have access to it."))
            });
        }
        Ok(self.model_id.clone())
    }
}

/// A stored Chat Completion (`store: true`).
pub struct StoredChat {
    pub completion: Value,
    /// The request messages, each with an `id` (for `/messages`).
    pub messages: Vec<Value>,
}

/// A stored Response (Responses API `store`, the default).
pub struct StoredResponse {
    pub response: Value,
    /// Input items as returned by `/input_items`.
    pub input_items: Vec<Value>,
    /// The full conversation in template-message form, including this
    /// response's output, for `previous_response_id`.
    pub history: Vec<Value>,
    /// Set while a background response is running.
    pub cancel: Option<Arc<AtomicBool>>,
}

/// Insertion-ordered maps of stored objects.
#[derive(Default)]
pub struct Store {
    pub chats: Ordered<StoredChat>,
    pub responses: Ordered<StoredResponse>,
}

impl Store {
    /// Changes so far; the saver writes the file when this moves.
    pub fn version(&self) -> u64 {
        self.chats.version + self.responses.version
    }

    fn to_json(&self) -> Value {
        let chats: Vec<Value> =
            self.chats.iter().map(|(_, c)| json!({ "completion": c.completion, "messages": c.messages })).collect();
        let responses: Vec<Value> = self
            .responses
            .iter()
            .map(|(_, r)| json!({ "response": r.response, "input_items": r.input_items, "history": r.history }))
            .collect();
        json!({ "version": 1, "chat_completions": chats, "responses": responses })
    }

    fn from_json(v: &Value) -> Self {
        let mut s = Store::default();
        let list = |k: &str| v[k].as_array().cloned().unwrap_or_default();
        for c in list("chat_completions") {
            let Some(id) = c["completion"]["id"].as_str().map(str::to_owned) else { continue };
            let messages = c["messages"].as_array().cloned().unwrap_or_default();
            s.chats.insert(id, StoredChat { completion: c["completion"].clone(), messages });
        }
        for r in list("responses") {
            let Some(id) = r["response"]["id"].as_str().map(str::to_owned) else { continue };
            let mut response = r["response"].clone();
            // A background response that was running when the server stopped.
            if matches!(response["status"].as_str(), Some("queued" | "in_progress")) {
                response["status"] = json!("failed");
                response["error"] = json!({ "code": "server_error", "message": "The server restarted before this response finished." });
            }
            let arr = |k: &str| r[k].as_array().cloned().unwrap_or_default();
            s.responses.insert(id, StoredResponse { response, input_items: arr("input_items"), history: arr("history"), cancel: None });
        }
        s.chats.version = 0;
        s.responses.version = 0;
        s
    }

    /// Load a saved store; an absent file is an empty store.
    pub fn load(path: &Path) -> Result<Self> {
        match std::fs::read(path) {
            Ok(bytes) => {
                let v: Value = serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))?;
                Ok(Self::from_json(&v))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(e).with_context(|| format!("read {}", path.display())),
        }
    }

    /// Serialize under the lock; write with [`write_atomic`] outside it.
    pub fn snapshot(&self) -> Vec<u8> {
        serde_json::to_vec(&self.to_json()).unwrap_or_default()
    }
}

/// Write via a temporary file and rename, so a crash never leaves half a file.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, bytes).with_context(|| format!("write {}", tmp.display()))?;
    std::fs::rename(&tmp, path).with_context(|| format!("rename to {}", path.display()))
}

pub struct Ordered<T> {
    map: HashMap<String, T>,
    order: Vec<String>,
    /// Bumped on every (possible) change.
    version: u64,
}

impl<T> Default for Ordered<T> {
    fn default() -> Self {
        Self { map: HashMap::new(), order: Vec::new(), version: 0 }
    }
}

impl<T> Ordered<T> {
    pub fn insert(&mut self, id: String, v: T) {
        self.version += 1;
        if self.map.insert(id.clone(), v).is_none() {
            self.order.push(id);
        }
    }

    pub fn get(&self, id: &str) -> Option<&T> {
        self.map.get(id)
    }

    pub fn get_mut(&mut self, id: &str) -> Option<&mut T> {
        self.version += 1;
        self.map.get_mut(id)
    }

    pub fn remove(&mut self, id: &str) -> Option<T> {
        let v = self.map.remove(id)?;
        self.version += 1;
        self.order.retain(|x| x != id);
        Some(v)
    }

    /// Items in creation order.
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = (&String, &T)> {
        self.order.iter().filter_map(|id| self.map.get(id).map(|v| (id, v)))
    }
}

/// Cursor pagination shared by the list endpoints: `after`, `limit`,
/// `order`. Returns the page and whether more items follow.
pub fn paginate<T: Clone>(items: Vec<(String, T)>, after: Option<&str>, limit: usize, desc: bool) -> (Vec<(String, T)>, bool) {
    let mut items = items;
    if desc {
        items.reverse();
    }
    let start = match after {
        Some(a) => items.iter().position(|(id, _)| id == a).map_or(items.len(), |p| p + 1),
        None => 0,
    };
    let rest: Vec<_> = items.into_iter().skip(start).collect();
    let has_more = rest.len() > limit;
    (rest.into_iter().take(limit).collect(), has_more)
}

pub fn list_object(data: Vec<Value>, has_more: bool) -> Value {
    let first = data.first().and_then(|v| v.get("id")).cloned().unwrap_or(Value::Null);
    let last = data.last().and_then(|v| v.get("id")).cloned().unwrap_or(Value::Null);
    serde_json::json!({ "object": "list", "data": data, "first_id": first, "last_id": last, "has_more": has_more })
}
