//! Shared server state and the in-memory stores behind `store: true`.

use std::collections::{HashMap, VecDeque};
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
#[derive(Clone)]
pub struct StoredChat {
    pub completion: Value,
    /// The request messages, each with an `id` (for `/messages`).
    pub messages: Vec<Value>,
}

/// A stored Response (Responses API `store`, the default).
#[derive(Clone)]
pub struct StoredResponse {
    pub response: Value,
    /// Input items as returned by `/input_items`.
    pub input_items: Vec<Value>,
    /// The full conversation in template-message form, including this
    /// response's output, for `previous_response_id`.
    pub history: Arc<History>,
    /// Set while a background response is running.
    pub cancel: Option<Arc<AtomicBool>>,
}

/// A conversation as a chain of turns that share their earlier turns, so a
/// chain of responses holds each message once.
pub struct History {
    prev: Option<Arc<History>>,
    own: Vec<Value>,
}

impl History {
    pub fn new(prev: Option<Arc<History>>, own: Vec<Value>) -> Arc<Self> {
        Arc::new(Self { prev, own })
    }

    pub fn empty() -> Arc<Self> {
        Self::new(None, Vec::new())
    }

    /// Every message, oldest first.
    pub fn messages(&self) -> Vec<Value> {
        let mut chain = vec![self];
        while let Some(p) = &chain.last().unwrap().prev {
            chain.push(p);
        }
        chain.iter().rev().flat_map(|h| h.own.iter().cloned()).collect()
    }
}

/// Most objects each store keeps; the oldest go first.
pub const MAX_STORED: usize = 10_000;

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

    /// Version 2: conversation turns are saved once, in `histories`, each
    /// naming the turn before it.
    fn to_json(chats: &[Arc<StoredChat>], responses: &[Arc<StoredResponse>]) -> Value {
        let chats: Vec<Value> = chats.iter().map(|c| json!({ "completion": c.completion, "messages": c.messages })).collect();
        let mut nodes: HashMap<*const History, usize> = HashMap::new();
        let mut histories: Vec<Value> = Vec::new();
        let mut node = |h: &Arc<History>| -> usize {
            let mut chain = Vec::new();
            let mut at = Some(h);
            while let Some(h) = at.filter(|h| !nodes.contains_key(&Arc::as_ptr(h))) {
                chain.push(h);
                at = h.prev.as_ref();
            }
            for h in chain.into_iter().rev() {
                let prev = h.prev.as_ref().map(|p| nodes[&Arc::as_ptr(p)]);
                nodes.insert(Arc::as_ptr(h), histories.len());
                histories.push(json!({ "prev": prev, "messages": h.own }));
            }
            nodes[&Arc::as_ptr(h)]
        };
        let responses: Vec<Value> =
            responses.iter().map(|r| json!({ "response": r.response, "input_items": r.input_items, "history": node(&r.history) })).collect();
        json!({ "version": 2, "chat_completions": chats, "histories": histories, "responses": responses })
    }

    fn from_json(v: &Value) -> Self {
        let mut s = Store::default();
        let list = |k: &str| v[k].as_array().cloned().unwrap_or_default();
        for c in list("chat_completions") {
            let Some(id) = c["completion"]["id"].as_str().map(str::to_owned) else { continue };
            let messages = c["messages"].as_array().cloned().unwrap_or_default();
            s.chats.insert(id, StoredChat { completion: c["completion"].clone(), messages });
        }
        let mut nodes: Vec<Arc<History>> = Vec::new();
        for h in list("histories") {
            let prev = h["prev"].as_u64().and_then(|i| nodes.get(i as usize).cloned());
            nodes.push(History::new(prev, h["messages"].as_array().cloned().unwrap_or_default()));
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
            // Version 1 saved each response's whole conversation.
            let history = match &r["history"] {
                Value::Array(a) => History::new(None, a.clone()),
                v => v.as_u64().and_then(|i| nodes.get(i as usize).cloned()).unwrap_or_else(History::empty),
            };
            s.responses.insert(id, StoredResponse { response, input_items: arr("input_items"), history, cancel: None });
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

    /// The stored objects, taken cheaply under the lock; serialize them with
    /// [`Snapshot::to_bytes`] outside it.
    pub fn snapshot(&self) -> Snapshot {
        Snapshot { chats: self.chats.iter().map(|(_, c)| c.clone()).collect(), responses: self.responses.iter().map(|(_, r)| r.clone()).collect() }
    }
}

pub struct Snapshot {
    chats: Vec<Arc<StoredChat>>,
    responses: Vec<Arc<StoredResponse>>,
}

impl Snapshot {
    pub fn to_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(&Store::to_json(&self.chats, &self.responses)).unwrap_or_default()
    }
}

/// Write via a temporary file and rename, so a crash never leaves half a file.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, bytes).with_context(|| format!("write {}", tmp.display()))?;
    std::fs::rename(&tmp, path).with_context(|| format!("rename to {}", path.display()))
}

pub struct Ordered<T> {
    map: HashMap<String, Arc<T>>,
    order: VecDeque<String>,
    /// Bumped on every (possible) change.
    version: u64,
}

impl<T> Default for Ordered<T> {
    fn default() -> Self {
        Self { map: HashMap::new(), order: VecDeque::new(), version: 0 }
    }
}

impl<T: Clone> Ordered<T> {
    /// Insert or replace; past [`MAX_STORED`] the oldest objects go.
    pub fn insert(&mut self, id: String, v: T) {
        self.version += 1;
        if self.map.insert(id.clone(), Arc::new(v)).is_none() {
            self.order.push_back(id);
            while self.order.len() > MAX_STORED {
                let old = self.order.pop_front().unwrap();
                self.map.remove(&old);
            }
        }
    }

    pub fn get(&self, id: &str) -> Option<&T> {
        self.map.get(id).map(|v| &**v)
    }

    /// Copies the object first if a snapshot still holds it.
    pub fn get_mut(&mut self, id: &str) -> Option<&mut T> {
        self.version += 1;
        self.map.get_mut(id).map(Arc::make_mut)
    }

    pub fn remove(&mut self, id: &str) -> Option<Arc<T>> {
        let v = self.map.remove(id)?;
        self.version += 1;
        self.order.retain(|x| x != id);
        Some(v)
    }

    /// Items in creation order.
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = (&String, &Arc<T>)> {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn response(id: &str, history: Arc<History>) -> StoredResponse {
        StoredResponse { response: json!({ "id": id, "status": "completed" }), input_items: Vec::new(), history, cancel: None }
    }

    /// A chain of responses saves each turn once and loads back whole.
    #[test]
    fn history_chain_round_trips() {
        let mut s = Store::default();
        let mut h = History::empty();
        for i in 0..50 {
            h = History::new(Some(h), vec![json!({ "role": "user", "content": format!("q{i}") }), json!({ "role": "assistant", "content": "a" })]);
            s.responses.insert(format!("resp_{i}"), response(&format!("resp_{i}"), h.clone()));
        }
        let bytes = s.snapshot().to_bytes();
        let v: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["histories"].as_array().unwrap().len(), 51);
        let back = Store::from_json(&v);
        let last = back.responses.get("resp_49").unwrap().history.messages();
        assert_eq!(last.len(), 100);
        assert_eq!(last[98]["content"], "q49");
        assert_eq!(back.responses.get("resp_0").unwrap().history.messages().len(), 2);
        // Version 1 files: a full conversation per response.
        let v1 = json!({ "version": 1, "responses": [{ "response": { "id": "r" }, "history": [{ "role": "user", "content": "x" }] }] });
        assert_eq!(Store::from_json(&v1).responses.get("r").unwrap().history.messages().len(), 1);
    }

    #[test]
    fn stores_drop_the_oldest_past_the_cap() {
        let mut o: Ordered<u32> = Ordered::default();
        for i in 0..MAX_STORED as u32 + 5 {
            o.insert(i.to_string(), i);
        }
        assert!(o.get("4").is_none() && o.get("5").is_some());
        assert_eq!(o.iter().count(), MAX_STORED);
    }
}
