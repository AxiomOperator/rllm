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
    /// Held while the store file is written, so the periodic and the
    /// shutdown save never write it at the same time.
    pub save_lock: Mutex<()>,
    /// The decision head, if the model is a decision model (`/v1/systemone`).
    pub decision: Option<Arc<crate::decision::Decider>>,
}

impl AppState {
    /// A decision model (Clef) has no text output: generation endpoints are
    /// refused (as llama.cpp's server does), only `/v1/systemone` answers.
    pub fn generation_allowed(&self) -> ApiResult<()> {
        if self.decision.is_some() {
            return Err(ApiError::invalid("This model is a decision model without text generation, use /v1/systemone", None));
        }
        Ok(())
    }

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

    /// 503 if `jobs` more generations would not be accepted now: the engine
    /// stopped, or the queue (`--max-queue`) is full. Streaming endpoints
    /// check this before sending their 200 head, so a full queue answers 503
    /// there too. Submission still re-checks exactly (a request that loses a
    /// race here gets an `error` event instead).
    /// Hold queue places for `jobs` generations before responding: a full
    /// queue (or a stopped engine) is a 503 now, before a stream's 200 head.
    pub fn reserve(&self, jobs: usize) -> ApiResult<Arc<crate::engine::Slots>> {
        self.engine.reserve(jobs).map_err(|e| ApiError::unavailable(format!("{e:#}")))
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
    /// Nodes in the chain, this one included.
    depth: usize,
    /// Serialized size of every message in the chain.
    chain_bytes: usize,
}

/// Most nodes in a conversation chain. Each turn adds two (its input and
/// its output), so `previous_response_id` chains stop at 1000 turns.
pub const MAX_HISTORY_DEPTH: usize = 2000;

impl History {
    pub fn new(prev: Option<Arc<History>>, own: Vec<Value>) -> Arc<Self> {
        let (depth, bytes) = prev.as_ref().map_or((0, 0), |p| (p.depth, p.chain_bytes));
        let chain_bytes = bytes + own.iter().map(json_size).sum::<usize>();
        Arc::new(Self { prev, own, depth: depth + 1, chain_bytes })
    }

    pub fn empty() -> Arc<Self> {
        Self::new(None, Vec::new())
    }

    /// Nodes in the chain, this one included.
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Serialized size of the whole conversation, shared turns included.
    pub fn chain_bytes(&self) -> usize {
        self.chain_bytes
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

/// Unlink the chain one node at a time: the default drop recurses once per
/// node (`Arc` → `History` → `Arc` …) and overflows the stack on a long
/// conversation.
impl Drop for History {
    fn drop(&mut self) {
        let mut prev = self.prev.take();
        while let Some(p) = prev {
            // Still shared (another response's chain): its last owner drops it.
            let Some(mut h) = Arc::into_inner(p) else { break };
            prev = h.prev.take();
        }
    }
}

/// Serialized JSON size of `v`, without building the bytes.
pub fn json_size(v: &Value) -> usize {
    struct Count(usize);
    impl std::io::Write for Count {
        fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
            self.0 += b.len();
            Ok(b.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut c = Count(0);
    let _ = serde_json::to_writer(&mut c, v);
    c.0
}

/// What an object costs its store, in (serialized JSON) bytes.
pub trait Weigh {
    fn weight(&self) -> usize;
}

impl Weigh for StoredChat {
    fn weight(&self) -> usize {
        json_size(&self.completion) + self.messages.iter().map(json_size).sum::<usize>()
    }
}

/// The whole conversation counts, shared turns included: a turn whose own
/// response was evicted stays in memory while a newer response's chain holds
/// it, so counting the chain per response keeps real memory under the budget.
impl Weigh for StoredResponse {
    fn weight(&self) -> usize {
        json_size(&self.response) + self.input_items.iter().map(json_size).sum::<usize>() + self.history.chain_bytes()
    }
}

/// Most objects each store keeps; the oldest go first.
pub const MAX_STORED: usize = 10_000;

/// Most bytes (by [`Weigh`]) each store keeps; the oldest objects go first.
pub const MAX_STORED_BYTES: usize = 256 << 20;

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

/// Write via a temporary file and rename, so a crash never leaves half a
/// file. The file is private to the user (0600) and its data reaches the
/// disk before the rename. Callers serialize saves ([`AppState::save_lock`]).
pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    use std::io::Write as _;
    let mut name = path.file_name().map(|n| n.to_os_string()).unwrap_or_default();
    name.push(format!(".{}.tmp", std::process::id()));
    let tmp = path.with_file_name(name);
    let write = || -> std::io::Result<()> {
        let mut o = std::fs::OpenOptions::new();
        o.write(true).create(true).truncate(true);
        #[cfg(unix)]
        std::os::unix::fs::OpenOptionsExt::mode(&mut o, 0o600);
        let mut f = o.open(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()
    };
    if let Err(e) = write() {
        let _ = std::fs::remove_file(&tmp);
        return Err(e).with_context(|| format!("write {}", tmp.display()));
    }
    std::fs::rename(&tmp, path).with_context(|| format!("rename to {}", path.display()))?;
    // Make the rename itself durable.
    #[cfg(unix)]
    if let Some(dir) = path.parent().map(|d| if d.as_os_str().is_empty() { Path::new(".") } else { d })
        && let Ok(d) = std::fs::File::open(dir)
    {
        let _ = d.sync_all();
    }
    Ok(())
}

pub struct Ordered<T> {
    /// Each object with its weight.
    map: HashMap<String, (Arc<T>, usize)>,
    order: VecDeque<String>,
    /// Sum of the weights.
    bytes: usize,
    /// Bumped on every (possible) change.
    version: u64,
}

impl<T> Default for Ordered<T> {
    fn default() -> Self {
        Self { map: HashMap::new(), order: VecDeque::new(), bytes: 0, version: 0 }
    }
}

impl<T: Clone + Weigh> Ordered<T> {
    /// Insert or replace; past [`MAX_STORED`] objects or [`MAX_STORED_BYTES`]
    /// the oldest objects go (never the one just inserted).
    pub fn insert(&mut self, id: String, v: T) {
        self.version += 1;
        let w = v.weight();
        self.bytes += w;
        match self.map.insert(id.clone(), (Arc::new(v), w)) {
            Some((_, old)) => self.bytes -= old,
            None => self.order.push_back(id.clone()),
        }
        self.evict(&id);
    }

    /// Drop the oldest objects other than `keep` until within the limits.
    fn evict(&mut self, keep: &str) {
        while self.order.len() > MAX_STORED || (self.bytes > MAX_STORED_BYTES && self.order.len() > 1) {
            let Some(pos) = self.order.iter().position(|x| x != keep) else { break };
            let old = self.order.remove(pos).unwrap();
            if let Some((_, w)) = self.map.remove(&old) {
                self.bytes -= w;
            }
        }
    }

    pub fn get(&self, id: &str) -> Option<&T> {
        self.map.get(id).map(|(v, _)| &**v)
    }

    /// Change an object in place (copying it first if a snapshot still
    /// holds it) and re-weigh it.
    pub fn update<R>(&mut self, id: &str, f: impl FnOnce(&mut T) -> R) -> Option<R> {
        let (v, w) = self.map.get_mut(id)?;
        self.version += 1;
        let r = f(Arc::make_mut(v));
        let new = v.weight();
        self.bytes = self.bytes - *w + new;
        *w = new;
        self.evict(id);
        Some(r)
    }

    pub fn remove(&mut self, id: &str) -> Option<Arc<T>> {
        let (v, w) = self.map.remove(id)?;
        self.version += 1;
        self.bytes -= w;
        self.order.retain(|x| x != id);
        Some(v)
    }

    /// Items in creation order.
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = (&String, &Arc<T>)> {
        self.order.iter().filter_map(|id| self.map.get(id).map(|(v, _)| (id, v)))
    }
}

/// Cursor pagination shared by the list endpoints: `after`, `limit`,
/// `order`. Returns the page and whether more items follow. The items can
/// be references, so only the page needs cloning.
pub fn paginate<K: AsRef<str>, T>(mut items: Vec<(K, T)>, after: Option<&str>, limit: usize, desc: bool) -> (Vec<(K, T)>, bool) {
    if desc {
        items.reverse();
    }
    let start = match after {
        Some(a) => items.iter().position(|(id, _)| id.as_ref() == a).map_or(items.len(), |p| p + 1),
        None => 0,
    };
    let has_more = items.len().saturating_sub(start) > limit;
    (items.into_iter().skip(start).take(limit).collect(), has_more)
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

    impl Weigh for u32 {
        fn weight(&self) -> usize {
            *self as usize
        }
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
        assert_eq!(back.responses.get("resp_49").unwrap().history.depth(), 51);
        // Version 1 files: a full conversation per response.
        let v1 = json!({ "version": 1, "responses": [{ "response": { "id": "r" }, "history": [{ "role": "user", "content": "x" }] }] });
        assert_eq!(Store::from_json(&v1).responses.get("r").unwrap().history.messages().len(), 1);
    }

    /// Dropping a very long chain must not recurse once per node.
    #[test]
    fn deep_history_drops_without_overflow() {
        let mut h = History::empty();
        for _ in 0..1_000_000 {
            h = History::new(Some(h), Vec::new());
        }
        // A shared middle node survives the drop of the newer part.
        let mut keep = h.clone();
        for _ in 0..10 {
            keep = History::new(Some(keep), Vec::new());
        }
        drop(h);
        assert_eq!(keep.depth(), 1_000_011);
        drop(keep);
    }

    #[test]
    fn stores_drop_the_oldest_past_the_cap() {
        let mut o: Ordered<u32> = Ordered::default();
        for i in 0..MAX_STORED as u32 + 5 {
            o.insert(i.to_string(), 1);
        }
        assert!(o.get("4").is_none() && o.get("5").is_some());
        assert_eq!(o.iter().count(), MAX_STORED);
        assert_eq!(o.bytes, MAX_STORED);
    }

    #[test]
    fn stores_drop_the_oldest_past_the_byte_budget() {
        let mut o: Ordered<u32> = Ordered::default();
        let w = (MAX_STORED_BYTES / 4) as u32;
        for i in 0..6 {
            o.insert(i.to_string(), w);
        }
        assert_eq!(o.iter().map(|(id, _)| id.as_str()).collect::<Vec<_>>(), ["2", "3", "4", "5"]);
        // Growing an object evicts older ones, never itself.
        o.update("5", |v| *v = MAX_STORED_BYTES as u32 * 2);
        assert_eq!(o.iter().map(|(id, _)| id.as_str()).collect::<Vec<_>>(), ["5"]);
        assert_eq!(o.bytes, MAX_STORED_BYTES * 2);
        o.remove("5");
        assert_eq!(o.bytes, 0);
    }

    #[test]
    fn paginate_after_and_order() {
        let items: Vec<(String, u32)> = (0..5).map(|i| (i.to_string(), i)).collect();
        let (page, more) = paginate(items.clone(), Some("1"), 2, false);
        assert_eq!(page.iter().map(|p| p.1).collect::<Vec<_>>(), [2, 3]);
        assert!(more);
        let (page, more) = paginate(items, Some("1"), 2, true);
        assert_eq!(page.iter().map(|p| p.1).collect::<Vec<_>>(), [0]);
        assert!(!more);
    }
}
