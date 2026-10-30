//! Counters for `GET /metrics` (Prometheus text format), updated by the
//! engine thread and the HTTP middleware.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

/// (method, route, status) -> (count, total seconds).
type HttpStats = BTreeMap<(String, String, u16), (u64, f64)>;

#[derive(Default)]
pub struct Metrics {
    // Engine (gauges).
    pub running: AtomicU64,
    pub waiting: AtomicU64,
    pub kv_blocks_total: AtomicU64,
    pub kv_blocks_free: AtomicU64,
    pub prefix_cache_entries: AtomicU64,
    pub prefix_cache_blocks: AtomicU64,
    // Engine (counters).
    pub requests_finished: AtomicU64,
    pub prompt_tokens: AtomicU64,
    pub cached_prompt_tokens: AtomicU64,
    pub generated_tokens: AtomicU64,
    pub embedded_tokens: AtomicU64,
    pub preemptions: AtomicU64,
    /// Speculative decoding: tokens the draft proposed, and how many the
    /// model accepted.
    pub draft_tokens: AtomicU64,
    pub draft_accepted: AtomicU64,
    /// Prompt lookup: tokens proposed and accepted.
    pub lookup_tokens: AtomicU64,
    pub lookup_accepted: AtomicU64,
    /// Prefix cache host tier.
    pub host_blocks_total: AtomicU64,
    pub host_blocks_free: AtomicU64,
    pub host_cache_entries: AtomicU64,
    pub host_offloaded_blocks: AtomicU64,
    pub host_restored_blocks: AtomicU64,
    pub steps: AtomicU64,
    pub step_tokens: AtomicU64,
    /// Time spent in forward passes and sampling, microseconds.
    pub busy_us: AtomicU64,
    http: Mutex<HttpStats>,
}

impl Metrics {
    pub fn add(c: &AtomicU64, n: usize) {
        c.fetch_add(n as u64, Ordering::Relaxed);
    }

    pub fn set(g: &AtomicU64, n: usize) {
        g.store(n as u64, Ordering::Relaxed);
    }

    pub fn record_http(&self, method: &str, route: &str, status: u16, seconds: f64) {
        let mut m = self.http.lock().unwrap();
        let e = m.entry((method.to_owned(), route.to_owned(), status)).or_default();
        e.0 += 1;
        e.1 += seconds;
    }

    pub fn render(&self, model: &str) -> String {
        let get = |a: &AtomicU64| a.load(Ordering::Relaxed);
        let mut s = String::new();
        let mut one = |name: &str, kind: &str, help: &str, v: f64| {
            let _ = writeln!(s, "# HELP rllm_{name} {help}\n# TYPE rllm_{name} {kind}\nrllm_{name}{{model=\"{model}\"}} {v}");
        };
        one("requests_running", "gauge", "Generations currently in the batch.", get(&self.running) as f64);
        one("requests_waiting", "gauge", "Generations queued for a batch slot or KV blocks.", get(&self.waiting) as f64);
        one("kv_blocks_total", "gauge", "KV cache blocks in the pool.", get(&self.kv_blocks_total) as f64);
        one("kv_blocks_free", "gauge", "Free KV cache blocks.", get(&self.kv_blocks_free) as f64);
        one("prefix_cache_entries", "gauge", "Cached prompt prefixes.", get(&self.prefix_cache_entries) as f64);
        one("prefix_cache_blocks", "gauge", "KV blocks referenced by the prefix cache.", get(&self.prefix_cache_blocks) as f64);
        one("requests_finished_total", "counter", "Finished generations.", get(&self.requests_finished) as f64);
        one("prompt_tokens_total", "counter", "Prompt tokens of finished generations.", get(&self.prompt_tokens) as f64);
        one("prompt_tokens_cached_total", "counter", "Prompt tokens served from the prefix cache.", get(&self.cached_prompt_tokens) as f64);
        one("generation_tokens_total", "counter", "Generated tokens.", get(&self.generated_tokens) as f64);
        one("embedding_tokens_total", "counter", "Tokens embedded.", get(&self.embedded_tokens) as f64);
        one("preemptions_total", "counter", "Generations paused to free KV blocks.", get(&self.preemptions) as f64);
        one("draft_tokens_total", "counter", "Tokens proposed by the draft model.", get(&self.draft_tokens) as f64);
        one("draft_tokens_accepted_total", "counter", "Draft tokens the model accepted.", get(&self.draft_accepted) as f64);
        one("lookup_tokens_total", "counter", "Tokens proposed by prompt lookup.", get(&self.lookup_tokens) as f64);
        one("lookup_tokens_accepted_total", "counter", "Prompt lookup tokens the model accepted.", get(&self.lookup_accepted) as f64);
        one("kv_host_blocks_total", "gauge", "Host tier blocks (pinned system memory).", get(&self.host_blocks_total) as f64);
        one("kv_host_blocks_free", "gauge", "Free host tier blocks.", get(&self.host_blocks_free) as f64);
        one("prefix_cache_host_entries", "gauge", "Cached prompt prefixes in the host tier.", get(&self.host_cache_entries) as f64);
        one("kv_offloaded_blocks_total", "counter", "KV blocks copied to the host tier.", get(&self.host_offloaded_blocks) as f64);
        one("kv_restored_blocks_total", "counter", "KV blocks copied back from the host tier.", get(&self.host_restored_blocks) as f64);
        one("engine_steps_total", "counter", "Forward passes.", get(&self.steps) as f64);
        one("engine_step_tokens_total", "counter", "Tokens processed by forward passes.", get(&self.step_tokens) as f64);
        one("engine_busy_seconds_total", "counter", "Time spent in forward passes and sampling.", get(&self.busy_us) as f64 / 1e6);
        let http = self.http.lock().unwrap();
        let _ = writeln!(s, "# HELP rllm_http_requests_total HTTP requests by route and status.\n# TYPE rllm_http_requests_total counter");
        for ((m, r, st), (n, _)) in http.iter() {
            let _ = writeln!(s, "rllm_http_requests_total{{method=\"{m}\",route=\"{r}\",status=\"{st}\"}} {n}");
        }
        let _ = writeln!(
            s,
            "# HELP rllm_http_request_seconds_total Time to the response head (streams: to the first byte).\n# TYPE rllm_http_request_seconds_total counter"
        );
        for ((m, r, st), (_, t)) in http.iter() {
            let _ = writeln!(s, "rllm_http_request_seconds_total{{method=\"{m}\",route=\"{r}\",status=\"{st}\"}} {t:.6}");
        }
        s
    }
}
