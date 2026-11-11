//! IDs, timestamps, the SSE plumbing and small helpers shared by the handlers.

use std::convert::Infallible;
use std::future::Future;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use serde_json::Value;
use tokio::sync::mpsc;
use tokio_stream::{Stream, StreamExt};
use tokio_stream::wrappers::UnboundedReceiverStream;

use crate::error::{ApiError, ApiResult};

/// Run CPU-heavy request work (template rendering, tokenizing up to the
/// body limit) on the blocking pool instead of an async worker.
pub async fn blocking<T: Send + 'static>(f: impl FnOnce() -> ApiResult<T> + Send + 'static) -> ApiResult<T> {
    tokio::task::spawn_blocking(f).await.map_err(|e| ApiError::server(format!("request task failed: {e}")))?
}

/// The sending half of a streaming (SSE) response.
pub struct SseSink {
    tx: mpsc::UnboundedSender<Result<Event, Infallible>>,
}

/// What [`SseSink::next`] saw first.
pub enum Recv<T> {
    Item(T),
    /// The source stream ended.
    Ended,
    /// The client went away (the response body was dropped).
    Gone,
}

impl SseSink {
    /// A `data:` event with `v`; false once the client is gone.
    pub fn data(&self, v: &Value) -> bool {
        self.tx.send(Ok(Event::default().data(v.to_string()))).is_ok()
    }

    /// A named event (`event: kind`) with `v`.
    pub fn event(&self, kind: &str, v: &Value) -> bool {
        self.tx.send(Ok(Event::default().event(kind).data(v.to_string()))).is_ok()
    }

    /// The `[DONE]` terminator of Chat Completions and Completions streams.
    pub fn done(&self) {
        let _ = self.tx.send(Ok(Event::default().data("[DONE]")));
    }

    /// An error event in OpenAI's shape, then `[DONE]`.
    pub fn error_done(&self, e: &ApiError) {
        self.data(&e.body());
        self.done();
    }

    /// The next item of `s`, unless the client disconnects first. Waiting
    /// here (rather than only noticing on the next send) is what cancels a
    /// generation still queued or prefilling when its client leaves: the
    /// caller returns on [`Recv::Gone`] and drops the generation's receiver.
    pub async fn next<S: Stream + Unpin>(&self, s: &mut S) -> Recv<S::Item> {
        tokio::select! {
            biased;
            _ = self.tx.closed() => Recv::Gone,
            v = s.next() => v.map_or(Recv::Ended, Recv::Item),
        }
    }
}

/// [`SseSink::next`] when there may be no client to watch (non-streaming
/// requests: dropping the handler's future already cancels them).
pub async fn next_from<S: Stream + Unpin>(sink: Option<&SseSink>, s: &mut S) -> Recv<S::Item> {
    match sink {
        Some(sink) => sink.next(s).await,
        None => s.next().await.map_or(Recv::Ended, Recv::Item),
    }
}

/// A streaming response produced by `f`, which runs as its own task. A
/// comment every 15 s keeps proxies and clients from timing out a stream
/// that waits in the queue or on a long prompt.
pub fn sse<F, Fut>(f: F) -> Response
where
    F: FnOnce(SseSink) -> Fut,
    Fut: Future<Output = ()> + Send + 'static,
{
    let (tx, rx) = mpsc::unbounded_channel();
    tokio::spawn(f(SseSink { tx }));
    Sse::new(UnboundedReceiverStream::new(rx)).keep_alive(KeepAlive::new().interval(Duration::from_secs(15))).into_response()
}

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// OpenAI's rule for function (tool) names: `^[a-zA-Z0-9_-]{1,64}$`. The
/// name is written into the prompt (forced tool calls), so nothing else is
/// let through.
pub fn valid_function_name(name: &str) -> bool {
    (1..=64).contains(&name.len()) && name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

/// Seconds since the Unix epoch.
pub fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

/// A fresh 64-bit random value (SplitMix64 over time and a process counter).
pub fn random_u64() -> u64 {
    let t = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_nanos() as u64);
    let mut z = t ^ COUNTER.fetch_add(0x9E37_79B9_7F4A_7C15, Ordering::Relaxed);
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// `prefix` followed by `len` random alphanumeric characters
/// (e.g. `chatcmpl-...`, `call_...`).
pub fn alnum_id(prefix: &str, len: usize) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    let mut s = String::with_capacity(prefix.len() + len);
    s.push_str(prefix);
    let mut bits = 0u64;
    for i in 0..len {
        if i % 10 == 0 {
            bits = random_u64();
        }
        s.push(ALPHABET[(bits % ALPHABET.len() as u64) as usize] as char);
        bits /= ALPHABET.len() as u64;
    }
    s
}

/// `prefix` followed by `len` random lowercase hex digits (e.g. `resp_...`).
pub fn hex_id(prefix: &str, len: usize) -> String {
    let mut s = String::with_capacity(prefix.len() + len);
    s.push_str(prefix);
    while s.len() < prefix.len() + len {
        s.push_str(&format!("{:016x}", random_u64()));
    }
    s.truncate(prefix.len() + len);
    s
}
