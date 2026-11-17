//! IDs, timestamps, the SSE plumbing and small helpers shared by the handlers.

use std::convert::Infallible;
use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
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

/// Most bytes of events a streaming response may have waiting for a client
/// that doesn't read them. Past this the client counts as gone and the
/// generation is cancelled, instead of the server buffering a whole
/// generation (with logprobs, gigabytes) for a stalled reader.
const MAX_PENDING_BYTES: usize = 32 << 20;

/// The sending half of a streaming (SSE) response.
pub struct SseSink {
    tx: mpsc::UnboundedSender<(Event, usize)>,
    /// Bytes sent but not yet taken by the response body.
    pending: Arc<AtomicUsize>,
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
    /// Queue `ev` (`n` bytes); false once the client is gone or has fallen
    /// [`MAX_PENDING_BYTES`] behind.
    fn push(&self, ev: Event, n: usize) -> bool {
        if self.pending.fetch_add(n, Ordering::Relaxed) + n > MAX_PENDING_BYTES {
            self.pending.fetch_sub(n, Ordering::Relaxed);
            return false;
        }
        self.tx.send((ev, n)).is_ok()
    }

    /// A `data:` event with `v`; false once the client is gone.
    pub fn data(&self, v: &Value) -> bool {
        let s = v.to_string();
        let n = s.len();
        self.push(Event::default().data(s), n)
    }

    /// A named event (`event: kind`) with `v`.
    pub fn event(&self, kind: &str, v: &Value) -> bool {
        let s = v.to_string();
        let n = s.len() + kind.len();
        self.push(Event::default().event(kind).data(s), n)
    }

    /// The `[DONE]` terminator of Chat Completions and Completions streams.
    pub fn done(&self) {
        let _ = self.tx.send((Event::default().data("[DONE]"), 0));
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
    let pending = Arc::new(AtomicUsize::new(0));
    tokio::spawn(f(SseSink { tx, pending: pending.clone() }));
    let events = UnboundedReceiverStream::new(rx).map(move |(ev, n): (Event, usize)| {
        pending.fetch_sub(n, Ordering::Relaxed);
        Ok::<_, Infallible>(ev)
    });
    Sse::new(events).keep_alive(KeepAlive::new().interval(Duration::from_secs(15))).into_response()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_stalled_client_is_cut_off() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let sink = SseSink { tx, pending: Arc::new(AtomicUsize::new(0)) };
        let big = Value::String("x".repeat(1 << 20));
        let sent = (0..64).take_while(|_| sink.data(&big)).count();
        assert!((30..=32).contains(&sent), "{sent}");
        // Once the body takes events, there is room again.
        while let Ok((_, n)) = rx.try_recv() {
            sink.pending.fetch_sub(n, Ordering::Relaxed);
        }
        assert!(sink.data(&big));
    }
}
