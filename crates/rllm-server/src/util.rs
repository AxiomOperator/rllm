//! IDs, timestamps and small helpers shared by the handlers.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static COUNTER: AtomicU64 = AtomicU64::new(0);

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
