//! Device-side token sampling: one token per logits row without copying the
//! row to the host. The reference implementation here is what the CPU
//! backend runs and what GPU kernels are tested against.
//!
//! Exact without sorting the vocabulary:
//! - temperature 0: argmax (lowest index on ties);
//! - otherwise the Gumbel-max trick, `argmax(x / T + g)` with `g` Gumbel
//!   noise, draws exactly from softmax(x / T);
//! - min_p first keeps the tokens whose probability is at least `min_p`
//!   times the most likely one's: `x >= max + ln(min_p)` on the scaled
//!   logits;
//! - top_k and top_p then restrict the candidates to `x >= threshold`, the
//!   threshold found by bisection on the value: the k-th largest value for
//!   top_k; for top_p the largest threshold whose candidates hold at least
//!   `top_p` of the probability (of the min_p and top_k set when given).
//!
//! The noise comes from a counter-based hash of (seed, counter, token), so a
//! request with a seed is reproducible whatever else is in the batch.

/// How to sample one row.
#[derive(Debug, Clone, PartialEq)]
pub struct RowSample {
    pub temperature: f32,
    /// 0: no limit.
    pub top_k: u32,
    /// 1.0: no limit.
    pub top_p: f32,
    /// 0.0: no limit.
    pub min_p: f32,
    pub seed: u64,
    /// Distinguishes the draws of one request (e.g. its position).
    pub counter: u64,
    /// Added to the logits first: logit bias, repetition penalties.
    pub adjust: Vec<(u32, f32)>,
}

/// Bisection steps for the top_k / top_p thresholds.
pub const BISECT_STEPS: u32 = 40;

/// Smallest temperature used as given; lower ones (but > 0) are raised to
/// it, so `x / T` stays finite (1 / T would overflow below ~3e-39, and
/// `0 * inf` is NaN). At this temperature sampling is greedy in practice.
pub const MIN_TEMPERATURE: f32 = 1e-4;

/// Uniform strictly inside (0, 1) from (seed, counter, token): splitmix64
/// finalizer over a mix of the three (the GPU kernels compute the same
/// bits). 23 bits: `(n + 0.5) * 2^-23` is exact for n < 2^23, so the largest
/// value is 1 - 2^-24 (with 24 bits the top one rounds to exactly 1.0, whose
/// Gumbel noise is +inf).
pub fn uniform(seed: u64, counter: u64, token: u32) -> f32 {
    let mut z = seed ^ counter.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (token as u64).wrapping_mul(0xD1B5_4A32_D192_ED03);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    ((z >> 41) as f32 + 0.5) * (1.0 / (1u64 << 23) as f32)
}

/// Gumbel noise for (seed, counter, token).
pub fn gumbel(seed: u64, counter: u64, token: u32) -> f32 {
    -(-uniform(seed, counter, token).ln()).ln()
}

/// Sample one token from a logits row (the adjustments are applied to a copy).
pub fn sample_row(logits: &[f32], s: &RowSample) -> u32 {
    let mut x = logits.to_vec();
    for &(id, d) in &s.adjust {
        if let Some(v) = x.get_mut(id as usize) {
            *v += d;
        }
    }
    if s.temperature <= 0.0 {
        // NaNs never win; with no candidate at all, token 0.
        let mut best = (f32::NEG_INFINITY, 0u32);
        for (i, &v) in x.iter().enumerate() {
            if v > best.0 {
                best = (v, i as u32);
            }
        }
        return best.1;
    }
    let inv_t = 1.0 / s.temperature.max(MIN_TEMPERATURE);
    x.iter_mut().for_each(|v| *v *= inv_t);
    let thr = threshold(&x, s.top_k, s.top_p, s.min_p);
    let mut best = (f32::NEG_INFINITY, 0u32);
    for (i, &v) in x.iter().enumerate() {
        if v >= thr {
            let key = v + gumbel(s.seed, s.counter, i as u32);
            if key > best.0 {
                best = (key, i as u32);
            }
        }
    }
    best.1
}

/// The candidate threshold for top_k / top_p over temperature-scaled `x`.
///
/// The bisection runs between the smallest and largest *finite* values:
/// -inf (masked tokens) would make every midpoint -inf and silently turn
/// the limits off. -inf and NaN are never candidates at a finite
/// threshold; +inf ones always are (and win the Gumbel draw). With no
/// finite value there is nothing to bisect: every non-NaN value competes.
///
/// Ties: every value equal to the threshold is kept, so top_k may keep more
/// than k tokens when several tie at the k-th value (as the GPU kernel does).
pub fn threshold(x: &[f32], top_k: u32, top_p: f32, min_p: f32) -> f32 {
    let (mut lo, hi) = x
        .iter()
        .filter(|v| v.is_finite())
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
    let mut thr = f32::NEG_INFINITY;
    if !(lo <= hi) {
        return thr;
    }
    let max = hi;
    if min_p > 0.0 {
        thr = hi + min_p.min(1.0).ln();
        lo = lo.max(thr);
    }
    if top_k > 0 && (top_k as usize) < x.len() {
        // Largest t with count(x >= t) >= k.
        let (mut a, mut b) = (lo, hi);
        for _ in 0..BISECT_STEPS {
            let mid = 0.5 * (a + b);
            if x.iter().filter(|&&v| v >= mid).count() >= top_k as usize { a = mid } else { b = mid }
        }
        thr = a;
        lo = a;
    }
    if top_p < 1.0 {
        let mass = |t: f32| -> f32 { x.iter().filter(|&&v| v >= t).map(|&v| (v - max).exp()).sum() };
        let total = mass(thr);
        let (mut a, mut b) = (lo, hi);
        for _ in 0..BISECT_STEPS {
            let mid = 0.5 * (a + b);
            if mass(mid) >= top_p * total { a = mid } else { b = mid }
        }
        thr = a;
    }
    thr
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(t: f32, k: u32, p: f32, seed: u64) -> RowSample {
        RowSample { temperature: t, top_k: k, top_p: p, min_p: 0.0, seed, counter: 0, adjust: Vec::new() }
    }

    #[test]
    fn greedy_and_adjustments() {
        let logits = [0.1, 2.0, 0.5, 2.0];
        assert_eq!(sample_row(&logits, &row(0.0, 0, 1.0, 0)), 1); // lowest index on ties
        let mut s = row(0.0, 0, 1.0, 0);
        s.adjust = vec![(1, -100.0)];
        assert_eq!(sample_row(&logits, &s), 3);
        let nan = [f32::NAN, 1.0, f32::NAN];
        assert_eq!(sample_row(&nan, &row(0.0, 0, 1.0, 0)), 1);
        assert_eq!(sample_row(&nan, &row(1.0, 2, 0.9, 0)), 1);
        assert_eq!(sample_row(&[f32::NAN; 4], &row(1.0, 2, 0.9, 0)), 0);
    }

    /// The extreme hash outputs map strictly inside (0, 1), with finite noise.
    #[test]
    fn uniform_is_open_interval() {
        let u = |z: u64| ((z >> 41) as f32 + 0.5) * (1.0 / (1u64 << 23) as f32);
        for z in [0, 1 << 41, u64::MAX, u64::MAX - (1 << 41), u64::MAX >> 1] {
            let v = u(z);
            assert!(v > 0.0 && v < 1.0, "{z:#x}: {v}");
            assert!((-(-v.ln()).ln()).is_finite(), "{z:#x}: {v}");
        }
        // The old 24-bit form rounded the top value to exactly 1.0.
        assert_eq!(((u64::MAX >> 40) as f32 + 0.5) * (1.0 / (1u64 << 24) as f32), 1.0);
        for t in 0..100_000 {
            let v = uniform(3, 9, t);
            assert!(v > 0.0 && v < 1.0 && gumbel(3, 9, t).is_finite());
        }
    }

    /// -inf logits (masked tokens) and tiny temperatures keep top_k / top_p working.
    #[test]
    fn limits_survive_inf_and_tiny_temperature() {
        let mut logits = vec![f32::NEG_INFINITY; 8];
        logits.extend([3.0, 2.9, 2.8, -1.0]);
        for c in 0..2000 {
            let mut s = row(1.0, 2, 1.0, 1);
            s.counter = c;
            assert!(matches!(sample_row(&logits, &s), 8 | 9));
            let mut s = row(1.0, 0, 0.3, 1);
            s.counter = c;
            assert_eq!(sample_row(&logits, &s), 8);
        }
        assert!(threshold(&logits, 2, 1.0, 0.0) > 2.8);
        // min_p: max + ln(min_p), and the stricter of it and top_k wins.
        let x = [2.0f32, 1.0, 0.5, 0.0];
        assert!((threshold(&x, 0, 1.0, 0.2) - (2.0 + 0.2f32.ln())).abs() < 1e-6);
        assert!(threshold(&x, 1, 1.0, 0.2) > 1.9);
        for t in [1e-30, 1e-45] {
            for c in 0..200 {
                let mut s = row(t, 3, 0.5, 2);
                s.counter = c;
                assert_eq!(sample_row(&[0.0, 1.0, 0.5, f32::NEG_INFINITY], &s), 1);
            }
        }
        // No finite value: +inf wins, all -inf gives token 0.
        assert_eq!(sample_row(&[f32::NEG_INFINITY, f32::INFINITY, 1.0], &row(1.0, 1, 0.5, 0)), 1);
        assert_eq!(sample_row(&[f32::NEG_INFINITY; 3], &row(1.0, 1, 0.5, 0)), 0);
    }

    /// Frequencies over many draws match softmax(x / T), restricted by top_k / top_p.
    #[test]
    fn distribution_matches_softmax() {
        let logits: Vec<f32> = (0..6).map(|i| -(i as f32) * 0.7).collect();
        let softmax = |t: f32, n: usize| {
            let e: Vec<f32> = logits[..n].iter().map(|v| (v / t).exp()).collect();
            let z: f32 = e.iter().sum();
            e.into_iter().map(|v| v / z).collect::<Vec<_>>()
        };
        let freq = |t: f32, k: u32, p: f32| {
            let mut f = vec![0f32; 6];
            let n = 40_000;
            for c in 0..n {
                let mut s = row(t, k, p, 7);
                s.counter = c;
                f[sample_row(&logits, &s) as usize] += 1.0 / n as f32;
            }
            f
        };
        let close = |a: &[f32], b: &[f32]| {
            for (i, (x, y)) in a.iter().zip(b).enumerate() {
                assert!((x - y).abs() < 0.01, "token {i}: {x} vs {y} ({a:?} vs {b:?})");
            }
        };
        close(&freq(1.0, 0, 1.0), &softmax(1.0, 6));
        close(&freq(0.5, 0, 1.0), &softmax(0.5, 6));
        close(&freq(1.0, 3, 1.0)[..3], &softmax(1.0, 3));
        // top_p 0.8: the first three tokens hold 0.86 of the mass, two hold 0.73.
        close(&freq(1.0, 0, 0.8)[..3], &softmax(1.0, 3));
    }
}
