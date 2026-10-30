//! Device-side token sampling: one token per logits row without copying the
//! row to the host. The reference implementation here is what the CPU
//! backend runs and what GPU kernels are tested against.
//!
//! Exact without sorting the vocabulary:
//! - temperature 0: argmax (lowest index on ties);
//! - otherwise the Gumbel-max trick, `argmax(x / T + g)` with `g` Gumbel
//!   noise, draws exactly from softmax(x / T);
//! - top_k and top_p restrict the candidates to `x >= threshold`, the
//!   threshold found by bisection on the value: the k-th largest value for
//!   top_k; for top_p the largest threshold whose candidates hold at least
//!   `top_p` of the probability (of the top_k set when both are given).
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
    pub seed: u64,
    /// Distinguishes the draws of one request (e.g. its position).
    pub counter: u64,
    /// Added to the logits first: logit bias, repetition penalties.
    pub adjust: Vec<(u32, f32)>,
}

/// Bisection steps for the top_k / top_p thresholds.
pub const BISECT_STEPS: u32 = 40;

/// Uniform in (0, 1) from (seed, counter, token): splitmix64 finalizer over a
/// mix of the three (the GPU kernels compute the same bits).
pub fn uniform(seed: u64, counter: u64, token: u32) -> f32 {
    let mut z = seed ^ counter.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (token as u64).wrapping_mul(0xD1B5_4A32_D192_ED03);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    ((z >> 40) as f32 + 0.5) * (1.0 / (1u64 << 24) as f32)
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
        let mut best = 0;
        for (i, &v) in x.iter().enumerate() {
            if v > x[best] {
                best = i;
            }
        }
        return best as u32;
    }
    let inv_t = 1.0 / s.temperature;
    x.iter_mut().for_each(|v| *v *= inv_t);
    let thr = threshold(&x, s.top_k, s.top_p);
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
pub fn threshold(x: &[f32], top_k: u32, top_p: f32) -> f32 {
    let (mut lo, hi) = x.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
    let max = hi;
    let mut thr = f32::NEG_INFINITY;
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
        RowSample { temperature: t, top_k: k, top_p: p, seed, counter: 0, adjust: Vec::new() }
    }

    #[test]
    fn greedy_and_adjustments() {
        let logits = [0.1, 2.0, 0.5, 2.0];
        assert_eq!(sample_row(&logits, &row(0.0, 0, 1.0, 0)), 1); // lowest index on ties
        let mut s = row(0.0, 0, 1.0, 0);
        s.adjust = vec![(1, -100.0)];
        assert_eq!(sample_row(&logits, &s), 3);
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
