/// Token sampling: greedy when `temperature == 0`, otherwise temperature +
/// top-k + top-p (nucleus).
#[derive(Debug, Clone)]
pub struct SamplerConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub seed: u64,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self { temperature: 0.0, top_k: 0, top_p: 1.0, seed: 0 }
    }
}

pub struct Sampler {
    cfg: SamplerConfig,
    rng: SplitMix64,
    /// Scratch for per-token weights, reused so each token doesn't touch
    /// fresh pages (a 262k vocabulary is 1 MB).
    weights: Vec<f32>,
}

impl Sampler {
    pub fn new(cfg: SamplerConfig) -> Self {
        let rng = SplitMix64(cfg.seed);
        Self { cfg, rng, weights: Vec::new() }
    }

    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        if self.cfg.temperature <= 0.0 {
            return argmax(logits);
        }
        // No truncation: sample the full distribution (no sort).
        if self.cfg.top_k == 0 && self.cfg.top_p >= 1.0 {
            let (inv_t, max) = (1.0 / self.cfg.temperature, max_of(logits));
            let w = &mut self.weights;
            w.resize(logits.len(), 0.0);
            let total = exp_weights(logits, max, inv_t, w);
            let mut r = self.rng.next_f32() * total;
            // Skip whole blocks by their sums, then walk the block.
            const B: usize = 512;
            for (bi, block) in w.chunks(B).enumerate() {
                let sum = sum_of(block);
                if r > sum {
                    r -= sum;
                    continue;
                }
                for (j, &x) in block.iter().enumerate() {
                    r -= x;
                    if r <= 0.0 {
                        return (bi * B + j) as u32;
                    }
                }
            }
            return argmax(logits);
        }
        let cand = self.head(logits);
        let kept: f32 = cand.iter().map(|c| c.1).sum();
        let mut r = self.rng.next_f32() * kept;
        for &(id, p) in &cand {
            r -= p;
            if r <= 0.0 {
                return id;
            }
        }
        cand.last().unwrap().0
    }

    /// The candidates truncated sampling draws from, as (id, probability
    /// under the full softmax), most likely first. Their sum is the kept
    /// mass; sampling draws from them in proportion. (temperature > 0 and
    /// top_k or top_p set.)
    fn head(&mut self, logits: &[f32]) -> Vec<(u32, f32)> {
        let inv_t = 1.0 / self.cfg.temperature;
        let max = max_of(logits);
        // Only the head of the distribution is sorted. The top 256 (or
        // top_k) candidates almost always hold the top_p mass; if not, the
        // head grows. The full normalizer only matters for top_p over the
        // whole vocabulary.
        let total = if self.cfg.top_k == 0 {
            let w = &mut self.weights;
            w.resize(logits.len(), 0.0);
            exp_weights(logits, max, inv_t, w)
        } else {
            0.0
        };
        let p = |l: f32| ((l - max) * inv_t).exp();
        let mut cand = if self.cfg.top_k > 0 {
            top_candidates(logits, self.cfg.top_k.min(logits.len()))
        } else {
            // Grow the head until it holds the top_p mass.
            let mut head = 256;
            loop {
                let head_n = head.min(logits.len());
                let c = top_candidates(logits, head_n);
                if head_n == logits.len() || c.iter().map(|c| p(c.1)).sum::<f32>() / total >= self.cfg.top_p {
                    break c;
                }
                head *= 16;
            }
        };
        // Probabilities relative to the kept set (top_k) or the whole vocabulary.
        let norm = if self.cfg.top_k > 0 { cand.iter().map(|c| p(c.1)).sum::<f32>() } else { total };
        for c in &mut cand {
            c.1 = p(c.1) / norm;
        }
        // Keep the smallest prefix whose probability mass reaches top_p.
        let mut cum = 0.0;
        let mut keep = cand.len();
        for (i, c) in cand.iter().enumerate() {
            cum += c.1;
            if cum >= self.cfg.top_p {
                keep = i + 1;
                break;
            }
        }
        cand.truncate(keep);
        cand
    }

    /// The distribution [`Self::sample`] draws from for these logits.
    pub fn dist(&mut self, logits: &[f32]) -> Dist {
        if self.cfg.temperature <= 0.0 {
            return Dist::Point(argmax(logits));
        }
        if self.cfg.top_k == 0 && self.cfg.top_p >= 1.0 {
            let mut w = vec![0f32; logits.len()];
            let total = exp_weights(logits, max_of(logits), 1.0 / self.cfg.temperature, &mut w);
            w.iter_mut().for_each(|x| *x /= total);
            return Dist::Dense(w);
        }
        let mut cand = self.head(logits);
        let kept: f32 = cand.iter().map(|c| c.1).sum();
        cand.iter_mut().for_each(|c| c.1 /= kept);
        Dist::Sparse(cand)
    }

    /// Draw a token from `d`.
    pub fn draw(&mut self, d: &Dist) -> u32 {
        match d {
            Dist::Point(id) => *id,
            Dist::Sparse(c) => draw_weighted(c.iter().copied(), self.rng.next_f32() * c.iter().map(|c| c.1).sum::<f32>()),
            Dist::Dense(w) => draw_weighted(w.iter().enumerate().map(|(i, &p)| (i as u32, p)), self.rng.next_f32() * sum_of(w)),
        }
    }

    /// Speculative sampling's verification of `drafted`, a token drawn
    /// from `q` (the draft model's distribution) where the target model's
    /// distribution is `p`: keep it with probability min(1, p/q), otherwise
    /// draw a replacement from max(0, p - q), normalized. The result is
    /// distributed exactly as `p`. Returns (token, whether `drafted` was kept).
    pub fn verify(&mut self, p: &Dist, q: &Dist, drafted: u32) -> (u32, bool) {
        let (pd, qd) = (p.prob(drafted), q.prob(drafted));
        if pd >= qd || self.rng.next_f32() * qd < pd {
            return (drafted, true);
        }
        let residual: Vec<(u32, f32)> = match p {
            Dist::Point(id) => return (*id, false),
            Dist::Sparse(c) => {
                let q = q.lookup();
                c.iter().map(|&(id, pv)| (id, (pv - q.prob(id)).max(0.0))).collect()
            }
            Dist::Dense(w) => {
                let q = q.lookup();
                w.iter().enumerate().map(|(i, &pv)| (i as u32, (pv - q.prob(i as u32)).max(0.0))).collect()
            }
        };
        let total: f32 = residual.iter().map(|c| c.1).sum();
        if total <= 0.0 {
            // Rounding left no residual mass (p and q nearly equal).
            return (self.draw(p), false);
        }
        (draw_weighted(residual.iter().copied(), self.rng.next_f32() * total), false)
    }
}

/// The token where cumulative weight first reaches `r` (the last one if
/// rounding leaves `r` unspent).
fn draw_weighted(items: impl Iterator<Item = (u32, f32)>, mut r: f32) -> u32 {
    let mut last = 0;
    for (id, w) in items {
        if w > 0.0 {
            last = id;
            r -= w;
            if r <= 0.0 {
                return id;
            }
        }
    }
    last
}

/// A token distribution: the probability [`Sampler::sample`] returns each
/// token with (for speculative decoding's verification).
#[derive(Debug, Clone)]
pub enum Dist {
    /// Greedy: always this token.
    Point(u32),
    /// Probabilities of these tokens; all others have none.
    Sparse(Vec<(u32, f32)>),
    /// Probability of every token.
    Dense(Vec<f32>),
}

impl Dist {
    pub fn prob(&self, id: u32) -> f32 {
        match self {
            Dist::Point(t) => (*t == id) as u32 as f32,
            Dist::Sparse(c) => c.iter().find(|c| c.0 == id).map_or(0.0, |c| c.1),
            Dist::Dense(w) => w.get(id as usize).copied().unwrap_or(0.0),
        }
    }

    /// Fast repeated `prob` lookups.
    fn lookup(&self) -> DistLookup<'_> {
        match self {
            Dist::Sparse(c) if c.len() > 32 => DistLookup::Map(c.iter().copied().collect()),
            d => DistLookup::Direct(d),
        }
    }
}

enum DistLookup<'a> {
    Direct(&'a Dist),
    Map(std::collections::HashMap<u32, f32>),
}

impl DistLookup<'_> {
    fn prob(&self, id: u32) -> f32 {
        match self {
            DistLookup::Direct(d) => d.prob(id),
            DistLookup::Map(m) => m.get(&id).copied().unwrap_or(0.0),
        }
    }
}

const LANES: usize = 16;

fn max_of(x: &[f32]) -> f32 {
    let mut lanes = [f32::NEG_INFINITY; LANES];
    let mut chunks = x.chunks_exact(LANES);
    for c in &mut chunks {
        for (m, &v) in lanes.iter_mut().zip(c) {
            *m = if v > *m { v } else { *m };
        }
    }
    chunks.remainder().iter().chain(&lanes).fold(f32::NEG_INFINITY, |a, &b| if b > a { b } else { a })
}

#[inline(always)]
fn sum_of(x: &[f32]) -> f32 {
    let mut lanes = [0f32; LANES];
    let mut chunks = x.chunks_exact(LANES);
    for c in &mut chunks {
        for (a, &v) in lanes.iter_mut().zip(c) {
            *a += v;
        }
    }
    chunks.remainder().iter().sum::<f32>() + lanes.iter().sum::<f32>()
}

/// e^x for x <= 0, to ~3e-5 relative: 2^(x·log2 e) as an exponent shift
/// times a polynomial for the fraction. Unlike `f32::exp` it vectorizes
/// (no calls, no saturating float-to-int casts).
#[inline(always)]
fn fast_exp(x: f32) -> f32 {
    let x = if x > -80.0 { x } else { -80.0 };
    let t = x * std::f32::consts::LOG2_E;
    // Round to nearest by adding 1.5 * 2^23: the integer lands in the low mantissa bits.
    const MAGIC: f32 = 12_582_912.0;
    let k = (t + MAGIC).to_bits() as i32 - MAGIC.to_bits() as i32;
    let f = (t - k as f32) * std::f32::consts::LN_2; // |f| <= ln2 / 2
    // e^f, Taylor to degree 5.
    let p = 1.0 + f * (1.0 + f * (0.5 + f * (1.0 / 6.0 + f * (1.0 / 24.0 + f * (1.0 / 120.0)))));
    f32::from_bits((p.to_bits() as i32).wrapping_add(k << 23) as u32)
}

/// `out[i] = exp((x[i] - max) / t)`; returns the sum.
fn exp_weights(x: &[f32], max: f32, inv_t: f32, out: &mut [f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        // SAFETY: the CPU supports the enabled features (checked above).
        return unsafe { exp_weights_avx2(x, max, inv_t, out) };
    }
    exp_weights_generic(x, max, inv_t, out)
}

/// The same code compiled for AVX2 + FMA (the default x86-64 target only
/// has SSE2): twice as wide.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn exp_weights_avx2(x: &[f32], max: f32, inv_t: f32, out: &mut [f32]) -> f32 {
    exp_weights_generic(x, max, inv_t, out)
}

#[inline(always)]
fn exp_weights_generic(x: &[f32], max: f32, inv_t: f32, out: &mut [f32]) -> f32 {
    for (o, &v) in out.iter_mut().zip(x) {
        *o = fast_exp((v - max) * inv_t);
    }
    sum_of(out)
}

/// ln Σ e^x, vectorized (for logprobs over a large vocabulary).
pub fn log_sum_exp(x: &[f32]) -> f32 {
    let max = max_of(x);
    if !max.is_finite() {
        return max;
    }
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        // SAFETY: the CPU supports the enabled features (checked above).
        return max + unsafe { exp_sum_avx2(x, max) }.ln();
    }
    max + exp_sum_generic(x, max).ln()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn exp_sum_avx2(x: &[f32], max: f32) -> f32 {
    exp_sum_generic(x, max)
}

#[inline(always)]
fn exp_sum_generic(x: &[f32], max: f32) -> f32 {
    let mut lanes = [0f32; LANES];
    let mut chunks = x.chunks_exact(LANES);
    for c in &mut chunks {
        for (a, &v) in lanes.iter_mut().zip(c) {
            *a += fast_exp(v - max);
        }
    }
    chunks.remainder().iter().map(|&v| fast_exp(v - max)).sum::<f32>() + lanes.iter().sum::<f32>()
}

/// The `k` largest logits as (id, logit), descending. One pass keeps a
/// small buffer above a rising threshold instead of sorting everything.
pub fn top_candidates(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
    let cap = (4 * k).max(64);
    let mut buf: Vec<(u32, f32)> = Vec::with_capacity(cap + LANES);
    let mut threshold = f32::NEG_INFINITY;
    let shrink = |buf: &mut Vec<(u32, f32)>| {
        buf.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1));
        buf.truncate(k);
        buf.iter().map(|c| c.1).fold(f32::INFINITY, f32::min)
    };
    for (ci, chunk) in logits.chunks(LANES).enumerate() {
        // Most chunks hold nothing above the threshold: check cheaply first.
        if chunk.iter().all(|&v| !(v > threshold)) {
            continue;
        }
        for (j, &v) in chunk.iter().enumerate() {
            if v > threshold {
                buf.push(((ci * LANES + j) as u32, v));
            }
        }
        if buf.len() >= cap {
            threshold = shrink(&mut buf);
        }
    }
    if buf.len() > k {
        shrink(&mut buf);
    }
    buf.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    buf
}

/// Index of the largest logit (the first one on ties; NaNs never win).
pub fn argmax(logits: &[f32]) -> u32 {
    // Two passes that vectorize: the maximum over 16 independent lanes,
    // then the first position holding it.
    const L: usize = 16;
    let mut lanes = [f32::NEG_INFINITY; L];
    let mut chunks = logits.chunks_exact(L);
    for c in &mut chunks {
        for (m, &x) in lanes.iter_mut().zip(c) {
            *m = if x > *m { x } else { *m };
        }
    }
    let max = chunks.remainder().iter().chain(&lanes).fold(f32::NEG_INFINITY, |a, &b| if b > a { b } else { a });
    logits.iter().position(|&x| x == max).unwrap_or(0) as u32
}

struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

#[cfg(test)]
mod speed {
    use super::*;

    /// `cargo test --release -p rllm-model sampler_speed -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn sampler_speed() {
        let n = 262_144;
        let mut rng = SplitMix64(1);
        let logits: Vec<f32> = (0..n).map(|_| rng.next_f32() * 30.0 - 15.0).collect();
        let time = |name: &str, f: &mut dyn FnMut() -> u32| {
            let reps = 200;
            let t0 = std::time::Instant::now();
            let mut acc = 0u32;
            for _ in 0..reps {
                acc = acc.wrapping_add(f());
            }
            println!("{name:>12}: {:7.1} us  ({acc})", t0.elapsed().as_secs_f64() / reps as f64 * 1e6);
        };
        time("argmax", &mut || argmax(std::hint::black_box(&logits)));
        let mut w = vec![0f32; n];
        time("exp_weights", &mut || exp_weights(std::hint::black_box(&logits), 15.0, 1.0, &mut w) as u32);
        time("max_of", &mut || max_of(std::hint::black_box(&logits)) as u32);
        time("sum_of", &mut || sum_of(std::hint::black_box(&logits)) as u32);
        let worst = (0..n).map(|i| (logits[i] - 15.0).exp()).zip(&w).map(|(a, b)| ((a - b) / a).abs()).fold(0f32, f32::max);
        println!("fast_exp max rel err {worst:e}");
        time("to_vec", &mut || std::hint::black_box(logits.to_vec())[7] as u32);
        let mut s = Sampler::new(SamplerConfig { temperature: 1.0, top_k: 0, top_p: 1.0, seed: 1 });
        time("temp 1.0", &mut || s.sample(&logits));
        let mut s = Sampler::new(SamplerConfig { temperature: 0.7, top_k: 0, top_p: 0.95, seed: 1 });
        time("top_p .95", &mut || s.sample(&logits));
        let mut s = Sampler::new(SamplerConfig { temperature: 0.7, top_k: 40, top_p: 1.0, seed: 1 });
        time("top_k 40", &mut || s.sample(&logits));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_logits(n: usize, seed: u64, scale: f32) -> Vec<f32> {
        let mut r = SplitMix64(seed);
        (0..n).map(|_| (r.next_f32() - 0.5) * scale).collect()
    }

    #[test]
    fn argmax_and_top_candidates_match_naive() {
        for seed in 0..20 {
            let l = random_logits(1000 + seed as usize * 37, seed, 20.0);
            let naive = l.iter().enumerate().fold(0, |b, (i, &x)| if x > l[b] { i } else { b });
            assert_eq!(argmax(&l) as usize, naive);
            let mut sorted: Vec<(u32, f32)> = l.iter().enumerate().map(|(i, &x)| (i as u32, x)).collect();
            sorted.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
            for k in [1, 5, 40, 256] {
                assert_eq!(top_candidates(&l, k), sorted[..k].to_vec());
            }
        }
        assert_eq!(argmax(&[1.0, 3.0, 3.0, 2.0]), 1);
    }

    #[test]
    fn fast_exp_is_accurate() {
        for i in 0..10_000 {
            let x = -(i as f32) * 0.008;
            let (a, b) = (x.exp(), fast_exp(x));
            assert!(((a - b) / a).abs() < 1e-5, "{x}: {a} vs {b}");
        }
    }

    /// Speculative verification reproduces the target distribution whatever
    /// the draft distribution, and `dist` matches what `sample` draws.
    #[test]
    fn verify_reproduces_the_target() {
        let target = [2.0f32, 1.0, 0.5, 0.0, -1.0, -3.0];
        let draft = [0.0f32, 2.0, 0.0, 1.0, -2.0, 0.0];
        for (seed, cfg) in [
            SamplerConfig { temperature: 1.0, top_k: 0, top_p: 1.0, seed: 0 },
            SamplerConfig { temperature: 0.7, top_k: 3, top_p: 1.0, seed: 0 },
            SamplerConfig { temperature: 1.0, top_k: 0, top_p: 0.8, seed: 0 },
            SamplerConfig { temperature: 0.0, top_k: 0, top_p: 1.0, seed: 0 },
        ]
        .into_iter()
        .enumerate()
        {
            let mut s = Sampler::new(SamplerConfig { seed: seed as u64 + 10, ..cfg.clone() });
            let (p, q) = (s.dist(&target), s.dist(&draft));
            let n = 200_000;
            let mut c = [0usize; 6];
            let mut d = [0usize; 6];
            for _ in 0..n {
                let x = s.draw(&q);
                c[s.verify(&p, &q, x).0 as usize] += 1;
                d[s.sample(&target) as usize] += 1;
            }
            for i in 0..6 {
                let (got, want, direct) = (c[i] as f32 / n as f32, p.prob(i as u32), d[i] as f32 / n as f32);
                assert!((got - want).abs() < 0.006, "cfg {seed} token {i}: verified {got} vs p {want}");
                assert!((direct - want).abs() < 0.006, "cfg {seed} token {i}: sampled {direct} vs p {want}");
            }
        }
    }

    /// Sampled frequencies follow softmax(logits / t) restricted as configured.
    #[test]
    fn sampling_follows_the_distribution() {
        let logits = [2.0f32, 1.0, 0.5, 0.0, -1.0, -3.0];
        let softmax = |t: f32, keep: usize| {
            let e: Vec<f32> = logits.iter().take(keep).map(|&l| ((l - 2.0) / t).exp()).collect();
            let s: f32 = e.iter().sum();
            e.into_iter().map(|x| x / s).collect::<Vec<_>>()
        };
        let freq = |cfg: SamplerConfig| {
            let mut s = Sampler::new(cfg);
            let mut c = [0usize; 6];
            let n = 200_000;
            for _ in 0..n {
                c[s.sample(&logits) as usize] += 1;
            }
            c.map(|x| x as f32 / n as f32)
        };
        let close = |got: [f32; 6], want: &[f32]| {
            for i in 0..6 {
                let w = want.get(i).copied().unwrap_or(0.0);
                assert!((got[i] - w).abs() < 0.006, "token {i}: got {} want {w} ({got:?})", got[i]);
            }
        };
        close(freq(SamplerConfig { temperature: 1.0, top_k: 0, top_p: 1.0, seed: 3 }), &softmax(1.0, 6));
        close(freq(SamplerConfig { temperature: 0.5, top_k: 0, top_p: 1.0, seed: 4 }), &softmax(0.5, 6));
        close(freq(SamplerConfig { temperature: 1.0, top_k: 3, top_p: 1.0, seed: 5 }), &softmax(1.0, 3));
        // top_p 0.8: the first three tokens hold 0.86 of the mass, two hold 0.73.
        close(freq(SamplerConfig { temperature: 1.0, top_k: 0, top_p: 0.8, seed: 6 }), &softmax(1.0, 3));
    }
}
