//! The decision head of Cloudflare's Clef (`general.architecture = clef`).
//!
//! Clef is a Qwen3.5 dense model (loaded as [`crate::Qwen35`]) with a small
//! head on top that scores the options of typed questions instead of
//! predicting tokens. The prompt (the GGUF's `systemone` chat template)
//! holds the state, then each question's instructions and its options;
//! every token is tagged with what it belongs to ([`Order`]). The head reads
//! the final hidden states of all prompt tokens and returns one score
//! (logit) per option, in prompt order; a softmax over each question's
//! options gives its probabilities.
//!
//! The math follows llama.cpp's `src/models/clef.cpp` (a port of
//! `JointSchemaHead` in the model repo's `joint_schema_model.py`):
//!
//! 1. `hidden` = LayerNorm(final hidden states); `memory` = hidden · P_memory
//!    (width 1024); `global` = the last token's hidden state; each question
//!    and option span is mean-pooled; `lexical` = the mean of the option
//!    tokens' output-embedding rows.
//! 2. One query per option (context + lexical + its question), refined by
//!    the routing blocks (cross-attention to `memory`, FFN).
//! 3. One vector per question: its text, an attention summary of its own
//!    options, `global` and a type embedding; refined by the joint blocks
//!    (self-attention over questions, cross-attention to `memory`, FFN).
//! 4. Score = s0 · prior + s2 · (s1 · cosine + MLP(features)), where prior is
//!    the cosine of an option's output embedding with its question + global.
//!
//! The head is small (about 125M parameters, dequantized to f32 at load) and
//! runs on the CPU, after the backbone produced the hidden states.

use anyhow::{Context, Result, bail, ensure};
use rayon::prelude::*;
use rllm_core::quant::dequantize_row;
use rllm_gguf::Gguf;

use crate::loader::{host_tensor, read_vector};

/// What a prompt token belongs to (llama.cpp's `llama_decision_order`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Order {
    /// Not read by the head's spans (state, template text).
    None = 0,
    /// The instructions of a yes/no question.
    QuestionNoul = 1,
    /// The instructions of a choice question.
    QuestionChoice = 2,
    /// The instructions of a score question.
    QuestionScore = 3,
    /// The text of an option (of the last question before it).
    Option = 4,
}

/// Whether `g` is a Clef model.
pub fn is_clef(g: &Gguf) -> bool {
    g.get_str("general.architecture").is_ok_and(|a| a == "clef")
}

/// A dense f32 matrix `[rows, cols]` (rows = outputs) and an optional bias.
struct Linear {
    w: Vec<f32>,
    b: Option<Vec<f32>>,
    rows: usize,
    cols: usize,
}

impl Linear {
    fn load(g: &Gguf, name: &str, bias: bool) -> Result<Self> {
        let t = host_tensor(g, &format!("{name}.weight"))?;
        ensure!(t.shape.len() == 2, "{name}.weight: expected a matrix, shape {:?}", t.shape);
        let (rows, cols) = (t.shape[0], t.shape[1]);
        let rb = t.dtype.row_bytes(cols);
        let src = t.bytes.as_slice();
        ensure!(src.len() == rows * rb, "{name}.weight: size mismatch");
        let mut w = vec![0f32; rows * cols];
        w.par_chunks_mut(cols).enumerate().for_each(|(r, o)| dequantize_row(t.dtype, &src[r * rb..(r + 1) * rb], o));
        let b = if bias { Some(read_vector(g, &format!("{name}.bias"))?) } else { None };
        if let Some(b) = &b {
            ensure!(b.len() == rows, "{name}.bias: {} values for {rows} outputs", b.len());
        }
        Ok(Self { w, b, rows, cols })
    }

    /// `x [m, cols]` → `[m, rows]`.
    fn apply(&self, x: &[f32]) -> Vec<f32> {
        let (n, k) = (self.rows, self.cols);
        assert_eq!(x.len() % k, 0, "linear: input width");
        let m = x.len() / k;
        let mut out = vec![0f32; m * n];
        // Tiles of input rows; each task streams the whole weight once.
        const TILE: usize = 16;
        out.par_chunks_mut(TILE * n).zip(x.par_chunks(TILE * k)).for_each(|(o, x)| {
            let rows = x.len() / k;
            for r in 0..n {
                let w = &self.w[r * k..(r + 1) * k];
                let bias = self.b.as_ref().map_or(0.0, |b| b[r]);
                for i in 0..rows {
                    o[i * n + r] = dot(&x[i * k..(i + 1) * k], w) + bias;
                }
            }
        });
        out
    }
}

/// LayerNorm weight and bias.
struct Norm {
    w: Vec<f32>,
    b: Vec<f32>,
}

impl Norm {
    fn load(g: &Gguf, name: &str, width: usize) -> Result<Self> {
        let w = read_vector(g, &format!("{name}.weight"))?;
        let b = read_vector(g, &format!("{name}.bias"))?;
        ensure!(w.len() == width && b.len() == width, "{name}: expected width {width}");
        Ok(Self { w, b })
    }

    /// LayerNorm of each row of `x` (biased variance).
    fn apply(&self, x: &[f32], eps: f32) -> Vec<f32> {
        let d = self.w.len();
        let mut out = vec![0f32; x.len()];
        out.par_chunks_mut(d).zip(x.par_chunks(d)).for_each(|(o, x)| {
            let mean = x.iter().sum::<f32>() / d as f32;
            let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
            let s = 1.0 / (var + eps).sqrt();
            for i in 0..d {
                o[i] = (x[i] - mean) * s * self.w[i] + self.b[i];
            }
        });
        out
    }
}

struct Attn {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
}

impl Attn {
    fn load(g: &Gguf, prefix: &str, il: usize) -> Result<Self> {
        let l = |s: &str| Linear::load(g, &format!("dec.blk.{il}.{prefix}{s}"), true);
        Ok(Self { q: l("q")?, k: l("k")?, v: l("v")?, o: l("o")? })
    }

    /// Multi-head attention of queries `q [n_q, d]` over `kv [n_kv, d]`, no mask.
    fn apply(&self, q: &[f32], kv: &[f32], n_heads: usize) -> Vec<f32> {
        let (k, v) = (self.k.apply(kv), self.v.apply(kv));
        self.attend(&self.q.apply(q), &k, &v, n_heads)
    }

    /// Attention from projected queries, keys and values.
    fn attend(&self, q: &[f32], k: &[f32], v: &[f32], n_heads: usize) -> Vec<f32> {
        let d = self.q.rows;
        let dh = d / n_heads;
        let (n_q, n_kv) = (q.len() / d, k.len() / d);
        let scale = 1.0 / (dh as f32).sqrt();
        let mut ctx = vec![0f32; n_q * d];
        ctx.par_chunks_mut(d).enumerate().for_each(|(i, out)| {
            let mut s = vec![0f32; n_kv];
            for h in 0..n_heads {
                let qh = &q[i * d + h * dh..i * d + (h + 1) * dh];
                for (j, s) in s.iter_mut().enumerate() {
                    *s = dot(qh, &k[j * d + h * dh..j * d + (h + 1) * dh]) * scale;
                }
                softmax(&mut s);
                let oh = &mut out[h * dh..(h + 1) * dh];
                for (j, &p) in s.iter().enumerate() {
                    for (o, &x) in oh.iter_mut().zip(&v[j * d + h * dh..j * d + (h + 1) * dh]) {
                        *o += p * x;
                    }
                }
            }
        });
        self.o.apply(&ctx)
    }
}

/// One block of the head. Routing blocks have `cross_norm_kv` and no self-attention.
struct Block {
    self_attn: Option<(Norm, Attn)>,
    cross_norm_kv: Option<Norm>,
    cross_norm: Norm,
    cross_attn: Attn,
    ffn_norm: Norm,
    ffn_up: Linear,
    ffn_down: Linear,
}

impl Block {
    fn ffn(&self, x: &[f32], eps: f32) -> Vec<f32> {
        let mut h = self.ffn_up.apply(&self.ffn_norm.apply(x, eps));
        h.par_iter_mut().for_each(|v| *v = gelu_erf(*v));
        self.ffn_down.apply(&h)
    }
}

pub struct ClefHead {
    hidden: usize,
    width: usize,
    n_heads: usize,
    eps: f32,
    blocks: Vec<Block>,
    n_routing: usize,
    hidden_norm: Norm,
    option_summary_norm: Norm,
    field_norm: Norm,
    option_norm: Norm,
    proj_memory: Linear,
    proj_question: Linear,
    proj_option_question: Linear,
    proj_global: Linear,
    proj_option_context: Linear,
    proj_option_lexical: Linear,
    scales: [f32; 3],
    /// `[3, width]`: noul, choice, score.
    type_embd: Vec<f32>,
    scorer: Linear,
    scorer_out: Vec<f32>,
    scorer_out_b: f32,
    /// The output embedding (`output.weight`, `[vocab, hidden]`), read a
    /// row at a time for the options' tokens.
    output: rllm_core::HostTensor,
}

/// A question's span: its type (0 noul, 1 choice, 2 score) and tokens.
#[derive(Debug, Clone, PartialEq, Eq)]
struct QuestionSpan {
    kind: usize,
    start: usize,
    end: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct OptionSpan {
    question: usize,
    start: usize,
    end: usize,
}

/// The question and option spans of a prompt's token tags (runs of equal
/// tags; an option belongs to the last question before it).
fn spans(order: &[Order]) -> Result<(Vec<QuestionSpan>, Vec<OptionSpan>)> {
    let (mut qs, mut os) = (Vec::new(), Vec::<OptionSpan>::new());
    let mut i = 0;
    while i < order.len() {
        let o = order[i];
        let mut end = i + 1;
        while end < order.len() && order[end] == o {
            end += 1;
        }
        match o {
            Order::None => {}
            Order::QuestionNoul | Order::QuestionChoice | Order::QuestionScore => {
                qs.push(QuestionSpan { kind: o as usize - 1, start: i, end });
            }
            Order::Option => {
                ensure!(!qs.is_empty(), "an option comes before any question");
                os.push(OptionSpan { question: qs.len() - 1, start: i, end });
            }
        }
        i = end;
    }
    ensure!(!qs.is_empty(), "the prompt has no question");
    for q in 0..qs.len() {
        ensure!(os.iter().any(|o| o.question == q), "question {} has no option", q + 1);
    }
    Ok((qs, os))
}

impl ClefHead {
    pub fn load(g: &Gguf) -> Result<Self> {
        ensure!(is_clef(g), "not a clef model");
        let meta = |k: &str| g.get_u64(&format!("clef.{k}")).with_context(|| format!("clef.{k}"));
        let n_routing = meta("decision.routing_block_count")? as usize;
        let n_joint = meta("decision.block_count")? as usize;
        let n_heads = meta("decision.head_count")? as usize;
        let hidden = meta("embedding_length")? as usize;
        let eps = g.get_f64("clef.attention.layer_norm_epsilon").unwrap_or(1e-5) as f32;
        ensure!(n_routing + n_joint <= 64 && n_heads > 0, "invalid decision head size");

        let proj = |s: &str| Linear::load(g, &format!("decision.proj_{s}"), false);
        let proj_memory = proj("memory")?;
        let width = proj_memory.rows;
        ensure!(proj_memory.cols == hidden, "decision.proj_memory: {} inputs for hidden size {hidden}", proj_memory.cols);
        ensure!(width % n_heads == 0, "decision head width {width} is not a multiple of {n_heads} heads");
        let norm = |s: &str, w: usize| Norm::load(g, s, w);

        let mut blocks = Vec::new();
        for il in 0..n_routing + n_joint {
            let p = |s: &str| format!("dec.blk.{il}.{s}");
            let routing = il < n_routing;
            blocks.push(Block {
                self_attn: if routing { None } else { Some((norm(&p("attn_norm"), width)?, Attn::load(g, "attn_", il)?)) },
                cross_norm_kv: if routing { Some(norm(&p("cross_attn_norm_kv"), width)?) } else { None },
                cross_norm: norm(&p("cross_attn_norm"), width)?,
                cross_attn: Attn::load(g, "cross_attn_", il)?,
                ffn_norm: norm(&p("ffn_norm"), width)?,
                ffn_up: Linear::load(g, &p("ffn_up"), true)?,
                ffn_down: Linear::load(g, &p("ffn_down"), true)?,
            });
        }
        let scales = read_vector(g, "decision.scales")?;
        let scales: [f32; 3] = scales.try_into().ok().context("decision.scales: expected 3 values")?;
        let type_embd = {
            let t = host_tensor(g, "token_types.weight")?;
            ensure!(t.shape == [3, width], "token_types.weight: shape {:?}", t.shape);
            let mut v = vec![0f32; 3 * width];
            dequantize_row(t.dtype, t.bytes.as_slice(), &mut v);
            v
        };
        let scorer = Linear::load(g, "decision.scorer", true)?;
        ensure!(scorer.cols == 4 * width && scorer.rows == width, "decision.scorer: unexpected shape");
        let scorer_out = read_vector(g, "decision.scorer_out.weight")?;
        let scorer_out_b = read_vector(g, "decision.scorer_out.bias")?;
        ensure!(scorer_out.len() == width && scorer_out_b.len() == 1, "decision.scorer_out: unexpected shape");
        let output = host_tensor(g, "output.weight")?;
        ensure!(output.shape.len() == 2 && output.shape[1] == hidden, "output.weight: shape {:?}", output.shape);
        Ok(Self {
            hidden,
            width,
            n_heads,
            eps,
            blocks,
            n_routing,
            hidden_norm: norm("decision.hidden_norm", hidden)?,
            option_summary_norm: norm("decision.option_summary_norm", width)?,
            field_norm: norm("decision.field_norm", width)?,
            option_norm: norm("decision.option_norm", width)?,
            proj_memory,
            proj_question: proj("question")?,
            proj_option_question: proj("option_question")?,
            proj_global: proj("global")?,
            proj_option_context: proj("option_context")?,
            proj_option_lexical: proj("option_lexical")?,
            scales,
            type_embd,
            scorer,
            scorer_out,
            scorer_out_b: scorer_out_b[0],
            output,
        })
    }

    /// Width of the hidden states [`Self::scores`] takes.
    pub fn hidden_size(&self) -> usize {
        self.hidden
    }

    /// The output embedding row of `token`.
    fn output_row(&self, token: u32, out: &mut [f32]) -> Result<()> {
        let rows = self.output.shape[0];
        ensure!((token as usize) < rows, "token {token} out of range");
        let rb = self.output.dtype.row_bytes(self.hidden);
        let t = token as usize;
        dequantize_row(self.output.dtype, &self.output.bytes.as_slice()[t * rb..(t + 1) * rb], out);
        Ok(())
    }

    /// One score per option, in prompt order. `hidden` holds the backbone's
    /// final (RMS-normalized) hidden states of every prompt token, `[n,
    /// hidden]`; `tokens` and `order` are the prompt's ids and tags.
    pub fn scores(&self, hidden: &[f32], tokens: &[u32], order: &[Order]) -> Result<Vec<f32>> {
        let (d, w) = (self.hidden, self.width);
        let n = tokens.len();
        ensure!(n > 0 && order.len() == n && hidden.len() == n * d, "decision head: {} hidden values for {n} tokens", hidden.len());
        let (qs, os) = spans(order)?;
        let (nq, no) = (qs.len(), os.len());
        let eps = self.eps;

        let hidden = self.hidden_norm.apply(hidden, eps);
        let memory = self.proj_memory.apply(&hidden);
        let global = &hidden[(n - 1) * d..];

        let mean = |start: usize, end: usize, out: &mut [f32]| {
            for t in start..end {
                add(out, &hidden[t * d..(t + 1) * d]);
            }
            out.iter_mut().for_each(|v| *v /= (end - start) as f32);
        };
        let mut question = vec![0f32; nq * d];
        for (q, o) in qs.iter().zip(question.chunks_exact_mut(d)) {
            mean(q.start, q.end, o);
        }
        let mut opt_ctx = vec![0f32; no * d];
        let mut lexical = vec![0f32; no * d];
        let mut row = vec![0f32; d];
        for (i, s) in os.iter().enumerate() {
            mean(s.start, s.end, &mut opt_ctx[i * d..(i + 1) * d]);
            let lex = &mut lexical[i * d..(i + 1) * d];
            for &t in &tokens[s.start..s.end] {
                self.output_row(t, &mut row)?;
                add(lex, &row);
            }
            lex.iter_mut().for_each(|v| *v /= (s.end - s.start) as f32);
        }
        let owner = |o: usize| os[o].question;

        // One query per option.
        let mut options = self.proj_option_context.apply(&opt_ctx);
        add(&mut options, &self.proj_option_lexical.apply(&lexical));
        let oq = self.proj_option_question.apply(&question);
        for o in 0..no {
            let q = owner(o);
            add(&mut options[o * w..(o + 1) * w], &oq[q * w..(q + 1) * w]);
        }

        // The options read the prompt.
        for b in &self.blocks[..self.n_routing] {
            let kv = b.cross_norm_kv.as_ref().expect("routing block").apply(&memory, eps);
            let a = b.cross_attn.apply(&b.cross_norm.apply(&options, eps), &kv, self.n_heads);
            add(&mut options, &a);
            let f = b.ffn(&options, eps);
            add(&mut options, &f);
        }

        // One vector per question.
        let mut fields = self.proj_question.apply(&question);
        {
            let mut summary = vec![0f32; nq * w];
            let scale = 1.0 / (w as f32).sqrt();
            for q in 0..nq {
                let mine: Vec<usize> = (0..no).filter(|&o| owner(o) == q).collect();
                let fq = &fields[q * w..(q + 1) * w];
                let mut s: Vec<f32> = mine.iter().map(|&o| dot(&options[o * w..(o + 1) * w], fq) * scale).collect();
                softmax(&mut s);
                let out = &mut summary[q * w..(q + 1) * w];
                for (&o, &p) in mine.iter().zip(&s) {
                    for (a, &x) in out.iter_mut().zip(&options[o * w..(o + 1) * w]) {
                        *a += p * x;
                    }
                }
            }
            add(&mut fields, &self.option_summary_norm.apply(&summary, eps));
            let g = self.proj_global.apply(global);
            for (q, s) in qs.iter().enumerate() {
                let f = &mut fields[q * w..(q + 1) * w];
                add(f, &g);
                add(f, &self.type_embd[s.kind * w..(s.kind + 1) * w]);
            }
        }

        // The questions read each other and the prompt.
        for b in &self.blocks[self.n_routing..] {
            let (norm, attn) = b.self_attn.as_ref().expect("joint block");
            let x = norm.apply(&fields, eps);
            let a = attn.apply(&x, &x, self.n_heads);
            add(&mut fields, &a);
            let x = b.cross_norm.apply(&fields, eps);
            let a = b.cross_attn.apply(&x, &memory, self.n_heads);
            add(&mut fields, &a);
            let f = b.ffn(&fields, eps);
            add(&mut fields, &f);
        }
        let fields = self.field_norm.apply(&fields, eps);

        // Prior: the option's output embedding against its question and the end of the prompt.
        let mut anchor = question.clone();
        for a in anchor.chunks_exact_mut(d) {
            add(a, global);
            l2_normalize(a);
        }
        let options = self.option_norm.apply(&options, eps);
        let mut scores = Vec::with_capacity(no);
        let mut features = vec![0f32; no * 4 * w];
        let mut cosines = Vec::with_capacity(no);
        for o in 0..no {
            let q = owner(o);
            let mut lex = lexical[o * d..(o + 1) * d].to_vec();
            l2_normalize(&mut lex);
            let prior = dot(&lex, &anchor[q * d..(q + 1) * d]);
            let (f, x) = (&fields[q * w..(q + 1) * w], &options[o * w..(o + 1) * w]);
            let (mut fu, mut xu) = (f.to_vec(), x.to_vec());
            l2_normalize(&mut fu);
            l2_normalize(&mut xu);
            cosines.push(dot(&fu, &xu));
            let feat = &mut features[o * 4 * w..(o + 1) * 4 * w];
            for i in 0..w {
                feat[i] = f[i];
                feat[w + i] = x[i];
                feat[2 * w + i] = f[i] * x[i];
                feat[3 * w + i] = (f[i] - x[i]).abs();
            }
            scores.push(prior);
        }
        let mut hid = self.scorer.apply(&features);
        hid.par_iter_mut().for_each(|v| *v = gelu_erf(*v));
        let [s0, s1, s2] = self.scales;
        for o in 0..no {
            let residual = dot(&hid[o * w..(o + 1) * w], &self.scorer_out) + self.scorer_out_b;
            scores[o] = scores[o] * s0 + (cosines[o] * s1 + residual) * s2;
        }
        if scores.iter().any(|s| !s.is_finite()) {
            bail!("the decision head produced a non-finite score");
        }
        Ok(scores)
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    // Eight partial sums so the loop vectorizes.
    let mut acc = [0f32; 8];
    let ((ca, ra), (cb, rb)) = (a.as_chunks::<8>(), b.as_chunks::<8>());
    let tail: f32 = ra.iter().zip(rb).map(|(x, y)| x * y).sum();
    for (x, y) in ca.iter().zip(cb) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    acc.iter().sum::<f32>() + tail
}

fn add(x: &mut [f32], y: &[f32]) {
    for (a, b) in x.iter_mut().zip(y) {
        *a += b;
    }
}

fn softmax(x: &mut [f32]) {
    let max = x.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
    let mut sum = 0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    x.iter_mut().for_each(|v| *v /= sum);
}

/// `x / max(‖x‖, 1e-12)`
fn l2_normalize(x: &mut [f32]) {
    let n = dot(x, x).sqrt().max(1e-12);
    x.iter_mut().for_each(|v| *v /= n);
}

fn gelu_erf(x: f32) -> f32 {
    0.5 * x * (1.0 + libm::erff(x * std::f32::consts::FRAC_1_SQRT_2))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spans_follow_runs_of_tags() {
        use Order::*;
        let order = [None, QuestionChoice, QuestionChoice, None, Option, Option, None, Option, QuestionNoul, Option, Option, None, Option];
        let (qs, os) = spans(&order).unwrap();
        assert_eq!(qs, vec![QuestionSpan { kind: 1, start: 1, end: 3 }, QuestionSpan { kind: 0, start: 8, end: 9 }]);
        assert_eq!(
            os,
            vec![
                OptionSpan { question: 0, start: 4, end: 6 },
                OptionSpan { question: 0, start: 7, end: 8 },
                OptionSpan { question: 1, start: 9, end: 11 },
                OptionSpan { question: 1, start: 12, end: 13 },
            ]
        );
    }

    #[test]
    fn spans_reject_bad_layouts() {
        use Order::*;
        assert!(spans(&[None, None]).is_err());
        assert!(spans(&[Option, QuestionNoul, Option]).is_err());
        assert!(spans(&[QuestionNoul, Option, QuestionScore]).is_err());
    }

    #[test]
    fn gelu_matches_known_values() {
        assert!((gelu_erf(1.0) - 0.841_344_7).abs() < 1e-6);
        assert!((gelu_erf(-2.0) + 0.045_500_26).abs() < 1e-6);
    }
}
