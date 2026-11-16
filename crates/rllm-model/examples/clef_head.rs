//! Run a Clef decision head on synthetic hidden states and write its inputs
//! and scores, for checking against an independent implementation:
//!
//! `cargo run --release -p rllm-model --example clef_head -- model.gguf out_dir`
//!
//! Writes `hidden.f32` (`[n, hidden]`), `tokens.u32`, `order.u8` and
//! `scores.f32` (one per option), little-endian.

use anyhow::{Context, Result};
use rllm_model::clef::{ClefHead, Order};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let (model, out) = (args.get(1).context("model path")?, std::path::Path::new(args.get(2).context("output dir")?));
    let g = rllm_gguf::Gguf::open(model)?;
    let t = std::time::Instant::now();
    let head = ClefHead::load(&g)?;
    eprintln!("head loaded in {:.1} s", t.elapsed().as_secs_f64());

    // Prompt layout: state, a choice question with 3 options, a noul
    // question with 2, a score question with 4, then a tail.
    use Order::*;
    let mut order = vec![None; 40];
    let push = |o: Order, n: usize, order: &mut Vec<Order>| order.extend(std::iter::repeat_n(o, n));
    push(QuestionChoice, 7, &mut order);
    for n in [3, 5, 2] {
        push(None, 4, &mut order);
        push(Option, n, &mut order);
    }
    push(None, 6, &mut order);
    push(QuestionNoul, 5, &mut order);
    for n in [3, 3] {
        push(None, 4, &mut order);
        push(Option, n, &mut order);
    }
    push(QuestionScore, 9, &mut order);
    for n in [1, 2, 4, 3] {
        push(None, 3, &mut order);
        push(Option, n, &mut order);
    }
    push(None, 12, &mut order);
    let n = order.len();

    // Deterministic pseudo-random hidden states (unit RMS, like the
    // backbone's normalized output) and token ids.
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let d = head.hidden_size();
    let hidden: Vec<f32> = (0..n * d).map(|_| ((next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 3.4).collect();
    let tokens: Vec<u32> = (0..n).map(|_| (next() % 150_000) as u32).collect();

    let t = std::time::Instant::now();
    let scores = head.scores(&hidden, &tokens, &order)?;
    eprintln!("{n} tokens scored in {:.3} s: {scores:?}", t.elapsed().as_secs_f64());

    std::fs::create_dir_all(out)?;
    let f32s = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    std::fs::write(out.join("hidden.f32"), f32s(&hidden))?;
    std::fs::write(out.join("tokens.u32"), tokens.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
    std::fs::write(out.join("order.u8"), order.iter().map(|&o| o as u8).collect::<Vec<u8>>())?;
    std::fs::write(out.join("scores.f32"), f32s(&scores))?;
    Ok(())
}
