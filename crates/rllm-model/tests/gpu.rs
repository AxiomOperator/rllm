//! GPU checks against a real model. Skipped unless `RLLM_TEST_MODEL` names a
//! GGUF file (and a CUDA device is present):
//!   RLLM_TEST_MODEL=models/Qwen3-0.6B-Q8_0.gguf cargo test --release -p rllm-model --test gpu

use rllm_cuda::CudaBackend;
use rllm_gguf::Gguf;
use rllm_core::KvType;
use rllm_model::{AnyModel, Generator};

/// Held by each test: they run one at a time. (Two backends recording
/// CUDA graphs at once in one process disturb each other's decode steps;
/// the binaries only ever have one.)
static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn setup() -> Option<(CudaBackend, Gguf)> {
    let path = std::env::var_os("RLLM_TEST_MODEL")?;
    let g = Gguf::open(std::path::Path::new(&path)).expect("open RLLM_TEST_MODEL");
    Some((CudaBackend::new(0).expect("CUDA device"), g))
}

/// Pseudo-random token ids below `vocab`.
fn tokens(n: usize, vocab: usize, seed: u64) -> Vec<u32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s % vocab as u64) as u32
        })
        .collect()
}

/// The same tokens give bit-identical logits every run (prefill matmuls
/// split along k sum their parts in a fixed order).
#[test]
fn prefill_is_deterministic() {
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let Some((b, g)) = setup() else { return };
    let model = AnyModel::load(&b, &g).unwrap();
    let prompt = tokens(700, rllm_model::Model::vocab_size(&model), 7);
    // Small chunks split k the most (few output tiles to spread over the SMs).
    let run = || {
        let mut g = Generator::new(&b, &model, 1024, 16).unwrap();
        let mut out = Vec::new();
        for (part, chunk) in prompt.chunks(350).zip([350, 40]) {
            out.extend(g.feed_all(part, chunk).unwrap());
        }
        // Decode steps (CUDA graphs after the first two).
        for t in 0..6 {
            out.extend(g.feed(&[t], 1).unwrap());
        }
        out
    };
    let first = run();
    for _ in 0..3 {
        let again = run();
        let diff = first.iter().zip(&again).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        assert_eq!(diff, 0, "{diff} of {} logits differ between runs", first.len());
    }
}

/// A step of several tokens of one sequence (how speculative decoding
/// verifies drafts) gives the same logits as feeding them one at a time, so
/// greedy output doesn't change with a draft model. Bit for bit on Qwen3;
/// Gemma 4's single-token steps use fused kernels of their own, so there
/// only closely (like batched requests vs one alone).
#[test]
fn verification_steps_match_single_steps() {
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let Some((b, g)) = setup() else { return };
    let model = AnyModel::load(&b, &g).unwrap();
    let vocab = rllm_model::Model::vocab_size(&model);
    let prompt = tokens(300, vocab, 3);
    let next = tokens(6, vocab, 4);
    for kv in [KvType::F16, KvType::Q8] {
        let mut one = Generator::with_kv(&b, &model, 1024, 16, kv).unwrap();
        let mut many = Generator::with_kv(&b, &model, 1024, 16, kv).unwrap();
        one.feed(&prompt, 512).unwrap();
        many.feed(&prompt, 512).unwrap();
        let singles: Vec<f32> = next.iter().flat_map(|&t| one.feed(&[t], 1).unwrap()).collect();
        let batched = many.feed_all(&next, 6).unwrap();
        let diff = singles.iter().zip(&batched).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        let scale = singles.iter().fold(0f32, |m, x| m.max(x.abs()));
        let worst = singles.iter().zip(&batched).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max) / scale;
        let top1 = singles
            .chunks_exact(vocab)
            .zip(batched.chunks_exact(vocab))
            .filter(|(a, b)| rllm_model::sampler::argmax(a) == rllm_model::sampler::argmax(b))
            .count();
        if g.get_str("general.architecture").unwrap() == "qwen3" {
            assert_eq!(diff, 0, "{kv}: {diff} of {} logits differ", singles.len());
        } else {
            // (Random tokens: near-ties may flip, so only the size is checked.)
            assert!(worst < 0.05, "{kv}: worst relative difference {worst}, top-1 {top1}/{}", next.len());
        }
    }
}

/// A q8 KV cache stays close to f16 on real text: the same next-token
/// prediction nearly everywhere and a tiny KL divergence (prefill rows, then
/// decode steps).
#[test]
fn q8_cache_tracks_f16() {
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let Some((b, g)) = setup() else { return };
    let model = AnyModel::load(&b, &g).unwrap();
    let vocab = rllm_model::Model::vocab_size(&model);
    let prompt = rllm_tokenizer::Tokenizer::from_gguf(&g).unwrap().encode(TEXT).unwrap();
    assert!(prompt.len() > 520, "{} tokens", prompt.len());
    let run = |kv| {
        let mut g = Generator::with_kv(&b, &model, 1024, 16, kv).unwrap();
        let mut out = g.feed_all(&prompt, 512).unwrap();
        for t in 0..8 {
            out.extend(g.feed(&[t], 1).unwrap());
        }
        out
    };
    let (a, q) = (run(KvType::F16), run(KvType::Q8));
    let rows = a.len() / vocab;
    let (mut agree, mut kl_sum) = (0, 0f64);
    for (ra, rq) in a.chunks_exact(vocab).zip(q.chunks_exact(vocab)) {
        agree += (rllm_model::sampler::argmax(ra) == rllm_model::sampler::argmax(rq)) as usize;
        kl_sum += kl_divergence(ra, rq);
    }
    let kl_mean = kl_sum / rows as f64;
    eprintln!("q8 vs f16: top-1 {agree}/{rows}, mean KL {kl_mean:.2e}");
    // Near-ties can flip; on average the distributions barely move.
    // (Measured 97-99% and KL 1e-3 to 4e-3 on Qwen3 and Nemotron.)
    assert!(agree * 100 >= rows * 95, "top-1 agreement {agree}/{rows}");
    assert!(kl_mean < 1e-2, "mean KL divergence {kl_mean}");
}

/// KL(p || q) of softmax(p_logits), softmax(q_logits), in nats.
fn kl_divergence(p: &[f32], q: &[f32]) -> f64 {
    let log_softmax = |x: &[f32]| {
        let max = x.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v)) as f64;
        let lse = x.iter().map(|&v| (v as f64 - max).exp()).sum::<f64>().ln() + max;
        x.iter().map(|&v| v as f64 - lse).collect::<Vec<f64>>()
    };
    let (lp, lq) = (log_softmax(p), log_softmax(q));
    lp.iter().zip(&lq).map(|(a, b)| a.exp() * (a - b)).sum::<f64>().max(0.0)
}

/// Varied English prose (written for this test), long enough for a full
/// 512-token prefill chunk plus more.
const TEXT: &str = "The harbor town woke slowly in winter. Fishing boats sat in rows along the stone quay, their nets folded and their engines cold, while gulls argued over scraps near the market hall. By seven the bakery had opened its shutters, and the smell of bread drifted down to the water where an old ferry waited for the first passengers of the day.

Most of the people who lived there had grown up with the sea as a neighbor rather than a view. They knew which winds brought storms and which only rattled the windows, and they could tell from the color of the water whether the afternoon would turn. Children learned to swim in the cold bay before they learned to ride bicycles, and every family kept a story about a relative who had been lost to the waves or saved by them.

In the nineteenth century the town had been famous for its shipyards. Wooden hulls rose on the slipways each spring and slid into the harbor each autumn, carrying timber and salt to ports across the northern sea. When steel replaced oak, the yards closed one after another, and the workers moved inland to the factories or stayed behind to mend nets and paint houses. The last yard became a museum, and its great crane still stands above the rooftops like the skeleton of some enormous bird.

Today the economy depends on a mixture of fishing, tourism and a small technology company that designs sensors for monitoring ocean currents. Its engineers spend half their time at desks and half on the water, dropping instruments over the side of a converted trawler and hauling them back weeks later, encrusted with barnacles and full of data. The measurements help climate scientists understand how heat moves through the ocean, and they also tell local captains where the cod are likely to gather.

Visitors come in summer for the beaches, the seafood and the long evenings when the sun barely sets. They walk the coastal path to the lighthouse, photograph the painted doors of the old quarter, and argue about whether the smoked herring at the harbor stall is better than the one sold by the station. In the evenings the pubs fill with music: fiddles, accordions and songs about sailors who never came home, sung loudly by people who have never been to sea.

Winter belongs to the residents again. The ferry runs twice a day instead of six times, the guesthouses close, and the town council meets in the library to discuss the budget, the sea wall and the perennial question of parking. Someone always proposes a new festival to bring tourists in the dark months; someone else always points out that the last one was cancelled because of a storm. The meeting ends, as it usually does, with tea, biscuits and an agreement to revisit the matter in spring.

Mathematics teachers at the local school use the harbor as a classroom. Students calculate tides from the phases of the moon, estimate the volume of the breakwater from its dimensions, and model the growth of the mussel farms with simple equations. One year a class measured the speed of waves entering the bay and discovered, to their delight, that their results matched a formula published more than a century earlier by a scientist who had never visited the town.";
