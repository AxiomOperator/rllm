# Model architectures and decoding techniques an inference engine needs (state as of 2026-11-11)

Scope: (a) which open model architectures have real demand in late 2026 and what each needs from the engine; (b) speculative and decoding techniques, with measured speedups and implementation cost. Written for rllm (Rust, GGUF, single GPU, CUDA + HIP; dense Qwen3, Llama 2/3, Mistral, Gemma 4; separate-draft speculation and prompt lookup).

Source note: the 2026 web is full of SEO and aggregator pages. Primary sources (arXiv, model cards, vendor blogs, vLLM docs) are marked as such. Community benchmark numbers are labelled. The research ran 17 tool calls, so some items are listed as gaps rather than verified.

## 1. Which open architectures are most used in late 2026, and what does each need from the engine?

### Takeaway
The model list in the brief is out of date. By late 2026, demand centres on **Qwen3.5/3.6/3.8**: hybrid Gated DeltaNet (linear attention) plus gated full attention, mostly MoE, with native MTP and vision. Next come **Gemma 4** (dense 31B, MoE 26B-A4B, edge E2B/E4B, with MTP drafters), **gpt-oss** (MoE, MXFP4 experts, attention sinks, SWA) and, for API and large hosts, **DeepSeek V4** (Flash 284B-A13B and Pro 1.6T-A49B; new compressed and sparse attention), **Kimi K2.6** and **GLM-5.x**. Plain dense Qwen3 still leads raw downloads at small sizes. For rllm, the gating features are **MoE (routing + grouped GEMM)**, **Gated DeltaNet / SSM-style recurrent layers with state rollback for speculation**, **MTP heads**, and **vision encoders** (Qwen3-VL is the #2 download).

### Cited Findings
**Demand signals**
- Hugging Face, 30-day downloads as of 2026-09-18. Top LLM/VLM rows: Qwen3-0.6B 22.5M; **Qwen3-VL-8B-Instruct 19.1M**; Qwen3-8B 13.0M; unsloth Qwen3-Coder-30B-A3B GGUF 12.8M; **Qwen3.6-35B-A3B-FP8 10.2M**; **gemma-4-26B-A4B-it 9.8M**; Qwen2.5-7B-Instruct 9.7M; **Qwen3.5-9B 9.3M**; **gemma-4-31B-it 9.0M**; nvidia Qwen3.6-35B-A3B-NVFP4 8.6M; **Qwen3.8-27B (FP8 7.6M + BF16 7.5M)**. Family totals in the top 60: Qwen 240.1M (36 rows), Gemma 32.6M, Llama 12.9M, gpt-oss 11.9M, DeepSeek 4.4M, Nemotron 3.5M. — [Digital Applied (HF Hub API snapshot, secondary)](https://www.digitalapplied.com/blog/most-downloaded-open-ai-models-hugging-face-september-2026)
- Downloads skew to small models: one analysis says models under 1B parameters took 83% of all-time downloads, another says about 75% for models under 10B. Hugging Face's spring 2026 report put Chinese open models at 41% of downloads. — [HF State of Open Source, Spring 2026](https://huggingface.co/blog/huggingface/state-of-os-hf-spring-2026); [tun.com summary](https://www.tun.com/home/hugging-faces-2026-open-model-report-qwen-leads-hype-vs-reality/) (secondary)
- OpenRouter, top 10 by tokens in June 2026: DeepSeek V4 Flash 10.9T (#1), Tencent Hy3 Preview 10.7T, DeepSeek V4 Pro 4.54T, DeepSeek V3.2 4.31T, **Kimi K2.6 3.72T**, Nemotron 3 Super 2.65T, plus closed Claude and Gemini models. — [OfficeChai, June 2026 (secondary)](https://officechai.com/miscellaneous/these-are-the-most-popular-ai-models-on-openrouter-june-2026/). This is five months old; check the live [OpenRouter rankings](https://openrouter.ai/rankings).
- Per-app OpenRouter pages also show "GLM 5.3 Flash" and DeepSeek V4 Flash among top models. — [OpenRouter app page](https://openrouter.ai/apps/url/https%3A%2F%2Frouter.systemsx.cloud%2F) (single app, weak signal)

**Qwen3.5 / 3.6 / 3.8: hybrid linear attention + MoE + MTP + vision**
- Qwen3.6-35B-A3B has 40 layers in 10 cycles of [3 × Gated DeltaNet + 1 × Gated Attention], each followed by an MoE FFN.
  - DeltaNet layers: 32 V heads and 16 QK heads.
  - Gated Attention layers: 16 Q heads and 2 KV heads, head dimension 128.
  - MoE: 256 experts, 8 routed + 1 shared per token, ~3B active.
  - Context: 262,144 tokens native, about 1.01M extended.
  - Input: image and video. Trained with MTP.
  - — [ApX model page](https://apxml.com/models/qwen36-35b-a3b); [vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.6-35B-A3B?variant=default) (the recipe says it shares the Gated-DeltaNet MoE architecture of Qwen3.5); [lilting.ch](https://lilting.ch/en/articles/qwen36-35b-a3b-agentic-coding-moe-hybrid). The official Qwen card was not fetched directly.
- In llama.cpp, MTP on hybrid models needed a separate PR (#22400). It adds a `keep_intermediates` path for GDN/SSM layers so speculative decoding can roll back a partially accepted draft. Per one report, "without this, MTP doesn't function on hybrid-attn models (27B)". — [llama.cpp PR search results / Reddit snapshot](https://reddit.sentinel-team.org/posts/1t65vl8/snapshots/2026-05-09T00%3A46%3A53.937033Z) (community)

**DeepSeek V4: new attention design, not plain MLA**
- V4-Pro is 1.6T total / 49B active; V4-Flash is 284B / 13B. Both support 1M context.
- Attention combines Compressed Sparse Attention (compressed KV plus top-k selection through a "Lightning Indexer") with Heavily Compressed Attention (dense attention over a heavily compressed KV).
- Against V3.2 at 1M context, V4-Pro needs about 27% of the per-token FLOPs and 10% of the KV cache. — [hyper.ai paper summary](https://hyper.ai/en/papers/DeepSeek-V4); [alphaXiv overview](https://www.alphaxiv.org/overview/deepseek-v4)
- Per-layer mix of dense MLA and CSA. Also: Hyper-Connections (mHC, multi-stream residual), hash-based routing in early MoE layers, and an MTP head. — [NVIDIA Megatron Bridge docs](https://docs.nvidia.com/nemo/megatron-bridge/latest/models/deepseek/deepseek-v4.html)

**gpt-oss: MoE, MXFP4, sinks, SWA**
- Alternates full attention and sliding-window attention (window 128) 1:1. A learned attention sink gives a global view.
- Only the MoE weights are MXFP4; attention and other layers are BF16.
- Unsloth found the reference window covers the last N tokens, not N+1, so the mask comparison must be strict (`<`). — [Unsloth docs](https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune/long-context-gpt-oss-training); [vLLM gpt-oss blog](https://vllm.ai/blog/2025-08-05-gpt-oss); [Kaitchup](https://kaitchup.substack.com/p/openai-gpt-oss-native-4-bit-moe-models)
- llama.cpp needs `--swa-full` to keep a full KV cache for SWA layers when prefix reuse is wanted. — [ik_llama.cpp discussion #758](https://gh.hhj520.top/ikawrakow/ik_llama.cpp/discussions/758)

**Gemma 4**
- Google released MTP drafters (Apache 2.0) on 2026-05-05 for 31B dense, 26B MoE and E2B/E4B.
- The drafters "utilize the target model's activations and share its KV cache".
- On E2B/E4B, the embedder uses clustering because the final logit computation becomes the bottleneck. — [Google blog (primary)](https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/)

**MoE on unified memory (Strix Halo)**
- Strix Halo, llama.cpp, community numbers (backend and quant vary):
  - Qwen3-30B-A3B: ~72–100 tok/s decode (100.04 tok/s at IQ4_XS on RADV Vulkan).
  - gpt-oss-120b MXFP4: 34.1 tok/s (Hardware Corner) to 53–56 tok/s (Vulkan).
  - Dense 70B: ~5 tok/s.
  - — [Hardware Corner](https://www.hardware-corner.net/strix-halo-llm-optimization/); [strix-halo-llm-perf](https://github.com/visorcraft/strix-halo-llm-perf); [Framework community](https://community.frame.work/t/ryzen-ai-max-not-so-much/77999)

### Inferences
- **Demand vs. coverage.** rllm's current coverage (dense Qwen3, Llama, Mistral, Gemma 4) matches the small dense end, which dominates downloads. The models people actually run on a 128 GB Strix Halo or a 48 GB Ada are now MoE: Qwen3.6-35B-A3B, Qwen3-Coder-30B-A3B, Gemma-4-26B-A4B and gpt-oss-20b/120b. Many of these are also hybrid (Qwen3.5+).
- **Priority order for rllm:**
  1. **MoE FFN** (top-k router with softmax or sigmoid, shared expert, grouped/indirect GEMM over quantized experts, an MXFP4 weight type for gpt-oss). This unlocks Qwen3-30B-A3B, Qwen3-Coder, Gemma-4-26B-A4B, gpt-oss and Qwen3.6.
  2. **Gated DeltaNet layers.** A chunked parallel scan is needed for prefill and a recurrent update for decode. The fixed-size recurrent state must be snapshotted per draft position so speculation can roll back. Prefix caching only works with checkpointed states, because recurrent state cannot be sliced the way paged KV can. This unlocks Qwen3.5/3.6/3.8, the most-downloaded new generation.
  3. **Attention sinks + SWA mask** for gpt-oss (small; rllm already has SWA for Gemma).
  4. **Vision encoder** (Qwen3-VL ViT + M-RoPE; Gemma vision).
  5. **MLA / DeepSeek-V4 CSA.** These models are large (≥284B), so they are mostly datacenter or multi-GPU. Low priority for single GPU, except on 128 GB unified-memory boxes with aggressive quantization.
- **Why MoE suits unified memory.** Decode speed is bound by bytes read per token. With ~3B active parameters, Qwen3-30B-A3B reads about as much per token as a 3B dense model, while its whole 18–20 GB fits in 128 GB of unified memory. A dense model of the same quality reads 10× more per token. That is why the community gets ~75–100 tok/s for 30B-A3B against ~5 tok/s for dense 70B on Strix Halo. Expert CPU offload matters less on unified memory, where there is no PCIe hop, and more on discrete GPUs (llama.cpp `--n-cpu-moe` / `-ot` overrides).

### Gaps
- No primary OpenRouter ranking newer than June 2026 was found. Exact late-2026 open-model API share is unverified.
- DeepSeek V4 support status in vLLM, SGLang and llama.cpp was not confirmed.
- GLM-5.x and Kimi K2.6 architecture details were not fetched. Kimi K2 is DeepSeek-V3-style MLA MoE, and GLM-4.5 had native MTP, from prior knowledge; neither was verified this session.
- Llama 4, Phi-4, Mistral/Magistral and Granite 4 (hybrid Mamba) were not researched. None appears in the top-60 download snapshot, apart from Llama at 12.9M total. Marked as low demand, unverified.
- No source found for llama.cpp `--n-cpu-moe` performance numbers.
- Qwen3.8 architecture was not verified. It is assumed to follow the 3.5/3.6 hybrid design.

## 2. MoE inference techniques

### Takeaway
What the engine needs is a fused MoE path: router → token sort/permute by expert → grouped GEMM over quantized experts → weighted unpermute, plus a shared expert. Decode at batch 1 is a gather-matvec over k experts. On unified-memory machines like Strix Halo, MoE turns spare capacity into speed. Speculation on MoE is less effective, because each draft token activates different experts and so adds new weight reads.

### Cited Findings
- A Core AI port of Qwen3.6-35B-A3B reports a 2.1× gain from a dedicated `gather_qmm` (gathered quantized matmul) kernel. — [HF model card](https://huggingface.co/mlboydaisuke/Qwen3.6-35B-A3B-CoreAI) (community)
- Google reports routing problems for the Gemma 4 26B MoE drafter at batch 1 on Apple Silicon. Batches of 4–8 reach up to ~2.2× locally. — [Google blog](https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/)
- One M1 Max tester saw only +13% from MTP on 26B-A4B, and a slowdown on 31B dense and E4B. — [lilting.ch](https://lilting.ch/en/articles/gemma-4-mtp-drafter-speed) (community)
- DFlash on Qwen3.5-35B-A3B gives up to 2.8× at concurrency 1, against up to 5.01× on dense Qwen3.5-9B. — [z-lab Qwen3.5-35B-A3B-DFlash card](https://huggingface.co/z-lab/Qwen3.5-35B-A3B-DFlash); [z-lab Qwen3.5-9B-DFlash card](https://huggingface.co/z-lab/Qwen3.5-9B-DFlash)

### Inferences
- Speculative verification on MoE reads the union of experts across k+1 tokens, so its cost grows with draft length more than it does on dense models. Expect smaller speedups than dense, as the DFlash 2.8× vs 5.0× pair suggests. rllm's adaptive draft length should take the expert-union size into account.
- Expert parallelism and multi-GPU are irrelevant to rllm's single-GPU scope today.

### Gaps
- No primary benchmark found for vLLM fused_moe / SGLang MoE kernel throughput on consumer GPUs.

## 3. Speculative decoding state of the art: speedups, acceptance, batching, support

### Takeaway
The field has moved from separate draft models to **heads that reuse the target's hidden states and KV cache**: EAGLE-3, native MTP (Qwen3.5+, DeepSeek, GLM, Gemma 4 drafters) and, newest, **DFlash** block-diffusion drafters. These give 2–5× at batch 1, against ~1.5× for a separate small draft model.
- **llama.cpp merged MTP on 2026-05-16** and reports 1.81× (Q4_K_M) and 2.44× (Q8_0) for Qwen3.6-27B on Strix Halo at temperature 0. That directly addresses rllm's weak +18% there.
- Gains shrink with temperature and batch size but stay positive up to batch ~64 for EAGLE-3 in SGLang.
- Model-free **suffix decoding** is the best method for agentic and repetitive workloads.

### Cited Findings
**EAGLE-3 (primary: [arXiv 2503.01840 v3](https://arxiv.org/html/2503.01840v3))**
- Mean speedup / acceptance length τ over 5 tasks (MT-bench, HumanEval, GSM8K, Alpaca, CNN/DM):

| Model | T=0, EAGLE-2 | T=0, EAGLE-3 | T=1, EAGLE-2 | T=1, EAGLE-3 |
|---|---|---|---|---|
| Llama-3.1-8B-Instruct | 3.23× / 4.11 | 4.44× / 6.23 | 2.80× / 3.62 | 3.45× / 4.92 |
| Llama-3.3-70B-Instruct | 2.85× / 3.78 | 4.12× / 5.88 | 2.65× / 3.54 | 3.95× / 5.66 |
| DeepSeek-R1-Distill-Llama-8B | 3.26× / 3.92 | 4.16× / 5.84 | 2.77× / 3.54 | 3.52× / 4.89 |
| Vicuna-13B | 4.22× / 4.83 | 5.51× / 6.62 | 3.76× / 4.41 | 4.65× / 5.67 |

- **Temperature 1 costs about 20% of the T=0 speedup on 8B models** (4.44× → 3.45× for Llama-3.1-8B).
- The method moves from feature prediction to token prediction, fuses features from several layers, and adds "training-time test". The tree depth went from 6 to 8, with the same 10-node expansion as EAGLE-2.
- Batching, Llama-3.1-8B on MT-Bench, chain drafts with no tree:
  - SGLang on H100, chain length 3: EAGLE-3 gives 1.81× at batch 2, 1.48× at 16, 1.32× at 32 and 1.38× at 64. EAGLE-1 falls to 0.88–0.99× at 24+.
  - vLLM, max chain 2: 1.75× at batch 2, 1.49× at 16, 1.36× at 32, 1.21× at 48 and 1.01× at 56.
  - The vLLM table's caption says A100 while the text says RTX 3090; the paper is inconsistent here.
- **EAGLE-3 head availability.** Qwen3 long lacked public EAGLE-3 drafts. SGLang's SpecForge (arXiv 2603.18567, 2026) released "SpecBundle" EAGLE-3 drafts for mainstream models, reporting up to 4.48× end-to-end on SGLang and 9.9× faster EAGLE-3 training for Qwen3-235B-A22B. — [SpecForge](https://arxiv.org/html/2603.18567)
- Mismatched EAGLE-3 drafts can collapse: Qwen2.5-VL/Qwen3-VL in SGLang reached τ ≈ 1.04, against 2.31 for Llama 3. — [SpecForge issue #310](https://gh.r-mj.com/sgl-project/SpecForge/issues/310)
- Red Hat Speculators reports 2.7× for Qwen3-32B (math, 2× A100). — [PremAI blog (secondary)](https://blog.premai.io/speculative-decoding-2-3x-faster-llm-inference-2026)

**Native MTP**
- llama.cpp PR #22673 "llama + spec: MTP Support" was merged on 2026-05-16, enabled with `--spec-type draft-mtp`. Recommended `--spec-draft-n-max 2`, with an optional q8_0 draft KV cache. A follow-up PR #23269 added refinements. — [SpecPicks](https://specpicks.com/reviews/llama-cpp-mtp-support-landed-qwen3-6-27b-at-2-44-on-a-strix-halo-2-17-on-a-rtx-3090-rig-news) (secondary); [llama.cpp PR #23269](https://github.com/ggml-org/llama.cpp/pull/23269)
- Measured with llama.cpp MTP on Qwen3.6-27B, single stream, temperature 0, median of 5 runs:

| Hardware | Quant | No MTP | MTP | Speedup |
|---|---|---:|---:|---:|
| Strix Halo, ROCm 7.0.2 | Q4_K_M | 11.7 tok/s | 21.2 tok/s | **1.81×** |
| Strix Halo, ROCm 7.0.2 | Q8_0 | 7.4 tok/s | 18.1 tok/s | **2.44×** |
| RTX 3090 | Q4_K_M | 38.7 tok/s | 59.5 tok/s | 2.17× |
| M2 Max | — | — | — | slower |

  - Draft acceptance: 79–89% on code, 62–70% on factual and structured text.
  - Early builds had a prompt-processing regression, later patched. — [SpecPicks](https://specpicks.com/reviews/llama-cpp-mtp-support-landed-qwen3-6-27b-at-2-44-on-a-strix-halo-2-17-on-a-rtx-3090-rig-news)
- Known llama.cpp issues: prompt processing ~14% slower with MTP enabled, and "parallel decoding with MTP enabled still remains highly suboptimal". — search summary of the llama.cpp PR thread / [dasroot.net](https://dasroot.net/posts/2026/05/llama-cpp-performance-surge-beta-mtp-support/) (secondary)
- GLM-5.2 (GLM_DSA) NextN/MTP is supported in a later llama.cpp release. Gemma 4 MTP is a WIP PR. — [llama.cpp release b10174 (newreleases)](https://newreleases.io/project/github/ggml-org/llama.cpp/release/b10174); [llama.cpp PR #23398](https://github.com/ggml-org/llama.cpp/pull/23398)
- vLLM's Ascend docs note that DeepSeek's MTP exposes only one layer of weights, so `num_speculative_tokens` > 1 (especially ≥ 3) is not guaranteed to help. — [vLLM Ascend spec-decode guide](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/speculative_decoding.html)
- Gemma 4 MTP drafters: up to 3× claimed; ~2× tok/s for 26B on an RTX PRO 6000. Tested on LiteRT-LM, MLX, Transformers and vLLM, with guidance for SGLang and Ollama. — [Google blog](https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/)

**DFlash (block diffusion drafter, primary: [arXiv 2602.06036](https://arxiv.org/pdf/2602.06036))**
- A lightweight block-diffusion drafter proposes a whole block in parallel, conditioned on the target's hidden features. Claims >6× lossless and ~2.5× better than EAGLE-3 on Qwen3-8B.
- vLLM, Qwen3.5-9B, concurrency 1 (paper Table 12): 4.0× Math500, 4.6× HumanEval, 3.0× MT-Bench.
- Model cards: Qwen3.5-9B up to 5.01× at concurrency 1 and 2.58× at concurrency 32. Qwen3.5-35B-A3B up to 2.8×, which the card says beats MTP. Block size 8 is recommended for concurrent serving and 16 for concurrency 1.
- vLLM support was via PR #40898, not confirmed merged. Drafts are published for Qwen3.5-4B/9B/27B/35B-A3B/122B-A10B and Qwen3.6-35B-A3B. — [z-lab Qwen3.5-9B-DFlash](https://huggingface.co/z-lab/Qwen3.5-9B-DFlash); [z-lab Qwen3.6-35B-A3B-DFlash](https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash); [z-lab Qwen3.5-35B-A3B-DFlash](https://huggingface.co/z-lab/Qwen3.5-35B-A3B-DFlash)
- The vLLM v0.20.2 docs list `method` values draft_model, ngram, suffix, mtp, eagle3 and **dflash**, which suggests DFlash did land upstream. This conflicts with the model cards' unmerged-PR note. — [vLLM v0.20.2 docs](https://docs.vllm.ai/en/v0.20.2/features/speculative_decoding/)

**Suffix decoding / n-gram**
- SuffixDecoding (arXiv 2411.04975) uses suffix trees over the prompt and earlier generations, with frequency-based, adaptive-length speculation.
  - SWE-Bench with OpenHands on vLLM (4× H100, TP4, concurrency 8): 1.8–4.5×, with the score unchanged at 37.2%.
  - v3 of the paper: up to 5.3× on agentic tasks, ~2.8× faster than EAGLE-2/3. — [arXiv](https://arxiv.org/html/2411.04975v3)
- Snowflake (Dec 2025): integrated in vLLM through Arctic Inference. After optimization it beats vLLM's n-gram at every concurrency level; it had been worse at 64 before. — [Snowflake blog](https://www.snowflake.com/en/engineering-blog/suffixdecoding-arctic-inference-vllm/)
- Arctic Inference's combined stack: 4× on SWE-Bench agents and up to 2.8× on open-ended chat. — [Snowflake](https://www.snowflake.com/en/engineering-blog/fast-speculative-decoding-vllm-arctic) (stale: vLLM v0.8.4 era)

**vLLM method list (v0.27.0 docs)**
- EAGLE, MTP, draft model, PARD, MLP, N-gram, suffix decoding, hidden-state extraction, a custom proposer (experimental) and dynamic speculative decoding.
- The docs say model-based methods give the largest latency cut, while n-gram and suffix give modest gains without adding load at peak traffic. — [vLLM v0.27.0 docs](https://docs.vllm.ai/en/v0.27.0/features/speculative_decoding/)

### Inferences
- **Why rllm got only +18% on Strix Halo.**
  - The 0.6B Q8 draft (~0.6 GB) reads about 12% of the 8B Q4_K_M target's bytes (~5 GB) per draft token. Proposing 5 tokens costs about 60% of a target step before verification. On bandwidth-bound Strix Halo (~39 tok/s target), that overhead eats most of the gain at 61% acceptance and temperature ~1.
  - Heads that share the target's KV and hidden states cost about one transformer layer per draft token, roughly 1/36 of the target. llama.cpp's 1.8–2.4× on the same hardware is consistent with this.
  - Temperature 1 alone should cost ~20% of the speedup (EAGLE-3 table).
- **Recommended rllm order:**
  1. **Native MTP** for Qwen3.5/3.6/3.8 (and GLM/DeepSeek later). The weights ship in the model, and with chain length 1–3 no training is needed. Requires the hybrid-layer state rollback from §1.
  2. **EAGLE-3 heads for Qwen3 and Llama 3** (SpecBundle and other public heads). One decoder layer reads fused low/mid/high hidden states; chain drafting first, trees later. A GGUF or safetensors loader for the head is needed.
  3. **Gemma 4 MTP drafters**, which share the KV cache.
  4. **DFlash**, which may beat all of the above at batch 1 but is newer and needs non-causal block attention in the drafter.
  5. **Suffix decoding** to replace or extend prompt lookup: a suffix tree over the prompt plus a global cache of earlier outputs. This is cheap, with no model, and is the biggest win for agentic and code-edit traffic.
- **Batching.** rllm stops speculating at ≥5 decoders, which is conservative next to EAGLE-3's 1.3–1.5× at batch 16–64 in SGLang. Cheap heads justify keeping speculation on at higher batch sizes with chain length 1–2.
- **Tree vs. chain.** The EAGLE papers get their batch-1 headline numbers with trees (60 draft tokens, depth 6–8), but both SGLang and vLLM used chains for serving benchmarks. With CUDA graphs capped at 8 tokens per step, rllm should start with chains and consider small trees (≤16 nodes) only for batch 1.

### Gaps
- Primary numbers for Medusa, lookahead decoding and tree vs. chain under batching were not re-fetched this session. From prior knowledge, Medusa-2 claims ~2.3–3.6× and lookahead decoding ~1.5–2.3×; both are superseded by EAGLE-3 and MTP. Treat as stale and unverified.
- No independent reproduction of DFlash numbers was found. Merge status of vLLM PR #40898 is unresolved: the model cards say pending, while the v0.20.2 docs list `dflash`.
- No source gives EAGLE-3 or MTP speedups on AMD RDNA3 apart from the llama.cpp MTP Strix Halo numbers.
- No exact acceptance length for Qwen3.5/3.6 native MTP at temperature 1 was found.

## 4. Other latency techniques: KV cache compression, long context, reasoning-length control

### Takeaway
4-bit KV via Hadamard/WHT rotation (the TurboQuant line) is now mainstream. llama.cpp merged the rotation and vLLM reportedly merged a `turboquant_k8v4` dtype in April 2026. rllm already rotates keys for its q8 cache, so 4-bit KV is a short step. DeepSeek V4-style compressed and sparse attention, and hybrid linear attention, have moved long-context savings into the architecture itself.

### Cited Findings
- vLLM reportedly merged TurboQuant KV compression in April 2026 (PR #38479) as a `turboquant_k8v4` cache dtype, with fused Triton kernels. — [TurboQuant+ README](https://github.com/TheTom/turboquant_plus) (project README, secondary). An older Kaitchup article says vLLM had only FP8/INT8 KV, which is likely stale. — [Kaitchup](https://kaitchup.substack.com/p/turboquant-finally-fast-and-widely)
- llama.cpp merged Hadamard KV rotation (#21038). The dedicated TQ4_0 PR (#20995) was closed. A q4_0 cache with rotation approximates turbo4, reported at +0.23% perplexity vs q8_0. — [TurboQuant+ README](https://github.com/TheTom/turboquant_plus)
- AMD has productionized TurboQuant on ROCm with vLLM for KV-bound agentic inference. — [ROCm blog](https://rocm.blogs.amd.com/artificial-intelligence/turboquant-vllm-agentic/README.html)
- DeepSeek V4-Pro uses 10% of V3.2's KV cache at 1M tokens. — [hyper.ai](https://hyper.ai/en/papers/DeepSeek-V4)

### Inferences
- A `--kv-type q4` (rotated K, possibly K8/V4 like vLLM's) halves KV memory again. That matters most for long-context agent traffic on 48 GB GPUs; on 128 GB unified memory it matters less, but it also cuts KV bandwidth at long context.
- Hybrid models (3/4 of layers linear) cut KV by about 4× by design, so supporting them is itself a long-context optimization.

### Gaps
- Eviction methods (H2O, SnapKV, etc.) were not researched this session. No evidence was found that vLLM, SGLang or llama.cpp ship them by default as of 2026.
- No sources were gathered on reasoning-length control or thinking budgets (e.g. Qwen `thinking_budget`, gpt-oss `reasoning_effort`). Not verified.

## 5. Structured output state of the art (xgrammar, llguidance)

### Takeaway
Both leading engines compute token masks in tens of microseconds per token on typical JSON schemas, by their own benchmarks. xgrammar is the default in vLLM and SGLang, and llguidance is integrated in llama.cpp. Independent 2026 measurements disagree sharply. For rllm, llguidance (Rust, MIT) is the natural fit, since it replaces the hand-rolled JSON-only enforcer and adds full JSON Schema, regex and Lark/GBNF grammars.

### Cited Findings
- XGrammar: under 40 µs per token for mask generation on JSON schema, and up to 3× faster than the best baseline. — [arXiv 2411.15100](https://arxiv.org/pdf/2411.15100)
- llguidance: ~50 µs of CPU time per token for a 128k tokenizer. A full JSON mask can take ~1.5 ms, but a "slicer" brings the average under 50 µs on JSON Schema Bench (10k schemas). Integrated in llama.cpp. — [llama.cpp llguidance.md](https://huggingface.co/spaces/YZ-TAN/flask-llama/raw/main/llama.cpp/docs/llguidance.md)
- An independent 2026 paper (Parser Stack Classification, arXiv 2608.03065) on Qwen2.5 (151,665-token vocab) measured llguidance at ~1,408 µs and ~72 µs on two grammars, and XGrammar at ~649,000 µs and ~54,000 µs. Its own PSC method takes a few µs. This contradicts XGrammar's own numbers, and the setup was not checked. — [arXiv 2608.03065](https://arxiv.org/pdf/2608.03065)
- A blog reports schema compile time of 0.12–0.30 s for XGrammar and under 60 ms for Guidance, with XGrammar the default in vLLM and SGLang. — [tianpan.co (secondary)](https://tianpan.co/blog/2026/04/16/grammar-constrained-generation-output-reliability)
- The benchmark both projects use is JSONSchemaBench. — [arXiv 2501.10868](https://arxiv.org/pdf/2501.10868)

### Inferences
- Masks computed on the CPU overlap with the GPU forward pass, so at ~50 µs they are effectively free at rllm's ~7–25 ms decode steps. Compile time per new schema (tool lists change per request) matters more, which favours llguidance's low compile cost.
- Constrained requests currently skip speculation in rllm. Grammar-aware drafting (masking the draft and verification with the same matcher) is a follow-up; xgrammar and llguidance both support rollback and jump-forward.

### Gaps
- No primary head-to-head xgrammar vs llguidance benchmark from late 2026 was found. Treat "xgrammar-2" claims as unverified.
