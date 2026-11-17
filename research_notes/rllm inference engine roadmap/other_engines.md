# Other LLM inference engines (beyond vLLM / SGLang / llama.cpp): state as of Nov 2026, and what rllm could adopt

Scope: TensorRT-LLM, MLC-LLM, ExLlamaV3/EXL3 + TabbyAPI, LMDeploy TurboMind, mistral.rs, candle, TGI, Aphrodite, PowerInfer, MAX/Mojo, Lemonade. Research date 2026-11-11. Budget-limited pass (~16 tool calls); every claim is dated where the source gives a date. Items marked **[stale]** come from sources older than about 12 months. Items marked **[vendor]** are self-reported by the project or vendor.

rllm reference points, from /Storage/Projects/rllm/chart.md and compared.md (2026-10-30):
- Qwen3-8B Q4_K_M decodes at 150 tok/s on an RTX 6000 Ada (960 GB/s). With a file of about 5 GB that is about 750 GB/s, or about 78% of peak bandwidth.
- At 16–64 clients on 8B 4-bit, rllm reaches 0.53–0.60× vLLM-AWQ. That gap is the main serving weakness.

---

## Q1. Which quantization formats beat GGUF K-quants on quality per bit (EXL3/QTIP, AQLM, HQQ, NVFP4/MXFP4), and what do they cost in kernel speed?

### Takeaway
EXL3 (a QTIP-derived trellis format) clearly beats GGUF K-quants and IQ4_XS on KL divergence per bit at 3–4 bpw. At 5–6.5 bpw the gap closes.
- **Decode cost:** in the only controlled same-model test found, its batch-1 decode was 15–30% slower than llama.cpp at matched file size. Its prefill was 2.3–2.5× faster.
- **Other formats:** QTIP/AQLM-class codebooks matter mostly at 2–3 bits, and AQLM decodes slowly.
- **FP4:** NVFP4 and MXFP4 are mainly a compute win on Blackwell tensor cores, not a quality-per-bit win. One ICLR 2026 paper finds FP4 is "not an automatic upgrade over INT4".

### Cited Findings
- **EXL3 definition (2026):** EXL3 is "a streamlined variant of QTIP" (Cornell RelaxML). It uses a fused Viterbi kernel and computes Hessians on the fly. Converting a model takes "a couple of minutes for smaller models, up to a few hours for 70B+" on one consumer GPU. The KV cache can be quantized to 2–8 bits. The README showed v0.0.6 in its install example; the Atomic Chat test below used ExLlamaV3 1.4.9, so the README snapshot looks out of date. CUDA ≥12.4 only, with no ROCm mentioned. — [exllamav3 GitHub](https://github.com/turboderp-org/exllamav3)
- **Controlled EXL3 vs GGUF test (Atomic Chat, 2026-09-14)**
  - **Setup:** Gemma-4-12B-it on an RTX 5090. ExLlamaV3 1.4.9 + TabbyAPI against llama.cpp commit 8172e6577. Both stacks were scored with one harness against BF16 logits.
  - **Mean KL divergence by format (lower is better):**

    | Format | Total file | Mean KLD |
    |---|---:|---:|
    | EXL3 3.00 bpw | 6.52 GiB | 0.0940 |
    | EXL3 4.00 bpw | 7.79 GiB (5.79 GiB of layer weights) | 0.0248 |
    | EXL3 5.00 bpw | 9.06 GiB | 0.0095 |
    | GGUF IQ4_XS (4.25 bpw) | 5.94 GiB | 0.0544 |
    | GGUF Q4_K_M (4.80 bpw) | 6.63 GiB | 0.0399 |
    | GGUF UD-Q4_K_XL (4.89 bpw) | 6.86 GiB | 0.0361 |
    | GGUF Q5_K_M (5.65 bpw) | 7.84 GiB | 0.0172 |
    | GGUF Q6_K (6.56 bpw) | 9.11 GiB | 0.0092 |

  - **Batch-1 decode at 2k prompt:** EXL3 4.0 ran at 113 tok/s, against 139 for Q4_K_M and 155 for IQ4_XS. EXL3 5.0 ran at 99, against 117 for Q6_K.
  - **Prefill at 2k:** EXL3 ran at about 4,077 tok/s against about 1,762 for llama.cpp. At 8k the numbers were 5,199 against 4,097.
  - **Caveats:** EXL3 keeps embeddings in high precision, so compare the File column. The "MCG" codebook is a legacy variant; the current "mul1" codebook had lower KLD and faster decode. This is one model on one GPU.
  - Source: [Atomic Chat](https://atomic.chat/blog/guides/exl3-vs-gguf)
- **Other community comparisons (blogs, less controlled)**
  - Qwen3.8-27B at about 3 bits: EXL3 KLD 0.0435 against 0.0554 for Q3_K_M with imatrix. At about 14–15 GB a custom GGUF won, 0.0159 against 0.0234 for EXL3 at 3.5 bpw. — [efraingaray.com](https://efraingaray.com/en/blog/exllamav3-vs-llamacpp-16gb/)
  - A model-card comparison reported EXL3 4.05 bpw at KLD 0.0066, UD-IQ4_XS at 0.0162 and UD-Q4_K_XL (about 5.2 bpw) at 0.0084. The same source warns that EXL3 calibration and evaluation used the same dataset, so the numbers may be overfit. — [mratsim/GLM-4.7-EXL3 (HF)](https://huggingface.co/mratsim/GLM-4.7-EXL3) (via search summary; not fetched directly)
- **QTIP paper (NeurIPS 2024) [stale but foundational]**
  - 2-bit Llama-2-70B Wikitext2 perplexity: QTIP 3.87, FP16 3.12, QuIP# 4.16, AQLM 3.83. AQLM is described as having "a slower decoder".
  - QTIP's bitshift trellis lets different segments of the weight sequence decode in parallel.
  - PV-Tuning (built on AQLM) "inherits its slow inference speed".
  - Sources: [QTIP arXiv 2406.11235](https://arxiv.org/pdf/2406.11235) and the [theorempath summary](https://theorempath.com/papers/qtip)
- **Proteus (ICML 2026 poster):** a lookup-free trellis-coded quantization for 2-bit models. It reports lower decode bandwidth than TCQ baselines at comparable throughput (740 vs 1020 GB/s on 70B). — [ICML 2026](https://icml.cc/virtual/2026/poster/63948)
- **Q-Palette (arXiv 2509.20214):** fused trellis-coded quantization kernels exist only for a few integer bit widths (2, 3, 4). — [arXiv 2509.20214](https://arxiv.org/pdf/2509.20214v2)
- **FP4 (Egiazarian et al., arXiv 2509.23202, ICLR 2026)**
  - GPTQ works well with FP4, while rotation methods suit MXFP4 poorly.
  - Their MR-GPTQ brings MXFP4 within 1–2% of NVFP4. NVFP4 costs 4.5 bits per element against 4.25 for MXFP4.
  - NVINT4 can beat NVFP4 when Hadamard rotation is used.
  - On large models both formats recover 98–99% of FP16 accuracy.
  - Source: [arXiv 2509.23202](https://arxiv.org/abs/2509.23202)
- **Nominal and real bpw differ:** a file labeled Q4_K_M is really about 4.8–4.9 bpw (4.798 in the Atomic Chat table). — [Atomic Chat](https://atomic.chat/blog/guides/exl3-vs-gguf)

### Inferences
- **Bandwidth arithmetic from the Atomic Chat numbers (5090 peak about 1792 GB/s):**
  - Q4_K_M: 6.63 GiB × 139 tok/s ≈ 990 GB/s, about 55% of peak.
  - EXL3 4.0: 5.79 GiB of layer weights × 113 tok/s ≈ 700 GB/s, about 39%.
  - So the trellis decode is ALU-bound at batch 1 on that card. EXL3 buys about 35–40% lower KLD at about 12% fewer bytes, and pays about 20% in decode speed.
- **Context for rllm:** rllm already reaches about 78% of bandwidth on Q4_K at 8B (Ada), so EXL3-style trellis decoding would lower rllm's decode speed.
- **What is worth adopting:**
  - EXL3's quality per bit is most compelling at 2–3.5 bpw, where GGUF Q2_K/Q3_K are weak and rllm has no support at all.
  - A cheaper first step is the IQ2/IQ3 family (lattice codebooks, already in GGUF and already used by rllm's IQ4_XS path). After that, consider an EXL3 loader only if 70B-on-24GB-class use matters.
- **FP4:** NVFP4/MXFP4 only pays off with FP4 tensor cores (sm_100/sm_120). The dev laptop is Blackwell and the Ada box is not. MXFP4 support in GGUF loading (gpt-oss-style) is a format-coverage item rather than a quality-per-bit item.

### Gaps
- No controlled 2025–2026 study compares AQLM or HQQ against GGUF K-quants at matched bytes. HQQ is calibration-free and roughly RTN-class in quality (background knowledge, not verified in this session).
- The ExLlamaV3 repo doc `doc/exl3.md` returned 404, so turboderp's own EXL3 vs GGUF/AWQ charts could not be read as numbers.
- No independent EXL3 benchmark at batch 8–64 was found.

---

## Q2. Which engines have the best single-stream decode efficiency (% of memory bandwidth), and how?

### Takeaway
No source found in 2025–2026 reports a clean percentage of bandwidth for these engines.
- **llama.cpp:** in the one controlled EXL3 test, llama.cpp at batch 1 beat ExLlamaV3 by 15–21% (about 55% of 5090 bandwidth on a 12B model).
- **TurboMind:** reports up to 86–93% bandwidth utilization in its attention kernels.
- **mistral.rs [vendor]:** claims Q8 decode 1.24× llama.cpp on a B200 (Gemma 4 E4B) and parity on GB10.
- **Older claims:** that EXL2 decodes 50–85% faster than llama.cpp is old, secondary, and not borne out for EXL3.

### Cited Findings
- **mistral.rs v0.8.2 README benchmarks [vendor]:** UQFF Q8 against llama.cpp Q8_0, in tok/s.
  - Gemma 4 E4B on B200: decode 241.4 against 194.4, prefill 27,706 against 11,992.
  - Gemma 4 26B-A4B on GB10: decode 46.8 against 46.4.
  - Against vLLM at BF16, mistral.rs loses: 26B-A4B decode on B200 was 159.6 against 220.2; prefill on GB10 was 592 against 3,879.
  - Latest release listed is v0.9.5 (date not shown).
  - Source: [mistral.rs GitHub](https://github.com/EricLBuehler/mistral.rs)
- **EXL3 against llama.cpp at batch 1 (RTX 5090, Sept 2026):** llama.cpp decodes 15–21% faster in file-size-matched pairs; ExLlamaV3 prefills up to 2.5× faster. — [Atomic Chat](https://atomic.chat/blog/guides/exl3-vs-gguf)
- **Older EXL2 claim [stale, secondary]:** EXL2 batch-1 decode on a 4090 is "50–85% faster" than GGUF; the methodology is unclear. — [insiderllm](https://insiderllm.com/guides/exllamav2-vs-llamacpp-speed-comparison/)
- **TurboMind attention with low-bit KV (paper, 2025-08-21):**
  - Reaches up to 86% and 93% memory bandwidth utilization.
  - With 8-bit KV, decode attention latency drops 7.6% on average (max 29.9%) and prefill 22.1% (max 48.7%), measured on Qwen3-8B AWQ.
  - Q is rearranged once per head per step in shared memory to match the low-bit K tile layout, and the KV is dequantized with integer-to-float conversion inside a multi-buffered pipeline.
  - Source: [arXiv 2508.15601](https://arxiv.org/html/2508.15601v1)
- **TurboMind INT4×FP16 GEMM:** at batch 1–16, latency improves 134% on average over FP16×FP16 (max 220%). — [arXiv 2508.15601](https://arxiv.org/html/2508.15601v1)
- **MLC-LLM (2023) [stale]:** the 7900 XTX reached about 80% of a 4090 and 94% of a 3090 Ti on Llama-2-7B/13B decode, because decode is memory-bound. — [MLC blog](https://blog.mlc.ai/2023/08/09/Making-AMD-GPUs-competitive-for-LLM-inference)

### Inferences
- **rllm's position:** at about 78% of bandwidth on 8B Q4_K (Ada), rllm is likely already at or near the top among GGUF engines for batch-1 decode. The low-hanging decode wins elsewhere come from:
  - CUDA graphs, which rllm already has;
  - fused dequantize-and-dot kernels, which rllm has via dp4a;
  - low-bit KV attention for long context (TurboMind's KV4/KV8 pipeline).
- **A concrete idea:** TurboMind's "rearrange Q once per step to the low-bit K layout" trick could help rllm's q8 KV decode at 16k–32k contexts. There rllm trails llama.cpp by 1.15× with f16 KV, though it leads with q8 KV.
- **A big lever not yet in rllm:** mistral.rs's large prefill lead over llama.cpp on Blackwell (2.3×) likely reflects FlashAttention 3 and cuBLASLt use on B200. rllm's own prefill already leads llama.cpp on small models. Learned speculation (EAGLE/MTP) also matters: a community Strix Halo run used MTP in llama.cpp (Qwen3.8-27B Q8_0 at 22 tok/s, [lilting.ch](https://lilting.ch/en/articles/qwen38-27b-evo-x2-rocm-mtp-test)).

### Gaps
- No engine publishes an achieved-bandwidth percentage for batch-1 decode in 2026, and no independent cross-engine batch-1 study on one GPU covering TensorRT-LLM, MLC, LMDeploy and mistral.rs was found.
- Independent (non-vendor) mistral.rs against llama.cpp numbers were not found.

---

## Q3. Which engines have the best mid-batch (8–64) quantized throughput, and how?

### Takeaway
LMDeploy TurboMind is the most relevant model for rllm's 8B 4-bit concurrency gap.
- **Its W4A16 GEMM:**
  - packs weights offline into an MMA-ready layout, so no runtime swizzle is needed;
  - dequantizes with integer-to-float + FMA on the ALUs while the tensor cores work the previous tile;
  - runs a ≥3-stage cp.async pipeline.
- **Results:** it matches FP16 GEMM at batch 64, where Marlin degrades by up to 20%. End to end it beats vLLM+Marlin by 13% on average (31% max).

### Cited Findings
- **TurboMind GEMM design (arXiv 2508.15601, Aug 2025)**
  - Low-bit weights are repacked offline so that each 16×16 tile becomes a 32×8 layout with no runtime swizzling.
  - Online dequantization is integer-to-float plus an FMA for the scales, overlapped with `mma.sync` on the previous tile and `cp.async` prefetch, with pipeline depth ≥3 on SM80+.
  - At batch 64 it matches FP16 kernels, while Marlin degrades by up to 20.3%.
  - On A100 at 16384³ it executes 64.66% more instructions than cuBLAS but takes only 2.89% more cycles.
  - Source: [arXiv 2508.15601](https://arxiv.org/html/2508.15601v1)
- **TurboMind end-to-end results (same paper)**
  - Against vLLM+Marlin on Qwen 8B/32B AWQ: throughput +13% on average (max 31% at high batch), TTFT 12% lower.
  - Against vLLM with FP8 KV, using TurboMind's 8-bit KV: throughput +50.6% on average (max 156% on A100).
  - Against TensorRT-LLM on Qwen 7B/14B AWQ (L40S and A100): throughput +119% on average.
  - GPUs tested: RTX 4090, L40S, A100, H100. Precisions: AWQ/GPTQ W4A16, KV8 and KV4, FP8.
  - Source: [arXiv 2508.15601](https://arxiv.org/html/2508.15601v1)
- **LMDeploy releases and claims**
  - **Releases:** docs PDF for 0.13.0 is dated 2026-05-12 ([lmdeploy docs](https://lmdeploy.readthedocs.io/_/downloads/en/stable/pdf/)). A review site names v0.17.0, released 2026-09-01, as the latest ([promptquorum](https://www.promptquorum.com/power-local-llm/lmdeploy-review)).
  - **PyPI wheels:** pre-built wheels for new releases could not be uploaded because the PyPI project hit its storage quota (same search, secondary).
  - **Throughput claims (secondary sources):**
    - About 1.36× vLLM on Llama-3-8B at batch 32 ([clore.ai](https://docs.clore.ai/guides/language-models/lmdeploy)).
    - About 16k against 12.5k tok/s for vLLM on H100 ([premai](https://www.premai.io/blog/vllm-vs-sglang-vs-lmdeploy-fastest-llm-inference-engine-in-2026/)).
    - The official docs claim up to 1.8× vLLM request throughput [vendor].
- **TensorRT-LLM (1.3.0 release-candidate series, through rc27 in 2026)**
  - The PyTorch backend is the default and the legacy TensorRT-engine backend has been removed.
  - AutoDeploy (beta) turns off-the-shelf PyTorch models into optimized graphs. It covers 100+ text LLMs and matched a hand-tuned baseline on a B200.
  - NVFP4 work continues across the RCs: Qwen3.5 NVFP4 loading, NVFP4 MoE per-expert scales, Kimi K3 NVFP4.
  - Sources: [TRT-LLM docs](https://nvidia.github.io/TensorRT-LLM/1.3.0rc27/legacy/key-features.html), [NVIDIA AutoDeploy blog](https://developer.nvidia.com/blog/automating-inference-optimizations-with-nvidia-tensorrt-llm-autodeploy/), [rc14 notes](https://newreleases.io/project/github/NVIDIA/TensorRT-LLM/release/v1.3.0rc14), [rc25 notes](https://newreleases.io/project/github/NVIDIA/TensorRT-LLM/release/v1.3.0rc25)
- **ExLlamaV3 serving:** continuous dynamic batching, speculative decoding, tensor and expert parallelism on consumer GPUs, and CPU offload for MoE. No throughput numbers are given. — [exllamav3 GitHub](https://github.com/turboderp-org/exllamav3)
- **mistral.rs serving:** PagedAttention with prefix caching on CUDA and Metal (the default on Metal since v0.9.5), FlashAttention V2/V3, and multi-GPU. It quantizes with ISQ (in-situ quantization at load time), GGUF 2–8 bit, GPTQ, AWQ, HQQ, FP8, BnB and UQFF prebuilt artifacts. — [mistral.rs GitHub](https://github.com/EricLBuehler/mistral.rs)

### Inferences
- **The concurrency gap:** rllm's gap at 16–64 clients (0.53–0.60× vLLM-AWQ) comes from using the dp4a/MMQ int8-activation path at batch 9–64. At those batch sizes a W4A16 tensor-core GEMM (Marlin- or TurboMind-style: offline repack into the mma fragment layout, integer-to-float dequant overlapped with mma) is what vLLM and LMDeploy use.
- **Applying it to Q4_K:** Q4_K's super-block layout (6-bit scales and mins per 32-weight sub-block) can be repacked at load time into a Marlin-like layout. rllm already loads GGUF and could keep a second, repacked copy only for the batched path, or repack exclusively.
- **Recommendation:** implement batch-sized MMQ tiles first, as planned in chart.md, then try a TurboMind-style W4A16 path for M = 8–64 as the larger fix. TurboMind's matching FP16 at batch 64 suggests the 0.6× could approach about 1×.
- **KV8 for throughput:** TurboMind's +50% from 8-bit KV against FP8 KV on vLLM suggests rllm's existing q8 KV, combined with a tensor-core decode-attention kernel, is a throughput lever at high concurrency, not just a memory saver.

### Gaps
- No primary benchmark of LMDeploy W4A16 at batch 8–64 on consumer GPUs (4090/Ada) with absolute tok/s; the paper reports relative numbers.
- No 2026 independent mid-batch numbers for TensorRT-LLM, MLC-LLM or mistral.rs.
- Aphrodite (renamed "Sonar" per one repo) is vLLM-based. Its 2026 release activity and its quant kernels (it historically bundled many: EXL2, AQLM, QuIP#, etc. — background knowledge) were not verified. — [dphnAI/sonar](https://github.com/dphnAI/sonar)

---

## Q4. Compilation approaches (TensorRT engines, TVM/MLC, MAX) against runtime NVRTC

### Takeaway
The trend in 2025–2026 is away from ahead-of-time engine builds.
- **TensorRT-LLM** removed its legacy TensorRT-engine backend in favor of a PyTorch backend plus AutoDeploy graph rewriting.
- **TVM/MLC** compiles ahead of time per target, which buys portability (Vulkan, Metal, WebGPU, ROCm) at the cost of a build step per model and target.
- **MAX/Mojo** compiles hardware-agnostic kernels and claims fast ports to new GPUs [vendor].
- rllm's runtime NVRTC/hiprtc with a cache is in the same spirit as MAX and candle's new cuTile: specialize at load time without a toolkit.

### Cited Findings
- **TensorRT-LLM:** the legacy TensorRT backend is removed, PyTorch is the default, and AutoDeploy (beta) converts PyTorch models to optimized graphs. AutoDeploy matched a manually optimized baseline on B200, with data from v1.3.0rc1. — [TRT-LLM docs](https://nvidia.github.io/TensorRT-LLM/1.3.0rc27/legacy/key-features.html), [NVIDIA blog](https://developer.nvidia.com/blog/automating-inference-optimizations-with-nvidia-tensorrt-llm-autodeploy/)
- **Modular MAX [vendor]**
  - Oct 2025: ported to MI355 in about 14 days and outperformed AMD's optimized vLLM fork by up to 2.2×. Adding a GPU "mostly involves updating a few kernels". — [Modular blog](https://www.modular.com/blog/achieving-state-of-the-art-performance-on-amd-mi355----in-just-14-days)
  - Release 25.6 claims one stack across NVIDIA, AMD and Apple GPUs. — [Modular 25.6](https://www.modular.com/blog/modular-25-6-unifying-the-latest-gpus-from-nvidia-amd-and-apple)
  - AMD's own vLLM+AITER numbers on MI355X dispute such framings. — [ROCm blog](https://rocm.blogs.amd.com/artificial-intelligence/scaling-ai-inference/README.html)
- **candle (2026 snapshot):** adds cuTile, an opt-in feature for writing JIT-compiled CUDA kernels in Rust. — [candle GitHub](https://github.com/huggingface/candle)
- **MLC-LLM [stale, 2023]:** uses one TVM compilation flow to target ROCm and Vulkan (Steam Deck APU). — [MLC blog](https://blog.mlc.ai/2023/08/09/Making-AMD-GPUs-competitive-for-LLM-inference)

### Inferences
- **Runtime NVRTC is a sound design:** rllm starts warm in about 0.7 s, needs no build step per model, and specializes for the exact GPU.
- **What compilers do that rllm doesn't:** shape specialization and autotuning (TVM/MAX search tile sizes per shape). rllm could cheaply extend runtime compilation with:
  - kernel variants specialized on batch size M, which directly addresses the fixed 64-token tile gap;
  - a small autotuning pass that times a few tile configurations at first run and caches the winner beside the compiled kernel cache.
- **What not to copy:** TensorRT-style offline engine builds, which NVIDIA itself abandoned in 2026.

### Gaps
- No independent 2026 benchmark of MAX against vLLM or llama.cpp, and no MAX data on consumer or RDNA hardware, was found.
- MLC-LLM's 2026 release status and current numbers were not found; the newest data seen is from 2023.

---

## Q5. Which engines run well on AMD consumer/APU GPUs (RDNA3/3.5) and on Apple, and with what numbers?

### Takeaway
- **AMD:** llama.cpp, packaged by AMD's Lemonade, dominates consumer AMD in 2026.
  - On Strix Halo, Vulkan still beats ROCm for decode in Phoronix's October 2026 tests, while ROCm leads prefill and TTFT.
  - Lemonade 11.9 (Sept 2026) added an experimental ROCm "HRX" llama.cpp backend for RX 7900 and Strix Halo.
  - ExLlamaV3, TensorRT-LLM and LMDeploy are CUDA-only.
- **Apple:** mistral.rs and candle support Metal, and mistral.rs made PagedAttention the default on Metal in v0.9.5.

### Cited Findings
- **Phoronix (Oct 2026):** Lemonade 2026.39.1 (bundling llama.cpp b10825) on two Strix Halo systems; Vulkan beats ROCm across models and scenarios. — [Phoronix](https://www.phoronix.com/review/lemonade-vulkan-rocm)
- **Lemonade 11.9 (Sept 2026):** experimental llama.cpp ROCm HRX backend for RDNA3 RX 7900 and Strix Halo. — [Phoronix news](https://www.phoronix.com/news/Lemonade-11.9-ROCm-HRx)
- **Community Strix Halo numbers (settings not comparable):**
  - About 50 tok/s on GPT-OSS-120B with llama.cpp ROCm, and 35 tok/s on Qwen3.5-122B with Vulkan. — [HN thread](https://hn.nuxt.dev/item/47612724)
  - Qwen3.8-27B Q8_0 with MTP on ROCm llama.cpp at 22 tok/s. — [lilting.ch](https://lilting.ch/en/articles/qwen38-27b-evo-x2-rocm-mtp-test)
  - Vulkan has a shared-memory leak on AMD driver 26.3.1 that ROCm does not, and ROCm is needed for Lemonade multimodal. — [lilting.ch](https://lilting.ch/en/articles/lemonade-strix-halo-verification)
- **RX 7900 XTX with llama.cpp:**
  - Qwen3.6-27B decode: ROCm 7.2.4 at 31.1 tok/s against Vulkan (Mesa 26.0-devel) at 13.3 tok/s. On that discrete card ROCm won, the reverse of the Strix Halo result. — [techyon](https://techyon.pages.dev/posts/strange-numbers-of-pp-and-tg-rx7900xtx-on-rocm-and-vulcan-with-qwen36-27b-nonmtp-and-mtp.html)
  - The llama.cpp ROCm performance thread is at [discussion #15021](https://github.com/ggml-org/llama.cpp/discussions/15021).
- **mistral.rs on Apple:** Metal backend with PagedAttention and prefix caching (default since v0.9.5); no ROCm is mentioned. — [mistral.rs GitHub](https://github.com/EricLBuehler/mistral.rs)
- **candle backends:** CUDA (+NCCL, cuDNN), CPU (MKL/Accelerate), WASM, and Metal kernels (a `candle-metal-kernels` crate). No ROCm or WGPU is mentioned. — [candle GitHub](https://github.com/huggingface/candle)
- **ExLlamaV3:** requires CUDA ≥12.4 and does not mention ROCm. — [exllamav3 GitHub](https://github.com/turboderp-org/exllamav3)
- **MAX:** claims NVIDIA, AMD (Instinct) and Apple GPU support in 25.6 [vendor]. — [Modular 25.6](https://www.modular.com/blog/modular-25-6-unifying-the-latest-gpus-from-nvidia-amd-and-apple)
- **PowerInfer:** its SmallThinker models (mid-2025, Apache-2.0) are co-designed with the engine, which fuses sparse FFN kernels and prefetches experts. The 21B-A3B and 4B-A0.6B models exceed 20 tok/s on consumer CPUs. — [arXiv 2507.20984](https://arxiv.org/html/2507.20984v1)

### Inferences
- **rllm's AMD position:** among non-llama.cpp engines, rllm's HIP backend on gfx1151 is unusual. None of ExLlamaV3, TurboMind, TRT-LLM or mistral.rs (per its README) target RDNA3.5. The yardstick on Strix Halo is llama.cpp Vulkan for decode, not llama.cpp ROCm.
- **Rebenchmark against Vulkan:** compared.md measures rllm HIP against llama.cpp ROCm (Qwen3-8B 39.3 against 42.8 tok/s). Phoronix shows Vulkan is faster for decode on Strix Halo, so rllm's real decode gap there may be larger. Rerun against a llama.cpp Vulkan build.
- **Apple:** no rllm Metal backend exists. mistral.rs and candle (Metal) and MLX are the competitors, and supporting Apple would need a new backend.

### Gaps
- No 2026 MLC-LLM numbers on RDNA3/3.5 or Apple.
- Phoronix's actual tok/s figures for Strix Halo were not extracted, because the snippets lacked them.
- No mistral.rs or candle Metal benchmarks against llama.cpp Metal or MLX were found.

---

## Engine-by-engine status summary (with rllm adoption judgement)

### Takeaway
The highest-value ideas for rllm, in order:
1. A TurboMind/Marlin-style W4A16 tensor-core GEMM (weights repacked at load time) for 8–64-token decode batches.
2. Batch-size-specialized kernel variants, with first-run autotuning cached beside the NVRTC cache.
3. Low-bit KV decode attention in TurboMind's style.
4. Lower-bit formats: IQ2/IQ3 first, EXL3 later.
5. Learned speculation (MTP/EAGLE).

Rust peers (mistral.rs, candle) are much broader in model and format coverage, but neither shows an AMD story. mistral.rs's own numbers put it close to llama.cpp on decode.

### Cited Findings
- **mistral.rs:** v0.9.5 latest; 45+ architectures, including multimodal, speech, image generation and embeddings; ISQ, GGUF 2–8 bit, GPTQ, AWQ, HQQ, FP8, BnB, UQFF; CUDA (FA2/FA3), Metal, multi-GPU; PagedAttention plus prefix cache. Vendor benchmarks: Q8 decode 1.00–1.24× llama.cpp, while losing to vLLM at BF16. — [mistral.rs GitHub](https://github.com/EricLBuehler/mistral.rs)
- **candle:**
  - **Character:** a "minimalist ML framework" with about 21.3k stars (2026), not a serving engine. It provides llama.cpp quant types via GGUF and FlashAttention v2, has no PagedAttention of its own (delegated to `candle-vllm`, `vllm.rs`, `atoma-infer`), and supports cuTile JIT kernels in Rust.
  - **Model coverage in examples:** LLaMA 1–3, Mistral, Mixtral, Phi 1–3, Gemma 1/2, Qwen1.5 and Qwen3-MoE, Mamba, RWKV, and others.
  - Source: [candle GitHub](https://github.com/huggingface/candle)
- **TGI:** in maintenance mode since 2025-12-11, and its GitHub repo was archived read-only on 2026-03-21 (secondary source). Hugging Face points users to vLLM, SGLang, llama.cpp and MLX. — [TGI docs](https://huggingface.co/docs/text-generation-inference/index), [aiwiki](https://aiwiki.ai/wiki/huggingface_tgi)
- **ExLlamaV3 + TabbyAPI:** the recommended server is TabbyAPI (OpenAI-compatible, Jinja templates). Supported architectures include Llama, Qwen2/3/3.5 (MoE and VL), Gemma 2/3/4, GLM 4/5, DeepSeek V3/V4, GPT-OSS, Phi 3/4, Command-R/A and Nemotron. — [exllamav3 GitHub](https://github.com/turboderp-org/exllamav3)
- **LMDeploy:** v0.17.0 (2026-09-01, per a secondary source). The TurboMind paper is the primary technical reference. — [promptquorum](https://www.promptquorum.com/power-local-llm/lmdeploy-review), [arXiv 2508.15601](https://arxiv.org/html/2508.15601v1)
- **TensorRT-LLM:** v1.3.0 RC series (rc27 seen); PyTorch backend, AutoDeploy beta, NVFP4 focus. — [TRT-LLM docs](https://nvidia.github.io/TensorRT-LLM/1.3.0rc27/legacy/key-features.html)
- **Aphrodite:** now branded Sonar and built on vLLM PagedAttention; the package is still `aphrodite-engine`. — [dphnAI/sonar](https://github.com/dphnAI/sonar)

### Inferences
- **Against mistral.rs:**
  - rllm's measured edges are serving at 4–64 clients, TTFT, bit-exact determinism, an AMD HIP backend and fast startup.
  - mistral.rs's edges are breadth (45+ architectures, multimodal, ISQ from safetensors, multi-GPU, Metal) and FA3 prefill on Hopper and Blackwell.
  - Neither engine has independent numbers.
  - ISQ (quantizing safetensors at load time) is a cheap way to gain HF checkpoint coverage without a GGUF conversion step, and is worth considering for rllm.
- **candle** is a library, not a competitor in serving. Its existence (and `candle-vllm`/`vllm.rs`) shows Rust serving stacks generally lag rllm on scheduler features.
- **TGI's archiving** removes the main Rust-router serving reference from Hugging Face. That leaves an opening for a Rust OpenAI-compatible server with GGUF support, which is rllm's niche.

### Gaps
- mistral.rs release dates and its speculative-decoding status were not shown on the README fetch.
- candle's latest version number was not found.
- MLC-LLM's 2026 status was not found.
- PowerInfer's GitHub activity in 2026 was not checked.
- No source was found for newer 2025–2026 Rust or C++ engines beyond those listed (for example, `vllm.rs` is mentioned only in candle's README).
