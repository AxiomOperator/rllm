# Hardware ceilings and best published results: Strix Halo / RDNA3-4 and NVIDIA Ada / Blackwell (as of 2026-11-11)

Scope: what the memory and compute ceilings are on rllm's two targets, the best numbers anyone has published, and which techniques reach them. rllm baselines, from /Storage/Projects/rllm/compare.md and review.md §1:
- Strix Halo, Qwen3-8B Q4_K_M: tg128 39.3 (≈46 with a draft model), llama.cpp ROCm 42.8; pp512 1333 vs 1303.
- RTX 6000 Ada, Qwen3-8B Q4_K_M: tg 143–150 vs 153; pp512 ~6.6–7.1k vs 9.4k.
- review.md §1 measured rllm's Strix Halo decode matvecs at 116–240 GB/s (fused GLU at ≈177 GB/s). It measured rllm MMQ at ≈18–22 int8 TOPS on Strix Halo and ≈112 TOPS on the Ada.

Notes on method:
- "Measured" means a published run with a stated tool.
- "Claim" means a figure without a reproducible method.
- "Inference" means my own arithmetic, with assumptions stated. Most community numbers are llama-bench on Llama 2 7B Q4_0 (3.56 GiB), the model used by the official llama.cpp scoreboards.

## Q1. Strix Halo memory bandwidth: theoretical vs achievable

### Takeaway
- The theoretical bandwidth is 256 GB/s (LPDDR5X-8000 × 256-bit). The GPU achieves about 212–215 GB/s, roughly 83–84% of that, according to community measurements.
- That puts the practical decode ceiling for Qwen3-8B Q4_K_M at about 45–46 tok/s, and the theoretical ceiling at about 55 tok/s.
- llama.cpp's 42.8 tok/s already reaches roughly 93% of the achievable bandwidth. rllm's 39.3 reaches about 86%.

### Cited Findings
- **rocm_bandwidth_test peak (measured):** about 212 GB/s on gfx1151 against 256 GB/s theoretical, with CPU→GPU copies at about 84 GB/s. The poster says this matches peak bandwidth tests by others on several Strix Halo systems. — [r/LocalLLaMA (randomfoo2) mirror](https://reddit.birdcat.cafe/r/LocalLLaMA/comments/1kmi3ra/amd_strix_halo_ryzen_ai_max_395_gpu_llm)
- **GPU peak bandwidth (measured):** "~215 GB/s max GPU MBW out of a 256 GB/s theoretical" (lhl, 2025-07-22). The stack was TheRock ROCm 7.0 nightlies and llama.cpp b5863. — [Framework Community: Strix Halo GPU LLM Performance Tests](https://community.frame.work/t/amd-strix-halo-ryzen-ai-max-395-gpu-llm-performance-tests/72521)
- **Knowledge-base figure:** ~215 GB/s measured, 84% of 256. The tool is not named. — [SoftAb gfx1151 hardware specs](https://whylucian.github.io/softab/docs/hardware-specs.html)
- **Caveat:** Secondary reviews say the ~210–215 GB/s figures are forum-measured, not vendor-certified. — [hardware-corner Strix Halo optimization](https://www.hardware-corner.net/strix-halo-llm-optimization/)
- **Further reading (not fetched):** Chips and Cheese has articles on the Strix Halo memory subsystem and its 32 MB Infinity Cache (MALL). — [Chips and Cheese: memory subsystem](https://old.chipsandcheese.com/?p=37437), [Chips and Cheese: Infinity Cache](https://old.chipsandcheese.com/?p=37421)

### Inferences
**Bytes per token.** Qwen3-8B Q4_K_M is about 5.03 GB. Decode reads every weight except the token embedding (about 0.36 GB of Q4_K), and it does read the Q6_K output head (about 0.5 GB). That makes ≈4.67 GB per token, plus KV reads, which are small at empty context.

**Decode ceiling** (tok/s = bandwidth ÷ bytes per token):

| Bandwidth | Qwen3-8B Q4_K_M ceiling |
|---|---:|
| 256 GB/s (theoretical) | ≈54.8 tok/s |
| 212–215 GB/s (achievable) | ≈45.5–46 tok/s |

**Effective bandwidth of today's engines:**

| Engine | tok/s | Effective GB/s | % of 215 | % of 256 |
|---|---:|---:|---:|---:|
| llama.cpp ROCm | 42.8 | ≈200 | ≈93% | ≈78% |
| rllm | 39.3 | ≈184 | ≈86% | ≈72% |
| Best Vulkan result (Llama 2 7B Q4_0, ≈3.76 GB read per token; see Q2) | 54.4 | ≈205 | ≈95% | ≈80% |

**Headroom for rllm decode** without speculation is therefore about 10–17%, which means ≈43–46 tok/s.
- rllm's fused GLU kernel at ≈177 GB/s is the main outlier against the ≈205–215 GB/s that the best kernels reach.
- review.md's ≈215 GB/s target is exactly the community-measured peak.

**Going beyond bandwidth ÷ bytes** requires reading fewer bytes per token. The options are:
- speculative decoding (rllm already does it),
- smaller quants (IQ4_XS at 4.25 bpw is about 8% fewer bytes than Q4_K_M's ~4.9 effective),
- batching.

**Infinity Cache.** The 32 MB MALL cannot hold 8B weights, so it does not raise the decode ceiling for single-sequence dense models. This is an inference; I did not fetch the Chips and Cheese numbers.

### Gaps
- No vendor-published or STREAM-style GPU bandwidth figure for gfx1151 was found, and none with a stated ROCm 7.x version.
- I found no measurement of whether achievable bandwidth differs between the HIP and Vulkan runtimes (for example, from page size or GTT vs VRAM carve-out).

## Q2. Best published decode and prefill on Strix Halo, ROCm vs Vulkan

### Takeaway
- **On the official llama.cpp scoreboards (Llama 2 7B Q4_0):** Vulkan reaches tg128 ≈54 (FA on) and pp512 ≈1700. ROCm reaches tg128 ≈50 and pp512 ≈1000.
- **In 2026 community runs:** Vulkan/RADV consistently wins token generation, by roughly 10–30% on MoE models. ROCm/HIP usually wins prompt processing, especially at long context.
- **No 8B dense Q4_K_M result above ≈45–47 tok/s was found on any stack.**

### Cited Findings
**Official llama.cpp scoreboards (Llama 2 7B Q4_0)**

| Source | Backend | Commit | pp512 FA off / on | tg128 FA off / on |
|---|---|---|---:|---:|
| [llama.cpp Vulkan discussion #10879](https://github.com/ggml-org/llama.cpp/discussions/10879) | Vulkan | fb4b273 | 1487 / 1704 | 51.81 / 54.38 |
| [llama.cpp ROCm discussion #15021](https://github.com/ggml-org/llama.cpp/discussions/15021) (firefox42) | ROCm | e60f241 | 911 / 1004 | 50.01 / 49.87 |
| Same thread (hspak) | ROCm | 1d0125b | 351 / 366 | 47.97 / 48.97 |

- The hspak row shows how much ROCm prefill varies by build and setup.
- The Vulkan scoreboard lists the device as "AMD Ryzen AI Max+ 300 Series." It does not name the driver (RADV or AMDVLK) or the date.

**Level1Techs thread** — [Level1Techs Strix Halo LLM benchmark results](https://forum.level1techs.com/t/strix-halo-ryzen-ai-max-395-llm-benchmark-results/233796)
- Vulkan fa=1, Llama 2 7B Q4_0: pp 1014.1, tg 45.8.
- Vulkan pp512 for this model rose from 884 tok/s (May 2025) to 1294 tok/s later. The rise came from llama.cpp and Mesa updates.

**Framework thread** (2025-07-22, llama.cpp b5863, TheRock ROCm 7.0 nightlies, Mesa 25.1.9 for Vulkan) — [Framework Community thread](https://community.frame.work/t/amd-strix-halo-ryzen-ai-max-395-gpu-llm-performance-tests/72521)

| Model | tg128 (Vulkan fa=1) | Best pp512 |
|---|---:|---:|
| Llama 2 7B Q4_K_M | 47.9 | 906 (HIP + hipBLASLt) |
| Shisa V2 8B (Llama 3.1 8B) i1-Q4_K_M | 42.0 | 878 |
| Shisa V2 70B Q4_K_M | 5.0 | 94.7 (rocWMMA) |

Comment from the same thread: ROCm 6.4.4 vs 7.0.1 on Qwen3-8B BF16 pp512 measured 1132 vs 326 tok/s. The poster says the official 7.0.1 release regressed badly compared with 6.4.4 and TheRock builds.

**hogeheer499 strix-halo-guide, 2026** (Vulkan RADV; Mesa 26.0.6 → 26.2.3; llama.cpp b9049 → b11146) — [BENCHMARKS.md](https://github.com/hogeheer499-commits/strix-halo-guide/blob/main/BENCHMARKS.md)

| Model | Build, date | pp512 | tg128 |
|---|---|---:|---:|
| Qwen3-30B-A3B-Instruct-2507 IQ4_XS | b9544, 2026-06-07 | 1438 | 103.18 |
| Qwen3-Coder-30B-A3B UD-Q4_K_XL | b11146, 2026-09-26 | 1437.5 | 96.3 |
| gpt-oss-120b MXFP4 | b11146, 2026-09-26 | 772 | 56.14 |
| Qwen3-Next-80B-A3B UD-Q4_K_XL | b11146 | 866 | 62.9 |
| Nemotron 3 Super 120B-A12B UD-IQ4_XS | 2026-06 | ~296 | ~18.5 |
| Llama 3.1 70B Q4_K_M (Ollama, Vulkan) | | 22–80 | 4.7–4.9 |

Same-machine HIP vs Vulkan comparisons from the same guide (builds not identical):

| Model | Metric | HIP | Vulkan | Winner |
|---|---|---:|---:|---|
| Qwen3-Coder-30B-A3B | pp16384 | 756 | 565 | HIP |
| Qwen3-Coder-30B-A3B | tg128 | 72.2 | 93.7 | Vulkan |
| Qwen3.6-35B-A3B | pp16384 | 1295 | 1038 | HIP |
| Qwen3.6-35B-A3B | tg128 | 52.7 | 62.2 | Vulkan |
| Qwen3-Next-80B | pp512 | 800 | 752 | HIP |
| Qwen3-Next-80B | tg128 | 49.6 | 59.1 | Vulkan |

- The guide also notes that the rocWMMA FlashAttention path was removed upstream (PR #26046).
- A third party (Soothill) measured Qwen3-Coder-30B-A3B at ROCm pp512 1345 vs Vulkan 1115, and Vulkan tg128 97.7 vs ROCm 73.7.

**visorcraft strix-halo-llm-perf** (single host, best config per model, kernel 6.18.x; rows undated, early 2026) — [visorcraft/strix-halo-llm-perf](https://github.com/visorcraft/strix-halo-llm-perf)

| Model | pp512 | tg128 |
|---|---:|---:|
| Llama 2 7B Q4_K_M | 1074 | 47.3 |
| Qwen3-30B-A3B Q4_K_M | 1142 | 86.1 |
| Llama 3.1 70B Q4_K_M | 81.6 | 5.1 |
| GPT-OSS-120B (via llama-server) | 120 | 53.4 |

The same repo's backend findings (2026-03-03):
- On Qwen3-Coder-Next Q6_K_XL, tg was best on Vulkan AMDVLK at 38.65 (RADV 36.80), and pp was best on ROCm 7.x nightlies at ~502.
- On Qwen3-Coder-Next Q4_K_M, RADV gave tg ~43.5 and ROCm 6.4.4 gave pp ~581.

**gpt-oss-120b on ROCm 7rc with WMMA flash attention:** pp512 775.6, tg16 46.1. — [Framework Community thread](https://community.frame.work/t/amd-strix-halo-ryzen-ai-max-395-gpu-llm-performance-tests/72521)

**Toolbox images.** kyuz0's toolboxes provide continuously rebuilt Vulkan (RADV and AMDVLK) and ROCm containers, with benchmark logs in docs/benchmarks.md and an interactive viewer (not fetched). — [kyuz0/amd-strix-halo-toolboxes](https://github.com/kyuz0/amd-strix-halo-toolboxes), [viewer](https://kyuz0.github.io/amd-strix-halo-toolboxes/)

**Phoronix (ROCm 7.0)** reported Vulkan fastest on Qwen3-8B and gpt-oss-20b for both tg and pp. I only saw this in search snippets: the page returned HTTP 403. — [Phoronix ROCm 7 Strix Halo](https://www.phoronix.com/review/amd-rocm-7-strix-halo/3)

### Inferences
**Dense 8B decode tops out at about 42–47 tok/s on every stack.**
- Shisa 8B Q4_K_M reached 42.0 on Vulkan.
- rllm's own llama.cpp ROCm baseline is 42.8 for Qwen3-8B. Qwen3-8B's 151k vocabulary makes its output head larger than Llama 3's.
- Llama 2 7B Q4_K_M reaches about 47–48 because it is smaller (3.80 GiB).
- If Vulkan's ~95% bandwidth efficiency (Llama 2 7B Q4_0) carried over to Qwen3-8B Q4_K_M, the best case would be about 44 tok/s. rllm at 39.3 is about 11% under that.

**Prefill:** rllm's pp512 of 1333 on Qwen3-8B Q4_K_M already matches or beats the best published 8B Q4_K numbers (≈880–1000 for Llama 3 8B in 2025). Vulkan's 1704 on Llama 2 7B Q4_0 was for a smaller model with a simpler quant.

**gpt-oss-120b** (≈5.1B active parameters, MXFP4) reaches about 56 tok/s on Vulkan. That implies about 3.7 GB per token if it runs at ≈205 GB/s, which is consistent with bandwidth-bound MoE decode.

### Gaps
- No same-build, same-day ROCm 7.1x vs RADV vs AMDVLK comparison on a dense 8B Q4_K_M model was found.
- Phoronix numbers could not be read (403).
- kyuz0's raw benchmark tables were not fetched.

## Q3. Why Vulkan beats ROCm on Strix Halo decode

### Takeaway
- The pattern is consistent across sources and generations of hardware (gfx1151, gfx1201, sometimes gfx1100): Vulkan wins tg and HIP wins pp.
- No source gives a measured, kernel-level cause. Everything below beyond that pattern is inference.

### Cited Findings
- **R9700 (RDNA4), Llama 2 7B:** Vulkan tg128 ≈124 vs ROCm ≈92. ROCm pp512 ≈2746 vs Vulkan ≈1944. — [llama.cpp ROCm discussion #15021](https://github.com/ggml-org/llama.cpp/discussions/15021)
- **RX 9070 XT** (ROCm 7.1.1, Fedora 44): Vulkan tg ≈132 vs ROCm ≈122. — [llama.cpp ROCm discussion #15021](https://github.com/ggml-org/llama.cpp/discussions/15021)
- **Maintainer comment** (IMbackK, about GCN/CDNA): "Bandwidth utilization is still fairly low", because the parts are register-starved with small caches. — [llama.cpp ROCm discussion #15021](https://github.com/ggml-org/llama.cpp/discussions/15021)
- **Strix Halo guide:** HIP for prompt-heavy work (RAG), Vulkan/RADV for chat and coding. No mechanism is given. — [strix-halo-guide](https://github.com/hogeheer499-commits/strix-halo-guide/blob/main/BENCHMARKS.md), [visorcraft](https://github.com/visorcraft/strix-halo-llm-perf)
- **RADV tuning:** The Vulkan scoreboard recommends `RADV_PERFTEST=nogttspill` for RADV performance issues. RADV driver updates gave about 13% faster prompt processing. — [llama.cpp Vulkan discussion #10879](https://github.com/ggml-org/llama.cpp/discussions/10879), [hardware-corner RADV update](https://www.hardware-corner.net/llama-cpp-amd-radv-vulkan-driver-update/)
- **Long context:** One source says ROCm drops sharply at long context while Vulkan stays stable. A Reddit commenter says the opposite: ROCm wins at depth. The two contradict each other. — [kyuz0 toolboxes and Level1Techs notes via search](https://forum.level1techs.com/t/strix-halo-llm-inference-notes/249466/9)

### Inferences
These are hypotheses for rllm to test with rocprof. None of them is a sourced fact.

1. **Different decode kernels.** ggml-vulkan's mul_mat_vec shaders are separate code, tuned per vendor with subgroup sizes and rows per workgroup, from ggml-hip's MMVQ, which is CUDA code hipified for wave32/64. The Vulkan path dequantizes to f16/f32 and does FMA. MMVQ quantizes activations to q8_1 and uses dp4a-equivalents (`v_dot4`), which costs an extra launch and more instruction pressure.
2. **Launch overhead.** The HIP runtime's launch and graph overhead on APUs is likely higher than Vulkan command-buffer replay. But rllm's rocprof shows only 2–7 µs gaps, so this is probably not the main factor for rllm.
3. **Memory allocation.** The HIP runtime allocates from GTT/UMA with possibly different page sizes or caching attributes than RADV device-local heaps.
4. **What this means for rllm.** rllm already uses a dp4a path. Its measured 177 GB/s on the fused GLU, against ≈205 GB/s for Vulkan-class kernels, points to kernel memory-access patterns rather than the runtime as the lever. Options: more bytes in flight per CU, wider loads, and fewer waves per row on the 4096-wide rows. Trying a dequant-to-f16 + FMA matvec variant (Vulkan-style) on gfx1151 is a cheap experiment.

### Gaps
- No maintainer post or profiler comparison (rocprof vs RGP) explaining the tg gap was found.
- The ggml-vulkan mul_mat_vec source was not examined in this pass.

## Q4. MoE on unified memory: why Strix Halo is popular

### Takeaway
- 128 GB of unified memory (up to ~96–120 GB usable by the GPU) holds 60–105 GB MoE models whole.
- Decode cost scales with active parameters, so 3–5B-active MoE models run at 55–100+ tok/s, while dense 70B runs at about 5 tok/s.

### Cited Findings
**Qwen3-30B-A3B** (all Vulkan RADV)
- IQ4_XS: tg128 100–103, pp512 ~1430 (b9467–b9544, 2026-06). — [strix-halo-guide](https://github.com/hogeheer499-commits/strix-halo-guide/blob/main/BENCHMARKS.md)
- Q4_K_M: tg128 86.1, pp512 1142. — [visorcraft](https://github.com/visorcraft/strix-halo-llm-perf)
- 2025 UD-Q4_K_XL: tg128 72.0. — [Framework](https://community.frame.work/t/amd-strix-halo-ryzen-ai-max-395-gpu-llm-performance-tests/72521)

**gpt-oss-120b MXFP4 (59 GiB)**
- Vulkan, b11146: tg128 56.1, pp512 772. — [strix-halo-guide](https://github.com/hogeheer499-commits/strix-halo-guide/blob/main/BENCHMARKS.md)
- llama-server: tg 53.4. — [visorcraft](https://github.com/visorcraft/strix-halo-llm-perf)
- An older Windows run measured ROCm pp4096 at 142 vs Vulkan 470, with Vulkan tg ≈40. — [search summary of Level1Techs/Framework posts](https://forum.level1techs.com/t/strix-halo-ryzen-ai-max-395-llm-benchmark-results/233796)

**Larger and dense models for contrast**

| Model | pp512 | tg128 | Source |
|---|---:|---:|---|
| Qwen3-235B-A22B Q3_K_M (105 GB) | 101 | 17.2 | [visorcraft](https://github.com/visorcraft/strix-halo-llm-perf) |
| MiniMax M2.5 Q3_K_M (102 GB) | | 32.8 | [visorcraft](https://github.com/visorcraft/strix-halo-llm-perf) |
| Llama 3.1 70B Q4_K_M (dense) | | 5.1 | [visorcraft](https://github.com/visorcraft/strix-halo-llm-perf) |

**Multi-user serving** (llama-server, b9010): aggregate output peaks at about 166–173 tok/s at np8–16 for 30B-A3B-class models. — [strix-halo-guide](https://github.com/hogeheer499-commits/strix-halo-guide/blob/main/BENCHMARKS.md)

### Inferences
**Effective bandwidth at these rates:**
- Qwen3-30B-A3B at 100 tok/s with IQ4_XS (≈3.3B active parameters × ~4.4 bpw ≈ 1.8–2.0 GB per token, plus shared and attention weights) implies roughly 180–200 GB/s. Strix Halo MoE decode therefore also runs near the bandwidth wall.
- Dense 70B at 5.1 tok/s × ~40 GB ≈ 204 GB/s, which confirms the ~205–215 GB/s practical ceiling from an independent model.

**For rllm,** an MoE path (grouped expert matvec, expert-sorted MMQ for prefill) is required to be relevant on this hardware. Strix Halo's value is the 128 GB, not the 8B speed.

### Gaps
- No GLM-4.5-Air or gpt-oss-20b numbers were found in the fetched sources. Phoronix's gpt-oss-20b numbers were not readable.

## Q5. NPU (XDNA 2) and ROCm 7.x for gfx1151 prefill

### Takeaway
- The NPU paths (Lemonade hybrid via OnnxRuntime-GenAI/RyzenAI, and FastFlowLM) are mainly about power and time-to-first-token on small models. They are not faster than the iGPU in raw prefill throughput on Strix Halo.
- ROCm 7.x gfx1151 GEMM reaches about 25–37 of 59.4 FP16 TFLOPS, and the results vary a lot by version.

### Cited Findings
**NPU throughput**
- FastFlowLM, NPU only, gpt-oss-20b: prefill ≈97 tok/s at a 27k prompt and ≈413 at 54k, decode 17–19 tok/s. The iGPU on the same post prefilled at ≈1643 tok/s. — [r/LocalLLaMA snapshot 2026-03](https://reddit.sentinel-team.org/posts/1rj3i8m/snapshots/2026-03-04T15%3A10%3A50.627029Z)
- Lemonade hybrid, Qwen3-8B-Hybrid on an EVO-X2: TTFT 0.44 s, decode ≈24 tok/s, well below llama.cpp iGPU decode. Hybrid needs the separate RyzenAI-Server package and ONNX/AWQ models, not GGUF. — [lilting.ch Lemonade on Strix Halo](https://lilting.ch/en/articles/lemonade-strix-halo-verification)
- Lemonade hybrid on Qwen3 4B: TTFT ≈4.5 s vs 66–67 s for llama.cpp Vulkan at an 18k-token input. This is a single Reddit report and the llama.cpp configuration is unknown. — [r/LocalLLaMA snapshot 2026-04](https://reddit.sentinel-team.org/posts/1s8p347/snapshots/2026-04-03T21%3A20%3A24.375333Z)
- AMD claims about 400 tok/s NPU+iGPU prefill for DeepSeek-R1-Distill-Llama-8B on Ryzen AI 300. This is a secondary report of an AMD claim. — [hardware-corner](https://hardware-corner.net/amd-targets-faster-local-llms)
- FastFlowLM on Linux supports XDNA 2, including Strix Halo. — [lemonade-server.ai FLM on Linux](https://lemonade-server.ai/flm_npu_linux.html)

**GEMM throughput on gfx1151**

| Configuration | Result | Source |
|---|---:|---|
| FP16/BF16 WMMA peak (512 ops/clk/CU × 40 CU × 2.9 GHz) | 59.4 TFLOPS | [SoftAb](https://whylucian.github.io/softab/docs/hardware-specs.html) |
| INT8 peak, same rate (RDNA3 has no 2× int8 WMMA rate) | 59.4 TOPS | [SoftAb](https://whylucian.github.io/softab/docs/hardware-specs.html) |
| hipBLASLt GEMM, mid-2025 | 36.9 TFLOPS (62%) | [SoftAb](https://whylucian.github.io/softab/docs/hardware-specs.html) |
| rocBLAS without hipBLASLt, mid-2025 | 5.1 BF16 TFLOPS | [r/LocalLLaMA mirror](https://reddit.birdcat.cafe/r/LocalLLaMA/duplicates/1kmi3ra) |
| TheRock 7.11 (2026-02-02), hipblaslt=0 | 32.7 TFLOPS | [SoftAb KB](https://whylucian.github.io/softab/KNOWLEDGE_BASE.html) |
| TheRock 7.11, default | 27.2 TFLOPS | [SoftAb KB](https://whylucian.github.io/softab/KNOWLEDGE_BASE.html) |
| FlyDSL PR benchmark of hipBLASLt (2026-05), by shape | ≈24–37 TFLOPS | [ROCm FlyDSL PR #567 mirror](https://github.zuike.workers.dev/ROCm/FlyDSL/pull/567) |
| FlyDSL's own WMMA kernel, 2048³ only | ≈34 TFLOPS | [ROCm FlyDSL PR #567 mirror](https://github.zuike.workers.dev/ROCm/FlyDSL/pull/567) |
| rocWMMA PyTorch GEMM patch | 21.6 TFLOPS (36%) | [search: glovepost/wmma_ops](https://github.com/glovepost/wmma_ops) |

So hipBLASLt helped on 2025 builds and hurt on the 2026-02 TheRock 7.11 build (+20% with it disabled).

**Other version notes**
- gfx1100 rocBLAS was reported 2.5–6× faster than gfx1151 rocBLAS in 2025 (ROCm issue #4748). — [Framework thread](https://community.frame.work/t/amd-strix-halo-ryzen-ai-max-395-gpu-llm-performance-tests/72521)
- A blog says ROCm 7.13 shipped Strix Halo optimizations, without numbers. — [sleepingrobots ROCm 7 toolbox](https://sleepingrobots.com/dreams/rocm7-toolbox-upgrade-strix-halo)

### Inferences
**rllm prefill utilization (Qwen3-8B).** pp512 needs ≈15.2 GFLOP per token (2 × ~7.6B weights read in matmuls). At 1333 tok/s that is ≈20 TOPS, or ≈34% of the 59.4 TOPS int8/fp16 peak. This matches review.md's 18–22 TOPS.

**Prefill headroom.** The best measured dense GEMM on gfx1151 is 33–37 TFLOPS (55–62%), so pp512 could plausibly reach ≈2200–2400 tok/s if MMQ hit hipBLASLt-class efficiency. That is ~1.7× today. Quantized MMQ carries dequant overhead, so 1.3–1.5× is a more realistic target.

**Data type.** RDNA3.5 int8 WMMA gives no rate advantage over fp16. int8's benefit is fewer bytes and registers, not peak throughput. RDNA4 (gfx1201) has a higher int8/fp8 WMMA rate; I recall roughly 2× fp16, but this was not verified in this pass.

**NPU.** The XDNA 2 NPU is not a credible prefill accelerator for rllm on Strix Halo: the iGPU prefills at more than 1300 tok/s on 8B. It is a low-power niche.

### Gaps
- No official AMD hipBLASLt/rocWMMA int8 measurements for gfx1151 on ROCm 7.1x were found.
- No measured NPU TOPS utilization was found.
- RDNA4 WMMA rates were not verified from AMD documentation in this pass.

## Q6. NVIDIA Ada (sm_89) and Blackwell (sm_120): tensor formats, peaks, and decode efficiency

### Takeaway
**Decode bandwidth efficiency on the llama.cpp CUDA scoreboard** (Llama 2 7B Q4_0):

| GPU | Peak bandwidth | Efficiency |
|---|---:|---:|
| RTX 6000 Ada | 960 GB/s | ~70% |
| RTX 4090 | 1008 GB/s | ~70% |
| RTX 5090 | 1792 GB/s | ~60–63% |
| RTX PRO 6000 Blackwell | 1792 GB/s | ~60–63% |

- rllm on the Ada sits at about 70–73% on Qwen3-8B, which equals llama.cpp's level.
- Engines with exotic weight formats (TRT-LLM AWQ, ExLlama) have been reported 50–85% faster than llama.cpp in older tests. Those numbers are not trustworthy as ceilings.
- Blackwell adds native FP4 (NVFP4/MXFP4) MMA, and llama.cpp gained NVFP4 tensor-core prefill on sm_120 in spring 2026.

### Cited Findings
**CUDA scoreboard** (Llama 2 7B Q4_0) — [llama.cpp CUDA discussion #15013](https://github.com/ggml-org/llama.cpp/discussions/15013)

| GPU | Commit | pp512 FA off / on | tg128 FA off / on |
|---|---|---:|---:|
| RTX 6000 Ada | b8e09f0 | 9229 / 10577 | 176.1 / 179.5 |
| RTX 4090 | | 11993 / 14771 | 186.2 / 189.0 |
| RTX 5090 | | 14073 / 14970 | 290.0 / 300.4 |
| RTX PRO 6000 Blackwell | | 14855 / 16619 | 274.2 / 281.1 |

**Vulkan scoreboard** (same model; NVIDIA rows run with coopmat2) — [llama.cpp Vulkan discussion #10879](https://github.com/ggml-org/llama.cpp/discussions/10879)

| GPU | pp512 (FA) | tg128 (FA) |
|---|---:|---:|
| RTX 5090 | 11796 | 273.7 |
| RTX 4090 | 10830 | 190.1 |
| RX 7900 XTX (Vulkan) | 3890 | 190.9 |
| RX 7900 XTX (ROCm, per [#15021](https://github.com/ggml-org/llama.cpp/discussions/15021)) | | 170 |
| RX 9070 XT | 5604 | 142.0 |
| AI PRO R9700 | 5908 | 152.7 |

**Peak specifications**
- **RTX 6000 Ada:** 960 GB/s, FP8 728.5 TFLOPS dense. Its "1,457 AI TOPS" is FP8 with sparsity. No FP4. — [HP NVIDIA RTX 6000 Ada datasheet](https://valuehub.hp.com/static/nvidia/docs/NVIDIA%20RTX%206000%20Ada%20Generation%20Datasheet.pdf), [getdeploying 5090 vs 6000 Ada](https://getdeploying.com/gpus/nvidia-rtx-5090-vs-nvidia-rtx-6000-ada)
- **RTX 5090:** 1,792 GB/s, FP4 tensor ≈1.68 PFLOPS dense (3.35 sparse). One third-party site lists INT8 at 838 TOPS. — [getdeploying](https://getdeploying.com/gpus/nvidia-rtx-5090-vs-nvidia-rtx-6000-ada), [hmc-tech spec comparison](https://hmc-tech.com/gpus/nvidia-rtx-pro-6000-blackwell-vs-nvidia-geforce-rtx-5090)

**FP4 on Blackwell in llama.cpp**
- llama.cpp gained native NVFP4 kernels on sm_120 around b8967 (late April 2026, PR #22196).
- An RTX 5090 user reported 43–68% faster prompt processing on Qwen3.6-27B-NVFP4, with tg unchanged.
- ik_llama.cpp has MXFP4. On DGX Spark (sm_121), MXFP4 prefill was reported 40% faster than NVFP4.
- All of this is blog and forum sourced, not official benchmarks. — [insiderllm FP4 in llama.cpp](https://insiderllm.com/guides/fp4-inference-llamacpp-nvfp4-mxfp4/), [ai-radar NVFP4 article](https://ai-radar.it/article/llama-cpp-nvfp4-nativo-accelera-l-elaborazione-dei-prompt-su-blackwell?lang=en), [NVIDIA forum MXFP4 PR thread](https://forums.developer.nvidia.com/t/llama-cpp-experimental-native-mxfp4-support-for-blackwell-pr/355639)

**Other engines vs llama.cpp (older results)**
- **TRT-LLM vs llama.cpp on an RTX 4090:** ≈170.6 vs 100.4 tok/s on Mistral-7B. This is a 2024 benchmark: TRT-LLM used AWQ, and llama.cpp was much older. — [Jan: benchmarking TensorRT-LLM](https://v2.jan.ai/blog/benchmarking-nvidia-tensorrt-llm)
- **ExLlamaV2 EXL2:** claimed 50–85% faster than llama.cpp GGUF at batch 1 on a 4090. This is a secondary aggregator with unclear methodology. — [insiderllm ExLlamaV2 vs llama.cpp](https://insiderllm.com/guides/exllamav2-vs-llamacpp-speed-comparison/)
- **ExLlamaV3:** the only number found is 27B EXL3 4 bpw at 122 tok/s on an RTX 5090, as a recipe claim. — [contextstudios recipe](https://www.contextstudios.ai/local-ai/miaai-lab--qwen3.8-27b-exl3-rtx-5090-one-click)
- **Llama 3.1 8B Q4_K_M on an RTX 5090,** community runs: ~207–229 tok/s (Ollama/llama.cpp). Other pages report 82–142, which looks like misconfigured or outdated runs. — [bmdpat receipts](https://bmdpat.com/receipts/rtx-5090/llama-3-1-8b.md), [insiderllm 5090 benchmarks](https://insiderllm.com/guides/rtx-5090-local-ai-benchmarks/)

### Inferences
**Bytes per token.** Llama 2 7B Q4_0 reads about 3.76 GB per token (3.83 GB file minus a ≈74 MB embedding).

| GPU | Measured tg (tok/s) | Effective bandwidth | % of peak |
|---|---:|---:|---:|
| RTX 6000 Ada | 179.5 | ≈675 GB/s | ≈70% of 960 |
| RTX 4090 | 189 | ≈711 GB/s | ≈70.5% of 1008 |
| RTX 5090 | 300 | ≈1128 GB/s | ≈63% of 1792 |
| RTX PRO 6000 Blackwell | 281 | ≈1057 GB/s | ≈59% of 1792 |
| RX 7900 XTX (Vulkan) | 191 | ≈718 GB/s | ≈75% of 960 |
| RX 9070 XT (Vulkan) | 142 | ≈534 GB/s | ≈83% of 640 |
| Strix Halo (Vulkan) | 54.4 | ≈205 GB/s | ≈80% of 256 |

- Fast GPUs show lower efficiency at small models because fixed per-token costs (launches, syncs, small kernels, sampling) are a larger share of a 3–6 ms step.

**rllm Qwen3-8B on the Ada:**

| Engine | tok/s | Effective bandwidth | % of 960 |
|---|---:|---:|---:|
| llama.cpp | 153 | ≈715 GB/s | ≈74% |
| rllm | 143–150 | ≈670–700 GB/s | ≈70–73% |

- The theoretical ceiling is ≈205 tok/s.
- A realistic ceiling at ~85–90% of peak is about 175–185 tok/s. That level is typical for well-tuned GDDR6 streaming kernels; it is an assumption, since no engine in the sources demonstrates it at batch 1.
- So rllm has maybe 15–25% decode headroom on the Ada, versus about 10–17% on Strix Halo.

**Prefill on the Ada.**
- llama.cpp's 9.4k tok/s × ≈15.2 GFLOP/token ≈ 143 TOPS, ≈20% of ≈728 dense int8/FP8 TOPS (assuming int8 dense equals FP8 dense on AD102).
- rllm at ≈7k tok/s ≈ 106 TOPS, matching review.md's ≈112 measurement.
- On the Ada, the int8 MMQ path (`mma.m16n8k32`) is the right instruction. Headroom comes from tiling, pipelining and epilogue, not from a new data type. FP8 mma (`m16n8k32 e4m3`) on sm_89 has the same peak as int8 and needs FP8 weights.
- **FP8 accumulation rate (unverified, from memory):** I recall that the NVIDIA Ada whitepaper lists GeForce RTX 4090 FP8 with FP32 accumulation at half the FP16-accumulate rate, while the RTX 6000 Ada runs both at full rate. If so, FP8 vs int8 on sm_89 depends on the SKU. This was not fetched or checked in this pass; verify against the Ada whitepaper's appendix tables before relying on it.

**Blackwell sm_120.**
- FP4 MMA (`mma.sync ... kind::f8f6f4 / mxf4nvf4` with block scales) roughly doubles peak over FP8/int8. It only helps prefill, and only with FP4-format weights.
- GGUF Q4_K would need requantizing to NVFP4/MXFP4, which is a different format.
- For decode, FP4 matters only through bytes: NVFP4 is ≈4.5 bpw, similar to Q4_K's 4.5–4.9.

### Gaps
- No primary-source batch-1 decode numbers were found for ExLlamaV3 or TRT-LLM 2026 releases on an 8B model on Ada or Blackwell, so their bandwidth efficiency is unverified.
- The official INT8 dense TOPS for the RTX 6000 Ada was not confirmed from NVIDIA's datasheet in this pass.
- No published llama.cpp MMQ TOPS-vs-peak measurement was found. The ≈20% above is inferred.
- Marlin-FP8/CUTLASS sm_120 kernel availability was not researched in depth: no sources were fetched on Marlin or CUTLASS sm_120 FP4 kernels.

## Q7. Practical ceilings summary (tokens/s = bandwidth ÷ bytes per token)

### Takeaway
rllm is at 70–86% of practical decode ceilings already, so the remaining big wins are:
- **Strix Halo decode:** kernel bandwidth (+10–17%) plus speculation and MoE support.
- **Strix Halo prefill:** MMQ efficiency (≈34% of peak today; ≈55–60% demonstrated by hipBLASLt), so ~1.3–1.7×.
- **Ada prefill:** llama.cpp already shows 1.35× is reachable, and the hardware allows more.
- **Ada decode:** fixed per-step costs and kernel streaming efficiency, so ~15–25% at most.

### Cited Findings
- Strix Halo achievable bandwidth is ≈212–215 GB/s of 256. — [Framework](https://community.frame.work/t/amd-strix-halo-ryzen-ai-max-395-gpu-llm-performance-tests/72521), [r/LocalLLaMA mirror](https://reddit.birdcat.cafe/r/LocalLLaMA/comments/1kmi3ra/amd_strix_halo_ryzen_ai_max_395_gpu_llm)
- Strix Halo best Vulkan tg128 for Llama 2 7B Q4_0 is 54.38. — [llama.cpp #10879](https://github.com/ggml-org/llama.cpp/discussions/10879)
- RTX 6000 Ada: 960 GB/s, 728.5 FP8 TFLOPS dense; llama.cpp tg128 179.5 on Llama 2 7B Q4_0. — [HP datasheet](https://valuehub.hp.com/static/nvidia/docs/NVIDIA%20RTX%206000%20Ada%20Generation%20Datasheet.pdf), [llama.cpp #15013](https://github.com/ggml-org/llama.cpp/discussions/15013)
- Strix Halo: 59.4 FP16/INT8 TOPS peak; best GEMM 36.9 TFLOPS. — [SoftAb](https://whylucian.github.io/softab/docs/hardware-specs.html)

### Inferences
All entries are my estimates, using ~4.67 GB per token for Qwen3-8B Q4_K_M and ≈15.2 GFLOP per prefill token.

| Target | Metric | Ceiling (theory / practical) | Best known | rllm now | Headroom |
|---|---|---|---|---|---|
| Strix Halo | tg, Qwen3-8B Q4_K_M | 54.8 / ≈45–46 | ≈43–44 if Vulkan-class efficiency (no measured run); llama.cpp ROCm 42.8 | 39.3 | +10–17% |
| Strix Halo | pp512, Qwen3-8B | ≈3900 at 59.4 TOPS / ≈2200–2400 at hipBLASLt-class 55–62% | rllm 1333 is the best 8B Q4_K figure found | 1333 | +30–70% |
| RTX 6000 Ada | tg, Qwen3-8B Q4_K_M | ≈205 / ≈175–185 (85–90%, assumed) | llama.cpp 153 | 143–150 | +15–25% |
| RTX 6000 Ada | pp512, Qwen3-8B | ≈48k at 728 TOPS / unknown | llama.cpp 9.4k (≈20% of peak) | ≈7k | ≥1.35× to match llama.cpp |
| RTX 5090 / PRO 6000 (sm_120) | tg, 8B Q4 | ≈380 / ≈320–340 | llama.cpp ≈300 (Llama 2 7B Q4_0), ≈207–229 on Llama 3.1 8B Q4_K_M | n/a | Fixed costs dominate at 3 ms steps |

### Gaps
- The 85–90% "practical" factor for NVIDIA GDDR is an assumption. No batch-1 LLM engine in the sources demonstrates more than ~75% at the 7–8B scale.
- I did not find nvbandwidth or STREAM measurements for the RTX 6000 Ada or RTX 5090 to set the achievable-bandwidth ceiling empirically, and I did not run any.
- Microbenchmarking on rllm's own boxes would close this; per user memory, ask before long GPU runs.
