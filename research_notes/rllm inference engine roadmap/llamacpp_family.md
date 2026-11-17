# llama.cpp family (llama.cpp, ik_llama.cpp, Ollama, LM Studio, KoboldCpp): what rllm lacks or does worse

Research date: 2026-11-11. Two kinds of source:
- **Primary, read directly:** the llama.cpp checkout vendored in this repo (`/Storage/Projects/rllm/third_party/llama.cpp`), commit `2e0d86b`, dated 2026-10-28. This is the same commit rllm benchmarked against in compared.md and compare.md. Code facts below are labelled "[local 2e0d86b]" and cite the matching GitHub path.
- **Web sources:** each is dated where the date is known.

rllm baseline, from `/Storage/Projects/rllm/chart.md`, `compared.md` and `compare.md`:
- 8 GGUF weight types, 3 architecture families, 1 GPU.
- MMQ uses a fixed 128×64 tile.
- MMVQ-style dp4a matvec handles up to 8 tokens; from 9 tokens on it switches to MMQ.
- Has a draft model and prompt-lookup speculation.
- Has only a q8 KV cache.

---

## 1. Quant types and kernels (MMQ, MMVQ, stream-K, tile sizing by batch, BLAS fallback)

### Takeaway
Mainline llama.cpp (2026-10-28) has about 30 quant types:
- Every one of them has an MMQ kernel. Since mid-2026 these include NVFP4 and the new Q1_0/Q2_0.
- MMQ picks its token-tile width J from 8 to 128 in steps of 8, choosing the J that needs the fewest column tiles.
- For Q4_K on Ampere and Ada it uses 128-row × J tiles with stream-K.

rllm's fixed 128×64 tile is the most transferable gap. It explains both the 9–32-sequence serving collapse and probably the Qwen3-8B pp512 deficit, since llama.cpp uses a 128×128 tile at 512 tokens. ik_llama.cpp adds about 20 more types (IQ*_K, IQ*_KS/KSS/KL, IQ*_KT trellis, ternary), but those files only load in ik_llama.cpp.

### Cited Findings

**Type list (mainline, [local 2e0d86b])**
- `ggml.h` defines 30-plus types: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K–Q6_K, Q8_K, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ1_S, IQ1_M, IQ4_NL, IQ4_XS, TQ1_0, TQ2_0, MXFP4 (id 39), NVFP4 (id 40, "4 blocks, E4M3 scale"), Q1_0 (41) and Q2_0 (42). The Q4_0_4_4/4_8/8_8 and IQ4_NL_4_x repack types were removed from GGUF and became runtime repacks — [ggml.h](https://github.com/ggml-org/llama.cpp/blob/2e0d86b/ggml/include/ggml.h)
- `ggml_cuda_should_use_mmq` gives MMQ support to Q1_0, Q2_0, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K–Q6_K, IQ1_S, IQ2_XXS/XS/S, IQ3_XXS/S, IQ4_XS, IQ4_NL, MXFP4 and NVFP4. TQ1_0, TQ2_0 and IQ1_M are absent, so they use the dequantize + cuBLAS or MMVQ paths — [mmq.cu](https://github.com/ggml-org/llama.cpp/blob/2e0d86b/ggml/src/ggml-cuda/mmq.cu)
- `ggml.h` describes a new src1 precision hint: `GGML_PREC_Q8` lets a backend quantize the F32/BF16/F16 activations to Q8_0, and `GGML_PREC_Q4` allows 4-bit activations (Q4_K, NVFP4, MXFP4). The MMQ templates carry this as a `prec_src1` parameter — [ggml.h](https://github.com/ggml-org/llama.cpp/blob/2e0d86b/ggml/include/ggml.h)
- `MMQ_ITER_K` is 256 and `MMQ_ITER_K_FP4` is 512. FP4 types use a doubled K step per iteration — [mmq.cuh](https://github.com/ggml-org/llama.cpp/blob/2e0d86b/ggml/src/ggml-cuda/mmq.cuh)

**Tile sizing by batch ([local 2e0d86b])**
- `ggml_cuda_mmq_get_J_best` loops `for (int J = 8; J <= 128 && ntiles_J_best > 1; J += 8)`.
  - It skips any J that has no config or whose shared memory exceeds the per-block limit.
  - It keeps the J with the fewest `ceil(ncols / J)` tiles.
  - A launch switch instantiates all 16 widths (8, 16, 24, …, 128) as template parameters.
  - Source: [mmq.cu](https://github.com/ggml-org/llama.cpp/blob/2e0d86b/ggml/src/ggml-cuda/mmq.cu), [mmq.cuh](https://github.com/ggml-org/llama.cpp/blob/2e0d86b/ggml/src/ggml-cuda/mmq.cuh)
- MMQ tile configs are now per-architecture tables: `mmq-config-{ampere,blackwell,cdna,gcn,pascal-dp4a,pascal-older,rdna2,rdna3,rdna3-5,rdna4}.cuh`. Each entry is (type, nthreads, occupancy, I rows, J cols, SRAM layout, K_vram, stream_k, fallback) — [ggml-cuda dir](https://github.com/ggml-org/llama.cpp/tree/2e0d86b/ggml/src/ggml-cuda)
- On Ampere/Ada, Q4_K uses 256 threads, occupancy 1, I=128 rows, J ∈ {8, 16, 24, 32, 40, 48, 64, 80, 96, 112, 128} and `stream_k=true` — [mmq-config-ampere.cuh](https://github.com/ggml-org/llama.cpp/blob/2e0d86b/ggml/src/ggml-cuda/mmq-config-ampere.cuh)
- On RDNA3.5 (gfx1151), Q4_0 uses 128 threads with 64-row tiles for J 16–80 and 112–128, but 256 threads with 128-row tiles at J=96. Other types (Q1_0, Q2_0, Q4_1, Q5_0) switch to 256 threads × 128 rows from J=48 or 64 up. Stream-K is off on RDNA3.5 — [mmq-config-rdna3-5.cuh](https://github.com/ggml-org/llama.cpp/blob/2e0d86b/ggml/src/ggml-cuda/mmq-config-rdna3-5.cuh)

**Stream-K ([local 2e0d86b])**
- When the config has `stream_k`, the launch works out tile efficiency as `100*ntiles/(nsm*waves)`.
  - On NVIDIA with ≥90% efficiency it launches one block per tile and skips the fixup.
  - Otherwise it launches `min(nsm, nwork)` blocks, plus a `mul_mat_q_stream_k_fixup` kernel when the tiles don't divide evenly.
  - Fused gate/up MMQ never uses stream-K.
  - Source: [mmq.cuh](https://github.com/ggml-org/llama.cpp/blob/2e0d86b/ggml/src/ggml-cuda/mmq.cuh)
- rllm tried a stream-K-style last-block reduction and measured it 13% slower, from register and occupancy cost — `/Storage/Projects/rllm/compare.md` (local)

**MMVQ / MMQ / BLAS thresholds ([local 2e0d86b])**
- `MMVQ_MAX_BATCH_SIZE 8`: the MMVQ kernels handle up to 8 tokens, the same as rllm's matvec — [mmvq.cuh](https://github.com/ggml-org/llama.cpp/blob/2e0d86b/ggml/src/ggml-cuda/mmvq.cuh)
- On NVIDIA Turing and newer, MMQ is always used for supported types at any batch size, unless built with `GGML_CUDA_FORCE_CUBLAS`. `MMQ_DP4A_MAX_BATCH_SIZE 64` only limits pre-tensor-core GPUs. MMQ also needs at least 48 KiB of shared memory per block — [mmq.cu](https://github.com/ggml-org/llama.cpp/blob/2e0d86b/ggml/src/ggml-cuda/mmq.cu)
- On RDNA3/3.5 (WMMA), MMQ is used except in these cases, where it falls back to dequantize + hipBLAS:
  - Q2_K above 128 tokens;
  - Q6_K above 128 tokens on RDNA3.0 and above 256 on RDNA3.5;
  - IQ2_XS/IQ2_S above 128 tokens except on RDNA3.5.
  - MoE with 64 or more experts always uses MMQ.
  - RDNA4 always uses MMQ.
  - Source: [mmq.cu](https://github.com/ggml-org/llama.cpp/blob/2e0d86b/ggml/src/ggml-cuda/mmq.cu)
- On CDNA (MFMA):
  - CDNA3 always uses MMQ, because rocBLAS/Tensile is slow there and hipBLASLt crashes.
  - Other CDNA parts use MMQ when there are more than 64 experts or at most 128 tokens; for Q4_0/1 and Q5_0/1; or for Q4_K/Q5_K up to 256 tokens.
  - Source: [mmq.cu](https://github.com/ggml-org/llama.cpp/blob/2e0d86b/ggml/src/ggml-cuda/mmq.cu)
- Float weights have their own kernels: `mmvf` (matvec) and `mmf` (a tensor-core matmul for small batches, 32 rows per block, 64 on CDNA). F16 and BF16 therefore don't go through cuBLAS at small batch — [mmf.cuh](https://github.com/ggml-org/llama.cpp/blob/2e0d86b/ggml/src/ggml-cuda/mmf.cuh)
- Fused paths exist for matvec + GLU (`should_fuse_mul_mat_vec_q/f`), MMQ gate/up/GLU (`ggml_cuda_should_fuse_mul_mat_q`), top-k MoE routing and a MoE weighted reduction — [ggml-cuda.cu](https://github.com/ggml-org/llama.cpp/blob/2e0d86b/ggml/src/ggml-cuda/ggml-cuda.cu)

**ik_llama.cpp quant types (README, last updated 2026-11-05)**
- IQK family: IQ2_K, IQ3_K, IQ4_K, IQ5_K and IQ6_K, plus `_KS`, `_KSS` and `_KL` variants (for example IQ2_KL).
- Row-interleaved `_R4` and newer `_R16` repacks: IQ4_KS_R16, IQ3_KS_R16, IQ5_KS_R16, Q8_K_R16 and Q8_0_R16.
- Trellis types: IQ1_KT, IQ2_KT, IQ3_KT and IQ4_KT (integer trellis), plus "Gyro Trellis" TQ2_T, TQK6/7 and TQP5–7.
- Ternary and low-bit types: IQ1_BN, IQ2_BN, Q2_0, PQ2_0 and PTQ1_0, with a lossless TQ1_0/TQ2_0 → PTQ1_0/PQ2_0 converter.
- KV and model support: Q8_KV for the KV cache, Hadamard transforms on the K and V caches, MXFP4 for gpt-oss, and regex-based custom quant mixes.
- Source: [ik_llama.cpp README](https://github.com/ikawrakow/ik_llama.cpp)
- ik_llama.cpp backends are CPU (AVX2+/NEON) and CUDA (Turing+) only. "ROCm, Vulkan, Metal, and AVX-only CPUs are not supported." It last synced with upstream in August 2024, so the formats have diverged — [ik_llama.cpp README](https://github.com/ikawrakow/ik_llama.cpp)

### Inferences
- **The 8B serving gap is a dispatch gap, not a kernel gap.** llama.cpp runs MMVQ up to 8 tokens, then MMQ with J = 16/24/32 for 9–32 tokens. rllm's 64-wide tile wastes 50–86% of its MMA work at 9–32 tokens (8 sequences: 905 tok/s; 9 sequences: 610). Templating rllm's MMQ on J ∈ {8, 16, 24, 32, 48, 64, 96, 128} with llama.cpp's "fewest column tiles that fit shared memory" rule is a direct port. NVRTC/hiprtc runtime compilation makes extra template instances cheap, since only the widths actually used get compiled.
- **The Qwen3-8B pp512 0.77× on the Ada is probably tile shape, not only profiling.**
  - At 512 tokens llama.cpp picks J=128, so its Q4_K tile is 128×128 with 256 threads at occupancy 1, plus stream-K.
  - rllm's 128×64 tile reads each weight tile twice as often per token.
  - On 0.6B models the smaller matrices may favour more, smaller tiles (rllm wins there), and on 8B they don't.
  - Worth testing a 128×128 variant first.
- **Not every type needs MMQ.** llama.cpp itself falls back to dequantize + BLAS for Q2_K and Q6_K at large batch on RDNA3. A cheap way to add Q2_K/Q3_K/Q4_0/Q5_0/IQ2/IQ3 coverage is dp4a matvec for decode plus dequantize-to-f16 + hipBLAS/cuBLAS for prefill, with MMQ added later per type.
- **Priority types, ranked by how common they are in published GGUFs:**
  1. Q4_0 and Q5_0/Q5_1 (simple to add; Q4_0 is the llama-bench reference format)
  2. Q3_K and Q2_K (Unsloth/bartowski low-bit mixes)
  3. MXFP4 (gpt-oss ships only in this format)
  4. IQ3_XXS/IQ3_S, IQ2_*, IQ1_* (codebook grids)
  5. NVFP4 (Blackwell hardware FP4)
  6. TQ1_0/TQ2_0 (niche)
- ik_llama.cpp's IQ*_K and KT types live only in the ik ecosystem. Supporting them would mean tracking a moving, single-author format, so they are low priority for rllm.

### Gaps
- No measured numbers for llama.cpp's MMQ throughput at each J. The PR discussions that justify the thresholds (#18202, #18537) were not fetched.
- No published per-type perplexity or speed table for ik_llama.cpp's IQ*_K/KT types. The README points to its wiki, which was not fetched.
- The new mainline Q1_0/Q2_0 and NVFP4 activation-precision paths (GGML_PREC_Q4) were not traced to a PR or date.

---

## 2. Backends (CUDA, ROCm/HIP, Vulkan, Metal, SYCL, CPU), multi-GPU, RPC

### Takeaway
- **Backends:** mainline has about 18 backend directories. Vulkan has cooperative-matrix MMQ and flash-attention shaders.
- **Strix Halo:** published Llama-2-7B Q4_0 rows put Vulkan ahead of ROCm on prompt processing (about 1.5–1.7k vs 0.9–1.0k t/s) and level or slightly ahead on decode (about 52–54 vs 50 t/s). Newer community results disagree, and some favour ROCm for prompts at large context.
- **Multi-GPU:** layer split (pipeline) is the default, row split is deprecated, and a new experimental `tensor` mode splits weights and KV cache over NCCL/RCCL.

### Cited Findings
- **Backend directories at 2e0d86b:** blas, cann, cpu, cuda, et, hexagon, hip, metal, musa, opencl, openvino, rpc, sycl, virtgpu, vulkan, webgpu, zdnn and zendnn. There is also a `ggml-backend-meta.cpp` "meta device" with split axes — [ggml/src](https://github.com/ggml-org/llama.cpp/tree/2e0d86b/ggml/src)
- **Vulkan shaders at 2e0d86b:**
  - matmul: `mul_mm.comp`, `mul_mm_cm2.comp` (NV coopmat2), `mul_mmq.comp` and `mul_mmq_cm1.comp` (int8 MMQ on KHR_coopmat)
  - flash attention: `flash_attn.comp`, `flash_attn_cm1.comp`, `flash_attn_cm2.comp`
  - split-K decode: `flash_attn_decode_phase_1/2.comp`, `flash_attn_split_k_reduce.comp`
  - sparse attention: `flash_attn_sparse_compact.comp`
  - Source: [vulkan-shaders](https://github.com/ggml-org/llama.cpp/tree/2e0d86b/ggml/src/ggml-vulkan/vulkan-shaders)
- **Vulkan scoreboard (discussion #10879; rows undated), Llama 2 7B Q4_0, pp512 / tg128:**

  | GPU | Commit | No FA | FA |
  |---|---|---|---|
  | "AMD Ryzen AI Max+ 300 Series" (Strix Halo) | fb4b273 | 1487 / 51.8 | 1704 / 54.4 |
  | RTX 5090, coopmat2 | ca71fb9 | 10382 / 264 | 11796 / 274 |
  | RTX 4090, coopmat2 | 4ae88d0 | 9452 / 188 | 10830 / 190 |
  | RX 7900 XTX | 304665f | 3727 / 183 | 3890 / 191 |

  Source: [llama.cpp discussion #10879](https://github.com/ggml-org/llama.cpp/discussions/10879)
- **ROCm scoreboard (discussion #15021, opened 2025-08-01), same model:**

  | GPU | Commit | No FA | FA |
  |---|---|---|---|
  | AI Max+ 395 (128 GB LPDDR5) | e60f241 | 911 / 50.0 | 1004 / 49.9 |
  | 8060S (different user) | 1d0125b | 351 / 48.0 | 366 / 49.0 |
  | RX 7900 XTX | 2f0c2db | 3552 / 167 | 3874 / 170 |
  | MI300X, `GGML_HIP_ROCWMMA_FATTN=ON` | ee3a9fc | — | 11946 pp (about 4037 without rocWMMA) |

  Source: [llama.cpp discussion #15021](https://github.com/ggml-org/llama.cpp/discussions/15021)
- In #15021, users on RX 7600 XT, R9700 and 7800 XT report ROCm well ahead of Vulkan on pp512 (for example R9700: 2746 vs 1944) and Vulkan ahead on tg128 (R9700: 124 vs 92) — [discussion #15021](https://github.com/ggml-org/llama.cpp/discussions/15021)
- **Strix Halo community data is mixed (2026):**
  - Phoronix (ROCm 7.0 era) found Vulkan RADV ahead on pp512 for Qwen3-8B and gpt-oss-20b.
  - A May 2026 Reddit benchmark (llama.cpp around b9187, Mesa 26.1) found ROCm 12–23% ahead on prompt processing, and Vulkan usually ahead on decode for 35B.
  - Sources: [Phoronix ROCm 7 Strix Halo](https://www.phoronix.com/review/amd-rocm-7-strix-halo/3), [Phoronix Lemonade ROCm vs Vulkan](https://www.phoronix.com/review/lemonade-vulkan-rocm/3), [r/LocalLLaMA snapshot 2026-05](https://reddit.sentinel-team.org/posts/1t4fkri/snapshots/2026-05-09T00%3A46%3A53.937033Z) (search-result summaries; the full posts were not read)
- **visorcraft strix-halo-llm-perf (2026, Fedora 43, ROCm 6.4/7.x, RADV and AMDVLK):**
  - Llama 2 7B Q4_K_M: 1,074 pp512 / 47.3 tg128 (backend not stated per row).
  - Qwen3-Coder-Next Q4_K_M: best prompt speed on ROCm 6.4.4 (about 581); best decode on Vulkan RADV (about 43.5).
  - The 120 W profile "usually wins for 7B+".
  - Unified-KV, no-mmap and batch tuning gave negligible gains.
  - Source: [visorcraft/strix-halo-llm-perf](https://github.com/visorcraft/strix-halo-llm-perf)
- **Multi-GPU at 2e0d86b:**
  - Split modes are `none`, `layer` (default; pipeline parallel, KV cache with its layer), `row` ("Deprecated… comparatively poor performance") and `tensor`.
  - `tensor` is "EXPERIMENTAL": it splits weights and KV cache via a meta device, uses NCCL (on by default) or RCCL (off by default, "not universally beneficial"), and needs FA plus an unquantized KV cache. It is unsupported for many MoE, hybrid and SSM architectures.
  - `--fit` (on by default) auto-sizes arguments to device memory.
  - Source: [docs/multi-gpu.md](https://github.com/ggml-org/llama.cpp/blob/2e0d86b/docs/multi-gpu.md)
- ik_llama.cpp has its own split mode `graph` (graph-parallel multi-GPU, PR 1022), which got "much better TG" on 2025-12-16 (PR 1067). The README warns of gibberish output in some graph-split or partial-offload configurations — [ik_llama.cpp README](https://github.com/ikawrakow/ik_llama.cpp)
- An RPC backend exists (`tools/rpc`, `--rpc`). visorcraft used it to span Strix Halo hosts in February 2026 — [tools dir](https://github.com/ggml-org/llama.cpp/tree/2e0d86b/tools), [visorcraft](https://github.com/visorcraft/strix-halo-llm-perf)
- **CUDA decode work (Aman's blog, 2025-12-01):**
  - Fusions covered: GEMV+GLU, top-k MoE (softmax + top-k), RMS_NORM+MUL(+ADD), adds and softcap.
  - Q, K and V projections run on concurrent streams (`GGML_CUDA_GRAPH_OPT=1`).
  - tg128 gains: +43% Qwen3-30B-A3B Q4_K_M on the 5090 (247 → 352) and +37% on the 4090 (198 → 271); gpt-oss-20B +27% (5090) and +17% (4090).
  - The author estimates gpt-oss-20B on the 5090 at about 80% of speed-of-light.
  - Source: [am17an blog](https://am17an.bearblog.dev/new-post/)

### Inferences
- **rllm's Strix Halo decode gap:**
  - Strix Halo's theoretical bandwidth is 256 GB/s. That figure is general knowledge, not from these sources.
  - Llama-2-7B Q4_0 (3.56 GiB) at 54 t/s on Vulkan works out to about 205 GB/s effective, roughly 80% of theoretical.
  - rllm's Mistral-7B Q4_K_M decode is 43.2 t/s vs llama.cpp ROCm at 46.3.
  - The reference to beat on gfx1151 decode is therefore probably llama.cpp **Vulkan**, not ROCm, since Vulkan tg is about 4–9% higher in the scoreboards.
  - Vulkan decode reportedly does well on RDNA. rllm could study its `mul_mat_vec` shader strategy (subgroup size, rows per workgroup) for its HIP matvec. The shader internals were not checked.
- **Tensor parallelism:** llama.cpp's `tensor` mode is a recent, experimental, NVIDIA-first design (meta device, NCCL all-reduce, FA required). rllm can build TP for its two-GPU plan without being behind a mature competitor. In the llama.cpp family, only ik_llama's `graph` split is reported to give real multi-GPU decode gains.
- **Vulkan:** a Vulkan backend would be a second shader language and a large investment for a single-binary engine. On Strix Halo, ROCm/HIP already matches Vulkan on prompts and trails it only slightly on decode. Vulkan is not a priority unless rllm needs Intel, older AMD or non-ROCm GPUs.

### Gaps
- No controlled late-2026 llama-bench row for Qwen3-8B or Llama-3-8B Q4_K_M on the 8060S with ROCm and Vulkan side by side, and kyuz0's toolbox benchmark tables were not reached.
- The scoreboard rows are undated, and their commits predate rllm's comparison build.
- Metal, SYCL and CPU repack-kernel details were not researched (out of rllm's near-term scope).

---

## 3. Model coverage (MoE, MLA, multimodal, embeddings/rerank, hybrid/SSM, expert offload)

### Takeaway
At 2e0d86b llama.cpp has about 165 architecture enums. They include every 2025–26 MoE family (Qwen3/3.5-MoE, gpt-oss as `OPENAI_MOE`, DeepSeek2/3.2/V4, GLM4-MoE/GLM-DSA, MiniMax-M2/M3, Kimi), hybrid/SSM models (Mamba/Mamba2, Qwen3-Next, Granite-Hybrid, Nemotron-H, Jamba, Falcon-H1, LFM2, RWKV6/7), vision and audio through `libmtmd`, and BERT-family embedding and rerank models. rllm has 3 dense families.

### Cited Findings
- **`src/llama-arch.h` at 2e0d86b lists, among others:**
  - Qwen and gpt-oss: QWEN2, QWEN2MOE, QWEN3, QWEN3MOE, QWEN35, QWEN35MOE, QWEN3NEXT, QWEN3VL, QWEN3VLMOE, OPENAI_MOE
  - DeepSeek: DEEPSEEK2, DEEPSEEK32, DEEPSEEK4, DEEPSEEK41
  - GLM: GLM4, GLM4_MOE, GLM_DSA, GLM5_NEXT
  - Others: MINIMAX_M2/M3, KIMI_LINEAR, KIMI_K3, LLAMA4, GEMMA3/3N/4, PHI2/PHI3/PHIMOE, MISTRAL3/4
  - Hybrid/SSM: MAMBA/MAMBA2, JAMBA, FALCON_H1, GRANITE_HYBRID, NEMOTRON_H(_MOE), LFM2/LFM2MOE/LFM3, RWKV6/7
  - Embedding encoders: BERT, NOMIC_BERT(_MOE), JINA_BERT_V2/V3, MODERN_BERT, NEO_BERT, EUROBERT, GEMMA_EMBEDDING, T5/T5ENCODER
  - Speculative heads: EAGLE3, DFLASH
  - Source: [llama-arch.h](https://github.com/ggml-org/llama.cpp/blob/2e0d86b/src/llama-arch.h)
- The CUDA backend has kernels for SSM and linear attention (`ssm-conv`, `ssm-scan`, `gated_delta_net` with an MMA variant, `gla`, `wkv`), MoE (`mmid`, `topk-moe`, `moe-weighted-reduction`, `add-id`), DeepSeek sparse attention (`lightning-indexer`, `dsv4-hc`) and `fwht` (Hadamard) — [ggml-cuda dir](https://github.com/ggml-org/llama.cpp/tree/2e0d86b/ggml/src/ggml-cuda)
- Multimodal lives in `tools/mtmd`: `clip.cpp`, `mtmd-audio.cpp`, image and video helpers, with test inputs `test-1.jpeg`, `test-2.mp3` and `test-3.mp4`. It is used through `--mmproj` — [tools/mtmd](https://github.com/ggml-org/llama.cpp/tree/2e0d86b/tools/mtmd)
- Server flags at 2e0d86b include `--embedding`, `--rerank`, `--lora`, `--cpu-moe`, `--n-cpu-moe`, `--override-tensor` (-ot) and `--fit` — [common/arg.cpp](https://github.com/ggml-org/llama.cpp/blob/2e0d86b/common/arg.cpp)
- **ik_llama.cpp MoE and MLA work:**
  - MLA (PR 188, 2025-02-09) and FlashMLA-3 on CPU (PR 273) and CUDA (PR 386, 2025-05-07).
  - Fused MoE FFN (PR 229).
  - Better hybrid MoE offload (PR 520, 2025-06).
  - A dynamic MoE expert cache (PR 2644) and on-demand tensor reload (PR 1989).
  - Auto-fit of offloaded tensors (PRs 1501/1504).
  - Sparse FA for DeepSeek-V4 and GLM-DSA (2026-10).
  - Source: [ik_llama.cpp README](https://github.com/ikawrakow/ik_llama.cpp)
- **Ollama (v0.40.x, October–November 2026):**
  - Recurrent and vision GGUF models gained concurrent requests (v0.40.15, 2026-11-03).
  - YaRN for Qwen3/3.5 (same release).
  - Bundled llama.cpp b12257 (v0.40.17-rc0, 2026-11-10).
  - An MLX backend on Apple silicon with DFlash speculation (v0.40.12, 2026-10-27).
  - Source: [Ollama releases](https://github.com/ollama/ollama/releases)

### Inferences
- **Highest-value coverage for rllm**, which already has dense attention, GQA and int8 MMQ:
  1. **Qwen2/2.5**: attention biases only.
  2. **Phi-3**.
  3. **Qwen3-MoE / Qwen3.5-MoE and gpt-oss**: needs grouped MMQ (`mmid`), top-k routing, and MXFP4 for gpt-oss.
  4. **A BERT-style encoder** for real embeddings and rerank.

  MoE is where llama.cpp-family users are in 2026: the benchmarks found nearly all use Qwen3.x-35B-A3B, gpt-oss, GLM or MiniMax. MoE also suits rllm's unified-memory AMD target, since 128 GB Strix Halo boxes mainly run 30B–120B MoE models.
- **Expert offload** (`--n-cpu-moe`, `-ot`) needs a fast CPU matmul, which rllm lacks (0.2–0.35× llama.cpp on prompts). On Strix Halo, unified memory makes offload unnecessary. Skip it until the CPU backend gets fast.
- **Hybrid/SSM** (Qwen3-Next, Nemotron-H, Granite-Hybrid) needs new kernels (gated delta-net, ssm-scan) and state caching that doesn't fit a paged KV model. It costs more than MoE.

### Gaps
- No throughput numbers for llama.cpp MoE on Strix Halo for a model rllm could target first (for example Qwen3-30B-A3B Q4_K_M). Secondhand figures vary widely; one cites 887 pp512 / 39.7 tg128 for Qwen3.5-35B-A3B Q4 on ROCm 7.2, unverified.
- KoboldCpp and LM Studio's runtime (llama.cpp and MLX engines) were not researched separately. Both track mainline llama.cpp, so their feature sets are largely mainline's.

---

## 4. Server features (speculation, grammar, tools, reasoning, slots, KV types, multi-user)

### Takeaway
llama-server (2026-10-28) has nine speculative modes:
- trained draft heads: EAGLE-3, MTP, DFlash, DSpark
- a separate draft model
- five n-gram variants

It also has GBNF grammars and llguidance, an automatic tool-call parser, unified KV, cache reuse, a host-RAM prompt cache, slot save/restore and 9 KV cache types.

rllm's measured lead in concurrency, TTFT and greedy-exact draft speculation stands. Its gaps are learned draft heads, grammar/regex, more KV types and more tool-call formats.

### Cited Findings
- **`common_speculative_type` at 2e0d86b:**
  - NONE
  - DRAFT_SIMPLE: a standalone draft model
  - DRAFT_EAGLE3
  - DRAFT_MTP: multi-token prediction
  - DRAFT_DFLASH
  - DRAFT_DSPARK: "DFlash + Markov head"
  - NGRAM_SIMPLE, NGRAM_MAP_K, NGRAM_MAP_K4V, NGRAM_MOD
  - NGRAM_CACHE: a 3-level n-gram cache
  - Source: [common/common.h](https://github.com/ggml-org/llama.cpp/blob/2e0d86b/common/common.h)
- **docs/speculative.md:**
  - EAGLE-3 is "a one-layer transformer" that reads the target's hidden states. Converted checkpoints exist for Llama-3.1-8B, Qwen3-1.7B/4B/8B/14B/32B/30B-A3B, gpt-oss-20b/120b, and Gemma-4-31B/26B-A4B.
  - DFlash "produces an entire block of draft tokens in a single forward pass (block diffusion)" and injects target hidden states into the draft's attention.
  - Source: [docs/speculative.md](https://github.com/ggml-org/llama.cpp/blob/2e0d86b/docs/speculative.md)
- **Draft-head speedups (community, May 2026):** on a 24 GB RTX 3090 with Qwen3.6-27B, ik_llama.cpp with MTP and BeeLlama with DFlash were both about 1.6× faster than mainline llama.cpp over a nine-prompt benchmark. Mainline later added MTP — [InsiderLLM shootout](https://insiderllm.com/guides/best-24gb-backend-shootout-ik-llama-beellama-llamacpp/), [dasroot.net 2026-05](https://dasroot.net/posts/2026/05/qwen-3-6-27b-24gb-vram-benchmarking-llama-cpp-ik-llama-cpp-beellama-vllm/) (search summaries)
- **ik_llama.cpp speculation:**
  - MTP for GLM-4.x/5, Qwen3.5/3.6/3.8, Gemma 4 and Step 3.7.
  - DFlash (PR 1970) and DFlash 2 (PR 2761).
  - DSpark (PR 2280).
  - n-gram (PR 1261) and suffix (PR 1646) self-speculation.
  - Source: [ik_llama.cpp README](https://github.com/ikawrakow/ik_llama.cpp)
- **Ollama:** Nemotron 3.5 Bolt E4B (an rllm target model) ships with a DFlash draft for speculation (v0.40.10, 2026-10-23) — [Ollama releases](https://github.com/ollama/ollama/releases)
- **Server flags at 2e0d86b:**
  - KV and caching: `--kv-unified`, `--cache-reuse`, `--cache-ram` (host-RAM prompt cache), `--ctx-checkpoints`, `--slot-save-path`, `--swa-full`, `--context-shift`
  - Constraints and chat: `--grammar`, `--json-schema`, `--jinja`, `--reasoning-format`
  - Speculation: `--spec-type`, `--draft-max`, `--mtp`, `--dflash`
  - Serving: `--models-dir` (router mode)
  - Source: [common/arg.cpp](https://github.com/ggml-org/llama.cpp/blob/2e0d86b/common/arg.cpp)
- **KV cache types (`-ctk/-ctv`):** F32, F16, BF16, Q8_0, Q4_0, Q4_1, IQ4_NL, Q5_0 and Q5_1. Quantized V needs FA. Tensor-split mode allows only unquantized KV — [common/arg.cpp](https://github.com/ggml-org/llama.cpp/blob/2e0d86b/common/arg.cpp), [docs/multi-gpu.md](https://github.com/ggml-org/llama.cpp/blob/2e0d86b/docs/multi-gpu.md)
- **Reasoning formats:** NONE, AUTO, DEEPSEEK_LEGACY and DEEPSEEK (streams `reasoning_content` deltas). Docs include `function-calling.md`, `autoparser.md` (automatic tool-call parser) and `llguidance.md` (grammar backend for Lark/regex/JSON Schema) — [common/common.h](https://github.com/ggml-org/llama.cpp/blob/2e0d86b/common/common.h), [docs/](https://github.com/ggml-org/llama.cpp/tree/2e0d86b/docs)
- **ik_llama.cpp KV work:** Q8_KV cache type (2025-02-19, PR 208); Hadamard on K (PRs 1033/1034) and V (PR 1527); fused `dequant_hadamard` for `-khad` (PR 1852, 2026-05) — [ik_llama.cpp README](https://github.com/ikawrakow/ik_llama.cpp)
- **rllm's own head-to-head (2026-10-29, RTX 6000 Ada):**
  - With Qwen3-8B plus a 0.6B draft, llama-server `draft-simple` was *slower* than no draft (126 vs 146 tok/s), while rllm reached 213.
  - At 4–8 clients llama-server was 1.5–3.4× behind rllm.
  - Source: `/Storage/Projects/rllm/compare.md` (local measurement)

### Inferences
- **Most transferable speculation gain: EAGLE-3, then MTP.**
  - Converted EAGLE-3 heads already exist for rllm's exact targets (Qwen3-1.7B/4B/8B/14B/32B, Llama-3.1-8B, Gemma-4).
  - The head is a single transformer layer that reads target hidden states, which fits rllm's verification path.
  - Speedups are about 1.6× in community tests where n-gram lookup doesn't help (free text). rllm's prompt lookup gave nothing on free text (142 → 142).
  - DFlash is promising (Ollama ships one for Nemotron E4B), but its block-diffusion drafter is a new architecture.
- **Grammar:** rllm enforces JSON structure but not value constraints. Embedding GBNF or porting an llguidance/xgrammar-style token-mask engine would close it. llguidance is Rust, so it could link directly; its licence and fit were not checked.
- **KV quant:** adding q4_0/q5_x KV types matters little for rllm's datacenter-style serving. ik_llama's Hadamard on the V cache is a cheap accuracy add to rllm's existing K-rotation.
- **Draft model prefix cache:** llama.cpp and vLLM have one and rllm doesn't (rllm's own note). This is a smaller gap than learned draft heads.

### Gaps
- No controlled acceptance-rate or speedup numbers for llama.cpp's EAGLE-3/MTP/DFlash on dense 8B targets. The community numbers are for a 27B model on a 3090 and come from search summaries, not full articles.
- The docs' n-gram variants (map_k, map_k4v, mod, cache) have no published per-mode numbers.

---

## 5. ik_llama.cpp improvements over mainline (with numbers)

### Takeaway
ik_llama.cpp's edge is CPU and hybrid CPU+GPU MoE (MLA/FlashMLA, fused MoE, row-interleaved repacks, expert cache), its exclusive quant types and early MTP/DFlash. On a pure single-GPU dense CUDA workload like rllm's, the evidence that it beats mainline is mixed and model-dependent. It doesn't support ROCm or Vulkan at all.

### Cited Findings
- The README publishes no speedup percentages and points to its wiki. Dated items from 2026:
  - Fused rms_norm + add (2026-04-13)
  - Much faster RNG sampling (2026-01-25)
  - Much faster long-context TG for GLM-4.5–4.7 and MiniMax-M2 (2026-01-28)
  - CUDA static per-stream cuBLAS workspace (2026-10-30)
  - CUDA MoE MMQ with one block per row tile and expert (PR 2831, undated)
  - Source: [ik_llama.cpp README](https://github.com/ikawrakow/ik_llama.cpp)
- **Community results (2026), all secondhand:**
  - Qwen3.5-35B MoE prompt processing: about 700 t/s on ik vs under 200 on mainline in one report.
  - Another report says mainline dominates prompt processing for Qwen3.5-35B at 180k context.
  - A 70B CPU+GPU test found mainline faster on UD-Q4_K_XL prompt processing.
  - One user reports a consistent ik regression.
  - Sources: [r/LocalLLaMA snapshot 2026-03](https://reddit.sentinel-team.org/posts/1ruew2g/snapshots/2026-03-16T20%3A46%3A16.316967Z), [hardware-corner](https://www.hardware-corner.net/guides/llama-cpp-vs-ik_llama-cpp/), [ik discussion #164](https://github.com/ikawrakow/ik_llama.cpp/discussions/164)
- ik_llama.cpp's README warns that Unsloth `_XL` files with f16 tensors are "likely to fail", and that `-rtr` (runtime repacking) moves k-quant matmuls onto the CPU in hybrid setups — [ik_llama.cpp README](https://github.com/ikawrakow/ik_llama.cpp)

### Inferences
- The transferable ideas are techniques, not formats:
  - fused rms_norm+add (rllm already has it)
  - per-expert MMQ blocking for MoE
  - an expert cache for offload
  - Hadamard on V
  - MTP heads
- ik's CPU repacking (`_R4`/`_R8`/`_R16` row interleaving for AVX2/AVX-512 VNNI) is the clearest model for fixing rllm's slow CPU backend if that ever becomes a priority.

### Gaps
- There is no rigorous 2026 single-GPU dense CUDA benchmark of ik_llama.cpp vs mainline for an 8B Q4_K_M model.

---

## 6. Latest llama-bench numbers: 8B Q4_K_M on Strix Halo (ROCm vs Vulkan) and RTX 4090/5090/Ada

### Takeaway
No public late-2026 row gives exactly Llama-3/Qwen3-8B Q4_K_M on the 8060S with ROCm and Vulkan side by side. The best anchors are below; most are Llama-2-7B Q4_0 rows from llama.cpp's two scoreboards:

| Hardware, backend | pp512 (t/s) | tg128 (t/s) |
|---|---:|---:|
| Strix Halo, ROCm | 911–1004 | about 50 |
| Strix Halo, Vulkan | 1487–1704 | 52–54 |
| RTX 5090, Vulkan | 10.4–11.8k | 264–274 |
| RTX 5090, CUDA (mirror) | about 14–15k | about 290–300 |
| RTX 4090, Vulkan | 9.5–10.8k | 188–190 |
| RTX 4090, Llama-3.3-8B Q4_K_M | — | about 138 (third-party) |

rllm's own same-box runs (compare.md, 2026-10-30) remain the most comparable numbers: Qwen3-8B Q4_K_M on gfx1151 with llama.cpp ROCm `5e03bdd87` gave 1302.65 pp512 and 42.84 tg128.

### Cited Findings
- **Strix Halo, ROCm:** AI Max+ 395, Llama 2 7B Q4_0 — 911 / 50.0 (no FA), 1004 / 49.9 (FA), commit e60f241 — [discussion #15021](https://github.com/ggml-org/llama.cpp/discussions/15021)
- **Strix Halo, Vulkan:** same model — 1487 / 51.8 (no FA), 1704 / 54.4 (FA), commit fb4b273 — [discussion #10879](https://github.com/ggml-org/llama.cpp/discussions/10879)
- **Strix Halo, Llama 2 7B Q4_K_M:** 1,074 pp512 / 47.3 tg128 (backend not stated, 2026) — [visorcraft](https://github.com/visorcraft/strix-halo-llm-perf)
- **RTX 5090, CUDA (Llama 2 7B Q4_0):** about 14–15k pp512 / 290–300 tg128, per a mirror of the CUDA scoreboard (#15013). The original thread was not fetched — [knightli summary 2026-04-23](https://knightli.com/en/2026/04/23/llama-cpp-gpu-benchmark-cuda-rocm-vulkan-scoreboard/), [discussion #15013](https://github.com/ggml-org/llama.cpp/discussions/15013)
- **RTX 5090 and 4090, Vulkan coopmat2 (Llama 2 7B Q4_0):** 5090: 11796 / 274 (FA); 4090: 10830 / 190 (FA) — [discussion #10879](https://github.com/ggml-org/llama.cpp/discussions/10879)
- **RTX 5090, Llama 8B Q8_0:** about 12,200 pp512 / about 143 tg128 — [HF post, csabakecskemeti](https://huggingface.co/posts/csabakecskemeti/458514085783723) (search summary)
- **RTX 4090, Llama-3.3-8B Q4_K_M:** about 138 tg, citing llama.cpp issue #11128 — [markaicode](https://markaicode.com/benchmarks/llamacpp-llama-33-rtx-5080-throughput-benchmark/) (third-party; low confidence)
- **rllm vs llama.cpp on the RTX 6000 Ada, Qwen3-8B Q4_K_M:** llama.cpp 9380 pp512 / 153 tg128, rllm 6640 / 150 — `/Storage/Projects/rllm/compared.md` (local)

### Inferences
- **On Strix Halo, Vulkan's pp512 lead over ROCm in the scoreboards (about 1.6–1.7×) is larger than rllm's lead over ROCm** (1.02× at 8B).
  - If a current Vulkan build still beats ROCm on prompts, rllm's real competitor on gfx1151 is llama.cpp Vulkan. rllm should benchmark against it on the Strix box.
  - Run `llama-bench` with the Vulkan build, `-fa 1`, on the same Qwen3-8B Q4_K_M, at a current build.
  - The scoreboard commits are older than 2e0d86b, and ROCm on gfx1151 improved a lot through 2025–26 (rllm measured 1303 pp512 on ROCm 7.14), so this needs re-measuring before drawing conclusions.
- **Decode:** Vulkan's roughly 54 t/s on a 3.56 GiB model means about 80% bandwidth use, against llama.cpp ROCm at a similar level. rllm's gap to close (0.90–0.96× ROCm) is matvec bandwidth efficiency. rllm could target about 80% of 256 GB/s as the bar.

### Gaps
- No kyuz0 toolbox table and no Level1Techs numbers were retrieved, and there's no dated late-2026 8B Q4_K_M ROCm-vs-Vulkan row.
- No RTX 6000 Ada rows from the community scoreboards were retrieved.
- The CUDA scoreboard's 4090 and 5090 rows were only seen through a mirror summary.
