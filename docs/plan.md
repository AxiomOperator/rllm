# rllm — Plan

A Rust LLM inference engine: llama.cpp-style local inference (GGUF, quantization) with vLLM-style serving (paged KV cache, continuous batching). NVIDIA first, AMD/ROCm second, both from the same codebase.

## Why

Because I want to build it. No need to beat existing engines to justify it. They're the benchmarks to measure against, not competitors.

## Non-goals (for now)

- Training or fine-tuning
- Multimodal models
- MoE models (until Phase 4)
- Model formats other than GGUF and SafeTensors
- Hardware other than NVIDIA (sm_86, sm_89) and AMD (gfx1151 Strix Halo)
- Pooling GPUs across machines into one logical device

## Key decisions

### 1. NVIDIA first, structured so ROCm isn't a rewrite

| Layer | NVIDIA (Phase 1) | AMD (Phase 3) |
| --- | --- | --- |
| Driver/runtime API | `cudarc` | HIP bindings (own thin `-sys` crate if nothing suitable exists) |
| GEMM | cuBLASLt | hipBLASLt / rocBLAS |
| Custom kernels | CUDA C++ `.cu` files, compiled at startup with NVRTC for the device arch | Same sources via `hipify` + hiprtc |
| Multi-GPU comms | NCCL | RCCL |

Rules that keep the ROCm port cheap:

- **Compile kernels at runtime (NVRTC), not at build time.** Building rllm needs no CUDA toolkit, only the driver plus the NVRTC and cuBLAS shared libraries. The kernels are compiled for the exact GPU (sm_86, sm_89, sm_120…) on first use. hiprtc gives the same model on AMD.
- **Write kernels in CUDA C++, not Rust-on-GPU.** HIP is close to source-compatible with CUDA C++, so one kernel source can serve both vendors. Rust GPU kernel toolchains have no AMD path.
- **No vendor types outside the backend crate.** Everything above the backend goes through a `Backend` trait (buffers, streams, kernel launches, GEMM).
- **Kernels must stay portable.** No headers (fp16 conversion is done by hand); no inline PTX; warp size is a parameter, never a hard-coded 32; vendor-only intrinsics sit behind small shims. RDNA (gfx1151) supports wave32, but keep the parameter anyway.
- **Every kernel has a CPU reference implementation** in Rust. Tests compare GPU output against it, so a new backend is checked against the same oracle.

### 2. Own the tensor layer, keep it small

No Candle and no general tensor framework. The engine needs a narrow set of ops (GEMM/quantized matmul, RMSNorm, RoPE, attention, SiLU/GELU, softmax/sampling, embedding lookup). A thin `Tensor` (device buffer + shape + dtype + strides) over the `Backend` trait is enough, and it keeps memory layout under our control.

### 3. Paged KV cache and a batch dimension from day one

Phase 1 runs one request at a time, but the forward pass still takes a batch of sequences, and the KV cache is a block allocator with per-sequence block tables. That way Phase 2 only adds the scheduler and the batching kernels; no rewrite of the core.

### 4. First model

A dense Qwen3 model (e.g. Qwen3-8B), as GGUF. Weight types in Phase 1: F16/BF16, Q8_0, and Q4_K + Q6_K (together these cover Q4_K_M files).

## Workspace layout

```
crates/
  rllm-gguf/       GGUF parser, metadata, tensor index (mmap)
  rllm-core/       Tensor, DType, Backend trait, quant block formats
  rllm-cpu/        CPU reference backend (correctness oracle first, speed later)
  rllm-cuda/       CUDA backend: cudarc, cuBLASLt, kernels/*.cu
  rllm-hip/        HIP backend: rllm-cuda's backend.rs and kernels over a HIP `gpu` module
  rllm-model/      Model configs + forward passes (Qwen3 first)
  rllm-kv/         Paged KV block allocator, block tables, prefix cache
  rllm-sched/      (Phase 2) Continuous-batching scheduler (for now: rllm-server/src/engine.rs)
  rllm-tokenizer/  HF `tokenizers` + chat templates (minijinja)
  rllm-server/     Axum OpenAI-compatible API, SSE, continuous batching, metrics (docs/server.md)
  rllm-cli/        CLI: run, bench, compare
```

## Hardware roles

| Hardware | Arch | Role |
| --- | --- | --- |
| RTX 6000 Ada, 48 GB | sm_89 | Main benchmark box |
| RTX PRO 3000 Blackwell laptop, 12 GB | sm_120 | Day-to-day development (power-limited clocks; don't benchmark here) |
| 2× RTX 4500 Ada | sm_89 | Concurrency/serving tests, 2-way tensor parallelism |
| 3× RTX A4500 | sm_86 | Ampere coverage (no FP8); 2-way TP + one spare, or pipeline parallelism |
| 5× Strix Halo, 128 GB | gfx1151 | ROCm backend, large GGUF models |
| 10 GbE | — | Cross-node routing and pipeline parallelism only, no cross-node TP |

TODO: list the actual machines (the GPU count and machine count don't map one-to-one yet).

## Correctness and benchmarking

Correctness harness, built in Phase 1 before any optimization:

- **Logit diff:** same prompt through rllm and llama.cpp (and/or HF transformers), compare the logits at each position with a set tolerance.
- **Perplexity:** fixed text (e.g. a wikitext slice), must be within a small tolerance of llama.cpp on the same GGUF.
- **Greedy decoding:** token-for-token match against the reference for N tokens.
- **Kernel tests:** each GPU kernel against the CPU reference, across shapes and dtypes.

Benchmark method (`rllm-cli bench`):

- Same GGUF file, context length, sampling settings and concurrency as the baseline
- Warmup runs excluded from results
- Report time to first token, time per output token, prefill tok/s, decode tok/s and aggregate throughput, separately
- Baselines: llama.cpp for single-stream/GGUF, vLLM for batched throughput

## Phases

Each phase ends at its exit criteria, not a date.

### Phase 1 — Correct single-stream inference on CUDA

GGUF loader, tokenizer and chat template, CPU reference backend, CUDA backend (cuBLASLt + custom kernels for dequant, RMSNorm, RoPE, paged attention, sampling), Qwen3 forward pass, CLI with streaming output, correctness harness.

**Exit:** passes the correctness harness for Q8_0 and Q4_K_M; decode tok/s at batch size 1 on the RTX 6000 Ada within 20% of llama.cpp.

### Phase 2 — Serving

Continuous batching, chunked prefill, prefix caching, quantized matmul kernels tuned for batches, OpenAI-compatible API with SSE, Prometheus metrics, profiling with Nsight.

**Exit:** API works with AegisLLM; at concurrency 16/32/64, throughput is benchmarked against vLLM and the gaps are understood.

### Phase 3 — ROCm and multi-GPU

HIP backend built from the hipified kernels, hipBLASLt, Strix Halo tuning (unified memory, bandwidth-bound decode), 2-way tensor parallelism (NCCL/RCCL), pipeline parallelism, model placement across nodes.

**Exit:** Phase 1 correctness harness passes on gfx1151; decode tok/s on Strix Halo benchmarked against llama.cpp (both its HIP and Vulkan backends); a model served across 2 GPUs with TP.

**Status (2026-10-30):** the HIP backend runs on gfx1151 and passes the correctness harness (parity tests, `rllm compare`, server checks); benchmarked against llama.cpp's ROCm build in `compare.md` (its Vulkan build not yet). How it is built:
- **One backend, two vendor layers.** `rllm-cuda/src/backend.rs` is compiled by both crates against a small `gpu` module: cudarc in `rllm-cuda`, dlopened HIP/hiprtc/hipBLAS in `rllm-hip` (no ROCm needed to build; one binary serves both vendors, `-d gpu` picks whichever is present).
- **One kernel source.** `kernels.cu` has a HIP prelude (f16 conversion, shuffles, dp4a) and RDNA3 WMMA versions of the tensor-core work: the MMQ inner loop and a prefill attention kernel. The PTX-only kernels are stubs on HIP.
- **gfx1151 specifics.** hipBLAS GEMMs write f16 (4× faster than f32 output there), scaled to keep range; BF16 weights go through f16. Integrated memory: `rllm tune` sizes from RAM and skips the host tier.
- **Left:** hipBLASLt (1.5× hipBLAS on f16 output), a tighter Q4_K repack for decode bandwidth, WMMA attention for 512-wide heads, gfx12 WMMA, multi-GPU.

### Phase 4 — Advanced

Speculative decoding, MoE (Qwen3 MoE), structured output, model hot-swapping, disaggregated prefill/decode, FP8 on Ada.

## Boundaries

- **AegisLLM** handles auth, routing, usage tracking and policy. rllm handles model execution and GPU memory only.
- **Licensing:** kernels ported or adapted from llama.cpp (MIT) or vLLM (Apache-2.0) keep their attribution in a `THIRD_PARTY` file. Pick rllm's own license before the first public push.

## Open questions

- ~~Is there a maintained HIP binding crate, or does rllm-hip need its own `-sys` crate?~~ Own thin bindings, loaded with dlopen (`rllm-hip/src/ffi.rs`): about 40 entry points.
- Attention kernel: write our own paged attention, or adapt FlashInfer-style kernels? FlashInfer is CUDA-only, so adapting it adds porting work in Phase 3.
- Is NVIDIA's newer Rust CUDA work worth using for host-side code? Check its maturity; kernels stay in CUDA C++ either way.
