# rllm vs vLLM vs llama.cpp: features

Written 2026-10-29. What rllm can and can't do next to the two engines it
borrows from: llama.cpp (local GGUF inference) and vLLM (high-throughput
serving).

**Sources:**
- **rllm:** this repository. Every rllm claim was checked against the code or measured.
- **llama.cpp:** commit `2e0d86b`. We built it and ran `llama-bench`, `llama-perplexity` and `llama-server` against rllm (numbers in [compare.md](compare.md)). Features we didn't exercise come from its documentation.
- **vLLM:** its documentation as of mid-2026. It has **not** been run here, so this file has no vLLM speed numbers. Both projects move quickly, so check their current docs before relying on a ✗ in their column.

Legend: ✓ supported · ◐ partial or limited · ✗ not supported.

## At a glance

| | rllm | llama.cpp | vLLM |
|---|---|---|---|
| Language, runtime | Rust, one binary | C/C++, one binary | Python + PyTorch |
| Typical use | GGUF serving on NVIDIA | Local inference on almost any hardware | Datacenter serving at high concurrency |
| Model architectures | 2 (Qwen3 dense, Gemma 4) | 100+ | 200+ (most Hugging Face architectures) |
| Weight formats | GGUF: F32, F16, BF16, Q8_0, Q4_K, Q6_K | GGUF, all quant types | safetensors (FP16/BF16, FP8, AWQ, GPTQ, NVFP4, …), GGUF experimental |
| Hardware | NVIDIA (CUDA), CPU (reference speed) | CPU, CUDA, Metal, Vulkan, HIP, SYCL and more | NVIDIA, AMD, Intel, TPU, CPU and more |
| Multi-GPU | ✗ | ✓ layer or row split | ✓ tensor, pipeline, data and expert parallel |
| Build needs | Rust; no CUDA toolkit (kernels compile at runtime with NVRTC) | C++ toolchain, plus a CUDA toolkit for GPU builds | pip wheels |

## Feature parity

Features all three have. rllm's versions are measured against llama.cpp
where a number exists.

| Feature | rllm | Notes |
|---|---|---|
| OpenAI chat completions, completions, embeddings, streaming | ✓ | Official `openai` SDK suite passes (47/47) on two models. |
| Continuous batching | ✓ | Requests join and leave each step. |
| Paged KV cache | ✓ | 16-token blocks. |
| Prefix caching | ✓ | Block-level, LRU eviction; covers chat turns, repeated system prompts, `n` choices. |
| Chunked prefill | ✓ | `--prefill-chunk`, interleaved with decoding. |
| Preemption when the KV cache is full | ✓ | The newest request pauses and resumes from the prefix cache. |
| Flash attention for prefill | ✓ | FlashAttention-2-style tensor-core kernels; split-context decode for long contexts. |
| CUDA graphs for decode | ✓ | Steps of up to 8 tokens. |
| Quantized matmuls with int8 activations | ✓ | MMQ (prefill) and dp4a matvec (decode), as in llama.cpp. |
| 8-bit KV cache | ✓ | `--kv-type q8`, keys Hadamard-rotated as in llama.cpp. Perplexity +0.19% (llama.cpp q8_0: +0.21%). vLLM uses FP8 instead. |
| Speculative decoding with a draft model | ✓ | Exact speculative sampling. vLLM also has EAGLE, MTP and n-gram drafting; see gaps. |
| JSON mode and `json_schema` | ◐ | Structure is enforced token by token; value constraints (`pattern`, `minimum`, `minLength`, …) are not. |
| Tool calling, including `required` and named tools | ✓ | Calls are constrained to the tools' schemas. Hermes/Qwen and Gemma 4 formats only. |
| Reasoning output (`reasoning_content`) | ✓ | `reasoning_effort` switches thinking on or off. |
| Chat templates from the GGUF (Jinja) | ✓ | minijinja. |
| Logprobs, `n`, `seed`, stop sequences, penalties, `logit_bias` | ✓ | |
| Prometheus metrics | ✓ | Engine, prefix cache, draft acceptance and HTTP series. |
| Tokenize / detokenize / apply-template endpoints | ✓ | llama-server style. |
| Perplexity and benchmark tools | ✓ | `rllm perplexity` and `rllm bench` use llama.cpp's methods, so numbers compare directly. |
| Matches llama.cpp's output | ✓ | Same tokenization; wikitext-2 perplexity within ±0.05% on Q8_0, BF16 and Nemotron, and 0.74% lower on Q4_K_M; KL near llama.cpp's own noise floor. |

## Where rllm leads

### Measured against llama.cpp (RTX 6000 Ada)

| Area | rllm | llama.cpp | |
|---|---:|---:|---|
| Serving, 4 concurrent clients, Qwen3-0.6B Q8_0 | 1874 tok/s | 545 tok/s | **3.4×** |
| Serving, 8 concurrent clients, Qwen3-0.6B Q8_0 | 2760 tok/s | 1176 tok/s | **2.3×** |
| Serving, 8 concurrent clients, Nemotron E4B Q8_0 | 744 tok/s | 485 tok/s | **1.5×** |
| Time to first token, 8 clients, Qwen3-0.6B | 14 ms | 262 ms | **19× sooner** |
| Chat follow-up turns, time to first token, Nemotron (sliding window) | 19–20 ms | 99–107 ms | **5× sooner** |
| Prompt processing (pp512), Qwen3-0.6B | 74–85k tok/s | 45–53k tok/s | **1.6×** |
| Generation (tg128), Qwen3-0.6B Q4_K_M | 809 tok/s | 718 tok/s | **1.13×** |
| Speculative decoding, Qwen3-8B + 0.6B draft, one request | 213 tok/s | 126 tok/s | **1.7×**; llama-server's speculation made it slower than no draft (146) |
| Decode at 32k context, Qwen3-8B, q8 KV cache | 88.0 tok/s | 79.5 tok/s (q8_0) | **1.11×**; also beats llama.cpp's f16 KV (82.3) |

### Features the others don't have, or don't do by default

- **Bit-identical results run to run.** Every GPU reduction sums in a fixed order, so the same tokens give the same logits on every run, and a test checks it on three models. llama.cpp's CUDA split-K and vLLM's default kernels sum in whatever order the hardware finishes. vLLM has a batch-invariant mode for this, at a throughput cost.
- **Speculative decoding that preserves greedy output exactly.** On Qwen3 a verification step computes the same logits as one-token steps, so greedy output with a draft is token for token the same as without one.
- **Complete OpenAI Responses API and stored completions.** These cover:
  - `previous_response_id`, background mode with cancel, and `input_items`;
  - stored chat completions with list, update, delete and messages endpoints;
  - persistence with `--store-file`.

  vLLM has a basic `/v1/responses`, and llama-server's support is newer and partial. Neither has the stored chat-completion endpoints.
- **Forced tool calls constrained to the schema in the model's native format,** including Gemma 4's non-JSON call syntax. Before constraining, Nemotron answered `"celsius"` for a `["c", "f"]` enum.
- **Fast start without Python.** A single binary; kernels compile on first run for the exact GPU, then load from a cache in about 0.7 s. vLLM typically takes tens of seconds to minutes (PyTorch import, `torch.compile`, graph capture).
- **A CPU reference for every kernel.** Every CUDA kernel has an f32 CPU reference implementation and a parity test against it. llama.cpp has its `test-backend-ops` tests; vLLM relies on comparing outputs with Hugging Face Transformers.

## Feature gaps

Ordered roughly by how much each one limits what rllm can serve.

### Models and formats

| Gap | llama.cpp | vLLM | Notes |
|---|---|---|---|
| Only two architectures (Qwen3 dense, Gemma 4) | ✓ 100+ | ✓ 200+ | No Llama, Mistral, Phi, DeepSeek or Qwen3-MoE yet. |
| MoE models | ✓ | ✓ | Planned for Phase 4 (`docs/plan.md`). |
| Six weight types | ✓ every GGUF type (Q2_K–Q6_K, Q4_0/Q5_0, IQ*, MXFP4, …) | ✓ FP8, AWQ, GPTQ, NVFP4, … | Q4_K_M and Q8_0 files work; Q5_K_M, IQ4_XS and similar don't load. |
| safetensors / Hugging Face checkpoints | ✗ | ✓ | GGUF only. |
| Multimodal (images, audio) | ✓ (libmtmd) | ✓ | Non-text input is rejected with 400. |
| LoRA adapters | ✓ | ✓ multi-LoRA serving | |
| Rerank and score endpoints | ✓ | ✓ | |
| Embedding models | ✓ | ✓ | rllm's embeddings are mean-pooled hidden states of a chat model, not a trained embedding model. |

### Hardware and scale

| Gap | llama.cpp | vLLM | Notes |
|---|---|---|---|
| One GPU per model | ✓ layer or row split | ✓ TP, PP, DP, EP | Tensor parallelism is Phase 3; no multi-GPU hardware in the dev setup yet. |
| NVIDIA only (sm_86 and newer) | ✓ Metal, Vulkan, HIP, SYCL, … | ✓ ROCm, Intel, TPU, … | ROCm is Phase 3; the kernels are written to be hipify-friendly. |
| Fast CPU inference | ✓ | ◐ | rllm's CPU backend is the f32 correctness reference: 0.2–0.35× llama.cpp on prompts, 0.5–0.94× on generation. |
| Partial GPU offload (model bigger than VRAM) | ✓ `-ngl` | ◐ CPU offload | The whole model must fit on one GPU. |
| FP8 compute and KV cache | ✗ | ✓ | Phase 4 (FP8 on Ada). |
| Several models per server | ✓ router mode | ◐ one per process | One model per `rllm serve`. |
| KV offload, disaggregated prefill/decode | ✗ | ✓ (LMCache, connectors) | |

### Serving and decoding

| Gap | llama.cpp | vLLM | Notes |
|---|---|---|---|
| Throughput at very high concurrency | ◐ | ✓ | Not benchmarked against vLLM yet. rllm's graph-replayed steps hold up to 8 tokens, and `--parallel` defaults to 4. The Phase 2 exit criterion (concurrency 16/32/64 vs vLLM) is still open. |
| Draft-free speculation (EAGLE, MTP, n-gram) | ◐ n-gram | ✓ | rllm only supports a separate draft model with the same tokenizer. |
| Prefix cache for the draft model | ✓ | ✓ | rllm's draft re-reads each request's whole prompt. |
| Speculation in the CLI | ✓ | n/a | `rllm run` has no `--draft`; only the server does. |
| Grammars and regex (GBNF, regex, full JSON Schema) | ✓ GBNF | ✓ xgrammar, guidance | rllm enforces JSON structure only, not value constraints. |
| Tool-call formats | ✓ many templates | ✓ many parsers | rllm parses Hermes/Qwen and Gemma 4 only. |
| Sampling options | ✓ min_p, repeat penalty, typical, mirostat, DRY, … | ✓ min_p, repetition penalty, … | rllm has temperature, top_p, top_k, frequency and presence penalties and `logit_bias`. |
| Web UI | ✓ | ✗ | |
| Fill-in-the-middle (`suffix`) | ✓ `/infill` | ◐ | Rejected with 400. |

### Performance gaps measured against llama.cpp

| Case | rllm | llama.cpp |
|---|---:|---:|
| Qwen3-8B Q4_K_M, prompt processing (pp512) | 6640 tok/s | 9380 tok/s (1.4×) |
| Qwen3-8B Q4_K_M, generation, empty context | 143 tok/s | 153 tok/s (1.07×) |
| Qwen3-8B Q4_K_M, generation at 32k, f16 KV | 71.4 tok/s | 82.3 tok/s (1.15×) |
| Qwen3-8B, prompt processing at 32k context | 2593 tok/s | 3261 tok/s (1.26×) |
| CPU backend, prompt processing | 0.2–0.35× | 1× |

rllm leads or ties on Qwen3-0.6B and Nemotron, the models it was tuned on.
The 8B gap is the next profiling target.

## Summary

- **Against llama.cpp**, rllm is a narrower engine that serves better.
  - On the models it supports, its output matches llama.cpp to within llama.cpp's own noise floor. Single-stream speed is equal or better, except on Qwen3-8B.
  - Under concurrent load it is 1.5–3.4× faster, with much lower time to first token.
  - What it lacks is breadth: architectures, quant types, hardware backends, multi-GPU, partial offload, multimodal.
- **Against vLLM**, rllm offers a single Rust binary that starts quickly, reads GGUF files and gives reproducible results.
  - vLLM is far ahead on model coverage, quantization formats, parallelism, hardware and advanced speculation.
  - Throughput at high concurrency hasn't been compared yet.
- **Next steps that close the most gaps:**
  1. Qwen3-8B kernel profiling.
  2. More architectures (Llama, Mistral) and the remaining K-quants (Q5_K, Q3_K, Q2_K).
  3. A vLLM benchmark at concurrency 16/32/64.
  4. Tensor parallelism once multi-GPU hardware is available.
