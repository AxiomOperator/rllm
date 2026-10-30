# rllm: where we are (2026-10-30)

Speed is shown as rllm ÷ llama.cpp on the same machine and model file:
above 1.00× means rllm is faster. Each bar is 10 blocks per 1.00×, and the
`│` after the tenth block marks parity. Full method and raw numbers are in
`compare.md`.

## At a glance

| | |
|---|---|
| GPUs | NVIDIA (CUDA, sm_86+), AMD RDNA3 (HIP/ROCm, tested on gfx1151); one binary, `-d gpu` picks either |
| Models | Qwen3 dense · Llama 2/3 · Mistral · Gemma 4 |
| Weights (GGUF) | F32 · F16 · BF16 · Q8_0 · Q4_K · Q5_K · Q6_K · IQ4_XS |
| Correctness | perplexity within noise of llama.cpp on every model and type checked; GPU kernels tested against a CPU reference on both vendors |
| Serving | OpenAI-compatible API, continuous batching, prefix cache with a RAM tier, q8 KV cache, speculative decoding (draft model and prompt lookup), sampling on the GPU |

## RTX 6000 Ada: rllm CUDA vs llama.cpp CUDA

| Model | Prompt (pp512) | | Generation (tg128) | |
|---|---:|---|---:|---|
| Qwen3-0.6B Q8_0 | **1.57×** | `▇▇▇▇▇▇▇▇▇▇│▇▇▇▇▇▇` | **1.07×** | `▇▇▇▇▇▇▇▇▇▇│▇` |
| Qwen3-0.6B Q4_K_M | **1.66×** | `▇▇▇▇▇▇▇▇▇▇│▇▇▇▇▇▇▇` | **1.13×** | `▇▇▇▇▇▇▇▇▇▇│▇` |
| Qwen3-0.6B Q5_K_M | **1.70×** | `▇▇▇▇▇▇▇▇▇▇│▇▇▇▇▇▇▇` | **1.09×** | `▇▇▇▇▇▇▇▇▇▇│▇` |
| Qwen3-0.6B IQ4_XS | **1.43×** | `▇▇▇▇▇▇▇▇▇▇│▇▇▇▇` | **1.14×** | `▇▇▇▇▇▇▇▇▇▇│▇` |
| Qwen3-0.6B BF16 | **1.61×** | `▇▇▇▇▇▇▇▇▇▇│▇▇▇▇▇▇` | **1.03×** | `▇▇▇▇▇▇▇▇▇▇│` |
| Llama-3.2-1B Q4_K_M | **1.06×** | `▇▇▇▇▇▇▇▇▇▇│▇` | 0.92× | `▇▇▇▇▇▇▇▇▇` |
| Nemotron E4B Q8_0 | **1.10×** | `▇▇▇▇▇▇▇▇▇▇│▇` | **1.01×** | `▇▇▇▇▇▇▇▇▇▇│` |
| Qwen3-8B Q4_K_M | 0.77× | `▇▇▇▇▇▇▇▇` | 0.98× | `▇▇▇▇▇▇▇▇▇▇` |

## Strix Halo (Radeon 8060S): rllm HIP vs llama.cpp ROCm

| Model | Prompt (pp512) | | Generation (tg128) | |
|---|---:|---|---:|---|
| Qwen3-0.6B Q8_0 | **1.36×** | `▇▇▇▇▇▇▇▇▇▇│▇▇▇▇` | **1.03×** | `▇▇▇▇▇▇▇▇▇▇│` |
| Qwen3-0.6B Q4_K_M | **1.44×** | `▇▇▇▇▇▇▇▇▇▇│▇▇▇▇` | **1.09×** | `▇▇▇▇▇▇▇▇▇▇│▇` |
| Qwen3-0.6B Q5_K_M | **1.43×** | `▇▇▇▇▇▇▇▇▇▇│▇▇▇▇` | **1.06×** | `▇▇▇▇▇▇▇▇▇▇│▇` |
| Qwen3-0.6B IQ4_XS | **1.36×** | `▇▇▇▇▇▇▇▇▇▇│▇▇▇▇` | **1.11×** | `▇▇▇▇▇▇▇▇▇▇│▇` |
| Qwen3-0.6B BF16 | **1.41×** | `▇▇▇▇▇▇▇▇▇▇│▇▇▇▇` | **1.13×** | `▇▇▇▇▇▇▇▇▇▇│▇` |
| Llama-3.2-1B Q4_K_M | **1.04×** | `▇▇▇▇▇▇▇▇▇▇│` | 0.90× | `▇▇▇▇▇▇▇▇▇` |
| Mistral-7B Q4_K_M | **1.06×** | `▇▇▇▇▇▇▇▇▇▇│▇` | 0.93× | `▇▇▇▇▇▇▇▇▇` |
| Nemotron E4B Q8_0 | **1.09×** | `▇▇▇▇▇▇▇▇▇▇│▇` | 0.96× | `▇▇▇▇▇▇▇▇▇▇` |
| Qwen3-8B Q4_K_M | **1.02×** | `▇▇▇▇▇▇▇▇▇▇│` | 0.92× | `▇▇▇▇▇▇▇▇▇` |
| Qwen3-8B, 16k context | 0.92× | `▇▇▇▇▇▇▇▇▇` | 0.93× | `▇▇▇▇▇▇▇▇▇` |
| Nemotron, 16k context | 0.88× | `▇▇▇▇▇▇▇▇▇` | 0.96× | `▇▇▇▇▇▇▇▇▇▇` |

## Serving under load: rllm vs vLLM (RTX 6000 Ada)

Total output tokens/s; each request is a ~130-token prompt and 192 output
tokens. Ratios are rllm ÷ vLLM.

| Clients | Qwen3-0.6B BF16: vLLM | rllm | ratio | Qwen3-8B 4-bit: vLLM (AWQ) | rllm (Q4_K_M) | ratio |
|---:|---:|---:|---|---:|---:|---|
| 1 | 445 | **502** | `▇▇▇▇▇▇▇▇▇▇│▇` 1.13× | 142 | **147** | `▇▇▇▇▇▇▇▇▇▇│` 1.04× |
| 16 | **4,619** | 4,452 | `▇▇▇▇▇▇▇▇▇▇` 0.96× | **1,733** | 912 | `▇▇▇▇▇` 0.53× |
| 32 | **8,136** | 6,695 | `▇▇▇▇▇▇▇▇` 0.82× | **2,749** | 1,480 | `▇▇▇▇▇` 0.54× |
| 64 | **11,324** | 8,625 | `▇▇▇▇▇▇▇▇` 0.76× | **3,472** | 2,093 | `▇▇▇▇▇▇` 0.60× |

At 64 clients, rllm's first token arrives sooner on the 0.6B model (45 ms
vs 91 ms). Before sampling moved to the GPU, rllm managed only 2,537 tok/s
there.

## Correctness: perplexity, rllm vs llama.cpp

wikitext-2, context 512, first 32 chunks. Lower is better, and the two
columns should match.

| Model | rllm | llama.cpp |
|---|---:|---:|
| Qwen3-0.6B Q5_K_M | 19.09 | 19.24 |
| Qwen3-0.6B IQ4_XS | 20.30 | 20.31 |
| Llama-3.2-1B Q4_K_M | 14.55 | 14.56 |
| Mistral-7B Q4_K_M | 5.95 | 5.96 |
| Mistral-7B IQ4_XS | 5.93 | 5.93 |

## What's next

| Gap | Size | Cause / fix |
|---|---|---|
| 8B serving with 9–32 clients | 0.53–0.60× vLLM | The quantized matmul uses fixed 64-token tiles; size them to the batch, as llama.cpp does |
| Qwen3-8B prompt on the Ada | 0.77× llama.cpp | Kernel profiling on sm_89 |
| AMD decode on the 1–8B models | 0.90–0.96× | Bandwidth tuning of the decode kernels |
| Coverage | | Phi, Qwen2, MoE; Q2_K/Q3_K, Q4_0/Q5_0 and the other IQ types; multi-GPU |
