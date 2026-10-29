Absolutely. We could build a high-performance LLM inference engine in Rust that combines the best capabilities of vLLM and llama.cpp. And given your existing NVIDIA and AMD hardware, I think this could become a substantial standalone project.

The important distinction is that I wouldn't recommend simply rewriting either application in Rust. I would design a new inference runtime with:

- Rust-native execution engine: Scheduling, memory management, model loading, concurrency, and distributed orchestration.
- llama.cpp-style flexibility: GGUF support, quantization, CPU/GPU hybrid inference, and heterogeneous hardware.
- vLLM-style performance: PagedAttention, continuous batching, prefix caching, speculative decoding, and optimized GPU execution.
- CUDA and ROCm: Native accelerator backends for NVIDIA and AMD hardware.
- Distributed inference: Run models across multiple GPUs and potentially multiple physical machines.
- OpenAI-compatible API: Work with your existing AI gateway and applications without requiring Python.

There is one important qualification: Rust alone doesn't make GPU inference faster. Most of the performance comes from optimized GPU kernels, memory movement, attention implementations, and request scheduling.

We can own the core architecture in Rust while using CUDA, HIP/ROCm, and carefully selected vendor libraries where those are the fastest options.

## 1. What already exists in Rust?

There are two projects particularly worth examining.

mistral.rs

Currently one of the closest examples of a Rust LLM-serving engine. It supports continuous batching, PagedAttention on supported GPUs, quantization, and streaming inference.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://docs.mistralrs.dev\&sz=32)

mistral.rs+1



Reference for our scheduler and inference-runtime architecture.

[GitHub repository](https://github.com/EricLBuehler/mistral.rs?utm_source=chatgpt.com)

Candle

A Rust tensor and machine-learning framework useful for building model execution code without adopting a complete inference server.

Potential foundation for tensor operations and model implementations.

[GitHub repository](https://github.com/huggingface/candle?utm_source=chatgpt.com)

llama.cpp

Particularly valuable as a reference for GGUF, CPU SIMD, quantized kernels, CUDA, HIP, and Vulkan.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub+1



[GitHub repository](https://github.com/ggml-org/llama.cpp?utm_source=chatgpt.com)

vLLM

Our primary reference for batching, KV-cache allocation, multi-GPU execution, and large-scale serving.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://docs.vllm.ai\&sz=32)

vLLM+1



[GitHub repository](https://github.com/vllm-project/vllm?utm_source=chatgpt.com)

My preference would be to study these implementations, adopt proven mathematical and scheduling concepts, and develop our own modular Rust engine rather than permanently depending on another inference server.

## 2. Proposed architecture

I would build a Rust-first, hardware-agnostic inference engine with a pluggable execution layer.

Client Applications

OpenAI API · Agents · Coding Assistants · RAG · Oxide UI

Rust Inference Server

Axum · Tokio · SSE Streaming · Authentication · Metrics

Inference Scheduler

Continuous Batching · Request Priorities · Chunked Prefill · Token Budgets

Model Runtime

GGUF · SafeTensors · Quantization · Tokenizer · Model Graph

Memory Manager

Paged KV Cache · Prefix Cache · GPU Allocator · Offload

Hardware Abstraction Layer

CUDA

NVIDIA

ROCm

AMD

CPU

AVX2/512

Local and Distributed Compute

RTX 6000 Ada · RTX 4500 Ada · A4500 · AMD Strix Halo

The architecture should separate the control plane from the actual tensor computation. That makes it possible to upgrade or replace a GPU kernel implementation without rewriting scheduling, API handling, model management, or distributed coordination.

## 3. Technology stack

| Component          | Proposed implementation                               |
| ------------------ | ----------------------------------------------------- |
| Core runtime       | Rust                                                  |
| API server         | Axum + Tokio                                          |
| Tensor abstraction | Custom traits and tensor descriptors                  |
| NVIDIA backend     | CUDA, cuBLASLt, custom kernels                        |
| AMD backend        | HIP, rocBLAS, hipBLASLt                               |
| CPU backend        | Rust + SIMD intrinsics                                |
| Model formats      | GGUF, SafeTensors                                     |
| Tokenization       | Hugging Face `tokenizers`                             |
| Scheduling         | Custom continuous-batching scheduler                  |
| KV cache           | Custom paged block allocator                          |
| Multi-GPU          | NCCL / RCCL adapters                                  |
| Multi-node         | gRPC/QUIC control plane, explicit tensor/KV transport |
| Observability      | OpenTelemetry + Prometheus                            |
| Deployment         | Docker Compose initially                              |

A particularly interesting recent development: NVIDIA introduced [CUDA Rust](https://github.com/NVIDIA/cuda-rust?utm_source=chatgpt.com) in September 2026, including mechanisms for writing CUDA GPU kernels in Rust itself.

However, NVIDIA still classifies the project as early-stage research, so I would use mature CUDA libraries for production-critical operations while experimenting with native Rust GPU kernels.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://nvidia.github.io\&sz=32)

CUDA Rust+2



For ROCm, Rust bindings such as `rocmrc` exist, but AMD GPU support would require substantial independent validation.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://docs.rs\&sz=32)

Docs.rs+1



## 4. What this would mean for your hardware

Your nine-machine AI deployment would be an excellent testing environment.

| Hardware                   | Recommended role                                       |
| -------------------------- | ------------------------------------------------------ |
| RTX 6000 Ada, 48 GB        | Main development and high-throughput inference         |
| 2× RTX 4500 Ada            | Parallel serving, scheduler and concurrency testing    |
| 3× RTX A4500 machine       | Multi-GPU tensor parallelism and communication testing |
| 5× Strix Halo, 128 GB each | ROCm, large GGUF models, CPU/GPU offloading            |
| 10 GbE network             | Distributed request routing and multi-node experiments |

The Strix Halo systems are particularly interesting because their large unified memory pools make them useful for running quantized models that exceed the dedicated VRAM of many discrete GPUs. The tradeoff is memory bandwidth and computation throughput.

I would not begin by trying to create one giant logical GPU spanning all nine machines. At 10 GbE, cross-node tensor communication can become a major bottleneck.

Instead, the engine should initially schedule complete models or independent model instances to appropriate nodes, then introduce tensor parallelism, pipeline parallelism, and distributed KV-cache transfer where benchmarks justify them.

## 5. Development roadmap

Phase 1 — Working inference engine

Estimated 4–8 weeks with focused development

Rust CLI, GGUF reader, model configuration, tokenization, basic transformer forward pass, CPU execution, CUDA execution, and streaming generation.

Goal: Generate correct output from one well-supported model family.

Phase 2 — Production-oriented performance

Estimated 8–16 additional weeks

Continuous batching, PagedAttention, KV-cache reuse, chunked prefill, custom quantized kernels, CUDA profiling, and OpenAI-compatible API.

Goal: Competitive throughput and latency on selected models.

Phase 3 — Heterogeneous hardware

Estimated 8–16 additional weeks

ROCm support, Strix Halo-specific optimizations, multi-GPU execution, multi-node scheduling, model placement, failure recovery, and memory offloading.

Goal: Operate across your NVIDIA and AMD nodes.

Phase 4 — Advanced inference

Ongoing optimization and expansion

Speculative decoding, MoE routing, structured generation, multimodal support, model hot-swapping, prefix-cache sharing, and distributed prefill/decode.

Goal: Expand model coverage and challenge mature inference engines on selected workloads.

These are planning estimates for an experienced small engineering team leveraging existing GPU libraries, not a commitment for a solo developer building all kernels from scratch. Matching vLLM's broad hardware and model support could take considerably longer.

## 6. The critical design decision

There are really two different projects we could undertake.

### Option A — Rust llama.cpp alternative

Focus on GGUF, quantization, CPU/GPU offloading, broad hardware compatibility, and maximum flexibility.

Lower initial complexity. Particularly relevant to your existing GGUF collection and Strix Halo machines.

### Option B — Rust vLLM alternative

Focus on PagedAttention, continuous batching, high-throughput GPU execution, multi-user serving, and distributed inference.

Higher complexity, but better suited to a multi-user inference platform.

I would actually choose a hybrid of both, but develop it in a deliberate order:

1. Build llama.cpp-style local inference with GGUF and CUDA.
2. Add the vLLM-style scheduler and memory architecture.
3. Introduce ROCm and heterogeneous execution.
4. Expand into distributed inference and larger-scale serving.

That gives us a usable engine relatively early instead of spending months building infrastructure without ever generating a token.

And I'd keep it independent of AegisLLM: AegisLLM would handle authentication, routing, usage tracking, and policy enforcement; the new Rust engine would own model execution and GPU memory.

This distinction is important if we eventually want to release the inference engine as a separate open-source or commercial product.

The first technical milestone I would target is concrete: load a Qwen-family GGUF model on the RTX 6000 Ada, generate output correctly, then benchmark it against llama.cpp using identical model weights, context lengths, sampling settings, and concurrency. That gives us a reliable foundation for every later optimization.
