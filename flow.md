# How rllm works

This file follows rllm from the command line down to the GPU kernels. The
diagrams are Mermaid, which GitHub renders. File names point to where each
step lives.

## 1. Crates

```mermaid
flowchart TD
    cli["rllm-cli<br/>the rllm binary: run, serve, bench, tune, …"]
    server["rllm-server<br/>HTTP API (axum) + engine thread"]
    model["rllm-model<br/>Qwen3 / Llama / Gemma 4 forward passes,<br/>KvPool, Generator, sampler"]
    tok["rllm-tokenizer<br/>BPE / SentencePiece + chat templates"]
    kv["rllm-kv<br/>block allocator, prefix cache"]
    gguf["rllm-gguf<br/>GGUF reader (mmap)"]
    core["rllm-core<br/>Backend trait, DType, dequant + sampling reference"]
    cpu["rllm-cpu<br/>CPU reference backend"]
    cuda["rllm-cuda<br/>backend.rs + gpu.rs (cudarc)"]
    hip["rllm-hip<br/>same backend.rs + gpu.rs (HIP via dlopen)"]
    kern["kernels.cu<br/>one source for both GPU vendors"]

    cli --> server & model & tok & gguf & cpu & cuda & hip
    server --> model & tok & kv
    model --> core & kv & gguf
    cpu & cuda & hip --> core
    cuda -. "NVRTC compiles" .-> kern
    hip -. "hiprtc compiles" .-> kern
```

Models are written against the `Backend` trait in `rllm-core`. The three
backends implement it, so the same model code runs on the CPU, NVIDIA and
AMD.

## 2. Commands

```mermaid
flowchart LR
    rllm(["rllm"]) --> run["run<br/>one prompt"] & serve["serve<br/>OpenAI-compatible server"]
    rllm --> bench["bench<br/>pp / tg / batched tg"] & ppl["perplexity"] & tune["tune<br/>recommend serve flags"]
    rllm --> cmp["compare / compare-ref<br/>logits vs CPU or llama.cpp"] & misc["inspect / tokenize"]
    run & bench & ppl & cmp --> dev{"-d"}
    serve --> dev
    dev -- "gpu (default)" --> pick{"CUDA device?"}
    pick -- yes --> C["CudaBackend"]
    pick -- no --> H["HipBackend"]
    dev -- "cuda[:N]" --> C
    dev -- "hip[:N]" --> H
    dev -- cpu --> P["CpuBackend"]
```

`Device::parse` and the `on_device!` macro in `rllm-cli/src/main.rs` pick the
backend. Each command is a generic function over `B: Backend`.

## 3. Startup (`rllm serve`)

```mermaid
sequenceDiagram
    participant CLI as rllm-cli
    participant G as Gguf (mmap)
    participant B as Backend
    participant M as AnyModel
    participant E as Engine thread
    participant H as HTTP server

    CLI->>G: open model.gguf (memory-mapped, no copy)
    CLI->>E: spawn engine thread (it owns the backend)
    E->>B: new(): load driver libs, compile kernels.cu<br/>(NVRTC / hiprtc, cached in ~/.cache/rllm/ptx), cuBLAS/hipBLAS
    E->>M: load: per tensor, upload + repack on the GPU<br/>(Q4_K → 4.625-bit layout, Q6_K, Q5_K, IQ4_XS, Q8_0)
    E->>E: KvPool (paged KV cache), prefix cache, optional host tier,<br/>optional draft model
    CLI->>H: tokenizer + chat template from GGUF metadata, routes
    H-->>CLI: listening on /v1
```

## 4. Lifecycle of a chat request

```mermaid
sequenceDiagram
    participant C as Client
    participant A as Handler (chat.rs)
    participant T as Tokenizer
    participant Gn as generation.rs
    participant E as Engine (engine.rs)

    C->>A: POST /v1/chat/completions
    A->>A: parse params (params.rs): sampling, tools,<br/>response_format → JSON constraint
    A->>T: render chat template (minijinja), encode
    A->>Gn: generate(prompt ids, GenParams)
    Gn->>E: submit Job::Generate (mpsc channel)
    loop every engine step
        E-->>Gn: GenEvent::Token {id, logprobs}
        Gn-->>A: detokenize incrementally; parse reasoning / tool calls (output.rs)
        A-->>C: SSE chunk (or buffered for non-streaming)
    end
    E-->>Gn: GenEvent::Done(stop | length | cancelled)
    A-->>C: final chunk + usage
```

`/v1/completions`, `/v1/responses` and `/v1/embeddings` follow the same path
with their own request and response shapes. Embeddings are a
`Job::Embed` that returns hidden states instead of tokens.

## 5. The engine loop (continuous batching)

One thread runs the loop, owns the backend and the KV pool, and serves every
request.

```mermaid
flowchart TD
    start([wait for jobs]) --> admit
    admit["Admit waiting requests while running &lt; --parallel<br/>• prefix cache: reuse KV blocks of a matching prompt prefix<br/>• host tier: copy matching blocks back from RAM"] --> plan
    plan["Plan the step<br/>• each decoding sequence: 1 token (+ drafts)<br/>• prompts: chunks within --prefill-chunk"] --> room{"enough KV blocks?"}
    room -- no --> evict["evict prefix cache (→ host tier if any),<br/>else preempt the newest sequence"] --> plan
    room -- yes --> draft
    draft["Speculation (decoding sequences, ≤ 4 running)<br/>• prompt lookup: copy what followed the last n-gram<br/>• draft model: propose k tokens (sequences ≤ 6144 tokens)"] --> fwd
    fwd["KvPool::step_sampled → Model::forward<br/>(one batched pass over every sequence's tokens)"] --> out
    out{"per sequence"}
    out -- "plain sampling" --> gpu_s["token sampled on the GPU<br/>(only the id comes back)"]
    out -- "logprobs / JSON constraint" --> host_s["logits row on the host:<br/>bias, penalties, constraint mask, sample"]
    out -- "drafts to verify" --> verify["accept / reject each draft<br/>(exact speculative sampling)"]
    gpu_s & host_s & verify --> emit["emit token(s) → client stream"]
    emit --> fin{"stop token, max_tokens,<br/>cancelled?"}
    fin -- yes --> retire["retire: KV blocks into the prefix cache"]
    fin -- no --> plan
    retire --> plan
```

`Scheduler::step` in `rllm-server/src/engine.rs` runs this loop. Metrics are
exposed at `/metrics`.

## 6. One forward pass (dense decoder: Qwen3, Llama, Mistral)

```mermaid
flowchart TD
    tok["token ids"] --> emb["embed"] --> n0["RMSNorm"]
    n0 --> layer
    subgraph layer["× n_layers"]
        qkv["Q, K, V projections (one launch)"] --> qkr["fused: Q/K RMSNorm (Qwen3 only) + RoPE (NeoX)<br/>+ write K, V into the paged cache (f16 or q8)"]
        qkr --> attn["paged attention over each sequence's blocks"]
        attn --> o["output projection"] --> res1["residual add + RMSNorm (fused)"]
        res1 --> ffn["gate·up → SiLU × up → down<br/>(fused GLU kernels)"] --> res2["residual add + next layer's RMSNorm (fused)"]
    end
    layer --> fn["final RMSNorm (wanted rows only)"] --> lm["vocab projection → logits"]
    lm --> fin{"finish_logits"}
    fin --> s1["sample_rows on device → ids"]
    fin --> s2["gather + copy rows to host"]
```

Gemma 4 (`gemma4.rs`) has the same outline plus its extras: sliding-window
and global layers with two KV pools, per-layer embeddings, logit soft-cap,
GELU, and 256/512-wide heads. For Llama, the Q/K rows are reordered at load
time so the NeoX RoPE kernel applies to them.

Steps of up to 8 tokens, and pure decode steps of up to 64 tokens, are
recorded once as a CUDA/HIP graph. Later steps with the same shape replay
that graph instead of launching about 300 kernels from the host.

## 7. Inside the backend: which kernel runs

```mermaid
flowchart TD
    mm["matmul(x, W)"] --> m{"rows m"}
    m -- "m ≤ 8 (decode)" --> mv["matvec: int8 activations · dp4a<br/>(long rows split over 2–4 warps on AMD)"]
    m -- "m > 8, quantized, large W" --> mmq["MMQ: int8 matrix cores<br/>mma.sync (NVIDIA) / WMMA (RDNA3)"]
    m -- "otherwise" --> gemm["dequant → f16 GEMM<br/>cuBLAS · hipBLAS / hipBLASLt (16-bit out, widened)"]

    at["attention"] --> a{"step"}
    a -- "decode, or ≤ 8 tokens" --> ad["attn_decode: one block per (token, KV head),<br/>long contexts split per token"]
    a -- "prefill, NVIDIA" --> fa["attn_fa (FlashAttention-2, tensor cores)"]
    a -- "prefill, AMD gfx11" --> wa["attn_wmma (WMMA, head dims split over waves)"]
    a -- "fallback" --> pl["attn_prefill / paged_attention (portable)"]
```

`rllm-cuda/src/backend.rs` holds all of this logic once. Each crate compiles
it against its own `gpu` module (`rllm-cuda/src/gpu.rs` or
`rllm-hip/src/gpu.rs`), which supplies streams, memory, launches, graphs and
BLAS. Vendor differences are small flags in that module, such as
`GEMM_HALF_OUT`, `MATVEC_SPLIT_K` and `wmma`, and `#ifdef RLLM_HIP` blocks in
`kernels.cu`.

## 8. Correctness checks along the flow

| Where | Check |
|---|---|
| Every GPU kernel | `tests/common/parity.rs`: GPU against the CPU reference on random inputs (CUDA and HIP) |
| Whole model | `rllm compare` (CPU logits at each step), `rllm compare-ref` (llama.cpp logit dumps), `rllm perplexity` |
| Engine invariants | `rllm-model/tests/common/gpu.rs`: deterministic steps, verification = single steps, q8 tracks f16, host tier round trip |
| Server | stress test (30k-token prompt + 7 concurrent requests), prompt lookup, host tier, concurrency benchmark (`compare.md`) |
