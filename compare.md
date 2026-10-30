# rllm vs llama.cpp: correctness comparison

Measured 2026-10-28 (GPU sections updated 2026-10-29, including an RTX 6000 Ada) on the dev laptop (RTX PRO 3000 Blackwell, Arrow Lake CPU
with AVX2 and AVX-VNNI). Both engines ran on the CPU for the correctness
comparison. The performance section also lists rllm's CUDA numbers.

- **llama.cpp:** commit `2e0d86b`, CPU-only build.
- **rllm:** CPU backend.

Method: `tools/llama-ref` dumps llama.cpp's token ids and full logits for a prompt
plus 16 greedy tokens. `rllm compare-ref` feeds rllm the *same token ids* and
compares the two at every position. See [tools/llama-ref/README.md](tools/llama-ref/README.md).

## How to read this

The two engines can't match bit for bit:

| Step | llama.cpp (CPU) | rllm |
|---|---|---|
| Activations entering a quantized matmul | rounded to 8 bits | f32 |
| KV cache | f16 | f32 |
| Attention accumulation (flash attention) | f16 | f32 |

So the bar for "same model" is llama.cpp's **noise floor**: how much its own
logits move when only its numeric settings change. That means comparing default
settings against an f32 KV cache with flash attention off. If rllm differs from
llama.cpp by about that much, the remaining gap is numerics, not a model bug.

**Metric:** mean KL divergence between the two engines' next-token
distributions, averaged over every position (in nats; lower is closer).

## Results

| Model | Weights | Tokenization | KL: rllm vs llama.cpp | KL: rllm (llama.cpp rounding) vs llama.cpp (f32 attention) | KL: llama.cpp vs itself (noise floor) | Top-1 agreement |
|---|---|---|---:|---:|---:|---|
| Qwen3-0.6B, plain text | BF16 | identical | 8.8e-5 | 6.5e-5 | 9.4e-5 | 54/54 |
| Qwen3-0.6B, plain text | Q8_0 | identical | 9.0e-3 | 1.9e-3 | 1.3e-3 | 54/54 · 53/54 |
| Qwen3-0.6B, chat | Q8_0 | identical | 4.5e-3 | 1.4e-3 | 1.5e-3 | 35/39 · 35/39 |
| Qwen3-0.6B, plain text | Q4_K_M | identical | 9.4e-3 | 8.8e-3 | 1.25e-2 | 51/54 · 50/54 |
| Nemotron 3.5 Bolt E4B (Gemma 4), chat | Q8_0 | identical | 3.7e-3 | 6.8e-3 | 6.0e-3 | 32/35 · 31/35 |
| Nemotron, 686-token prompt (longer than the 512-token sliding window) | Q8_0 | identical | 5.4e-4 | — | — | 689/692 |
| Tokenizer stress test, both tokenizers | — | identical (130 tokens each) | — | — | — | — |

Top-1 agreement is shown for the default comparison, then for the matched-rounding
comparison. Almost every top-1 disagreement is at a low-confidence position
inside the prompt or chat template. The only disagreements in generated text
were in the Qwen3 runs, where llama.cpp's own two settings also disagree. On the
long prompt, both engines answer the retrieval question correctly ("5").

## Chart: difference relative to llama.cpp's noise floor

Each bar is rllm's KL against llama.cpp divided by llama.cpp's KL against
itself. **At or below 1.0 (the line) means rllm differs from llama.cpp no more
than llama.cpp differs from itself.**

```mermaid
xychart-beta
    title "rllm vs llama.cpp, KL / llama.cpp noise floor (<= 1.0 is within noise)"
    x-axis ["Qwen3 BF16", "Qwen3 Q8_0 plain", "Qwen3 Q8_0 chat", "Qwen3 Q4_K_M", "Nemotron Q8_0"]
    y-axis "ratio" 0 --> 7
    bar [0.93, 6.83, 3.04, 0.75, 0.61]
    bar [0.69, 1.41, 0.98, 0.70, 1.14]
    line [1, 1, 1, 1, 1]
```

The first bar in each group is rllm's default f32 numerics. The second is rllm
mimicking llama.cpp's activation rounding (`RLLM_CPU_LLAMACPP_NUMERICS=1`),
compared against llama.cpp with f32 attention.

The tall default bars for Q8_0 come from a known numeric difference, not a bug.
llama.cpp rounds activations to 8 bits before every Q8_0 matmul, and rllm
doesn't. When rllm applies the same rounding, the ratio drops to about 1.

## CUDA vs CPU reference (rllm only)

`rllm compare` runs the same tokens through the CUDA backend and the CPU
reference: the prompt in one batch, then 24 decode steps, which run through
CUDA graph replay.

| Model | Max relative logit diff | Top-1 agreement |
|---|---:|---|
| Qwen3-0.6B Q8_0 | 2.3e-2 | 25/25 |
| Qwen3-0.6B Q4_K_M | 2.6e-2 | 25/25 |
| Qwen3-0.6B BF16 | 2.8e-3 | 25/25 |
| Nemotron Q8_0 | 2.9e-2 | 25/25 |

The CUDA backend now uses the same reduced-precision shortcuts as llama.cpp's
CUDA backend:
- quantized matmuls round activations to int8
- the KV cache is f16
- prefill attention uses f16 tensor cores

So it no longer matches rllm's f32 CPU reference to 1e-6 the way the first
version did. The per-kernel parity tests (`crates/rllm-cuda/tests/common/parity.rs`)
check each kernel against the CPU at tolerances that match each format.

## Performance: prompt processing and generation

Same method as llama.cpp's `llama-bench`:
- **pp512 (prompt processing):** one 512-token prompt processed as a single batch.
- **tg128 (generation):** 128 tokens decoded one at a time from an empty context.

Each figure is tokens per second, mean ± standard deviation over 3 runs after
one warmup. The commands are `llama-bench -t 8 -p 512 -n 128 -r 3` and
`rllm bench -p 512 -n 128 -r 3`.

**Setup:**
- Intel Core Ultra 9 285H with 16 cores (6 performance, 8 efficient, 2
  low-power), no hyperthreading.
- Both engines run on the CPU with 8 threads. A sweep of 6, 8 and 16 threads
  found 8 fastest for both. 16 threads is slower because the efficient cores
  drag down the synchronized work. llama-bench defaults to 3 threads here.
- This section compares CPU against CPU. rllm's CUDA numbers (current, after
  the GPU optimization work) are listed for reference. For llama.cpp's own GPU backend, see
  [GPU vs GPU](#gpu-vs-gpu-llamacpp-cuda-vs-rllm-cuda) below.
- The laptop GPU runs at power-limited clocks (about 900 MHz SM under load), so
  CUDA numbers here are well below what the RTX 6000 Ada box should do.

### Prompt processing (pp512, tokens/s)

| Model | llama.cpp CPU | rllm CPU | rllm ÷ llama.cpp (CPU) | rllm CUDA |
|---|---:|---:|---:|---:|
| Qwen3-0.6B Q8_0 | 413.82 ± 2.50 | 142.72 ± 1.69 | 0.34× | 17011.01 ± 301.72 |
| Qwen3-0.6B Q4_K_M | 571.19 ± 7.40 | 145.57 ± 1.77 | 0.25× | 17225.80 ± 86.11 |
| Qwen3-0.6B BF16 | 409.74 ± 7.28 | 145.40 ± 2.30 | 0.35× | 18208.22 ± 56.88 |
| Nemotron 3.5 Bolt E4B Q8_0 (7.5B) | 66.75 ± 0.44 | 13.48 ± 0.09 | 0.20× | 3479.10 ± 29.20 |

### Generation (tg128, tokens/s)

| Model | llama.cpp CPU | rllm CPU | rllm ÷ llama.cpp (CPU) | rllm CUDA |
|---|---:|---:|---:|---:|
| Qwen3-0.6B Q8_0 | 53.90 ± 0.56 | 42.03 ± 0.41 | 0.78× | 238.13 ± 0.94 |
| Qwen3-0.6B Q4_K_M | 69.95 ± 1.37 | 34.08 ± 0.18 | 0.49× | 279.45 ± 0.30 |
| Qwen3-0.6B BF16 | 32.30 ± 0.53 | 30.20 ± 0.21 | 0.94× | 164.98 ± 0.21 |
| Nemotron 3.5 Bolt E4B Q8_0 (7.5B) | 8.84 ± 0.11 | 6.24 ± 0.06 | 0.71× | 46.13 ± 1.63 |

### Charts (CPU vs CPU)

The first bar in each group is llama.cpp, the second is rllm.

```mermaid
xychart-beta
    title "Prompt processing, pp512 on CPU (tokens/s, higher is better)"
    x-axis ["Qwen3 Q8_0", "Qwen3 Q4_K_M", "Qwen3 BF16", "Nemotron Q8_0"]
    y-axis "tokens/s" 0 --> 600
    bar [413.82, 571.19, 409.74, 66.75]
    bar [142.72, 145.57, 145.40, 13.48]
```

```mermaid
xychart-beta
    title "Generation, tg128 on CPU (tokens/s, higher is better)"
    x-axis ["Qwen3 Q8_0", "Qwen3 Q4_K_M", "Qwen3 BF16", "Nemotron Q8_0"]
    y-axis "tokens/s" 0 --> 75
    bar [53.90, 69.95, 32.30, 8.84]
    bar [42.03, 34.08, 30.20, 6.24]
```

### rllm CUDA vs rllm CPU

```mermaid
xychart-beta
    title "rllm CUDA speedup over rllm CPU (x)"
    x-axis ["Qwen3 Q8_0 pp", "Qwen3 Q8_0 tg", "Nemotron pp", "Nemotron tg"]
    y-axis "speedup (x)" 0 --> 260
    bar [119.2, 5.7, 258.1, 7.4]
```

### Reading the numbers

- **Generation on CPU is within reach.** rllm reaches 0.49–0.94× llama.cpp.
  Decode is limited by memory bandwidth, and both engines read every weight once
  per token. The gap is widest on Q4_K. llama.cpp multiplies the 4-bit values
  directly with integer instructions (AVX-VNNI), while rllm's CPU backend first
  expands each weight row to f32.
- **Prompt processing on CPU is 3–5× behind.** rllm's CPU backend is the
  correctness reference: it expands one weight row at a time and does plain f32
  dot products. llama.cpp uses repacked weights, 8-bit integer matrix kernels and
  cache-blocked GEMM. Closing this gap is optimization work on the CPU backend,
  not a correctness issue.
- **rllm on CUDA** is 118–258× faster than rllm's CPU backend for prompt
  processing and 5.5–8.2× for generation, even at this laptop's power-limited
  clocks. See the GPU section below for how it compares with llama.cpp's CUDA
  backend.

## GPU vs GPU: llama.cpp CUDA vs rllm CUDA

Two GPUs:

| | Laptop | Benchmark box |
|---|---|---|
| GPU | RTX PRO 3000 Blackwell Laptop (sm_120, 46 SMs) | RTX 6000 Ada (sm_89, 142 SMs, 48 GB) |
| Clocks | power-limited, about 1 GHz SM | up to 3.1 GHz SM, 300 W |
| CPU | Core Ultra 9 285H | Core i9-14900K |
| OS | Fedora 45 | Fedora 44 |

**How llama.cpp's CUDA backend was built:** CUDA 13.4 toolkit (nvcc, ptxas,
NVVM, cudart, CCCL, cuBLAS) assembled from NVIDIA's PyPI packages into
`.cuda/tk`, since there's no system toolkit and no sudo. Host compiler GCC
16.2, llama.cpp `2e0d86b`. The laptop build targets `sm_120a`
(`third_party/llama.cpp/build-cuda`). The Ada build
(`build-cuda89`) targets `sm_89` with `GGML_NATIVE=OFF` (AVX2, AVX-VNNI, FMA,
F16C) and `LLAMA_OPENSSL=OFF`, so the binaries run on the box's Fedora 44.
The box has no Rust or CUDA toolchain: rllm, llama.cpp and their CUDA
libraries were built on the laptop and copied to `~/rllm-bench` there (rllm
compiles its kernels for sm_89 with NVRTC on first run).

**How it was run:**
- llama.cpp: `llama-bench -ngl 99 -p 512 -n 128 -r 3` with default settings
  (flash attention on auto, f16 KV cache).
- rllm: `rllm bench -d cuda -r 3`.
- Each model ran llama.cpp then rllm back to back, so both see the same
  thermal state.
- The Ada box also runs an idle `llama-server` (32 GB, 0% utilization during
  the runs; checked before and after).

### Speed on the RTX 6000 Ada (tokens/s)

Final run, 2026-10-29.

| Model | pp512, llama.cpp CUDA | pp512, rllm CUDA | pp512, rllm ÷ llama.cpp | tg128, llama.cpp CUDA | tg128, rllm CUDA | tg128, rllm ÷ llama.cpp |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B Q8_0 | 47280.77 ± 10369.29 | 74120.87 ± 52.46 | **1.57×** | 636.16 ± 16.45 | 683.42 ± 0.52 | **1.07×** |
| Qwen3-0.6B Q4_K_M | 45029.89 ± 9279.18 | 74905.97 ± 53.60 | **1.66×** | 718.24 ± 10.93 | 809.21 ± 0.42 | **1.13×** |
| Qwen3-0.6B BF16 | 52833.88 ± 14033.81 | 84889.99 ± 67.86 | **1.61×** | 478.44 ± 2.44 | 493.86 ± 2.01 | **1.03×** |
| Nemotron 3.5 Bolt E4B Q8_0 (7.5B) | 12641.96 ± 696.79 | 13943.39 ± 273.69 | **1.10×** | 129.59 ± 0.27 | 131.34 ± 0.03 | **1.01×** |

llama.cpp's prompt numbers on the 0.6B model swing by ±20% between its three
repetitions; rllm's don't. The BF16 and Nemotron rows were re-run back to
back after the last decode changes (float-weight fused FFN; fused
norm + gate/up for Gemma). Nemotron generation is 1.014× (131.34 ± 0.03 vs
129.59 ± 0.27), outside both engines' spread; a 64-chunk perplexity check
after the change gave 114.64 vs llama.cpp's 114.86.

**The Ada found problems the laptop hid.** At 3 GHz a decode step is 1.5–8 ms,
so fixed per-token costs that were invisible at 1 GHz showed up. The first Ada
run had rllm at 0.68× llama.cpp for Nemotron generation and 0.95–0.97× for the
Qwen3 models; the fixes are in the table below (the rows marked *Ada*).

### Speed on the laptop (tokens/s)

Final run, 2026-10-29, both engines back to back per model. The SM clock was
about 1 GHz, and the laptop ran warmer than in earlier runs, so absolute
numbers are lower than before for both engines. Run-to-run spread here is
about ±3%, so ratios within a few percent of 1.0 are ties.

| Model | pp512, llama.cpp CUDA | pp512, rllm CUDA | pp512, rllm ÷ llama.cpp | tg128, llama.cpp CUDA | tg128, rllm CUDA | tg128, rllm ÷ llama.cpp |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B Q8_0 | 14357.03 ± 619.89 | 17011.01 ± 301.72 | **1.18×** | 218.53 ± 0.89 | 238.13 ± 0.94 | **1.09×** |
| Qwen3-0.6B Q4_K_M | 13742.72 ± 940.56 | 17225.80 ± 86.11 | **1.25×** | 264.12 ± 0.88 | 279.45 ± 0.30 | **1.06×** |
| Qwen3-0.6B BF16 | 15750.89 ± 619.57 | 18208.22 ± 56.88 | **1.16×** | 150.54 ± 0.88 | 164.98 ± 0.21 | **1.10×** |
| Nemotron 3.5 Bolt E4B Q8_0 (7.5B) | 3428.00 ± 23.30 | 3479.10 ± 29.20 | 1.01× | 44.93 ± 1.74 | 46.13 ± 1.63 | 1.03× |

Nemotron on the laptop is a tie within noise in this run; the previous run,
in a cooler state, measured 1.04× for prompt processing.

**Before and after the GPU work** (rllm CUDA on the laptop):

| Model | pp512 before | pp512 after | speedup | tg128 before | tg128 after | speedup |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B Q8_0 | 2333.69 | 17011.01 | 7.3× | 108.72 | 238.13 | 2.2× |
| Qwen3-0.6B Q4_K_M | 2345.30 | 17225.80 | 7.3× | 102.34 | 279.45 | 2.7× |
| Qwen3-0.6B BF16 | 2322.38 | 18208.22 | 7.8× | 112.06 | 164.98 | 1.5× |
| Nemotron Q8_0 | 703.46 | 3479.10 | 4.9× | 19.41 | 46.13 | 2.4× |

### Charts

The first bar in each group is llama.cpp CUDA, the second is rllm CUDA.

```mermaid
xychart-beta
    title "RTX 6000 Ada: prompt processing, pp512 (tokens/s)"
    x-axis ["Qwen3 Q8_0", "Qwen3 Q4_K_M", "Qwen3 BF16", "Nemotron Q8_0"]
    y-axis "tokens/s" 0 --> 90000
    bar [47280.77, 45029.89, 52833.88, 12641.96]
    bar [74120.87, 74905.97, 84889.99, 13943.39]
```

```mermaid
xychart-beta
    title "RTX 6000 Ada: generation, tg128 (tokens/s)"
    x-axis ["Qwen3 Q8_0", "Qwen3 Q4_K_M", "Qwen3 BF16", "Nemotron Q8_0"]
    y-axis "tokens/s" 0 --> 900
    bar [636.16, 718.24, 478.44, 129.59]
    bar [683.42, 809.21, 493.86, 131.34]
```

```mermaid
xychart-beta
    title "rllm CUDA / llama.cpp CUDA (1.0 = parity): RTX 6000 Ada"
    x-axis ["Q8_0 pp", "Q8_0 tg", "Q4_K_M pp", "Q4_K_M tg", "BF16 pp", "BF16 tg", "Nemotron pp", "Nemotron tg"]
    y-axis "ratio" 0 --> 1.8
    bar [1.57, 1.07, 1.66, 1.13, 1.61, 1.03, 1.10, 1.01]
    line [1, 1, 1, 1, 1, 1, 1, 1]
```

```mermaid
xychart-beta
    title "rllm CUDA / llama.cpp CUDA (1.0 = parity): laptop"
    x-axis ["Q8_0 pp", "Q8_0 tg", "Q4_K_M pp", "Q4_K_M tg", "BF16 pp", "BF16 tg", "Nemotron pp", "Nemotron tg"]
    y-axis "ratio" 0 --> 1.4
    bar [1.18, 1.09, 1.25, 1.06, 1.16, 1.10, 1.01, 1.03]
    line [1, 1, 1, 1, 1, 1, 1, 1]
```

### Correctness on GPU

`llama_ref` was built against each CUDA build and run with all layers on the
GPU, then compared with `rllm compare-ref -d cuda` on the same machine. The
noise floor is llama.cpp's own CUDA output vs its own CPU output on the same
prompt.

| Model | KL: rllm vs llama.cpp, laptop | Floor, laptop | Top-1, laptop | KL: rllm vs llama.cpp, Ada | Floor, Ada | Top-1, Ada |
|---|---:|---:|---|---:|---:|---|
| Qwen3-0.6B Q8_0 | 8.93e-3 | 1.60e-3 | 52/54 | 8.37e-3 | 1.16e-3 | 54/54 |
| Qwen3-0.6B Q4_K_M | 1.20e-2 | 2.40e-2 | 52/54 | 1.16e-2 | 7.08e-3 | 51/54 |
| Qwen3-0.6B BF16 | 1.28e-4 | 1.23e-4 | 54/54 | 8.44e-5 | 4.61e-5 | 54/54 |
| Nemotron Q8_0, chat | 4.4e-3 – 5.2e-3 (4 runs) | 4.13e-3 | 33–35/35 | 6.3e-3 – 6.9e-3 (2 runs) | 5.59e-3 | 34/35 |
| Nemotron Q8_0, 686-token prompt | 5.3e-4 | — | 701/702 | — | — | — |

Tokenization is identical in every case. `rllm compare` (CUDA vs rllm's CPU
reference) also passes on the Ada: no top-1 disagreements on Q4_K_M, BF16 and
Nemotron over 25 steps, and one on Q8_0, at a position where the CPU
reference's top two tokens are 0.029 logits apart.

- **Nemotron is about 1.1–1.2× llama.cpp's floor on both GPUs; BF16 is at it
  in absolute terms (KL below 1e-4).** When these were measured, rllm's CUDA
  output varied slightly from run to run: split-K matmuls summed partial
  results with `atomicAdd`, whose order isn't fixed. On the 35-position chat
  prompt that moved the KL between 3.5e-3 and 6.8e-3 across recent builds.
  Since then the sums run in a fixed order and the same tokens give
  bit-identical logits on every run (see
  [Deterministic reductions](#deterministic-reductions-q8-kv-cache-and-speculative-decoding)).
- **Q8_0 sits above the floor,** as in every comparison since the start, and
  Q4_K_M is above the (smaller) Ada floor. The disagreements are at
  low-confidence positions in the prompt; generated text matches.

### Perplexity (wikitext-2 test set, RTX 6000 Ada)

`rllm perplexity` uses llama-perplexity's method: tokenize the whole file
once, cut it into 512-token chunks, start each chunk with BOS when the model
uses one, and score the second half of each chunk (255 next-token predictions
per chunk). Both engines ran on the full `wiki.test.raw` from wikitext-2-raw-v1
with `-c 512`, all layers on the GPU:

| Model | Chunks | llama.cpp CUDA | rllm CUDA | rllm − llama.cpp |
|---|---:|---:|---:|---:|
| Qwen3-0.6B Q8_0 | 584 | 21.9497 ± 0.19166 | 21.9569 ± 0.19191 | +0.03% |
| Qwen3-0.6B Q4_K_M | 584 | 22.9116 ± 0.20046 | 22.7416 ± 0.19891 | −0.74% |
| Qwen3-0.6B BF16 | 584 | 21.8891 ± 0.19150 | 21.8864 ± 0.19148 | −0.01% |
| Nemotron 3.5 Bolt E4B Q8_0 | 576 | 128.4028 ± 1.86834 | 128.3503 ± 1.86730 | −0.04% |

Same chunk counts, so the tokenizations agree over the whole file. Every
difference is far inside the ± (the standard error, computed the same way
by both tools). Nemotron's high value is the model's, not rllm's: it is a
chat model scored on raw encyclopedia text, and llama.cpp gives the same
number. A short laptop check (8 chunks, Qwen3 Q8_0) gave 21.66 vs 21.62.

```sh
rllm perplexity model.gguf wiki.test.raw -c 512 [--chunks N] [-d cuda|cpu]
llama-perplexity -m model.gguf -f wiki.test.raw -c 512 -ngl 99
```

### What was fixed

The first GPU-vs-GPU run found rllm at 0.13–0.21× llama.cpp for prompt
processing and 0.38–0.75× for generation. Changes, roughly in order of impact:

| Area | Problem found | Fix |
|---|---|---|
| Prefill attention | 57% of prefill time; one block per token-head walking positions one at a time | FlashAttention-2-style kernels `attn_fa_d{64,128,256,512}`: `mma.m16n8k16`, online softmax, K/V shared across the GQA group, `cp.async` double-buffered K/V tiles, `ldmatrix(.trans)` fragment loads, per-head-dim tile and warp layouts |
| Decode attention | One block per query head, a shuffle reduction per position, serial V loads | `attn_decode_g{G}_d{HD}`: one block per KV head serving its whole GQA group, lane-per-position scoring, batched V loads, templated so state stays in registers |
| Decode matvec | Byte-wise reads of GGUF blocks, about 35–130 GB/s | Weights repacked on upload into aligned device formats; int8 activations with `dp4a` (same idea as llama.cpp's MMVQ); Q/K/V and gate/up share one quantization and one launch (`matmul_many`) |
| KV cache | f32: double the attention bandwidth and memory | f16 KV cache; f16 staging is now plain 16-byte copies |
| Prefill matmuls | Every weight re-expanded to f16 on every pass | int8 tensor-core MMQ (`mma.m16n8k32`, `ldmatrix` fragments, 128×64 block tiles, split-K picked by a wave-cost model) for quantized weights of 4M+ elements; BF16 weights passed to cuBLAS as-is; f16 copies of small f32 matrices; 32 MB cuBLAS workspace |
| Activation quantization | A separate quantize launch before every quantized matmul | Quantization fused into the producing kernels (norms, `qkv_norm_rope_store`, the FFN kernels, and on large GPUs prefill attention), only where a consumer reads the int8 copy; the f32 output is skipped when nothing reads it. The standalone `quantize_q8` uses 8 lanes per 32-value unit with vector loads (37 → 18 µs per call) |
| Prefill attention output (*Ada*) | One standalone quantize launch per layer before the output projection | On GPUs with 80+ SMs the FlashAttention epilogue stages each warp's output slab in the freed K/V shared memory and writes the int8 copy (one lane per 32-value unit): +2% Nemotron prefill on the RTX 6000 Ada. On the 46-SM laptop the same epilogue measured 1.2% slower, so there it stays a separate launch (`RLLM_FA_QUANT=0/1` overrides) |
| FFN, prefill | gate, up, activation and down as separate launches with f32 intermediates | `mmq_glu_*` computes gate and up in one MMQ (each block holds 64 gate rows and the matching 64 up rows), stages both in shared memory (one swizzled 32 KB buffer, so 3 blocks per SM still fit), and each thread applies SiLU/GELU and writes one 32-value unit of the down projection's int8 input. Its cost is within 1% of a plain MMQ of the same size |
| FFN, decode (*Ada*) | gate/up matvec, then a separate activation launch per layer | `matvec_glu_*`: for quantized weights, 32 rows per block, each warp computes its gate and up rows, and the block writes the down projection's int8 input directly; for F16/BF16/F32 weights, each warp writes act(gate) * up (BF16 decode on the Ada +1.7%) |
| Logit softcap (*Ada*) | Gemma's `cap * tanh(x / cap)` ran on the host over all 262,144 logits every token: about 3 ms per token at Ada speeds | `softcap` backend op, run on the GPU before the copy (Nemotron decode on the Ada 88 → 123 tok/s) |
| Per-step host overhead (*Ada*) | The decode graph launched only after the host had walked all ops for the step (240 µs per Qwen step), and every op read environment variables | Debug switches read once at startup; the replay graph launches as soon as the step's last index array is written, so the remaining walk overlaps the GPU; models skip the identity logits gather in decode |
| Gemma post-attention norm, decode (*Ada*) | A one-row `rms_norm_add` launch between the attention output projection and the FFN | `norm_add_gated_ffn`: the gate/up matvec blocks recompute the residual update and the FFN norm in a prologue and read the int8 input from shared memory (`matvec_glu_norm_*`); the residual moves to a second buffer and back, so no block overwrites what another reads (+0.7% Nemotron decode on the Ada) |
| Gemma per-layer inputs, decode (*Ada*) | Four small launches per layer (norm, F32 matvec, gather, activation), each mostly launch latency | `ple_gate`: one kernel recomputes the cheap norm per block, does the 256×2560 matvec with 4 warps per row, and applies the gated activation in its epilogue |
| Float matvec | One warp per row left most SMs idle on 256-row weights | Rows split across 4 warps when rows are few (kw is a template parameter: a runtime stride cost the common case 10%) |
| Ada (sm_89) codegen | The plain MMQ kernels kept their 64 accumulators in a 256-byte local-memory stack frame on sm_89 (not on sm_120); `rms_norm_add` needed 96 registers × 1024 threads and failed to launch (`CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES`) on the real card | Store loops fully unrolled; `__launch_bounds__(1024)` on the norm kernels. A unit test compiles all kernels for sm_89 and sm_86, and `ptxas -arch=sm_89` shows no stack frames or spills |
| Norms | `rms_norm_add` used 1024 threads for 2560-wide rows and re-read its inputs | Row values kept in registers, 8 per thread (prefill 94 → 62 µs per call); decode-sized calls keep wide blocks |
| Split-K | Fixed 5% cost per extra split | Penalty scaled by 2560 / k, so long-k matmuls (the FFN down projection) split further |
| Launch overhead | about 25 kernels per layer, fresh allocations per step | Fused add+RMSNorm, RMSNorm+RoPE, Q/K-norm+RoPE+KV-store, and Gemma's post-norm+residual+next-norm; CUDA graphs for decode steps (arena allocations, pinned index uploads captured as memcpy nodes, graphs invalidated when a captured buffer is reallocated) |
| Startup | NVRTC compile about 6 s per run | Compiled PTX cached in `~/.cache/rllm/ptx` (startup about 0.7 s) |

Tried and reverted because they measured slower: pinned staging for the
logits copy (cudarc's pinned memory is write-combined, so reading it on the
host halved Nemotron decode), split-k and unrolling for the quantized matvec
(no change), and the fused attention-output quantization described below.

One bug was found and fixed along the way. Pinned staging buffers for index
uploads were reused before their asynchronous copy had run. cudarc's own guard
for this is off, because event tracking is disabled context-wide. This
corrupted Gemma prefill once prompts had several attention tiles. It's now
guarded with per-buffer events, and `index_reupload_back_to_back` covers it.

Debug switches: `RLLM_PROFILE=1` gives a per-kernel event profile;
`RLLM_NO_GRAPHS=1`, `RLLM_MMQ=0|1`, `RLLM_MMQ_MIN=<elements>`, `RLLM_NO_FA=1`,
`RLLM_NO_TC_ATTN=1`, `RLLM_NO_FUSED_QUANT=1`, `RLLM_NO_GLU_MMQ=1` (prefill and
decode fused FFN), `RLLM_FA_QUANT=0|1` and `RLLM_BF16_VIA_F16=1` force alternative paths for A/B
checks. They are read once at startup.

## Serving: `rllm serve` vs `llama-server` (RTX 6000 Ada)

Run 2026-10-29. Both servers loaded the same GGUF on the RTX 6000 Ada:
- **llama.cpp:** `llama-server` from the same sm_89 build as above, `-ngl 99 -fa on -np 8 -c 32768`.
- **rllm:** `rllm serve --parallel 8 --ctx 4096`.

The client was a standard-library Python script on the same machine, so no network was involved. It made streaming chat completions, greedy, with the end-of-turn tokens biased away so every request runs to its full length. Scenarios:
- one request at a time: a 540–670-token prompt, 256 tokens out, median of 3
- 1, 4 and 8 concurrent clients: 2 requests each, about 120-token prompts, 192 tokens out
- a 4-turn chat on a 2–2.6k-token system prompt

The docker `llama-server` on the box was idle throughout.

| | Qwen3-0.6B Q8_0, llama-server | Qwen3-0.6B Q8_0, rllm | rllm ÷ llama | Nemotron E4B Q8_0, llama-server | Nemotron E4B Q8_0, rllm | rllm ÷ llama |
|---|---:|---:|---:|---:|---:|---:|
| Single request: time to first token | 103 ms | 13 ms | 7.9× faster | 130 ms | 49 ms | 2.7× faster |
| Single request: decode, tok/s | 490.8 | 547.4 | **1.12×** | 118.5 | 119.8 | **1.01×** |
| 1 client: total output, tok/s | 432 | 650 | **1.50×** | 115 | 127 | **1.10×** |
| 4 clients: total output, tok/s | 545 | 1874 | **3.44×** | 233 | 450 | **1.93×** |
| 8 clients: total output, tok/s | 1176 | 2760 | **2.35×** | 485 | 744 | **1.53×** |
| 8 clients: mean time to first token | 262 ms | 14 ms | | 305 ms | 58 ms | |
| Chat, turn 1: time to first token | 338 ms | 50 ms | | 385 ms | 170 ms | |
| Chat, turns 2–4: time to first token | 16–17 ms | 12–13 ms | | 99–107 ms | 19–20 ms | |

Notes:
- **Prefix reuse:** both servers reuse the cached chat prefix on later turns. llama-server's Nemotron turns still took about 100 ms, presumably because its sliding-window cache can't be reused across turns.
- **How rllm batches:** decode steps of up to 8 requests replay CUDA graphs; each forward pass also takes one prompt chunk; requests are sampled on separate threads. See `docs/server.md` for the design.
- **Batching numerics:** batched requests run the multi-row int8 kernels (the ones prefill uses), so their logprobs differ from a lone request's by about 0.01–0.02 on average.
- **Fixes during this benchmark:**
  - Server per-token overhead was first 0.6 ms on Nemotron (8.25 ms per step against 7.63 ms for raw decode). The causes were a 1 MB logits copy per token, a scalar argmax, and the engine thread landing on E-cores.
  - Sampling now works in place and is vectorized, and the engine thread pins itself to the P-cores. A step now takes 7.69 ms.
  - Temperature sampling over the 262k vocabulary went from 0.86 to 0.08 ms per token (laptop CPU).

## Deterministic reductions, q8 KV cache and speculative decoding

Three later additions, measured on the RTX 6000 Ada (details and laptop
numbers in `docs/server.md`).

**Deterministic reductions.** Prefill matmuls split along k (MMQ) with three
or more splits now write their partial tiles to a buffer that a small kernel
sums in split order; two splits still `atomicAdd` into a zeroed output,
which is order-independent (x + y = y + x). Same tokens, same logits, every
run: checked bit for bit on Qwen3-0.6B (forced through MMQ), Qwen3-8B and
Nemotron on both GPUs (`crates/rllm-model/tests/common/gpu.rs`). The cost is in
prefill only: pp512 0–1.8% slower on the Ada (Nemotron within noise,
Qwen3-8B 6560 vs 6678 tok/s), 1.5–3% on the laptop. (Summing in the last
block to finish each tile, llama.cpp's stream-k style, measured 13% slower
here: it cost registers and occupancy.)

**q8 KV cache** (`--kv-type q8`): int8 with a scale per 32 values, keys
rotated by a Walsh–Hadamard matrix as llama.cpp does. Perplexity is within
noise of f16 on Qwen3-8B and Nemotron (+0.19% on Qwen3-0.6B at ctx 512,
llama.cpp's q8_0 +0.21% on the same chunks). Decode attention also got a
split-context (flash-decoding) path for long contexts, which helps f16 as
well. Its split count is now chosen per token rather than per step, so a
verification step matches one-token steps bit for bit at any depth (the
model tests caught the mismatch on the 20-WGP Strix Halo GPU), and batched
decode at depth gets more parallelism: 8 Qwen3-8B requests at 32k, 489 →
568 tok/s on the Ada (`rllm tune --measure`; one request unchanged).

| Qwen3-8B Q4_K_M | rllm f16 | rllm q8 | llama.cpp f16 | llama.cpp q8_0 |
|---|---:|---:|---:|---:|
| tg64, empty context | 143.1 | 141.0 | 153.3 | 146.9 |
| tg64 @ 8k context | 110.4 | 121.8 | 125.5 | 124.9 |
| tg64 @ 32k context | 71.4 | **88.0** | 82.3 | 79.5 |
| pp512 @ 32k context | 2593 | 2477 | 3261 | 3180 |

(`rllm bench --depth N --kv-type q8` and `llama-bench -d N -ctk q8_0 -ctv
q8_0 -fa 1`.) Nemotron: q8 decodes 7% faster at 32k context and 1.7%
slower with an empty context.

**Speculative decoding** (`rllm serve --draft`): Qwen3-8B Q4_K_M with
Qwen3-0.6B Q8_0 as the draft, same harness as the serving section.

| Qwen3-8B Q4_K_M | rllm | rllm + draft | llama-server | llama-server + draft |
|---|---:|---:|---:|---:|
| Single request, 6 mixed prompts, tok/s | 141 | **213** | 146 | 126 |
| Single request, list continuation, tok/s | 140 | **246** | 143 | 134 |
| 4 clients: total output, tok/s | 475 | **515** | 401 | 189 |
| 8 clients: total output, tok/s | 762 | 745 | 528 | 355 |

Greedy output with the draft is token for token the same as without it
(six prompts compared). llama-server used `--spec-type draft-simple` with its
default 3 draft tokens; 8 was slower still.

## Prompt lookup and the prefix cache's host tier

Two ways to put the idle CPU and system memory to work, measured on the RTX
6000 Ada with Qwen3-8B Q4_K_M (details in `docs/server.md`).

**Prompt lookup** (`--prompt-lookup`): drafts copied from earlier in the
sequence, verified like a draft model's, so output is unchanged (greedy
output identical in all twelve comparisons).

| Greedy, tok/s | rllm | lookup | draft model | both |
|---|---:|---:|---:|---:|
| Rename a variable in 20 lines of code | 136 | **360** | 271 | **363** |
| Short story | 142 | 142 | 169 | 170 |
| Copy a paragraph and add a sentence | 136 | **280** | 238 | **285** |

**Host tier** (`--host-cache-gb`): evicted prefix-cache blocks move to
pinned system memory and come back over PCIe on a hit. A 2546-token prompt
repeated after two others displaced it from a 4096-token pool: time to first
token 381 ms recomputed, 39 ms restored (375 MB copied back).

## AMD Strix Halo: llama.cpp ROCm vs rllm HIP

| | |
|---|---|
| GPU | Radeon 8060S in a Ryzen AI MAX+ 395 (gfx1151, RDNA 3.5, 40 CUs, wave32) |
| Memory | 128 GB LPDDR5X shared with the CPU (100 GB addressable by the GPU) |
| Software | Ubuntu 26.04, ROCm 7.14 |

**How it was run:** llama.cpp is the box's own ROCm build (0.6.0-dev,
`5e03bdd87`) in a container, `llama bench -ngl 99 -r 3` (f16 KV, flash
attention auto). rllm was built on the box with no ROCm SDK headers
involved (it loads HIP, hiprtc and hipBLAS at runtime) and run as
`rllm bench -d hip -r 3`, each model right after llama.cpp. The box also
holds an idle `llama-server` (gemma-4-31B, 56 GB; 0% GPU before the runs).

### Speed (tokens/s), 2026-10-30

llama.cpp columns from back-to-back runs earlier the same day; rllm from
the final build.

| Model | pp512, llama.cpp | pp512, rllm | rllm ÷ llama.cpp | tg128, llama.cpp | tg128, rllm | rllm ÷ llama.cpp |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B Q8_0 | 12518.87 ± 1371.83 | 16985.93 ± 61.95 | **1.36×** | 223.86 ± 0.58 | 231.32 ± 0.00 | **1.03×** |
| Qwen3-0.6B Q4_K_M | 12256.07 ± 1047.65 | 17641.81 ± 74.88 | **1.44×** | 285.25 ± 0.54 | 311.45 ± 0.14 | **1.09×** |
| Qwen3-0.6B BF16 | 11138.34 ± 998.76 | 15670.04 ± 28.74 | **1.41×** | 135.11 ± 0.04 | 153.26 ± 0.05 | **1.13×** |
| Qwen3-0.6B Q5_K_M | 12110.17 ± 1031.69 | 17345.95 ± 86.36 | **1.43×** | 268.81 ± 0.61 | 286.13 ± 0.15 | **1.06×** |
| Qwen3-0.6B IQ4_XS | 12724.50 ± 1415.50 | 17278.64 ± 101.00 | **1.36×** | 299.23 ± 0.81 | 333.36 ± 0.12 | **1.11×** |
| Qwen3-8B Q4_K_M | 1302.65 ± 16.23 | 1333.43 ± 1.78 | **1.02×** | 42.84 ± 0.02 | 39.34 ± 0.01 | 0.92× |
| Nemotron 3.5 Bolt E4B Q8_0 | 2140.94 ± 62.85 | 2338.66 ± 3.15 | **1.09×** | 37.75 ± 0.03 | 36.15 ± 0.02 | 0.96× |
| Llama-3.2-1B Q4_K_M | 7483.22 ± 479.55 | 7781.68 ± 26.80 | **1.04×** | 212.13 ± 0.10 | 190.40 ± 0.05 | 0.90× |
| Mistral-7B v0.3 Q4_K_M | 1294.30 ± 9.19 | 1366.00 ± 1.56 | **1.06×** | 46.32 ± 0.04 | 43.20 ± 0.01 | 0.93× |

At 16k tokens of context (one run each):

| Model | pp512 @ d16384, llama.cpp | rllm | tg32 @ d16384, llama.cpp | rllm |
|---|---:|---:|---:|---:|
| Qwen3-8B Q4_K_M | 787.66 | 727.28 (0.92×) | 29.39 | 27.44 (0.93×) |
| Nemotron E4B Q8_0 | 1363.62 | 1194.10 (0.88×) | 33.78 | 32.50 (0.96×) |

How the HIP backend got there (Qwen3-8B pp512 / Nemotron pp512):

| Step | Qwen3-8B | Nemotron |
|---|---:|---:|
| First working port: dequantize + hipBLAS, plain tiled attention | 349 | 575 |
| hipBLAS writes f16 (then widened to f32): 4× faster GEMMs on gfx1151 | 869 | 1139 |
| BF16 weights through the f16 GEMM | | 1366 |
| Prefill attention on RDNA3 matrix cores (`attn_wmma`) | 956 | |
| Int8 MMQ on matrix cores (`wmma_s8`), as on NVIDIA | 1327 | 1934 |
| `attn_wmma` for 512-wide heads (head dims split over waves) | | 2330 |

And for decode and long contexts:

| Change | Effect |
|---|---|
| Q4_K repacked at 4.625 bits per weight (was 5): 6-bit scales kept as bytes, d/dmin per super-block | Qwen3-8B tg128 34.4 → 35.8 (and 144 → 150 on the RTX 6000 Ada) |
| Decode matvecs split each long row over 4 waves (2 for 2048-wide Q4 rows), as llama.cpp does | BF16 0.6B 104 → 153, Qwen3-8B 35.8 → 40.2, Q8 0.6B 216 → 236 |
| `attn_wmma`: queries held as matrix fragments, one softmax reduction per 32-key chunk, 8 waves sharing each K/V chunk | Qwen3-8B pp512 @ 16k 579 → 735 |
| Decode attention with fewer warps where 8 don't fit in 64 KB of LDS (512-wide heads) | Nemotron tg32 @ 16k 5.0 → 32.5 |
| hipBLASLt for 16-bit-output GEMMs from 4096 × 4096 weights up (1.5× hipBLAS there; slower on small ones) | Qwen3-8B pp512 with MMQ off 935 → 1232 |

### Correctness on gfx1151

`rllm compare` (logits against the CPU reference at every step, 8 decode
steps after a 50-token prompt), the same command on the RTX 6000 Ada for
reference:

| Model, KV cache | Worst relative diff, HIP | Same on CUDA (sm_89) | Argmax mismatches, HIP |
|---|---:|---:|---:|
| Qwen3-0.6B BF16, f16 | 9.9e-4 | | 0/9 |
| Qwen3-0.6B Q8_0, f16 | 2.1e-2 | | 0/9 |
| Qwen3-8B Q4_K_M, f16 | 1.5e-2 | 1.6e-2 | 0/9 |
| Qwen3-8B Q4_K_M, q8 | 1.9e-2 | 1.8e-2 | 0/9 |
| Nemotron E4B Q8_0, f16 | 5.2e-2 | 4.1e-2 | 0/9 |
| Nemotron E4B Q8_0, q8 | 4.8e-2 | 6.8e-2 | 0/9 |

All 14 GPU-vs-CPU kernel parity tests (`cargo test -p rllm-hip`) pass, as do
the model tests (`--test gpu_hip`: determinism, verification steps,
q8 vs f16, host tier) on Qwen3, Llama and Nemotron, and
the server checks used on the Ada (a 30k-token prompt with 7 concurrent
requests; prompt lookup; draft model; host-tier offload and restore) pass
on gfx1151.

## More weight types and model families

**Q5_K and IQ4_XS** (Q5_K_M and IQ4_XS files) join Q4_K, Q6_K and Q8_0. On
the device Q5_K is Q4_K's layout plus one high bit per weight (5.625 bits);
IQ4_XS keeps its 4-bit codebook indices and a scale per 32 (4.5 bits, as in
GGUF), decoded four at a time with a byte permute (`__byte_perm` /
`v_perm_b32`). Both use the matvec, MMQ and dequantize paths of the other
K-quants, and both backends' parity tests cover them.

**Llama and Mistral** (GGUF architecture `llama`): the Qwen3 decoder without
Q/K norms. Llama's GGUF stores Q and K for ggml's adjacent-pair RoPE; rllm
reorders those rows at load time so its NeoX-style RoPE kernels apply
unchanged. Llama 3's `rope_freqs` scaling, its `llama-bpe` tokenizer and its
`strftime_now` template call are supported, as is the SentencePiece
tokenizer of Llama 2 / Mistral (merges derived from the token scores).

Perplexity (wikitext-2, ctx 512, first 32 chunks), rllm against llama.cpp
on the same GPU:

| Model | Strix Halo: rllm | llama.cpp ROCm | RTX 6000 Ada: rllm | llama.cpp CUDA |
|---|---:|---:|---:|---:|
| Qwen3-0.6B Q5_K_M | 19.095 | 19.241 | 19.092 | 19.239 |
| Qwen3-0.6B IQ4_XS | 20.304 | 20.312 | 20.302 | 20.315 |
| Llama-3.2-1B-Instruct Q4_K_M | 14.547 | 14.560 | 14.550 | 14.556 |
| Mistral-7B-Instruct-v0.3 Q4_K_M | 5.952 | 5.962 | | |
| Mistral-7B-Instruct-v0.3 IQ4_XS | 5.927 | 5.929 | | |

All within ±0.16–0.74 (the estimates' own error). RTX 6000 Ada speed for the
new types and Llama: Qwen3-0.6B Q5_K_M 80036 / 763 tok/s (pp512 / tg128;
llama.cpp 47077 / 698), IQ4_XS 70867 / 827 (49568 / 727), Llama-3.2-1B
51278 / 651 (48244 / 707).

## Concurrency: `rllm serve` vs vLLM (RTX 6000 Ada)

vLLM 0.32.0 (PyTorch 2.13, CUDA 13) in a Python 3.12 venv on the Ada box,
`vllm serve <model> --gpu-memory-utilization 0.28 --max-model-len 2048
--max-num-seqs 64`; rllm `serve --ctx 2048 --parallel 64 --kv-tokens 49152`.
The client (`bench_conc.py`, stdlib only) runs N streaming clients, 3N
requests, each a ~130-token prompt and exactly 192 greedy output tokens
(end-of-turn tokens biased away). An idle llama-server holds 32 GB of the
GPU throughout.

| Output tokens/s | c1 | c16 | c32 | c64 |
|---|---:|---:|---:|---:|
| Qwen3-0.6B BF16, vLLM | 445 | 4619 | 8136 | 11324 |
| Qwen3-0.6B BF16, rllm before | 481 | 3206 | 4350 | 2537 |
| Qwen3-0.6B BF16, rllm | **502** | 4452 | 6695 | 8625 |
| Qwen3-8B AWQ (4-bit), vLLM | 142 | 1733 | 2749 | 3472 |
| Qwen3-8B Q4_K_M, rllm | **147** | 912 | 1480 | 2093 |

Mean time to first token at c64: rllm 45 ms vs vLLM 91 ms (0.6B); 602 vs
541 ms (8B). What changed, and what is left:
- **Sampling on the device.** Each step used to copy every sequence's full
  logits row to the host (64 × 151936 floats, 39 MB, into a fresh pageable
  buffer) and sample there; at 64 sequences that took most of the 24 ms
  step. Tokens are now sampled on the GPU (`sample_rows`: Gumbel-max, with
  top_k / top_p thresholds by bisection, logit bias and penalties applied
  first) and only ids come back; logprobs, JSON-constrained and draft
  verification rows still go to the host. `--host-sampling` restores the
  old path (2524 tok/s at c64).
- **Graphs for larger decode steps:** pure decode steps of up to 64 tokens
  are recorded and replayed (was 8): +18–21% at 16–32 sequences.
- **Left (8B):** from 9 sequences on, decode leaves the matvec kernels for
  MMQ, whose 64-token tiles waste most of their work at 9–32 tokens
  (8 sequences: 905 tok/s; 9: 610). llama.cpp sizes its MMQ tile to the
  batch; rllm doesn't yet. vLLM's AWQ kernels are also a different
  quantization (no K-quant scales to unpack).

## Known gaps

- **Prefill activation quantization on small GPUs:** on the RTX 6000 Ada no
  standalone quantize launch remains (decode or prefill). On the 46-SM laptop
  GPU prefill keeps one per layer (the attention output, about 18 µs), because
  all three fused designs measured slower there (0.5–1.2%); the fused path is
  enabled from 80 SMs up.
- **Close margins:** Nemotron generation on the Ada (1.01×) and Nemotron on
  the laptop are the closest results; small changes in either engine could
  flip them.
- **Laptop numbers at the end of the session:** the laptop's battery ran down
  to 13% and the GPU was power-capped (both rllm builds then measured the same
  ~12 tok/s), so the last decode change (float-weight fused FFN) was
  benchmarked on the Ada only; the laptop table is from the run before it.
- **Qwen3-8B Q4_K_M vs llama.cpp (Ada):** llama.cpp prefills 1.4× faster
  (pp512 9380 vs 6640 tok/s) and decodes 7% faster with an empty context; at
  32k context its f16 decode is 15% faster than rllm's f16 (rllm's q8 beats
  both). The earlier tables cover Qwen3-0.6B and Nemotron, where rllm leads
  or matches.
- **q8 KV prefill:** steps longer than 8 tokens expand the blocks they read
  into f16 before the tensor-core kernels run: 1–4% slower prefill at depth.
- **Speculative decoding on Gemma 4:** its one-token steps use fused kernels
  the multi-token verification step doesn't, so greedy output with a draft
  matches as closely as batched requests do rather than bit for bit (Qwen3
  matches exactly). The draft model has no prefix cache: each request's
  draft cache reads its whole prompt at the first decode step, one reason
  the draft only runs for sequences up to 6144 tokens (at 17k tokens it
  had cut decode speed by 24%). `rllm run` has no `--draft`; the server does.
- **Several backends in one process:** two `CudaBackend`s recording CUDA
  graphs at once (only possible in tests) disturb each other's decode
  steps, so the GPU tests take a lock. The binaries create one backend.
- **AMD (gfx1151):** decode trails llama.cpp by 4–10% on the 7–8B models
  and Llama-3.2-1B, and prefill at 16k context by 8–12%. hipBLAS on gfx1151
  is 4× slower with f32 output, so GEMMs write f16 scaled by 1/256 and widen
  it (overflow past 1.7e7). Only gfx11 (RDNA3) has the WMMA kernels; other
  AMD targets run the plain kernels and are untested.
- **Batched decode of 9–32 sequences:** MMQ's fixed 64-token tile (see the
  vLLM comparison above).
- **Q4_K on CPU:** emulating llama.cpp's activation rounding doesn't narrow the
  gap. llama.cpp uses repacked Q4_K kernels on AVX2, whose rounding isn't
  mirrored. The gap is still within llama.cpp's own noise floor.
- **CPU backend speed:** the CPU backend is the f32 correctness reference and
  has not been optimized (see the CPU tables above).
- **Not built yet:** tensor parallelism across GPUs (the 2× RTX 4500 Ada setup
  in `docs/plan.md` is a later phase), so there is nothing to benchmark there
  yet.
