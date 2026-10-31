# Performance review, bottom to top (2026-10-30)

This review covers rllm from the GPU kernels up to the HTTP layer. It has two
sources:
- **Measurements** taken for it on the RTX 6000 Ada (CUDA) and the Strix Halo
  (HIP), using `rllm bench`, the engine's step timing, rocprofv3 traces and
  the server.
- **Four code reviews**, one per layer: kernels, the GPU backend's host side,
  model/KV/sampler/loader, and the server/engine/tokenizer.

Claims from the code reviews were checked against the code, and the
cheap-to-test ones were measured. Items marked *measured* have numbers
behind them; the rest are estimates from reading the code. File references
are `file:line` as of commit d4e6351.

## 1. Where the time goes (measured)

| Scenario | Where the time goes | What follows |
|---|---|---|
| Decode, 1 sequence, Qwen3-8B, Ada | 6.75 ms/step at 150 tok/s; engine overhead ≈ 0 (forward is 99.9% of the step) | Kernel bandwidth only; already 0.98× llama.cpp |
| Decode, 1 sequence, Qwen3-8B, Strix Halo | rocprof: 95% of wall time is kernels, gaps p50 2 µs / p90 7 µs. `matvec_glu_silu_q4` takes 295 µs/layer (≈177 GB/s), other matvecs 116–240 GB/s | Kernel memory efficiency is the lever (target ≈ 215 GB/s) |
| Prefill pp512, Qwen3-8B, Strix Halo | rocprof: 0.8% idle; MMQ is 94% of kernel time (`mmq_glu` 168 ms, `mmq_q4` 135 ms, `mmq_q6` 58 ms per 3 passes) | Host-side changes won't help prefill; MMQ efficiency (≈18–22 int8 TOPS of ~59) will |
| Prefill pp512, Qwen3-8B, Ada | ≈7.1k tok/s vs llama.cpp 9.4k; `mmq_glu_silu_q4` is about half the pass (≈112 TOPS of a ~700 TOPS int8 peak) | Same: MMQ epilogue, pipelining and tile shapes |
| Batched decode, Qwen3-8B, 9–32 sequences | 8 seqs: 905 tok/s; 9: 610; 16: 1053 (16.7 ms/step) | MMQ's fixed 64-token tile at small m (§3, K3) |
| Server, 64 clients, Qwen3-0.6B | 7.4 ms/step vs 6.2 ms for the same batch without the server; mixed prefill+decode steps run without graphs | Engine/host overhead ≈ 1.2 ms/step (§3, S1–S3) |
| Cached 30k-token prompt, Qwen3-0.6B | client sees 67–73 ms, the engine 7–8 ms; ≈ 60 ms is JSON parsing, template rendering and **re-tokenizing 30k tokens** in the handler | Cache tokenization per segment (§3, S6) |
| Startup, Qwen3-8B | load 1.35 s (Ada) / 0.6 s (Strix Halo), ≈ 2 s end to end; uploads are serial per tensor | Pinned double-buffered upload (§3, H12) |
| First run, empty kernel cache, Qwen3-0.6B | Ada 1.38 s vs 0.86 s warm; **Strix Halo 10.9 s vs 0.59 s warm**: hiprtc compiles every instantiation in one translation unit | Compile in parallel modules, or lazily per kernel family (§3, H13) |
| Short request (9-token prompt, 1 token), Qwen3-0.6B, server | 4.1 ms client-side, 3 ms engine | Fine as is |

**Refuted by measurement.** These were checked and are not worth doing:
- **Pool release at every sync costs eager steps time.** Holding freed memory
  in the pool gave no difference in prefill on either vendor (Ada 8B
  7088–7139 vs 7114–7290 tok/s; Strix Halo 1331.6 vs 1331.2).
- **Q4 matvec ALU (the redundant `sx` dp4a).** Removing it changed nothing on
  AMD decode (35.76 vs 35.78 tok/s). It may still help NVIDIA at m = 4–8.
- **Logits for non-final prefill chunks (M2).** Skipping them changed
  chunked prefill of a 15.8k-token prompt by ≤ 1% (0.6B 0.64 → 0.63 s, 8B
  3.02–3.05 s either way): one vocab matvec per 512-token chunk is small next
  to the chunk.
- **Profile-mode gaps.** The 30% idle that `RLLM_PROFILE` showed in prefill
  is an artifact of its per-launch events. rocprof shows < 1% idle.

## 2. Correctness bugs found (fix first)

Status: B1–B3 were confirmed by reading the code; B4–B8 come from the
code reviews and were not reproduced. **All but B7 are now fixed** (see
section 6); B7 is still only documented.

| # | Where | Problem | Fix |
|---|---|---|---|
| B1 | `rllm-cuda/src/backend.rs` `ensure_len` (used by `gemm`) | **Confirmed:** it reallocates the GEMM scratch (`wh`/`xh`/`oh`) without invalidating captured graphs. With graphs now covering 9–64-token decode steps, a float-weight model (BF16/F16) can replay a graph that reads and writes freed scratch after a larger prefill. | Call `invalidate_graphs()` on growth (or refuse to grow in Capture/Replay), as `qscratch` does |
| B2 | `rllm-tokenizer` StreamDecoder + `rllm-server/src/generation.rs` at Done | Text the decoder holds back (an incomplete UTF-8 tail) is never flushed when a request ends; the decoder has no flush at all | Flush on Done |
| B3 | `rllm-kv` `make_room` / `evict` | It can evict (and offload over PCIe) every prefix-cache entry while freeing no blocks, when the entries' blocks are still used by running sequences | Pick victims by blocks they would actually free (refcount == 1) |
| B4 | `rllm-model/src/generate.rs` `step_sampled` | The `sample.len()` check runs after `seq.append`, so a mismatch leaves sequences grown (breaks all-or-nothing) | Check before appending |
| B5 | `rllm-core/src/sample.rs` greedy | Returns 0 if `logits[0]` is NaN (unlike `sampler::argmax`) | Skip NaNs |
| B6 | backend graph capture | An error during Capture leaves the stream capturing, and `g.graphs[&key]` panics if a key was invalidated mid-step | End/discard the capture on error; use `get` and fall back to eager |
| B7 | `sync_index` pinned buffers in graph modes | Not fenced in Measure/Capture/Replay. Safe today only because every step ends with a full sync | Needed before any cross-step pipelining (S1) |
| B8 | engine `step` | When KV blocks run out, speculative draft tokens can cause a preemption: drafts aren't dropped first | Retry with `spec = 0` before preempting |

## 3. Improvements, by layer

Impact: ●●● large (more than 10% on a headline number), ●● medium, ● small.
Effort: S < 1 day, M a few days, L a week or more.

### Kernels (`kernels.cu`)

| # | Item | Helps | Impact | Effort |
|---|---|---|---|---|
| K1 | **Matvec: several rows per wave instead of splitting one row over 4 waves.** At k = 4096, Q4 with kw = 4 gives each lane exactly one 16 B load, then a reduction, an LDS write and a barrier. That fits the ≈ 30 µs fixed cost and 116–180 GB/s measured on mid-size matrices. Use R = 2–4 rows per wave with all loads issued up front, and share the activation loads across rows (K2). | AMD decode (0.90–0.96× → ≈ 1.0×), also NVIDIA | ●●● | M |
| K2 | **Reuse activation loads across rows.** Each lane loads 32 B of `xq` per unit per row, more than the weight bytes, and 16× more at m = 8. | Batched decode m ≤ 8, both vendors | ●● | M (with K1) |
| K3 | **Batch-sized MMQ tiles** (BN ∈ {16, 32, 64}, as llama.cpp's `mmq_x`), or raise `MV_MAX_M` to 16. Fixes the 905 → 610 tok/s drop at 9 sequences. | 8B serving at 9–32 clients (0.53–0.60× vLLM) | ●●● | M |
| K4 | **MMQ epilogue in integers per super-block.** Scale per 256 (q8_K-style activations), accumulate unit dots × 6-bit scales in int32 (no overflow: ≤ 3e7), and do one float FMA per 8 units. Today each 32-deep result gets a convert plus 2–3 FMAs, which compete with WMMA on RDNA3's VALU. Check perplexity. | Prefill on both vendors (Ada 0.77×) | ●●● | L |
| K5 | **Pipeline MMQ staging.** Load chunk k+1 into registers (or `cp.async` on NVIDIA) while chunk k computes; `MMQ_CU = 8` halves the barriers. Only 4 warps per block, and one block per WGP on gfx1151. | Prefill, both | ●● | M |
| K6 | **MMQ on HIP: contiguous scale reads.** Permute rows when staging so each lane's 8 `Ad`/`Am` values are two `ds_read_b128` instead of 8 strided reads. Should also lower the 167 VGPRs. | AMD prefill | ●● | S |
| K7 | **`matvec_glu` uses 1024-thread blocks.** 67% occupancy cap on sm_89, a 2.7-wave tail, and a 64-register limit. Use 256 threads with 4 rows per warp. It also lacks the row split, and it is the biggest decode kernel on AMD (295 µs/layer). | Decode, both | ●● | M |
| K8 | **`sample_rows`: top_k / top_p take ≈ 80 vocabulary passes** in one block. Use a radix select on float bits, then work on the ≤ k candidates in shared memory; split large vocabularies over blocks; skip tokens that cannot win the Gumbel draw (`v < max − 20.2`, exact). | Sampled (temperature > 0) decode with top_k/top_p | ●● | M |
| K9 | **`attn_wmma` and `attn_prefill_tc`: V-transpose stores are 8–16-way bank-conflicted.** Map threads key-fastest or write packed half2. | Long-context prefill | ●● | S |
| K10 | **`attn_wmma`: ≈ 240 VGPRs, no prefetch, no `full`-chunk fast path.** Move Q to LDS (fits at HD 128), prefetch the next K/V chunk into the freed registers, and copy attn_fa's `full` predicate. | AMD prefill at 16k (0.88–0.92×) | ●● | M |
| K11 | **q8 KV prefill re-expands the whole context to f16 every chunk** (`kv_dequant_blocks`). Dequantize during the attention kernels' K/V staging instead. | Long-context q8 prefill (≈ 5% today) | ●● | M |
| K12 | **`attn_decode` f16 V uses 2-byte loads, PER per position.** Use one 8-byte load of consecutive dims (the Q8 path already does). | Long-context decode | ● | S |
| K13 | **`attn_decode` scores re-read q from LDS for every K element.** Use packed f16 dot products; for a Q8 cache, int8 q with dp4a. | Long-context decode | ●● | M |
| K14 | **`attn_decode` splits sized by cache capacity.** Splits with no positions should exit immediately (the graph shape stays the same); the combine stops at the used splits. | Decode early in long allocations | ● | S |
| K15 | **Non-temporal loads for weights** (`__builtin_nontemporal_load` / `L1::no_allocate`), so streamed weights don't evict KV and activations from L2/MALL. | Decode at depth; needs an A/B test | ● | S |
| K16 | Small ones: `matvec_glu_norm` issues weight loads only after its norm prologue; `quantize_q8`/`quant_unit` divide instead of multiplying by 1/d; `add_rms_norm`/`rms_norm`/`half_to_f32` use scalar loads; rope recomputes `powf`/`sincos` per head; `dequant4`/`embed` use byte loads; `attn_fa` loads K fragments as scalars (use `ldmatrix`). | various | ● each | S |

### GPU backend, host side (`backend.rs`, `gpu.rs`)

| # | Item | Helps | Impact | Effort |
|---|---|---|---|---|
| H1 | **Attention metadata is rebuilt for every layer.** The padded block-table matrix, three `sync_index` memcmps, `build_tiles`, and the q8 staging lists (with a HashMap) are recomputed per layer, also in graph replay. Build once per step (with M1). | High concurrency, long context: 0.3–3 ms/step | ●● | S–M |
| H2 | **`sync_index` stalls the host on the GPU when a key's contents change within a step** (Gemma 4 on HIP: `attn_tiles` differ between 256- and 512-wide layers; q8 staging between window and global layers). That is a full host↔GPU serialization per layer. Use a key per variant or a per-step pinned ring buffer. | Gemma 4 prefill (HIP especially), q8 prefill | ●● | M |
| H3 | **Scratch growth invalidates every captured graph** (`qscratch`/`glu_scratch`/`split_part` grow to exact size; arena growth clears graphs as the batch grows from 8 to 16 to 64), forcing recaptures. Pre-size for the maximum batch, grow by powers of two, and keep graph scratch separate from eager scratch. | Serving with varying batch sizes | ●● | S–M |
| H4 | **Graph keys are too fine, and there are no graphs above 64 sequences or for mixed steps.** Bucket batch sizes (pad to 1, 2, 4, 8, 16, 24, 32, 48, 64, 128), add LRU eviction, and capture the decode part of mixed steps (S2). | Serving | ●●● | M–L |
| H5 | **Per-launch host costs ≈ 0.3–0.6 µs:** a `format!` name, a `HashMap<String>` lookup, two mutexes, and argument-Vec growth. Resolve kernels into an array indexed by (kind, Fmt) at init and keep the mode in an atomic. | ≈ 0.15–0.3 ms/step eager; also delays graph launch in replay | ● | M |
| H6 | **hipBLASLt plans are keyed by m.** Every new prefill length runs the heuristic on the host. Cache by (n, k, type, m-bucket). | HIP prefill with float weights | ● | M |
| H7 | Per-layer `alloc` of `q_rot` / split `part` in eager steps (costly on HIP). Make them persistent scratch. | HIP eager steps | ● | S |
| H8 | `sample_rows`: re-uploads zeros to `sample_out` every step, one event per upload, pageable readback. Use a pinned persistent readback, and capture sampling inside the graph when there are no adjustments. | Every serving step: 15–40 µs | ● | S |
| H9 | `sync_index` copies the whole pinned capacity (up to 2×), creates an event per upload, and allocates `data.to_vec()` twice. | Every step | ● | S |
| H10 | Full-logit `to_host` goes into a fresh zeroed pageable Vec. Use a persistent pinned buffer and return only the needed rows. | logprobs / JSON / spec rows: ms per step | ●● | S–M |
| H11 | HIP 16-bit GEMM path: 4 launches per matmul (convert, dequant, GEMM, widen); exact-size scratch growth. Let producers emit an f16 copy (as `qout` does for int8) and cache f16 copies of small quantized weights. | HIP prefill for small models | ● | M |
| H13 | **hiprtc compiles every kernel in one unit:** 10.3 s on a first run on gfx1151 (cached afterwards in `~/.cache/rllm/ptx`). Split `kernels.cu` into a few modules compiled on parallel threads, or compile each kernel family on first use. | First run on AMD; every kernel edit | ●● | M |
| H12 | Model load: one staging buffer and a sync per tensor; pageable synchronous copies on HIP; no `madvise`. Use pinned double buffers that overlap reads, H2D copies and repacking, and sync once. | Startup (≈ 1.5–3× faster load) | ●● | M |

### Model, KV cache, sampler (`rllm-model`, `rllm-kv`, `rllm-core`)

| # | Item | Helps | Impact | Effort |
|---|---|---|---|---|
| M1 | **Block tables are cloned per sequence per step** (`generate.rs:106`), then rebuilt per layer (H1). Keep one persistent padded table in `KvPool` and update it incrementally. | High concurrency | ●● | M |
| M2 | Non-final prefill chunks still compute one logits row and copy it to the host. *Confirmed in code, measured at ≤ 1%* of chunked prefill. Worth doing only together with S1, where it removes a sync between chunks. | Long-prompt prefill | ● | S |
| M3 | **The prefix cache is linear everywhere:** `lookup` compares against every entry, `insert` scans with `starts_with`, `evict` is a min-scan, `offload` is O(E²), and `publish` counts entries every loop. Use a block-hash (radix) index with per-tier LRU lists and cached counters. | Admission with many or long cached prompts: ms → µs | ●● | M–L |
| M4 | **Final norm not fused** when every row is wanted. Pass `out_norm` as the last layer's "next" norm. | Every decode step: one kernel | ● | S |
| M5 | **Gemma per-layer-embedding prologue: 7 ops → 4.** Fold the scale into the norm's epsilon (exact) and use `rms_norm_add`. | Gemma 4 | ● | S |
| M6 | Speculative decoding allocates full-vocabulary vectors per draft token and per rejection (`Dist::Dense`, `verify`). Reuse scratch. | 1–4 clients with a draft model: 0.3–1 ms/step | ●● | S–M |
| M7 | GGUF metadata arrays (152–262k tokens, merges, scores) are parsed into `Vec<Value>` with a String per entry. Parse typed arrays. | Startup ≈ 10–40 ms | ● | M |
| M8 | Gemma's sliding-window KV pool is sized for the full context. Free blocks that fall out of the window. | KV memory, so more concurrency on Gemma | ●● | L |
| M9 | CPU backend (the reference, low priority): matmul loop order thrashes the cache, the transpose is serial, there is no AVX2/F16C dispatch, rope recomputes sin/cos per head, and attention allocates per (token, head). | `-d cpu` speed (2–4×) | ● | S–M |

### Server and engine (`rllm-server`, `rllm-tokenizer`)

| # | Item | Helps | Impact | Effort |
|---|---|---|---|---|
| S1 | **No host/GPU overlap between steps.** Plan, emit, admit and publish all run while the GPU idles. When every row is sampled on the device, launch step N+1 from step N's ids still on the device, and emit step N while it runs (needs B7). | Serving: most of the ≈ 1.2 ms/step gap at 64 clients | ●●● | L |
| S2 | **Mixed prefill+decode steps run eagerly** (no graph key when n > 8 and the step isn't pure decode), and under steady arrivals that is most steps at 64 clients. Run the decode part from its graph, or cap or alternate prefill chunks while many sequences decode. | 64-client throughput and inter-token jitter | ●●● | M |
| S3 | **`emit` wakes a tokio task per sequence per step** (≈ 64 futex wake-ups inside the GPU-idle window). Batch them into one message per step to a dispatcher. | Serving: 0.1–0.3 ms/step | ●● | S–M |
| S4 | **≈ 15 per-step Vecs and HashMaps** in `step` (spec, takes, proposals, spec_tokens, on_device, sample…), plus `drop_cancelled` rebuilding the queue every loop. Keep scratch in `Scheduler`. | Serving | ● | S |
| S5 | **Host sampling spawns one OS thread per row per step** (logprobs, JSON, `--host-sampling`). Use a persistent pool and chunk the rows. | Those workloads: ≈ 1 ms/step at 64 | ●● | S |
| S6 | **Requests re-tokenize the whole prompt** (*measured*: ≈ 60 ms of a 67 ms cached 30k-token request). Special tokens already split the text before BPE, so caching tokenized segments by hash is exact. Also: the chat template is recompiled per request (compile once), and render/encode run on a tokio worker (use `spawn_blocking`). | TTFT for long and multi-turn chats | ●●● | S–M |
| S7 | **Preempted sequences are re-admitted immediately and preempted again** (churn: cache insert/lookup/evict/offload every step). Re-admit only when free blocks cover the need plus K steps of growth. | Memory-tight serving | ●● | S |
| S8 | Prefill order is strictly first-in-first-out: a large prompt at the head blocks smaller ones, and the budget goes in admission order. Split the budget among prefilling sequences or favour short prompts. | TTFT tail with mixed prompt lengths (16 clients: 268 ms mean) | ●● | S–M |
| S9 | **Speculation moves full rows to the host:** about 14 MB/step at 4 sequences × 6 rows; `propose` allocates a zeroed `running × vocab` buffer per step; k serial draft rounds. Compare greedy on device ids, and keep persistent buffers. | 1–4 clients with draft: +10–20% | ●● | M |
| S10 | **JSON constraint per token:** a full-vocabulary `order` Vec, `select_nth` over 151k, up to 256 deep clones of `JsonState`, and per-character allocations. Bucket tokens by first character, use a scratch state, and cache masks per state. | `response_format` requests: 100–500 µs → tens of µs per token | ●● | M |
| S11 | Streaming: each token builds a `json!` object and crosses three channels and two task wake-ups. Pre-render the chunk prefix and suffix, merge the tasks, and coalesce ready tokens into one event. | CPU under load | ● | M |
| S12 | The stream decoder re-decodes up to 33 ids per token. Use a prefix/read-offset window of about 5 (or `tokenizers::DecodeStream`). | Per token | ● | S |
| S13 | Streaming with `n > 1` (or several prompts) runs the choices serially. Spawn them and merge. | n× latency | ●● | S–M |
| S14 | Prompt lookup rescans the whole history every step. Keep an incremental n-gram index. | Long sequences with `--prompt-lookup` | ● | S |
| S15 | Penalties rebuild `adjust` from the whole `counts` map and upload it every step, and `counts` is maintained even with no penalties. | Requests with penalties | ● | S–M |

## 4. Suggested order

1. **Bugs B1–B8.** B1 is a real use-after-realloc for BF16/F16 models since
   larger graphs landed. All are small.
2. **Cheap wins with clear payoff:**
   - S6: tokenization cache, compiled template, `spawn_blocking`.
   - S5: sampler thread pool.
   - S7: preemption hysteresis.
   - H3: pre-sized scratch.
   - H8 and H9: sampling and index-upload trimming.
   - K6, K9, K12: small kernel fixes.
3. **Decode bandwidth on AMD:** K1, K2 and K7 (several rows per wave, shared
   activation loads, smaller GLU blocks). This is the main lever for the
   0.90–0.96× decode gap.
4. **Serving at 9–64 clients:** K3 (batch-sized MMQ tiles), S2 (graphs for
   mixed steps), S3 and S4, then S1 (pipelined steps) as the larger
   follow-up. Together these target the 0.53–0.76× gap to vLLM.
5. **Prefill:** K4 and K5 (integer MMQ epilogue, pipelined staging), then
   K10 and K11 for long context.
6. **Startup and memory:** H12, M7, M8.

## 5. Coverage

What was reviewed (code reading) and measured, per part of the system:

| Part | Files | Reviewed | Measured |
|---|---|---|---|
| GPU kernels | `rllm-cuda/kernels/kernels.cu` (all kernels) | yes | decode and prefill traces (rocprof), kernel bandwidths, occupancy (`kernel_info`) |
| GPU backend, host side | `rllm-cuda/src/{backend,gpu}.rs`, `rllm-hip/src/{gpu,ffi}.rs` | yes | idle time per pass, pool-release A/B, startup, cold compile |
| Backend trait, dequant, sampling reference | `rllm-core` | yes | sampling parity |
| Models | `rllm-model/src/{qwen3,gemma4,generate,lib,loader,sampler}.rs` | yes | head-skip A/B, decode/prefill per model |
| KV cache, prefix cache | `rllm-kv` | yes | cached-prompt latency |
| GGUF reader | `rllm-gguf` | yes | load time |
| CPU backend | `rllm-cpu` | yes (low priority) | — |
| Engine and scheduler | `rllm-server/src/engine.rs` | yes | engine step timing at 1, 16 and 64 clients |
| HTTP, streaming, output parsing, JSON constraint | `rllm-server/src/*.rs` | yes | TTFT, cached-prompt breakdown, concurrency vs vLLM |
| Tokenizer and templates | `rllm-tokenizer` | yes | tokenize time for 30k tokens |
| CLI and `rllm tune` | `rllm-cli` | partly (bench and tune are dev tools) | — |

## 6. Second review: findings and fixes (2026-10-30)

A second pass (security, robustness, correctness) found the items below.
All are fixed except where noted.

**Security and robustness**

| Finding | Fix | Checked by |
|---|---|---|
| JSON schema `$ref` inside `allOf` grows exponentially (memory exhaustion) | Compile under a work budget (depth 64, 20,000 steps); too large is 400 | unit test; e2e |
| Special-token strings in user text become real tokens | User/tool text is wrapped in private-use markers, and added tokens are matched only outside them | tokenizer test on 4 models; e2e |
| Queued requests not cancelled on disconnect; no queue limit | The generation task watches the client channel; `--max-queue` (256) answers 503 | build; e2e |
| `/v1/completions` prompts × `best_of` unbounded | At most 128 sequences | e2e |
| Open CORS; any content type; DNS rebinding | CORS opt-in (`--cors-origin`); non-JSON POST is 415; foreign `Host` on loopback is 403 | e2e |
| An engine-thread panic left `/health` saying ok | An alive flag cleared on thread exit; `/health` and submissions give 503 | build |
| Malformed GGUF files crash the process | Checked counts, offsets, sizes, alignment, dims and nesting depth | unit tests |
| CPU sampler panics on all-NaN logits; GPU returns id 0x7fffffff; greedy reference picks a NaN at 0 (B5) | NaNs count as -inf everywhere; no candidate is token 0 | unit tests |
| HIP module never unloaded; hipBLAS handle leaked when the workspace allocation fails | `hipModuleUnload` on drop; handle destroyed on that error | build; HIP parity |

**Correctness**

| Finding | Fix | Checked by |
|---|---|---|
| Prefix cache: a failed host restore reused a stale entry index and could hand out another request's live blocks | Rescan matches after a failed restore | unit test (fails without the fix) |
| `logit_bias` and penalties on the same token overwrote each other on the GPU | Merge per logit before upload | GPU parity |
| Graph key lacked the token-to-sequence split, so a replay could use another step's attention tiles | Key includes the split | build; GPU model tests |
| Tensor-core prefill attention dropped dims at head sizes 320/352/416/480 | Even per-warp splits only (320 = 4 × 80); others take the portable path | GPU parity, new test (fails without the fix) |
| CPU q8 reference skipped key rounding at head dims 48/80/112 | Rotate, round in 32-value groups, rotate back | build (the GPU needs multiples of 32) |
| JSON dead end reported `stop` with invalid JSON | Ends with an error | build |
| JSON constraint on SentencePiece vocabularies (Llama 2, Mistral) saw tokens without their leading space | Token texts decoded after an anchor token | e2e on Mistral: an enum value went from `New\u0020York\u0020City` (21 tokens, mean logprob −1.39) to `New York City` (8 tokens, −0.02) |
| Vocabulary shorter than the embedding table could index out of bounds in the JSON path | Ids past the tokenizer's vocabulary are never valid | build |
| JSON instruction as a system message broke templates without a system role (Mistral) | It leads the first user message there | e2e on Mistral |
| Deleted responses came back on completion; non-streamed disconnects stayed `in_progress` | Completion updates only an existing entry; a drop guard marks it `failed` | e2e |
| Stored-object store unbounded; response chains quadratic; snapshot serialized under the lock | 10,000 per store; chains share turns (memory and file, format v2, v1 still loads); snapshot of `Arc`s, serialized outside the lock | unit tests |
| `truncation: "auto"` re-rendered once per dropped message on an async worker, and broke role alternation | Bisection over whole turns, in `spawn_blocking` | e2e on Mistral |
| Llama/Mistral ignored sliding window and RoPE scaling | Sliding window applied; unsupported scaling types rejected at load | build |
| B1: GEMM scratch growth left graphs pointing at freed memory | Growth invalidates graphs (an error inside Capture/Replay) | GPU model tests (BF16) |
| B2: decoder text held back at the end was lost | Flushed at Done, through the stop matcher | build |
| B3: eviction offloaded entries that freed no blocks | Only entries holding sole references are evicted, and nothing when they can't free enough | unit test |
| B4: sample-count check after sequences grew | Checked first | build |
| B6: a failed capture left the stream capturing; `graphs[&key]` could panic | The next step ends a stale capture; lookups return errors | build; GPU tests |
| B8: drafts could force a preemption | Drop drafts before preempting | build |

Verified on the laptop (CUDA parity, workspace tests), the RTX 6000 Ada
(CUDA parity, GPU model tests on Qwen3-0.6B Q8_0/BF16 and Llama-3.2-1B) and
Strix Halo (HIP parity, GPU model tests, a 17-check end-to-end server run on
Mistral-7B). Speed on the Ada matches the previous build within noise (Qwen3-0.6B
Q8_0 and BF16, Qwen3-8B Q4_K_M; pp512, tg128 and tg128 x32).

