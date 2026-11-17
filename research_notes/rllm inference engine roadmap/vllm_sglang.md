# vLLM and SGLang (late 2026): what explains their throughput lead, and what rllm could adopt

Research date: 2026-11-11. Version context: vLLM async scheduling default since v0.14.0 (Jan 2026); docs exist up to v0.25.0 (compilation API pages). SGLang newest release found: v0.5.20 (2026-09-18). Older material is marked with its date.

rllm baseline (from local files `chart.md`, `compared.md`): RTX 6000 Ada, ~130-token prompt / 192 output tokens. Qwen3-8B Q4_K_M vs vLLM AWQ: 0.53x / 0.54x / 0.60x at 16 / 32 / 64 clients. Qwen3-0.6B BF16: 0.96x / 0.82x / 0.76x. rllm leads at 1 client and on 0.6B TTFT. rllm's own diagnosis is that the 8B gap comes from MMQ's fixed 64-token tile at 9–32 tokens per step. rllm captures CUDA graphs only for decode steps of up to 8 tokens.

## 1. Scheduler, batching, CUDA graphs, and hiding CPU overhead per step

### Takeaway
Both engines now aim for no host-device sync at all. The CPU prepares step N+1 while the GPU runs step N. Inputs are built on the GPU from persistent per-request state. Full CUDA graphs cover uniform decode batches, including spec-decode batches. The BF16 0.6B gap at 32–64 clients (0.76–0.82x) is the signature of CPU overhead per step. The 8B 4-bit gap at 16–64 clients (0.53–0.60x) is mainly the GEMM (section 3).

### Cited Findings
- **vLLM async scheduling** has been the default since v0.14.0 (Jan 2026; PR #27614) and can be disabled with `--no-async-scheduling`. At launch it excluded pipeline parallel, the CPU backend, and spec decode other than MTP/EAGLE. — [newreleases.io v0.14.0](https://newreleases.io/project/github/vllm-project/vllm/release/v0.14.0); [search summary of release notes](https://newreleases.io/project/pypi/vllm/release/0.16.0)
- In v0.16.0 (Feb 2026), pipeline parallel gained async scheduling support, giving +30.8% E2E throughput and +31.8% TPOT (PR #32618). — [newreleases.io v0.16.0](https://newreleases.io/project/pypi/vllm/release/0.16.0)
- v0.19.0 reportedly made async scheduling work with speculative decoding without pipeline stalls ("zero-bubble"). Secondary source only. — [autokaam summary](https://autokaam.pages.dev/news/vllm-v0-19-0-zero-bubble-scheduling-gemma-4/)
- **vLLM Model Runner V2** (MRV2, blog 2026-03-24) was experimental as of v0.18.0, enabled with `VLLM_USE_V2_MODEL_RUNNER=1`, and was not yet the default. — [vLLM blog: MRV2](https://vllm.ai/blog/mrv2)
  - **Persistent batch:** each request gets a stable row in a fixed-size state table. Per-step inputs (`input_ids`, `positions`, `query_start_loc`, `seq_lens`) are gathered and built on the GPU by Triton kernels rather than on the CPU.
  - **No sync:** it targets zero CPU-GPU sync. Rejection-sampling results are consumed on-device, and outputs are copied to the host asynchronously on a separate stream.
  - **Sampler:** a Gumbel-max sampler avoids materializing softmax and uses stateless in-kernel RNG. Top-k logprobs are computed only over the top-k logits.
  - **Measured:** Qwen3-0.6B on 1x GB200 reached ~25K output tok/s vs 16K for MRV1 (+56%), purely from runner and CPU-overhead work on a small model. GLM-4.7-FP8 with MTP=1 on 4x GB200 showed 6.3% lower mean TPOT.
- **SGLang zero-overhead batch scheduler** (v0.4, Dec 2024, older). It runs one batch ahead: CPU work (scheduling, memory allocation, prefix matching, radix-cache operations) overlaps the GPU's current batch. It is on by default (`--disable-overlap` turns it off). — [LMSYS SGLang v0.4](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)
  - Nsight showed no GPU idle time across 5 consecutive decode batches with the Triton backend, and a minor gap with FlashInfer.
  - Gains were 1.1x throughput over v0.3. The benefit is largest for small models and large TP.
  - The post states unoptimized engines can spend up to half their time on CPU overhead.
- **SGLang Spec V2** (default spec engine, June 2026) adds two overlap points. — [LMSYS DFlash/Spec V2](https://lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2/)
  - Host cleanup for batch N-1 (stop-token detection, metadata updates) overlaps GPU work on batch N.
  - Host KV allocation (`prepare_for_decode`) for batch N overlaps GPU work on batch N-1.
  - Measured: **Qwen3-8B on 1x B200 at concurrency 32 went from ~11.4k to ~15.3k tok/s (+33%)** from removing host-device sync alone.
- **vLLM CUDA graph modes** — [vLLM design doc: CUDA graphs](https://docs.vllm.ai/en/latest/design/cuda_graphs.html)
  - The modes are NONE, PIECEWISE, FULL, FULL_DECODE_ONLY and FULL_AND_PIECEWISE.
  - The default on V1 with piecewise compilation is FULL_AND_PIECEWISE: full graphs for uniform decode batches, and piecewise graphs (attention eager, everything else captured) for prefill or mixed batches.
  - Uniform decode means `max_query_len = 1` or `1 + num_spec_tokens`. A batch pads to the smallest captured graph that covers it, and runs eager if none fits.
  - Attention backends declare their support level (ALWAYS / UNIFORM_BATCH / UNIFORM_SINGLE_TOKEN_DECODE / NEVER). For example, FA3 = ALWAYS, FA2 = UNIFORM_BATCH, FlashInfer = UNIFORM_SINGLE_TOKEN_DECODE.
  - Capture sizes are configured separately: `decode_cudagraph_capture_request_sizes` (by request count) and `prefill_cudagraph_capture_token_sizes` (by token count).
- SGLang's GLM-5.2 serving write-up says the overlap scheduler lets the CPU prepare the next step while the GPU runs, but the authors needed extra changes to realize the gains in practice. Secondary source. — [winzheng GLM5.2 write-up](https://www.winzheng.com/en/review/sglang-glm52-optimization)

### Inferences
- rllm's 0.6B BF16 ratio falls from 0.96x at 16 clients to 0.76x at 64. The GEMM there is plain BF16, so the loss most likely comes from per-step host work and syncs. That is the same overhead MRV2 (+56% on 0.6B) and Spec V2 (+33% on 8B at c=32) removed.
- The following are realistic in Rust:
  1. A one-step-ahead pipeline. Schedule and allocate KV for step N+1 while step N runs. Copy sampled tokens to pinned memory on a side stream and read them back a step late. Stop-token checks lag one step, and the extra token is discarded.
  2. Persistent per-request GPU state, with a small kernel that builds positions, seq_lens and block tables on the device.
  3. Extending CUDA graphs past rllm's current 8-token limit: full graphs for uniform decode at batch sizes up to 64–256, padded to captured sizes, plus graphs for spec-verify batches with 1+k tokens per request.
- Rust has an inherent advantage over Python here: rllm avoids the Python overhead that vLLM fights. So rllm's residual overhead is mostly syncs, not interpreter cost. Profile with Nsight to confirm GPU idle gaps between steps.

### Gaps
- I did not find vLLM's current default chunked-prefill token budget (`max_num_batched_tokens`) or its default CUDA graph capture size list for v0.2x.
- I did not verify whether MRV2 became the default after v0.18.

## 2. Attention and KV cache: backends, prefix caching, KV quantization, cascade attention

### Takeaway
vLLM uses block-hash prefix caching. SGLang uses a radix tree. Both reach the same reuse; rllm already has the block-level equivalent plus a RAM tier. Backend choice in vLLM matters mainly for CUDA graph compatibility. FP8 KV is the main KV feature rllm lacks, and on Ada it is mostly a capacity win, not a speed win.

### Cited Findings
- **vLLM automatic prefix caching** — [vLLM prefix caching design (v0.8.x docs, early 2025)](https://docs.vllm.ai/en/v0.8.4/design/v1/prefix_caching.html)
  - Each block's hash is hash(parent hash, the block's tokens, extra keys such as LoRA ID and multimodal hashes). Only full blocks are cached.
  - Eviction is LRU through a free queue. A request's tail block is evicted first, because it is less likely to be shared.
  - SHA256 hashing has been an option since v0.8.3 for multi-tenant collision safety.
- **SGLang RadixAttention / radix cache:** the radix-tree cache is integrated with the scheduler. The Rust `sglang-router` (cache-aware load balancer across workers, 8x A100) gave up to 1.9x throughput and raised cache hit rate from 20% to 75% on a multi-prefix workload. This is a multi-replica feature (Dec 2024). — [LMSYS SGLang v0.4](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)
- SGLang's DFlash implementation avoids storing target features separately, so the radix cache stays shareable across requests with the same prefix, including the draft model's KV. — [LMSYS DFlash/Spec V2](https://lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2/)
- **Attention backends and CUDA graph capability** (FA3 / FA2 / FlashInfer / Triton / MLA variants) — see section 1 and the [vLLM design doc: CUDA graphs](https://docs.vllm.ai/en/latest/design/cuda_graphs.html). A model takes the minimum capability across its backends, and the graph mode is downgraded to match.
- One vendor test (RunPod, cited in a secondary blog) reported that under multi-turn, high-concurrency cache pressure SGLang held ~30–31 tok/s per user while vLLM fell from 22 to 16. It is not a controlled comparison. — [techsy vLLM vs SGLang](https://techsy.io/en/blog/vllm-vs-sglang)

### Inferences
- rllm's block-level LRU prefix cache with a pinned-RAM tier is already on par in kind. What rllm lacks is a prefix cache for the draft model (noted in `compared.md`), which SGLang's DFlash/EAGLE path and vLLM both have.
- Cascade/shared-prefix attention reads a long shared system prompt's KV once per batch instead of once per request. It matters only for long shared prefixes at high batch; the 130-token benchmark won't show it. It is lower priority.
- On Ada (sm_89), FP8 E4M3 KV is supported in hardware conversion. Compared with rllm's q8 KV (int8 plus scales, Hadamard-rotated keys), FP8 KV gives a similar footprint, so the gain would be modest. rllm already beats llama.cpp at 32k with q8 KV.

### Gaps
- I did not retrieve current (2026) vLLM docs on FP8 KV accuracy and speed numbers, or on cascade attention status and default. The search returned only prefix-caching pages.
- I found no 2026 head-to-head of RadixAttention vs vLLM APC hit rates on identical workloads.

## 3. Quantized GEMM at batch 8–64: Marlin, Machete, AWQ/GPTQ, FP8; GGUF status

### Takeaway
vLLM's AWQ path on an Ada card runs Marlin-family W4A16 kernels. They keep activations in FP16 and stream pre-shuffled 4-bit weights with async copies and deep pipelining. This gives near-ideal 4-bit bandwidth speedup up to batch 16–32, which is exactly where rllm's int8-activation MMQ with a fixed 64-token tile wastes work. GGUF in vLLM is "highly experimental and under-optimized" and has moved to an out-of-tree plugin. SGLang's first-class GGUF support is documented only for Ascend. That leaves a real niche for a fast GGUF server.

### Cited Findings
- **MARLIN paper** (Frantar et al., arXiv 2408.11743, PPoPP 2025): design techniques — [arXiv HTML](https://arxiv.org/html/2408.11743v1)
  - Weights are reshuffled offline into 16×64 tiles stored contiguously, for 128-bit coalesced loads.
  - `cp.async` global-to-shared loads, with an `evict_first` cache hint on weights so they don't push activations out of L2.
  - A 4-stage pipeline, plus double buffering of shared-memory to register reads, overlapped with tensor-core MMA.
  - XOR-swizzled activation tiles in shared memory for conflict-free `ldmatrix`.
  - INT4→FP16 dequant via bit masks and a single `lop3`, two weights per 32-bit register, interleaved with MMA.
  - Each warp owns a fixed 64-wide output sub-tile and splits K, with a logarithmic reduction in shared memory.
  - "Striped" partitioning spreads work evenly across all SMs (tiles span columns) and avoids extra global reductions.
  - Group scales are reloaded from shared memory per sub-tile.
- **MARLIN numbers** — [arXiv HTML](https://arxiv.org/html/2408.11743v1)
  - Single layer: near the ideal ~3.9x over FP16 up to batch ~16–32 (A10), tapering to ~1.5x at batch 128 as the work becomes compute-bound.
  - End-to-end in vLLM, Llama-2-7B speedup over FP16 at batch 1 / 16 / 32 / 128:

    | GPU | b=1 | b=16 | b=32 | b=128 |
    |---|---:|---:|---:|---:|
    | A10 | 2.93x | 2.74x | 2.26x | 1.20x |
    | RTX 3090 | 2.69x | 2.30x | 1.84x | 1.11x |

  - Llama-2-13B on A6000: 3.17x / 2.77x / 2.39x / 1.23x.
- **Machete** (Red Hat / Neural Magic, 2024-10-14, older) — [Red Hat developers: Introducing Machete](https://developers.redhat.com/articles/2024/10/14/introducing-machete-mixed-input-gemm-kernel)
  - A CUTLASS 3.5.1-based mixed-input GEMM for Hopper (TMA and WGMMA), in vLLM 0.6.2+ for w4a16/w8a16 and GPTQ. It exists because "Marlin was specifically designed for Ampere… and struggles on Hopper."
  - It shares Marlin's key trick: weight pre-shuffling.
  - Vendor claim: Llama 3.1 70B on one H100 at 5 req/s with median TTFT under 250 ms and TPOT under 100 ms.
- On Ada GPUs (L4), blockwise FP8 falls back to Marlin kernels in vLLM. Title-level evidence only; not fetched. — [levelup.gitconnected article](https://levelup.gitconnected.com/blockwise-fp8-on-ada-gpus-why-your-l4-falls-back-to-marlin-measured-e74a0dfa3024)
- **vLLM GGUF status** (latest docs, dated 2026-06-12): "GGUF support in vLLM is highly experimental and under-optimized at the moment", "might be incompatible with other features", and "GGUF support has migrated to OOT vllm-gguf-plugin". — [vLLM docs: GGUF](https://docs.vllm.ai/en/latest/features/quantization/gguf/)
  - It is positioned mainly as a way to reduce memory footprint.
  - The docs recommend the base model's HF tokenizer, because GGUF tokenizer conversion is "time-consuming and unstable".
  - Only single-file GGUFs load (an older docs version says multi-file ones must be merged first). — [vLLM docs v0.18.0](https://docs.vllm.ai/en/v0.18.0/features/quantization/gguf/)
- **SGLang GGUF:** GGUF is documented for Ascend NPUs, dequantized to FP16/BF16 at load. For CUDA, I found no first-class GGUF support. An older quantization page lists GGUF as "will soon support". — [SGLang Ascend quantization](https://docs.sglang.io/docs/hardware-platforms/ascend-npus/ascend_npu_quantization); [SGLang quantization (older)](https://docs.sglang.ai/backend/quantization.html)
  - A cookbook page warns the GGUF path is "not optimized". — [SGLang cookbook](https://docs.sglang.io/cookbook/autoregressive/Meta/MuseGlimmer)

### Inferences
- **Why W4A16 beats int8 MMQ at batch 8–64:** at M=9–64 the GEMM is still weight-bandwidth bound. Marlin-style kernels:
  - keep the tile M at 16 (the MMA M dimension), so M=17 costs two m16 tiles, not a 64-row tile;
  - need no activation quantization pass;
  - stream weights through a deep async pipeline with offline-shuffled layout and cheap `lop3` dequant.

  rllm's fixed 64-token MMQ tile does up to 7x redundant work at M=9, and it also pays to quantize activations to int8 and decode the Q4_K superblock layout in the inner loop.
- **Most transferable fix for rllm:** a GGUF-native "Marlin-like" W4A16 kernel for Q4_K/Q5_K/Q6_K/IQ4_XS at decode batches of 2–64, with three parts:
  1. At load time, repack GGUF blocks into a tensor-core-friendly interleaved layout with pre-expanded scales/mins. Q4_K's 6-bit packed sub-block scales can be unpacked to FP16 per group of 32.
  2. FP16 activations with `mma.m16n8k16`, multi-stage `cp.async` pipelining, and striped SM partitioning.
  3. M-tile sizes of 16/32/48/64 chosen by batch.

  Repacking at load is allowed, since GGUF on disk is unchanged. This is the same kind of transformation vLLM applies to AWQ/GPTQ checkpoints for Marlin.
- A cheaper intermediate step is what rllm's chart already proposes: MMQ tile sizes matched to the batch, as llama.cpp does with `mmq_x` 8–128. That likely recovers part of the gap; a Marlin-style W4A16 kernel likely recovers more. Unverified; measure both.
- **FP8 W8A8** on Ada (sm_89 FP8 tensor cores) is an option for Q8_0-class models; GGUF has no FP8 type, so it would mean converting at load. Lower priority than 4-bit.
- **Positioning:** no major server has fast GGUF. vLLM's is an under-optimized plugin, and SGLang's is effectively absent on CUDA. "Fastest GGUF server at concurrency" is a defensible niche for rllm.

### Gaps
- I found no 2026 published numbers for vLLM's GGUF plugin throughput vs AWQ.
- I did not confirm which AWQ kernel vLLM selects on sm_89 in current versions (AWQ-Marlin is the documented path for Ampere and newer, per older docs; not re-verified here).
- I did not retrieve the paper tables for Machete vs Marlin on Ada; Machete targets Hopper only.

## 4. Speculative decoding: EAGLE-3, MTP, n-gram, DFlash; keeping it worthwhile under batching

### Takeaway
Learned draft heads (EAGLE-3, MTP, and now DFlash block-diffusion drafters) reach acceptance lengths of about 3–4.8 tokens and 2–3.3x speedups at low batch. The gain shrinks with batch size (EAGLE-3: 1.38x at batch 64 in SGLang). Both engines now run spec decode inside the overlapped, sync-free scheduler, which is what keeps it beneficial at moderate batch.

### Cited Findings
- **EAGLE-3** (Li et al., arXiv 2503.01840, NeurIPS 2025): up to 6.5x speedup over vanilla decoding, ~1.4x over EAGLE-2, and **1.38x throughput at batch size 64 in SGLang**. Acceptance length keeps rising with training data, whereas EAGLE-2 plateaus. — [NeurIPS 2025 abstract](https://proceedings.neurips.cc/paper_files/paper/2025/hash/c7b5a35ea98b62512a869c19ea7b03cb-Abstract-Conference.html); [arXiv](https://arxiv.org/html/2503.01840v3)
  - A secondary overview gives 373 vs 158 tok/s at batch 1 on SGLang v0.4.4. Not verified against the paper. — [alphaxiv overview](https://www.alphaxiv.org/overview/2503.01840v3)
- **DFlash vs EAGLE-3** (Z Lab initial implementation, Qwen3-4B, 5-layer drafters; LMSYS, 2026-06-15). — [LMSYS DFlash/Spec V2](https://lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2/)
  - DFlash is a block-diffusion drafter: one parallel forward pass produces a whole block of draft tokens, and target hidden states are injected into the drafter's KV at every layer.
  - Acceptance length / speedup:

    | Task | EAGLE-3 | DFlash |
    |---|---|---|
    | GSM8K | 4.2 / 2.1x | 4.2 / 3.3x |
    | HumanEval | 4.3 / 2.2x | 4.0 / 3.2x |
    | MT-Bench | 3.1 / 1.4x | 3.0 / 2.2x |

  - A 5-layer DFlash drafter producing 16 tokens has lower draft latency than a 1-layer EAGLE-3 drafter producing 4.
  - On Qwen3.5-397B-A17B (8x B200, HumanEval, concurrency 1) DFlash reached >4.3x baseline throughput and 1.5x native MTP. It beat MTP from concurrency 1 to 32.
- **SGLang Spec V2** became the default speculative engine in 2026; its overlap scheduling is described in section 1. — [LMSYS DFlash/Spec V2](https://lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2/)
- **vLLM:** async scheduling initially supported only MTP/EAGLE spec decode (v0.14). MRV2 supports only EAGLE/EAGLE-3/MTP among spec methods (as of v0.18), consumes rejection-sampling output on-device, and uses `idx_mapping` indirection rather than expanding request state. — [vLLM blog: MRV2](https://vllm.ai/blog/mrv2); [newreleases v0.14.0](https://newreleases.io/project/github/vllm-project/vllm/release/v0.14.0)
- Search results also mention "P-EAGLE" as a 2026 vLLM update. Not verified. — [besthub summary](https://www.besthub.dev/articles/vllm-s-four-major-2026-updates-semantic-router-athena-nemotron-3-super-p-eagle-and-model-runner-v2-3c8e94adf657)

### Inferences
- rllm's draft-model speculation (1.7x on Qwen3-8B + 0.6B at 1 request) is competitive at batch 1. The EAGLE-3 gain over a separate draft model comes from reusing target hidden states, which makes the drafter cheap (one layer).
- **EAGLE-3 is transferable:** it is a small extra decoder layer plus an FC fusing three target hidden layers. Checkpoints exist for Llama-3.x/Qwen3 on Hugging Face as safetensors, so rllm would need a small safetensors loader for the head alone. Moderate effort; high value at 1–16 clients.
- DFlash drafters are newer and fewer (Z Lab collection). Watch them, but treat as lower priority.
- **To keep spec decoding beneficial under batching:**
  1. Run the verify step inside the CUDA-graph path (uniform 1+k query length).
  2. Do rejection sampling on the GPU.
  3. Adapt k, or disable speculation, as batch size grows; the gain is ~1.4x at batch 64 even for EAGLE-3.
  4. Give the draft a prefix cache, which rllm lacks today.

### Gaps
- I found no 2026 single-GPU, 8B-class EAGLE-3 vs n-gram vs draft-model sweep across concurrency in vLLM.
- I found no official vLLM speculators acceptance-rate tables.

## 5. Structured output, tool calling, and reasoning parsers

### Takeaway
Both engines use xgrammar (default) or llguidance. The per-token mask cost is in the tens of microseconds on JSON schemas, overlapped with the GPU forward pass, so structured output costs little throughput. llguidance is a Rust crate (llama.cpp already integrates it), making it the most direct way for rllm to get full JSON Schema/regex/CFG support.

### Cited Findings
- **XGrammar** (arXiv 2411.15100) reports under 40 µs per token for JSON Schema and CFG (JSON), and up to 3x over the best baseline on JSON Schema. — [XGrammar paper](https://arxiv.org/pdf/2411.15100)
  - SGLang v0.4 (Dec 2024) claimed up to 10x faster JSON decoding than other open-source solutions with `--grammar-backend xgrammar`. — [LMSYS SGLang v0.4](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)
- **llguidance** (Rust) costs about 50 µs of CPU per token with a 128k-token vocabulary. A full mask for a typical JSON schema takes about 1.5 ms, but the "slicer" optimization brings the average over JSONSchemaBench under 50 µs. — [llguidance GitHub](https://github.com/guidance-ai/llguidance)
  - llama.cpp integrates it. — [llama.cpp llguidance doc mirror](https://huggingface.co/spaces/YZ-TAN/flask-llama/raw/main/llama.cpp/docs/llguidance.md)
- **Conflicting benchmarks:** a 2026 parser-stack-classification paper's table lists LLGuidance far faster than XGrammar on some grammar types (µs vs hundreds of ms per token). The columns were garbled in the extraction and are not verified. Secondary blogs claim xgrammar has a slight TPOT edge but slower compile (0.12–0.30 s vs under 60 ms for Guidance) for dynamic schemas. — [arXiv 2608.03065](https://arxiv.org/pdf/2608.03065); [tianpan blog](https://tianpan.co/blog/2026/04/16/grammar-constrained-generation-output-reliability)
- JSONSchemaBench (arXiv 2501.10868) is the standard coverage and speed benchmark for structured-output backends. — [JSONSchemaBench](https://arxiv.org/pdf/2501.10868)

### Inferences
- rllm's JSON mode enforces structure only, not value constraints. Embedding the `llguidance` crate (native Rust, no FFI) would add full JSON Schema (`pattern`, `minimum`, ...), regex, Lark/GBNF grammars, and low compile latency for per-request schemas.
- Compute the mask on the CPU while the GPU runs the forward pass, which requires the one-step-ahead pipeline from section 1. Apply it on the GPU as a bitmask.
- **Tool and reasoning parsers:** vLLM and SGLang ship many per-model parsers (Hermes, Llama-3 JSON, Mistral, DeepSeek, Qwen3-coder XML, and others). rllm has Hermes/Qwen and Gemma 4. Adding parsers is cheap per format and grows with model coverage.

### Gaps
- I did not fetch current vLLM/SGLang docs listing default grammar backends and parser lists for late 2026. I am relying on the general knowledge that xgrammar is the default in both, not re-verified here.
- I found no controlled head-to-head of xgrammar vs llguidance per-token cost from a neutral party.

## 6. Multi-GPU and scale features (brief, low relevance for single GPU)

### Takeaway
Most of the 2026 headline work in both engines (DP-attention, EP for MoE, PD disaggregation, Rust cache-aware routers, GB200/B200 NVFP4) targets multi-GPU datacenter serving. It does not explain the single-GPU gap rllm measured.

### Cited Findings
- SGLang v0.4 introduced DP attention for DeepSeek: up to 1.9x decode throughput on 8x H100. It also introduced a cache-aware router written in Rust (claimed 2x faster than Python alternatives). — [LMSYS SGLang v0.4](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)
- vLLM v0.16 made pipeline parallel work with async scheduling (+30.8% throughput). — [newreleases v0.16.0](https://newreleases.io/project/pypi/vllm/release/0.16.0)
- SGLang v0.5.20 (2026-09-18) notes that v0.5.19 was the last release with CUDA 12 wheels. — [ai-tldr v0.5.20](https://ai-tldr.dev/releases/sglang-v0-5-20/)
- MRV2 did not yet support LoRA as of v0.18. — [vLLM blog: MRV2](https://vllm.ai/blog/mrv2)

### Inferences
- **Relevant for a single GPU:**
  - Multi-LoRA serving (a batched LoRA kernel like Punica/S-LoRA). GGUF LoRA adapters exist in the llama.cpp ecosystem.
  - Multimodal input for Gemma 4 and Qwen VL models.
  - A Rust router across several rllm processes, cache-aware like `sglang-router`. Useful for multi-replica deployments behind AegisLLM.
- **Not relevant now:** TP, PP and EP, PD disaggregation, and DP-attention.

### Gaps
- I did not research LoRA throughput numbers or multimodal implementation details for 2026 versions.

## 7. Published benchmark numbers, 8B-class models on a single GPU

### Takeaway
Reliable, current, neutral 8B single-GPU concurrency sweeps are scarce. The best primary data points are from the engines' own blogs on B200/GB200, which are not comparable to Ada. rllm's own Ada measurements vs vLLM are the most relevant data available.

### Cited Findings
- **SGLang Spec V2 overlap:** Qwen3-8B on 1x B200 at concurrency 32 went from ~11.4k to ~15.3k tok/s (engine authors). — [LMSYS DFlash/Spec V2](https://lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2/)
- **vLLM MRV2:** Qwen3-0.6B on 1x GB200 reached ~25K output tok/s, vs 16K for MRV1. — [vLLM blog: MRV2](https://vllm.ai/blog/mrv2)
- **AIMultiple (via secondary blogs):** Llama 3.1 8B BF16 on 1x H100, 1,000 prompts × 10 runs: SGLang 16,215 tok/s vs vLLM 12,553 tok/s. Concurrency settings and the vLLM version are unclear. — [techsy vLLM vs SGLang](https://techsy.io/en/blog/vllm-vs-sglang); [morphllm vLLM benchmarks](https://www.morphllm.com/vllm-benchmarks)
- **Llama 3.3 70B FP8 on H100** (Spheron, via blogs): the two engines are within 3–5% from 1 to 100 concurrent requests; the gap is said to be larger at 8B. — [techsy vLLM vs SGLang](https://techsy.io/en/blog/vllm-vs-sglang)
- One blog says SGLang's TTFT p95 was 5–8% lower than vLLM's at every concurrency level tested; the model is not stated. — [techsy vLLM vs SGLang](https://techsy.io/en/blog/vllm-vs-sglang)
- **MARLIN end-to-end on consumer and workstation cards** (2024, older vLLM): ~2.3x over FP16 at batch 16 on RTX 3090, and 2.77x at batch 16 on A6000 (13B). — [arXiv HTML](https://arxiv.org/html/2408.11743v1)

### Inferences
- SGLang is often 0–30% ahead of vLLM on 8B-class single-GPU throughput, but the evidence is weak (secondary, version-unclear). If rllm reaches vLLM parity at 16–64 clients, SGLang would likely still be modestly ahead.
- The biggest single lever for rllm is the 4-bit GEMM at M=16–64, which explains the 0.53x. The second is CPU/sync overhead, which explains the 0.6B decline.

### Gaps
- I found no published 2026 vLLM/SGLang numbers on RTX 6000 Ada, L40S, or RTX 4090/5090 for 8B 4-bit at 16–64 concurrency.
- The AIMultiple primary page was not fetched; its figures are second-hand.
