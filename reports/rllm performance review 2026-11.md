# rllm performance review, November 2026

This review covers performance only. It looks at the working tree after this session's changes: the security and correctness fixes, the recurrent-state checkpoints and the split Gated DeltaNet (GDN) kernels. Nothing was benchmarked for the review itself, so gains are estimates from bytes moved, kernel launches and passes over the data.

**Verification markers.**
- ✔: re-checked by hand at the cited line.
- C: traced in the code by the reviewer.
- P: plausible, but the size of the gain is unmeasured.

**Baselines.**
- Qwen3.6-35B-A3B Q4_K_M on Strix Halo (this session): prompt processing 1431 tok/s at 512 tokens (pp512) and 1714 tok/s at 4096 (pp4096), decode 59.6 tok/s.
- Same model on the RTX 6000 Ada: pp512 6281 tok/s, decode 207 tok/s.

Items already on the roadmap are not repeated: batch-sized MMQ tiles, the W4A16 GEMM, the sync-free scheduler, EAGLE-3, quantized KV below q8, and the MTP head duplication and reload.

## Summary: what to do first

| # | Change | Where it helps | Est. gain | Effort |
|---|---|---|---|---|
| 1 | Drop superseded checkpoints when a conversation's new one is added | Hybrid prefix-cache hit rate with 8 agents | Avoids 10–20 s re-prefills on misses | S |
| 2 | Shortest-remaining-prompt-first prefill, and let small requests pass a blocked head | Time to first token under mixed load | Seconds to tens of seconds for short requests | S |
| 3 | Cut kernel launches in decode (about 5–6 per layer) | Decode on Ada; Strix | +8–15% decode on Ada, +3–6% on Strix | S each |
| 4 | Device sampler: radix select instead of about 84 bisection passes | Every request with temperature > 0 | +6–10% decode on Ada, +2–3% on Strix | M |
| 5 | Compact device weight layouts (Q6/Q4/IQ4 scale blocks) | Bandwidth-bound decode everywhere | about +3% on Strix, +1.5–2% on Ada | M |
| 6 | Batched or remapped MTP rollback (`keep_step`) | MTP decode, especially 8-way | up to 3% single stream, more at 8-way | S/M |
| 7 | Greedy MTP verification on the device; persistent pinned readback | Single-stream MTP | 4–8% per step on Ada | M |
| 8 | Graphs for mixed steps of up to 64 tokens | 8-way MTP verification steps | 1.5–3 ms per step on HIP | M |
| 9 | Expert-sized MoE tiles (16/32 rows) | MoE prefill (pp512 vs pp4096 gap) | toward pp4096 rates at pp512 | M |
| 10 | Cheap host fixes: JSON top-k, penalty lists, prefix matching, `make_room` | Structured output, penalties, large caches | 0.5–2 ms per token or step in those cases | S |

## 1. Scheduler and prefix cache (host)

**H1. Checkpoints evict each other.** *Fixed in this session: a new checkpoint now drops the ones it extends (tested in `rllm-kv`).* (✔ `rllm-kv/src/lib.rs`, `PrefixCache::checkpoint`)
- **Mechanism:** a new checkpoint never drops the older checkpoint it extends. Each conversation therefore holds two slots, the previous turn's checkpoint (just touched by the hit) and the new one.
- **Effect:** with 8 agents taking turns and `--state-checkpoints` at the default 8, 16 live checkpoints compete for 8 LRU slots. That cyclic access pattern makes LRU miss almost every time, and each miss re-prefills the whole conversation (20k tokens is about 20 s on Strix Halo).
- **Fix:** when adding a checkpoint, release checkpoint entries whose tokens are a prefix of the new one. Their blocks are shared, so only the state slot is freed. A branch or edit of an older turn then loses its checkpoint, which is the right trade.

**H2. First-come-first-served prefill blocks short prompts.** (C `engine.rs` step sort, `invariant_plan`)
- **Mechanism:** a 300-token chat waits behind every chunk of a 30k-token agent prompt. In batch-invariant mode, prefill chunks alternate with decode steps, so that is about 120 steps.
- **Fix:** give the prefill budget to the sequence with the fewest remaining prompt tokens. Chunk boundaries come from each sequence's own `kv.len`, so batch invariance and checkpoint alignment are unchanged.

**H3. Head-of-line admission, with repeated work on the blocked head.** (C `engine.rs` `admit`)
- **Mechanism:** a head request that doesn't fit stops all admission. Each engine iteration also redoes its cache lookup. For hybrid models that includes `resume_checkpoint`'s state-slot allocation and a state copy of tens of MB, which is then freed again.
- **Fix:** let a bounded number of smaller requests pass the head, with aging. Retry the head only when free blocks or the running set changed.

**H4. `copy_state` goes through a scratch buffer.** (✔ `qwen35.rs` `copy_state`)
- It allocates a scratch buffer on every call and moves the state twice. A direct copy between non-overlapping slots of one buffer is legal (a backend copy-within).
- **Gain:** halves about 60 MB of traffic per resume or checkpoint, about 0.3 ms on Strix.

**H5. `make_room` and the host tier.** (P `rllm-kv` `freeable`, `offload`)
- **Mechanism:**
  - `freeable()` hashes every cached block each time the pool is short.
  - Eviction frees only what is needed, so once the cache is full, every shortfall takes this slow path.
  - Host-tier offload copies whole entries over PCIe inside the step.
- **Fix:**
  - Keep a per-block count of cache references (`Vec<u16>`).
  - Evict down to a low watermark rather than exactly the need.
  - Offload between steps.

**H6. Prefix matching compares token by token.** (C `rllm-kv` `best_matches`)
- **Fix:** compare whole blocks with slice equality, which is about 10× faster. Longer term, use a block-hash chain or a radix tree.
- **Gain:** matters with a shared 16k-token tools prompt and many entries, and multiplies with H3's retries.

## 2. Sampling and speculation (host and device)

**S1. Device sampling makes about 84 full-vocabulary passes in one block.** (✔ `kernels.cu` `SAMPLE_BISECT 40`)
- **Mechanism:** with top_k and top_p both set (Qwen's recommended settings), each sampled row reads about 1 MB of logits 84 times on a single SM.
- **Fix:**
  - A radix or histogram select for top-k: 2–3 passes, with shared-memory bins.
  - Compact the survivors into shared memory, then run top-p, min-p and Gumbel over them only.
  - Optionally split the vocabulary across several blocks per row.

**S2. MTP verification copies whole vocabulary rows to the host.** (P `engine.rs` `Want::All`, `rllm-model` `to_host`)
- **Mechanism:** about 4 × 1 MB per step goes into newly allocated pageable memory.
- **Fix:**
  - Greedy requests: sample the verification rows on the device (temperature 0) and compare ids, so no rows are copied.
  - Everything else: read back into a persistent pinned buffer.

**S3. MTP draft rounds sync with the host every round.** (P `engine.rs` `propose_mtp`)
- **Fix:** for greedy and top-k cases, pick the draft token on the device and chain the rounds without host round trips. Return only the ids and the q entries that verification needs.

**S4. The host sampler builds dense vocabulary-sized vectors per draft row.** (P `sampler.rs` `Dist::Dense`, `verify` residual)
- **Mechanism:** this is the path for clients that send only `temperature`.
- **Fix:** keep q as logits plus a normalizer in the reused scratch, and draw the residual in two streaming passes instead of building a vector.

**S5. Penalty adjust lists are rebuilt, globally sorted and uploaded every step.** (P `engine.rs` `device_sample`, `backend.rs` sample)
- **Effect:** with `presence_penalty` and long reasoning outputs, that is tens of thousands of entries per step.
- **Fix:**
  - Interim: keep the counts sorted so rows concatenate without a global sort.
  - Real fix: per-slot count arrays on the device, with penalties applied in the sample kernel.

**S6. The JSON constraint partially sorts the whole vocabulary for every token.** (✔ `engine.rs:1771` `select_nth_unstable_by`)
- **Fix:** use the existing single-pass `top_candidates(logits, 256)`, and keep the full order only for the fallback.
- **Gain:** about 1–2 ms per structured token.

**S7. `keep_step` rollback.** (C `qwen35.rs` `keep_step`)
- **Mechanism:** two eager `copy_span` launches per recurrent layer and rejected sequence, about 60 launches and 120 MB moved per sequence. At 8-way MTP that is hundreds of eager launches per step.
- **Fix:**
  - Short term: one batched copy kernel driven by a pointer table.
  - Better: keep states and snapshots in one per-layer pool and make "keep" a slot-index remap with no copy.

## 3. Decode kernels and launches (device)

**K1. About 18 launches per layer (about 720 per token); several are pure waste.** (C)
On Ada, 207 tok/s is 4.8 ms per token, of which weight streaming is about 2.1 ms. Launch gaps are the biggest remaining cost there. The six fixes together save about 220 launches per token, which is +8–15% decode on Ada and +3–6% on Strix. Each is small:

- **a. The FFN input is quantized to int8 three times per MoE layer.**
  - Cause: every launch clears the cached int8 copy (✔ `backend.rs:333`), including float matvecs that only read the input. `moe_ffn` also quantizes again on its own.
  - Fix: keep the cached copy across launches that only read the input, and let `moe_ffn` use `qscratch`.
- **b. The attention gate (`sigmoid_mul`) invalidates the int8 output the attention kernels already wrote.**
  - Fix: apply σ(gate) in the attention epilogue, both decode and FA/WMMA.
- **c. The GDN output norm is three steps (rms_norm, silu_mul, then quantizing for `wout`).**
  - Fix: one kernel that writes int8 only. It also saves about 48 KB per token per layer of prefill traffic.
- **d. `conv1d_silu` and `gdn_prep` can be one kernel for decode.**
- **e. `moe_combine` can fold into the next `add_rms_norm`.**
- **f. `MATVEC_TARGETS = 3`** (✔ `backend.rs:91`), but the GDN path (qkv, z, α, β) and the attention path (q, gate, k, v) pass four weights, so each takes two launches.
  - Fix: raise it to 4.

**K2. Graphs stop at 8-token mixed steps.**
- **Mechanism:** at 8-way concurrency, MTP verification steps of 16–32 tokens run eagerly, about 720 launches per step.
- **Fix:** key graphs on the per-sequence run pattern, for mixed steps of up to 64 tokens.

**K3. Device weight layouts are larger than GGUF.** (C `backend.rs:190`, `kernels.cu` scale decoders)

| Format | Device bits per weight | GGUF bits per weight |
|---|---|---|
| Q6_K | 7.0 | 6.56 |
| Q4_K | 4.625 | 4.5 |
| IQ4_XS | 4.5 | 4.25 |
| Q3_K | 7.0 (stored as Q6) | 3.44 |

- **Mechanism:** the 248k×2048 output head is Q6_K, so it alone reads about 28 MB extra per token.
- **Fix:** compact scale blocks (Q6: 18 bytes, Q4: 16 bytes, IQ4: 10 bytes).
- **Gain:** about 80–90 MB less read per token on 35B-A3B Q4_K_M, which is about +3% decode on Strix.

**K4. Router weights are read as f32 although an f16 copy is resident.** (P `backend.rs` `f16_copy`)
- **Effect:** `ffn_gate_inp` is 2 MB per layer, 80 MB per token.
- **Fix:** read the f16 copy in decode and drop the duplicate. That saves about 40 MB per token and 80 MB of VRAM.

**K5. Decode with m > 1 re-reads experts shared by several tokens.** (C `moe_matvec*`)
- **Effect:** about 11% duplicate expert reads at 8-way, likely 30–50% for an MTP verification of one sequence.
- **Fix:** for 1 < m ≤ 8, group the pairs by expert so one warp per row serves every token routed to that expert.

**K6. HIP block shape.**
- **Mechanism:** `matvec_glu` and `moe_matvec_glu` use 1024-thread blocks, leaving about a 20% wave tail on 20 WGPs. Half the lanes idle on the 512-wide expert down projection.
- **Fix:** 256-thread blocks with 8 rows, or two rows per warp.

## 4. Prefill kernels (device)

**P1. MoE prefill tiles are about 25% full at pp512** (16 rows per expert in 64-row tiles).
- This is the likely cause of most of the gap between pp512 (1431) and pp4096 (1714) on Strix.
- **Fix:** expert-sized tiles of 16 or 32 rows for `mmq_id_*`. This is the MoE case of the roadmap's batch-sized tiles.

**P2. `moe_scan` runs on one thread.** (✔ `kernels.cu:1081`)
- **Effect:** a serial loop over 256 experts per MoE layer, about 30–100 µs.
- **Fix:** a one-block shared-memory scan. Worth about 1% of prompt time.

**P3. Further headroom in the new GDN scan.** (C `kernels.cu` `gdn_scan_impl`)
- **Mechanism:** v, g and b are global loads inside the serial token loop, and one sequence gets only 128 blocks of 4 warps.
- **Fix:** stage v, g and b per 8-token tile in shared memory, and use 8 lanes per column (256 threads).
- **Gain:** about 2× on the scan, which is 3–5% of Ada pp512 and about 1% on Strix.
- **The chunked form, measured:** a chunked (WY) recurrence on plain SIMT is now implemented and passes the parity tests. It is opt-in with `RLLM_GDN_CHUNKED=1` and was measured on Strix Halo: pp512 1256 tok/s vs 1431 for the column-split scan, and pp4096 1440 vs 1714. Its chunk products cost about twice the scan's work per token. A chunked kernel only pays with matrix cores (WMMA or tensor cores), so measure it on Ada before going further.

**P4. `conv1d_silu_par` loads the conv weights as scalars per tap.**
- **Fix:** a float4 load for kernel width 4, and fuse with `gdn_prep`. Minor.

## 5. Server plumbing (off the engine thread)

- **Responses streaming:** rebuilds each item's whole text on every piece (✔ `responses.rs:418,425`). That is quadratic: about 0.5 GB of copying for a 16k-token reasoning output. Fix: append in place.
- **`StreamDecoder`:** re-decodes up to 33 ids per token. A window of about 8 is enough.
- **Per-token JSON building and three channel hops:** a few µs per token. Not worth changing.
- **Batch-invariant mode:** the strict 1:1 alternation of prefill and decode doubles inter-token latency during prompt bursts. A configurable decode:prefill ratio would let users trade it against time to first token.

## Suggested order

1. **H1, H2 and S6** (each small): H1 fixes a regression risk in this session's checkpoint feature, H2 is the largest latency win for mixed traffic, and S6 is a one-line structured-output speedup.
2. **K1 a–f and K4**: launch and byte cuts in decode, each small and testable with the existing parity suite. Measure on Ada once its GPU is free, since that is where launch overhead dominates.
3. **S1** (radix sampler), then **S2/S3/S7** (sync-free and copy-free MTP), which also prepare for the one-step-ahead scheduler.
4. **K3** (compact weight layouts) and **P1** (expert-sized tiles): the larger bandwidth wins.
5. **K2, K5 and P3**, then the remaining host items (H3–H6, S4, S5).

## Implementation status (2026-11-17)

All of the suggestions above were implemented and tested on both GPUs:
- **Strix Halo** (gfx1151, HIP) and the **RTX 6000 Ada** (sm_89, CUDA): all 24 parity tests and the 5 model-level GPU tests on Qwen3.6-35B-A3B pass on both.
- The CPU suite passes.

Measurements are Qwen3.6-35B-A3B Q4_K_M, same session, same binary with and without each change where possible.

| Item | What was done | Result on gfx1151 |
|---|---|---|
| H1 | A checkpoint replaces the ones it extends | one slot per conversation |
| H2 | Shortest-remaining-prompt-first prefill, with aging | chunking unchanged; batch invariance kept |
| H3 | Up to 4 requests may pass one that doesn't fit (each blocked one is passed at most 8 times); admission isn't retried until something changed | no repeated checkpoint copies |
| H4, S7 | `Backend::copy_layers`: one launch per span across every layer, used by `copy_state` and `keep_step` | about 120 launches down to 3 per copy |
| H5 | Freeable blocks counted with a per-block array; eviction goes 1/64 of the pool past the need; a step's worth of blocks is kept free between steps | evictions out of step planning |
| H6 | Prefix matching compares whole blocks | — |
| S1 | Sampler: top-k bound from a 2048-bin value histogram, candidates compacted in shared memory, exact k-th and top-p there | **sampled decode 51.2 → 57.6 tok/s (greedy 58.1)** |
| S2 | Greedy verification on the device (ids only); logits readback into a recycled buffer | MTP output identical to no-MTP |
| S3 | Greedy MTP drafts' argmax on the device | as S2 |
| S4 | Host sampler: unnormalized dense distribution, residual streamed | — |
| S5 | Penalty lists merged in id order; the backend skips its sort | — |
| S6 | JSON constraint uses `top_candidates` | — |
| K1a | Float matvecs keep the input's int8 copy | one quantize fewer per MoE layer |
| K1b | `sigmoid_mul` writes the int8 copy for `wo` | one quantize fewer per attention layer |
| K1c | `norm_gate`: one fused kernel | one launch fewer per GDN layer |
| K1d | The conv writes normalized q/k; the recurrence computes the gates while staging | `gdn_prep` gone from the model path |
| K1e | `moe_ffn_add_norm`: combine, residual and norm in one kernel | bit-identical to unfused |
| K1f | `MATVEC_TARGETS` 4 | one launch fewer per layer |
| K4 | F32 decode matvecs read the f16 copy (as prefill does) | router bytes halved |
| P2 | `moe_scan` is a one-block parallel scan | — |
| P3 | Scan stages v, g and b; 8 lanes per column tried | 4 lanes kept (8: 173 vs 175 tok/s 8-way) |
| P4 | float4 conv taps | — |
| P1 | 32-pair MoE tiles (`mmq_id32`, 4×1 warps) below 48 pairs per expert | **pp512 1441 → 1603, pp1024 1716 → 1827** |
| K2 | Graphs for mixed steps up to 64 tokens | correct; such steps don't occur yet (speculation stops at 8 tokens per step) |
| K5 | Pairs grouped by expert for multi-token decode | slower (169 vs 176 8-way; MTP 74.9 vs 77.5): opt-in `RLLM_MOE_GROUP=1` |
| K6 | Smaller MoE gate/up blocks | no gain (1024: 59.7, 256: 59.5): default kept, `RLLM_MOE_GLU_THREADS` |
| K3 | Compact Q4/Q5 (16 B), Q6 (18/20 B), IQ4 (10 B) scale blocks | slower (Q4 packed: decode 59.7 → 58.3; Q6: 59.6 → 59.4): reverted, noted in kernels.cu |
| Chunked GDN | WY-form recurrence over 16/32-token chunks | slower without matrix cores (pp512 1256 vs 1431): opt-in `RLLM_GDN_CHUNKED=1` |
| §5 | Responses text appended in place; stream decoder window 8; `--decode-per-prefill` | — |

End state on Strix Halo against the build deployed before this work: pp512 1389 → 1603 tok/s (+15%), pp4096 1635 → 1737 (+6%), greedy decode 59.6 → 59.7, sampled decode 50.7 → 57.6 (+14%), MTP over ten prompts 77.6 → 77.7 tok/s (same acceptance, 67%), 8-way decode about 175 either way. Decode here is bandwidth-bound, so the launch cuts are expected to show on the Ada box.

**On the RTX 6000 Ada** (CUDA, measured after the box was restarted and its GPU was free), against the previous Ada build:

| | Before | After |
|---|---|---|
| pp512 | 6282 tok/s | 7218 tok/s (+15%) |
| pp4096 | 7201 tok/s | 8962 tok/s (+24%) |
| Greedy decode | 207 tok/s | 214.6 tok/s (+3.7%) |
| 8-way decode | 783 tok/s | 780 tok/s |
| Sampled decode, Qwen's settings | 133.9 tok/s | 212.6 tok/s (+59%) |
| Sampled decode, top-p only | 159.7 tok/s | 211.2 tok/s (+32%) |

The Strix Halo defaults hold on Ada:
- the chunked GDN kernel is slower (pp4096 7674 vs 8962);
- grouped MoE decode is slower (8-way 733 vs 780, 4-way 525 vs 561);
- 32-pair MoE tiles help (pp512 7218 vs 7129);
- the gate/up block size is a wash (256: 215.6, 1024: 214.9).
