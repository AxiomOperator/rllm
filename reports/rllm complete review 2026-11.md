# rllm complete review, November 2026

Fourth review, written 2026-11-16 at commit `5a53d61`. It covers every crate, plus a comparison against six reference projects: [lm-inference-engines](https://github.com/lapp0/lm-inference-engines), [ExLlamaV2](https://github.com/turboderp-org/exllamav2) (and V3), [Strata](https://github.com/Niko1221/Strata), [vAttention](https://github.com/microsoft/vattention), [Awesome-LLM-Inference-Engine](https://github.com/sihyeong/Awesome-LLM-Inference-Engine) and [Narwhal](https://github.com/athrael-soju/Narwhal). Nothing was built or run for this review; all findings come from reading the code. A finding marked ✔ was re-checked by hand at the line given. Items fixed in the earlier reviews (`review.md` §2, §6, §7) are not repeated.

## The short version

1. **Qwen3.5/3.6 models get no prefix reuse.** These are what most of the fleet runs. Every chat turn and agent step re-prefills the whole conversation. Strata, vLLM and SGLang solve this with recurrent-state checkpoints. This is the biggest user-visible win available. ✔ `qwen35.rs:35`, `lib.rs:146`, `engine.rs:403`
2. **The agent dashboard has a stored XSS that can steal the fleet token**, and the fleet token travels over plain HTTP. Fix the XSS before anything else in the agent. ✔ `dashboard.html:85`, `sys.rs:332`
3. **Gated DeltaNet prefill runs one token at a time** in one kernel. vLLM and llama.cpp use a chunked tensor-core form, so hybrid-model prompt speed is far below what the GPU can do. ✔ `kernels.cu:1993`
4. **The prefix cache is filled only when a request finishes.** Concurrent requests with the same system prompt, and `n>1`, never share KV. ✔ `engine.rs:848`
5. **Servers ignore SIGTERM.** Every agent-managed (systemd) server stops without draining or saving stored chats. ✔ `lib.rs:248,277`
6. Of the October roadmap, MoE (#5) and Gated DeltaNet + MTP (#10) are done. Batch-sized MMQ tiles (#1), the sync-free scheduler (#4), W4A16 (#6), llguidance (#7) and q4 KV (#11) have not been started. Fixed 64-pair tiles hurt MoE even more than dense models: with 256 experts, tiles are 75–97% empty.

## 1. Security (agent and server)

| # | Sev | Finding | Where | Fix |
|---|---|---|---|---|
| S1 | **High** ✔ | Stored XSS. `esc()` turns `'` into `&#39;`, which the browser decodes back to `'` before running `onclick`. A model file name with quotes (allowed by `safe_name`) runs script that can read `localStorage['rllm-agent-token']`. It can be planted by a pasted fetch URL or by any user who can write to a model directory. | `rllm-agent/.../dashboard.html:85`, uses at 159–358; `sys.rs:332` | Stop building inline handlers from strings: use `data-*` attributes plus `addEventListener`. Also restrict `safe_name` to `[A-Za-z0-9._-]`. Add a CSP (`script-src 'self'`) and `X-Frame-Options`. |
| S2 | High | One fleet-wide token over plain HTTP. A deploy trusts a SHA-256 sent by the same peer as the binaries, so a LAN man-in-the-middle gets code execution on every box through the rolling deploy and `.deploy/parity`. A deploy also accepts any file name, not just the three in `BINARIES`. | `deploy.rs:366-399`, `http.rs:25` | Accept only `BINARIES` names now. Later, sign releases (an ed25519 key held on the laptop), use per-peer tokens, and add TLS or a WireGuard-only listener. |
| S3 | High | Secrets in process command lines, readable by any local user via `ps` or `/proc/*/cmdline`: the curl `-H Authorization` header, `systemd-run --setenv=RLLM_API_KEY=` and `/api/servers` echoing `--api-key`. | `deploy.rs:384`, `models.rs:276`, `servers.rs:205,275` | Pass headers to curl with `-H @file` or `--config -` on stdin, use `EnvironmentFile=` with mode 0600, and redact `--api-key` in listings. |
| S4 | Med | SSRF and disk fill through `/api/models/fetch`: curl `-L` with no `--proto =https,http`, `--globoff`, `--max-filesize`, or a block on loopback or link-local addresses. The result can be read back via `/api/files/models`. | `models.rs:183,262,274` | Add those flags, reject private and loopback targets unless allowed, and enforce the expected size during the transfer. |
| S5 | Med | Agent-started servers default to `0.0.0.0` with no API key. | `servers.rs:52,262` | Default to 127.0.0.1, or require `serve_api_key` whenever the host is not loopback. |
| S6 | Med | The fleet proxy buffers whole responses (proxying a model file means tens of GB in RAM). A percent-decoded `rest` inserted into the request line allows CRLF header injection. | `http.rs:37`, `lib.rs:471` | Stream the body, and reject `\r\n` in the path. |
| S7 | Low ✔ | The server API key comparison is not constant-time (the agent's is). | `rllm-server/src/lib.rs:399` | Use the agent's constant-time compare. |
| S8 | Low | `Gguf::open` turns metadata arrays into `Vec<Value>` with no cap, about 32 B per element. A 1 GB hostile array can take about 32 GB of RAM, which matters now that the agent fetches from any URL. | `rllm-gguf/src/lib.rs:508` | Cap the total elements, or store typed arrays. |

## 2. Correctness

| # | Sev | Finding | Where |
|---|---|---|---|
| C1 | Med ✔ | `SeqRuns::check` doesn't require distinct state slots or disjoint snapshot ranges. The CPU backend builds `&mut` slices from `SyncPtr` in rayon jobs, so safe code can create aliased `&mut` (undefined behaviour). | `rllm-core/src/lib.rs:186`, `rllm-cpu/src/lib.rs:495,513,571,606` |
| C2 | Med | Two fetches of the same name share `.partN` and `.try` files and corrupt each other. Cancelling one deletes the other's parts. A resume never checks `Content-Range`, so leftovers from another source are joined unchecked; only a known SHA-256 catches it. | `rllm-agent/src/lib.rs:251`, `models.rs:252,282` |
| C3 | Med | A rolling deploy reports "installed; agent restarted" after any polling error and moves on to the next box. | `rllm-agent/src/lib.rs:496-500` |
| C4 | Med | The disk check requires `size + 2 GiB`, but joining parts peaks at about 1.9× the size, and concurrent fetches aren't counted. The fit check uses current free memory, so two starts close together both pass. `servers.json` is updated without a lock. | `models.rs:439,499`, `servers.rs:257,305` |
| C5 | Med | A client that stops reading (but keeps the socket open) makes the server buffer the whole generation through unbounded channels. There is no per-request timeout. | `rllm-server/src/util.rs:89`, `generation.rs:88,93` |
| C6 | Low | The same `seed` gives different sampled output depending on load: device sampling draws from the token position, while host and draft sampling use a stateful stream. That breaks the batch-invariant promise. | `engine.rs:1173,1520,1559` |
| C7 | Low ✔ | `min_p` is accepted but silently ignored (there is no implementation). | `sampler.rs`, chat request types |
| C8 | Low | `--max-queue` can be exceeded: `accept` lowers `queued` before `waiting` is published. | `engine.rs:180,723,752` |
| C9 | Low | Validation allows `ssm.state_size` up to 256, but the GPU kernel handles only 64 and 128, so such a model loads and then fails every forward. `keep_step` doesn't check `kept` against the sequence's own snapshot count. The Clef head checks only some tensor shapes, so a malformed file panics on the first request. | `qwen35.rs:177,911`, `backend.rs:2560`, `clef.rs:282-343` |
| C10 | Low | `tune --measure` always steps 8 sequences but sizes recurrent state for `parallel + 2`, so it fails on hybrids with parallel below 6. `stale/kill --services` kills managed servers on every port except a hard-coded 8097. | `rllm-cli/src/tune.rs`, `tools/stale.sh:45-66` |
| C11 | Low | `moe_route` on the server side lets a NaN logit win and poison the weights. The kernel (checked) treats NaN as −inf. | `rllm-core/src/lib.rs:254-262` |

The reviews also checked these and found them correct: slot, KV and draft-KV accounting; preemption; cancellation; stop strings that span tokens; UTF-8 hold-back; logprobs; MTP `keep_step` rollback; host-tier restore; GGUF bounds and alignment; MoE ordering; Q3_K repack; `attn_decode` 64-bit offsets; and the HIP Send/Sync impls.

## 3. Performance

| # | Finding | Evidence | Fix | Gain (estimate) |
|---|---|---|---|---|
| P1 | **No prefix reuse on hybrid models.** Both `is_recurrent` and the default batch-invariant mode disable it. | ✔ `lib.rs:146`, `engine.rs:403` | Snapshot the conv and SSM state at block or turn boundaries and store it beside the KV blocks, keyed by the block hash. Restore snapshot plus KV on a hit. Strata adds a tail checkpoint and a last-turn-boundary checkpoint (time to first edit −33%). vLLM and SGLang (Mamba radix cache) do the same. About 60 MB per checkpoint on 35B-A3B, so keep a few per conversation and spill them to the host tier. | Agent and chat turns: re-prefill drops from the whole history to the new turn, often 10–50× less prompt work. |
| P2 | **GDN prefill is serial per token.** One block per (head, sequence); each token costs 2 block reductions, 4 barriers and dependent 128-long FMA chains. | ✔ `kernels.cu:1993-2050` | Chunked gated delta rule (WY form, chunks of 64, tensor-core GEMMs), as in FLA, vLLM and llama.cpp. Quick win: split the dot-product accumulators 4 ways. | Prompt speed on all Qwen3.5/3.6 models, roughly 0.25 s saved per 8k tokens per pass |
| P3 | `conv1d_silu` uses one thread per channel and walks every token. | `kernels.cu:1950` | Use one thread per (token, channel). | ~16 ms/layer at 8k tokens |
| P4 | Prefix cache is filled only at retire, so concurrent requests and `n>1` duplicate prefill and KV. | ✔ `engine.rs:848,865` | Register each block as soon as it fills (vLLM, SGLang). For `n>1`, prefill once and fork. | Shared system prompts, `n>1` |
| P5 | MoE grouped tiles are a fixed 64 pairs per expert; with 256 experts they are 75–97% empty. | `MMQ_BN = 64`, `moe_scan` | Roadmap #1: batch-sized tiles (8–128) chosen by llama.cpp's fewest-tiles rule. | MoE prefill and batched decode |
| P6 | `mtp_draft` copies n×131k logits to the host every round and scatters them into a −inf-filled 248k vector. | `qwen35.rs:986-998` | Sample drafts on the device with the existing `sample_rows`. | One sync and a ~1 MB copy saved per draft round |
| P7 | MTP head weights are duplicated: `head` is loaded but unused when `draft_head` is set, and `draft_head` copies a prefix of the output head. | ✔ `qwen35.rs:606-618` | Make `head` lazy, and use a row-offset view instead of a copy. | ~0.2–1 GB VRAM |
| P8 | Load reads split tensors from disk twice, because pages are released after the first half (`attn_q`, fused `gate_up_exps`). | `loader.rs:83`, `qwen35.rs:521,540` | Split both halves in one pass, then release. | ~12 GB less I/O on 35B-A3B |
| P9 | Recurrent state is f32 (~60 MB per sequence) and gets a full snapshot per verified draft token. | `qwen35.rs` | Snapshot the per-token (k, v, g, β) instead and replay the accepted prefix. Consider a bf16 state. | MTP step bandwidth |
| P10 | Host-side: `OutputParser` re-scans the whole buffer on every push (quadratic over long tool calls); prompt lookup does an O(context) scan per step; `is_special` is a linear scan; a new thread per sequence for host sampling every step. | `output.rs:192`, `engine.rs:591,1187`, `tokenizer/lib.rs:333` | Search incrementally, keep an n-gram index, a hash set and a persistent pool. | Long outputs, high batch |
| P11 | `/v1/systemone` copies all prompt rows to the host (~335 MB at 32k tokens), and the Clef head runs on the CPU. | `decision.rs:412`, `engine.rs:1452` | Copy only the tagged spans; run the head through the backend matmuls. | Long-context decisions |

## 4. What the reference projects do that rllm doesn't

**ExLlamaV2/V3** (V2 archived 2026-03; V3 at v1.6.7, very active). The ideas worth taking:
- A **Q4/Q6 KV cache**, Hadamard-rotated. ExLlamaV2 measured Q4 as more accurate than FP8 ([qcache_eval](https://github.com/turboderp-org/exllamav2/blob/master/doc/qcache_eval.md)). rllm already rotates keys for q8, so this is an incremental change, and it doubles context or batch size per GB.
- **llguidance** grammars (adopted in V3 v1.4.1), including grammars combined with speculative drafting.
- **Hybrid draft + n-gram** speculation and **adaptive draft length** (v1.4.3, v1.6.1).
- **MoE expert offload** to pinned host RAM (v1.5.0).
- A **KV tier to RAM and disk** for preempted sequences, not just prefixes.
- **DRY, XTC, banned strings and a reasoning-token budget**. These are cheap to add and popular with local users.

**Strata** (C++ on ggml; 44.9k stars, two months old, very fast commit rate). It targets the same Strix Halo class of machine:
- **GDN state checkpoints** at the tail and turn boundaries (P1 above).
- **Free memory computed as `MemAvailable` minus 6 GiB** for unified memory. The agent's `plan` should do the same: it currently has no safety margin and ignores the q8 staging buffer.
- **Acceptance-gated prompt lookup**: draft only where measured acceptance pays.
- A **tuned hipBLASLt table** per gfx1151.
- The recommended **`ttm.pages_limit`** boot setting for a 112 GiB GTT.

**vAttention** (Microsoft, ASPLOS 2025). It keeps each request's KV contiguous in virtual memory with CUDA VMM:
- Its gains come from reusing vendor non-paged attention kernels. rllm writes its own paged kernels and shares blocks for the prefix cache, so the core idea doesn't fit.
- Worth taking: a **growable KV pool** that reserves virtual address space and maps 2 MB chunks on demand. On a 128 GB APU, KV memory could then grow and shrink with load instead of being fixed at startup.
- HIP has `hipMemMap`, but the ROCm docs warn of incorrect results on RDNA for aliased mappings. Check `hipDeviceAttributeVirtualMemoryManagementSupported` on gfx1151 first.

**Narwhal** (prefill/decode disaggregation controller, small project). Disaggregation itself doesn't fit: one GPU per box, Ethernet between boxes. Two ideas carry over:
- **Predictive admission**: return 429 with `Retry-After` when the projected TTFT, from a fitted prefill cost curve, would break an SLO.
- **SLO-aware routing**: the agent could send each request to the node with the lowest projected TTFT.

**lm-inference-engines** is stale (last commit 2024-12). **Awesome-LLM-Inference-Engine** is current (26 engines). Against their taxonomy, rllm still lacks:
- Below-8-bit KV cache
- Grammar and regex constraints
- EAGLE, Medusa or DFlash drafters
- LoRA
- Vision input
- Multi-GPU
- MXFP4, Q4_0, Q5_x, Q2_K and IQ2/IQ3
- Beam search

Of these, only multi-GPU and beam search are low value for this fleet.

**Serving features missing compared with vLLM, SGLang and llama-server:**
- Request priorities and fair scheduling. Admission is strict FIFO and blocks behind the first request that doesn't fit.
- Per-request deadlines, and rate limits.
- A readiness endpoint separate from `/health`.
- `max_model_len` in `/v1/models`.
- Sampling and stop options: `min_p` (C7), `repetition_penalty`, `ignore_eos`, `stop_token_ids`, `min_tokens`.
- Multiple forced tool calls.
- TTFT and inter-token latency histograms.
- FIM (`suffix`).

**Agent features missing compared with llama-swap, GPUStack and Ollama:**
- `Restart=on-failure` and a health watchdog (servers run with `Restart=no`).
- Load on request and idle unload behind one endpoint.
- Fleet-wide routing.
- Multi-GPU device selection (always GPU 0).
- Jobs that survive an agent restart.
- Agent metrics.

## 5. Recommended order

| # | Work | Why | Effort |
|---|---|---|---|
| 1 | **Agent security batch**: the XSS (S1), deploy file allowlist (S2, part 1), secrets off command lines (S3), curl hardening (S4), server default host and key (S5), proxy CRLF and streaming (S6), constant-time server key (S7) | Live exposure on the LAN today; all are small | 2–3 days |
| 2 | **SIGTERM drain**, plus a bounded SSE channel and a per-request deadline (C5) | Every managed server is affected | 1 day |
| 3 | **Correctness batch**: C1 slot checks, C2/C3/C4 agent job races, C7 `min_p`, C9 load-time validation, C11 | Cheap, and removes undefined behaviour and silent wrong results | 2–3 days |
| 4 | **Hybrid prefix cache with state checkpoints** (P1), and drop batch-invariant as the reason prefix caching is off | Largest user-visible win on the models the fleet actually runs | 1–2 wk |
| 5 | **Chunked GDN prefill plus a parallel conv1d** (P2, P3) | Prompt speed on every Qwen3.5/3.6 model | 1–2 wk |
| 6 | **Batch-sized MMQ tiles** (old roadmap #1), measured on MoE first (P5) | Still not started; matters more for MoE | 1–2 wk |
| 7 | **Incremental prefix registration and `n>1` fork** (P4) | Concurrent clients sharing system prompts | 3–5 days |
| 8 | **Q4/Q6 KV cache** (ExLlama-style, reusing the Hadamard rotation) | 2× context per GB; big on 48 GB and for long agent sessions | 1 wk |
| 9 | **MTP and load cleanups** (P6–P9) | VRAM, load time and per-step syncs | 3–5 days |
| 10 | **llguidance** and the missing sampler options (DRY, `repetition_penalty`, `stop_token_ids`, …) | Structured output parity; local-user expectations | 1–2 wk |
| 11 | **Agent operations**: Restart and watchdog, `MemAvailable`-minus-headroom planning, idle unload and load-on-request, signed releases | Fleet reliability | 1–2 wk |
| 12 | **Sync-free one-step-ahead scheduler** (old #4), then batched MTP beyond 4 sequences | The throughput ceiling at concurrency | 3–5 wk |
| Later | MXFP4/Q4_0/Q5_x (gpt-oss, QAT Gemma), W4A16 GEMM, EAGLE/DFlash, MoE expert offload, growable VMM KV pool, predictive admission, vision, LoRA | Valuable, but behind the items above | — |
| Skip | Prefill/decode disaggregation, full vAttention, multi-GPU, Vulkan/Metal, EXL3 format, beam search | Wrong fit for one-GPU-per-box GGUF serving | — |

Items 1–3 are about a week in total and remove every known security and correctness defect. Items 4–5 are where most of the speed is for this fleet's actual workload: Qwen3.6 hybrids serving multi-turn agents.
