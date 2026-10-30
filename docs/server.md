# OpenAI-compatible server

`rllm serve` exposes a loaded GGUF model over the OpenAI HTTP API, so the
official SDKs and other OpenAI clients work against it unchanged. It follows
the API reference at <https://developers.openai.com/api/reference/overview>
for the endpoints, request parameters, response objects, streaming events and
error format.

```sh
rllm serve models/Qwen3-0.6B-Q8_0.gguf --port 8080
rllm serve model.gguf --host 0.0.0.0 --port 8080 --ctx 8192 --parallel 8 \
    --api-key "$KEY" --model-name my-model --strict-model -d cuda:0 \
    --store-file store.json
# A draft model for speculative decoding, and an 8-bit KV cache:
rllm serve models/Qwen3-8B-Q4_K_M.gguf --draft models/Qwen3-0.6B-Q8_0.gguf --kv-type q8
# Speculation without a draft model, and 16 GiB of system RAM for the prefix cache:
rllm serve models/Qwen3-8B-Q4_K_M.gguf --prompt-lookup --host-cache-gb 16
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="unused")
r = client.chat.completions.create(model="my-model", messages=[{"role": "user", "content": "Hi"}])
```

| Flag | Default | Meaning |
|---|---|---|
| `--host`, `--port` | `127.0.0.1`, `8080` | Listen address. |
| `-d` | `cuda` | `cuda`, `cuda:N` or `cpu`. |
| `--ctx` | 8192 | Most tokens (prompt + output) one request may use. |
| `--parallel` | 4 | Most requests decoded together in one batch. More wait in a queue. |
| `--kv-tokens` | `ctx × parallel` | Size of the KV cache shared by all requests, in tokens. Allocated once at startup. |
| `--no-prefix-cache` | off | Don't keep finished requests' KV for reuse. |
| `--kv-type` | `f16` | KV cache storage: `f16`, or `q8` (8-bit, about half the memory; see [KV cache type](#kv-cache-type)). |
| `--draft` | none | Draft model for speculative decoding: a small GGUF with the same tokenizer (see [Speculative decoding](#speculative-decoding)). |
| `--draft-tokens` | 5 | Most tokens the draft (or prompt lookup) proposes per step. |
| `--prompt-lookup` | off | Speculative decoding without a draft model: propose the tokens that followed the latest few tokens' earlier occurrence in the prompt or output (see [Prompt lookup](#prompt-lookup)). |
| `--host-cache-gb` | 0 | Pinned system memory (GiB) for the prefix cache's host tier: evicted prefixes move there and come back on a hit (see [Host tier](#host-tier)). |
| `--api-key` | `$RLLM_API_KEY`, else none | Require `Authorization: Bearer <key>`. |
| `--model-name` | file name without `.gguf` | Model id reported by `/v1/models` and in responses. |
| `--strict-model` | off | Answer requests naming another model with 404 `model_not_found`, as OpenAI does. Off by default: any model name is served by the loaded model, so clients that hard-code e.g. `gpt-4o` still work. |
| `--prefill-chunk` | 512 | Prompt tokens per forward pass (long prompts are split, so running requests keep decoding in between). |
| `--store-file` | none | Save stored chat completions and responses to this JSON file (every 2 s when changed, and at shutdown) and load it at start. |
| `-q`, `--quiet` | off | Don't log a line per finished generation. |

Environment: `RLLM_API_KEY` (as `--api-key`), `RLLM_ENGINE_TIMING=1` (print
where engine step time goes every 256 steps, draft model included), `RLLM_NO_PIN=1` (see
[Performance](#performance)).

## Choosing settings: `rllm tune`

`rllm tune model.gguf` reads the model's metadata and the machine (free GPU
memory, system RAM, CPU) and prints a ready-to-run `rllm serve` command with
the reason for each setting:
- **`--kv-type`:** q8 for contexts of 8k or more (half the memory, faster decoding at depth), or when f16 wouldn't fit two full contexts.
- **`--kv-tokens`:** all GPU memory left after weights, runtime buffers, the draft model and a margin (4% of free memory, at least 0.5 GiB).
- **`--ctx`:** the model's trained context, up to 32768; less if memory doesn't allow it.
- **`--parallel`:** 8, or fewer if the pool holds less than about 2k tokens per request.
- **`--draft`:** the smallest GGUF next to the model with the same tokenizer, if it is at least 4× smaller and memory allows. Skipped with `--parallel 5` or more: speculation pauses with that many requests decoding.
- **`--prompt-lookup`:** on.
- **`--host-cache-gb`:** a quarter of available RAM, at most 4× the KV pool and 64 GiB, as long as 8 GiB stays available.

It picks the GPU with the most free memory, so it accounts for other
processes on a shared GPU. `--ctx`, `--parallel`, `--draft`, `--no-draft`
and `-d` constrain the plan, and `--json` prints it for scripts.

The estimate never touches the GPU beyond asking for its free memory.
`--measure` loads the model and runs the plan for real (one request
prefilled to the full context, then 8 decoding together, about 10–20 s on
the RTX 6000 Ada). Freed memory stays reserved during the run, so the
lowest free memory it sees includes every temporary buffer's peak. It then:
- replaces the estimated weights and runtime memory with measured ones, resizing the KV pool to leave exactly the margin;
- shrinks the pool if the run runs out of memory;
- drops the draft if its decode step takes more than 30% of the model's.

The estimates are calibrated against `--measure` on the RTX 6000 Ada.
Weights came within 2% for Qwen3-0.6B Q8_0/BF16, Qwen3-8B Q4_K_M and
Nemotron E4B Q8_K_XL; K-quants take about 11% more on the GPU than in the
file. On the same box, the recommended commands for Qwen3-8B (with a draft)
and Nemotron served a 30k-token prompt plus 7 concurrent requests with no
errors and 1.3 GiB to spare.

## Endpoints

All under `/v1`. Unknown paths, and OpenAI resources this server doesn't
implement, return OpenAI's `404 Invalid URL (METHOD /path)` error.

| Endpoint | Status |
|---|---|
| `GET /models`, `GET /models/{model}` | The served model. |
| `DELETE /models/{model}` | 403: only fine-tuned models can be deleted. |
| `POST /chat/completions` | Full: streaming, `n`, tools, JSON mode, logprobs, stored completions. |
| `GET /chat/completions` | List stored completions (`model`, `metadata[k]=v`, `after`, `limit`, `order`). |
| `GET`, `POST`, `DELETE /chat/completions/{id}` | Retrieve, update `metadata`, delete. |
| `GET /chat/completions/{id}/messages` | The stored request messages, paginated. |
| `POST /completions` | Legacy completions: string, string-array, token and token-array prompts; `echo`, `logprobs`, `best_of`, `n`, streaming. `suffix` (insertion) is rejected. |
| `POST /responses` | Text and function-call output, reasoning items, streaming events, `previous_response_id`, background mode, `store` (on by default). |
| `GET`, `DELETE /responses/{id}`, `POST /responses/{id}/cancel` | Retrieve, delete, cancel a background response. |
| `GET /responses/{id}/input_items` | Paginated input items. |
| `POST /responses/input_tokens` | Prompt token count for a Responses request. |
| `POST /embeddings` | String, string-array and token inputs; `dimensions`; `float` or `base64`. |
| `GET /health` | `ok` (outside `/v1`, no auth). |
| `GET /metrics` | Prometheus metrics (outside `/v1`; behind the API key if one is set). |
| `POST /tokenize`, `POST /detokenize`, `POST /apply-template` | llama.cpp-style helpers (outside `/v1`): `{"content", "add_special", "with_pieces"}` → `{"tokens"}`; `{"tokens"}` → `{"content"}`; `{"messages", "tools"?}` → `{"prompt"}`. |

Every response carries `x-request-id`, `openai-processing-ms` and
`openai-version: 2020-10-01`; `X-Client-Request-Id` is validated and echoed.
CORS is open, so browser clients can call the server directly.

Errors use OpenAI's shape, `{"error": {"message", "type", "param", "code"}}`,
with the same status codes: 400 `invalid_request_error` (naming the bad
parameter in `param`), 401 for a missing or wrong key (`invalid_api_key`),
404 for unknown models, objects and URLs, 400 `context_length_exceeded`,
500 `server_error`, 503 if the engine stopped.

## Request parameters

Chat Completions:
- **Sampling:** `temperature`, `top_p`, `seed` (reproducible for the same seed and server), `frequency_penalty`, `presence_penalty`, `logit_bias`, plus the common `top_k` extension.
- **Length and stopping:** `max_completion_tokens` (or the deprecated `max_tokens`) and up to 4 `stop` sequences. A stop sequence ends the output and is not included.
- **Choices and logprobs:** `n` (1–128), `logprobs` and `top_logprobs` (up to 20). Logprobs come from the model's raw distribution.
- **Streaming:** `stream` and `stream_options.include_usage`.
- **Tools:** `tools` (function tools), `tool_choice` (`none`, `auto`, `required`, a named function, `allowed_tools`), `parallel_tool_calls`, and the deprecated `functions` / `function_call`.
  - Tool definitions go through the model's own chat template.
  - Tool calls are parsed from the model's native format: Qwen/Hermes `<tool_call>` JSON, or Gemma 4 `<|tool_call>call:…`.
  - `required` and named choices are enforced by starting the model's output with the tool-call opening, then constraining the call to the tools' names and parameter schemas (in either format), so the arguments always match a tool's `parameters`.
- **`response_format`:** `json_object` and `json_schema`.
  - Decoding is constrained token by token, and stops when the value is complete. `json_object` output is always a valid JSON object.
  - `json_schema` output always matches the schema's structure (see [Structured output](#structured-output)). The schema is also given to the model in the system prompt.
- **`reasoning_effort`:** `none` or `minimal` turn the model's thinking mode off; any other value turns it on. Without it, the chat template's default applies.
  - Thinking text is returned as `message.reasoning_content` (the de facto field used by compatible servers), or `delta.reasoning_content` when streaming.
  - Usage counts it in `completion_tokens_details.reasoning_tokens`.
- **Storage:** `store` and `metadata` (in memory; saved to disk with `--store-file`).
- **Usage:** `prompt_tokens_details.cached_tokens` counts prompt tokens served from the prefix cache.
- **Accepted but without effect:** `user`, `safety_identifier`, `prompt_cache_key`, `service_tier` (reported as `default`) and `prediction`.
- **Rejected with 400:**
  - non-text content parts (`image_url`, `input_audio`, `file`)
  - `audio` / `modalities: ["audio"]` output
  - `web_search_options`
  - custom (non-function) tools

Responses:
- **Input:**
  - `input` as a string, or as items: messages with text content, `function_call` and `function_call_output`.
  - Reasoning items from earlier turns are kept as input items but not replayed into the prompt.
- **Generation:** `instructions`, `max_output_tokens`, `temperature`, `top_p`, `top_logprobs` / `include: ["message.output_text.logprobs"]`.
- **Tools:** `tools` (function tools), `tool_choice`, `parallel_tool_calls`, `max_tool_calls`.
- **Output format:** `text.format` (`text`, `json_object`, `json_schema`, as above), and `reasoning.effort`.
- **Conversation:** `previous_response_id` (the stored conversation, without the earlier `instructions`), and `truncation: "auto"`, which drops the oldest turns until the prompt fits.
- **Storage and execution:** `store`, `background` (requires `store`), `metadata`, `stream`.
- **Output items:**
  - `reasoning` items (`content: [{"type": "reasoning_text"}]`)
  - `message` items (`output_text`)
  - `function_call` items
- **Status:**
  - `status` is `completed`, `incomplete` (with `incomplete_details.reason: "max_output_tokens"`), `cancelled` or `failed`.
  - Background responses start as `queued`, and `cancel` stops them.
- **Streaming events:**
  - `response.created`, `response.in_progress`
  - `response.output_item.added` / `.done`, `response.content_part.added` / `.done`
  - `response.output_text.delta` / `.done`, `response.reasoning_text.delta` / `.done`
  - `response.function_call_arguments.delta` / `.done`
  - `response.completed` / `response.incomplete` / `response.failed`, and `error`
  - Every event carries its `sequence_number`.
- **Rejected with 400:** `conversation`, `prompt` (templates), `context_management`, built-in tools (web search, file search, …), and non-text input.

## Structured output

`json_schema` formats and forced tool calls are compiled into a validator
that runs on every candidate token. Supported keywords:
- `type`, including type lists such as `["string", "null"]`, and `nullable`
- `properties`, `required`, `additionalProperties`
- `items`, `prefixItems`, `minItems`, `maxItems`
- `enum`, `const`
- `anyOf` / `oneOf`, `allOf` (merged)
- local `$ref`s (`#/$defs/…`, `#/definitions/…`), recursive ones included

As in OpenAI's structured outputs, an object that lists `properties`
without `additionalProperties` gets no other keys. Keys may come in any
order, never twice, and the object can only close once every `required`
key is present. Value constraints (`pattern`, `format`, `minimum`,
`minLength`, …) are not enforced. The model sees the schema in its prompt
and usually follows them.

Forced tool calls use the same validator. Hermes calls are JSON
(`{"name": …, "arguments": …}`). Gemma 4 calls use Gemma's own syntax
(`name{key:<|"|>text<|"|>,n:1}`): bare keys, strings between the
`<|"|>` token.

Embeddings are the model's final hidden states, mean-pooled over the input's
tokens and L2-normalized. `dimensions` truncates and renormalizes. These are
general-purpose sentence vectors from a chat model, not a trained
retrieval-embedding model.

## How requests run

- **One model, one engine thread, continuous batching.** The engine owns a paged KV cache of `--kv-tokens` tokens, shared by all requests.
  - Each forward pass takes one token from every decoding request, plus up to `--prefill-chunk` prompt tokens from requests still reading their prompt.
  - New requests join the batch as soon as a slot (`--parallel`) and KV blocks are free. Finished ones leave without waiting for the rest.
  - Steps of up to 8 tokens (decoding requests, or drafts being verified) replay CUDA graphs. Steps that mix in prompt chunks run directly.
  - Decode attention splits long contexts over several thread blocks per head and merges them in a fixed order (2× faster on Qwen3-0.6B at 8k context on the laptop GPU).
  - HTTP handling, tokenization, chat templating, stop sequences and output parsing run on the async runtime.
- **Prefix cache.** When a request finishes, its KV blocks stay cached. A later request whose prompt starts with the same tokens reuses those blocks instead of recomputing them. That covers the next turn of a chat, a repeated system prompt, `n` choices or `previous_response_id`.
  - Only whole 16-token blocks are shared, so the last few tokens before the difference are recomputed.
  - Least recently used entries are evicted when blocks run out: dropped, or with `--host-cache-gb` moved to system memory (see [Host tier](#host-tier)).
- **When the KV cache is full.** Cached prefixes are evicted first. If that isn't enough, the newest running request is paused, and its blocks go into the prefix cache. It resumes later by recomputing (often reusing) its tokens. Clients only see a short stall.
- **Sampling.** Each request's logits are processed on the host: logprobs, `logit_bias`, penalties, the JSON constraint, then sampling. With several requests in a step, this runs on one thread per request.
- **Numerics.** A request batched with others runs through the multi-row int8 kernels (the ones prefill uses), and a lone request through the single-row decode kernels.
  - Logprobs agree to about 0.01–0.02 on average, but greedy text can drift apart after a few dozen tokens.
  - Results are reproducible for the same mix of concurrent requests: every GPU reduction sums in a fixed order, so the same tokens give bit-identical logits on every run. (Prefill matmuls split along k used to `atomicAdd` their parts in whatever order they finished.)
- **Cancellation.** A client that disconnects (a closed stream, a dropped connection) cancels its generation at the next step and frees its slot. Stop sequences cancel the rest of the generation too.
- **`n` and `best_of`.** The choices run concurrently, so they share batches. They use seeds `seed`, `seed + 1`, …. Streaming with `n > 1` still sends the choices one after another.
- **Storage.** Stored chat completions and responses live in memory.
  - With `--store-file` they are written to that JSON file (atomically, at most every 2 s, and at shutdown) and loaded at start.
  - A background response still running at shutdown loads as `failed`.
- **Logging.** One line per finished generation: prompt tokens (cached), generated tokens, time to first token, decode rate, total time and finish reason.

## Speculative decoding

With `--draft`, a small model with the same tokenizer (for example
Qwen3-0.6B for Qwen3-8B) proposes a few tokens for each decoding request,
and one forward pass of the main model checks them all. Every accepted token
is a main-model step saved.
- **Exact.** Verification follows the speculative sampling rule (accept a drafted token with probability min(1, p/q), otherwise draw from the leftover distribution), so output is distributed exactly as without a draft. Greedy output is token for token the same on Qwen3: a verification step computes the same logits as one-token steps. On Gemma 4 models, whose one-token steps use their own fused kernels, it matches as closely as batched requests do.
- **Request features.** Logit bias, frequency and presence penalties, logprobs, stop sequences and `n` all apply as usual (the draft sees the bias and penalties too). Requests with a JSON constraint (`json_object`, `json_schema`, forced tool calls) decode without drafts.
- **Adaptive.** Each request proposes one token more than it last had accepted, up to `--draft-tokens`, so drafts stay short where the text is hard to predict. After three verifications in a row that accept nothing, the request decodes without drafts for 8 steps.
- **Batching.** Drafts are shortened so a step stays within 8 tokens (one request: up to 7 drafts; two: 3 each; three or four: 1 each). With 5 or more requests decoding, the batch alone keeps the GPU busy and speculation pauses.
- **Short sequences only.** The draft model drafts for sequences up to 6144 tokens; longer ones decode with prompt lookup only (if on) and release their draft cache. Beyond that length the draft's attention and its catch-up over the prompt cost more than it saves:

| Qwen3-8B + 0.6B draft, RTX 6000 Ada | 901 tokens | 4111 | 8331 | 16751 |
|---|---:|---:|---:|---:|
| No draft | 137.7 tok/s | 132.1 | 124.0 | 111.2 |
| Draft at every length (before) | 178.9 | 161.3 | 121.6 | 84.0 |
| Draft up to 6144 tokens (now) | 179.1 | 161.6 | 124.0 | 111.8 |

- **Draft cache.** The draft model has its own KV pool of 4 × 6144 tokens (or the main pool's size, if smaller): enough for every sequence that can draft at once. Each request's draft cache catches up lazily (its whole prompt at the first decode step) and is cut back to the accepted tokens after each verification. It has no prefix cache. If the pool runs out, that step goes without drafts.
- **Pairs need the same vocabulary.** The server checks the logits width and every token string at startup.

| Qwen3-8B Q4_K_M, RTX 6000 Ada | rllm | rllm + Qwen3-0.6B draft | llama-server | llama-server + draft |
|---|---:|---:|---:|---:|
| Single request, 6 mixed prompts (mean decode) | 141 tok/s | 213 tok/s (1.51×) | 146 tok/s | 126 tok/s |
| Single request, list continuation | 140 tok/s | 246 tok/s (1.76×) | 143 tok/s | 134 tok/s |
| 4 clients, total output | 475 tok/s | 515 tok/s | 401 tok/s | 189 tok/s |
| 8 clients, total output | 762 tok/s | 745 tok/s | 528 tok/s | 355 tok/s |

The six prompts are code, an explanation, a list, a translation, a story and
JSON (200 tokens out at most, greedy). Code and JSON gain most (1.8–1.9× on
the laptop GPU), the story least (1.15×); 58% of drafted tokens were
accepted. llama-server ran `-md` with `--spec-type draft-simple` and its
default `--spec-draft-n-max 3` (8 was slower still); in this build its
speculative mode lowered throughput.

## Prompt lookup

`--prompt-lookup` drafts without a second model. Before each decode step,
the engine looks for the sequence's last 4 tokens (then 3, then 2) earlier in
its prompt or output, and proposes the tokens that followed the most recent
match. The main model verifies them in the same pass as a draft model's.
- **Free when it misses.** The search is host-side (microseconds) and only proposes when it finds a match; free-form text rarely matches, and then the step is a plain decode step.
- **Exact.** A copied token is a draft drawn with probability 1, so the speculative sampling rule (accept with probability p, else draw from p without that token) keeps output distributed exactly as without lookup. Greedy output is identical on Qwen3.
- **Shares the draft machinery.** Proposals follow the same adaptive length (`--draft-tokens`), pause after repeated misses, stay within 8-token steps, and switch off with 5 or more requests decoding. JSON-constrained requests don't use it.
- **With `--draft`,** lookup goes first and the draft model covers the steps where lookup finds nothing.

| Qwen3-8B Q4_K_M, RTX 6000 Ada, greedy | rllm | `--prompt-lookup` | `--draft` 0.6B | both |
|---|---:|---:|---:|---:|
| Rename a variable in 20 lines of code | 136 tok/s | 360 tok/s (2.64×) | 271 tok/s (1.98×) | 363 tok/s (2.66×) |
| Short story | 142 tok/s | 142 tok/s (1.00×) | 169 tok/s (1.19×) | 170 tok/s (1.19×) |
| Copy a paragraph and add a sentence | 136 tok/s | 280 tok/s (2.06×) | 238 tok/s (1.75×) | 285 tok/s (2.10×) |

All twelve outputs matched the plain run's word for word. On the server
benchmark (8 clients) throughput was unchanged (772 vs 782 tok/s, within
noise); with one client it rose 1.35×, since that benchmark's outputs repeat
themselves. Metrics: `rllm_lookup_tokens_total` and
`rllm_lookup_tokens_accepted_total`.

## Host tier

`--host-cache-gb N` gives the prefix cache a second tier in pinned system
memory. When device blocks run out, the least recently used cached prefix
is copied to the host instead of dropped; a later prompt that matches it
further than anything on the GPU copies it back instead of recomputing it.
- **How.** Host memory is pinned and mapped into the GPU's address space, and one kernel per layer and part (K, V, q8 scales) moves a list of blocks over PCIe. The copies run on the same stream as the forward passes, so there is no host synchronization: a freed device block can be reused by the very next step.
- **Exact.** Blocks come back bit for bit (a GPU test overwrites every device block in between and compares logits, f16 and q8, Qwen3 and Gemma 4).
- **Bookkeeping.** The host tier evicts least recently used entries in turn (at most 2048). A prefix already covered by a host entry isn't copied twice; a restored entry leaves the host when the device has all of it. Only the matched blocks are copied back.
- **Sizing.** One block of 16 tokens takes `16 × layers × kv_heads × head_dim × 4` bytes in f16 (2.4 MB for Qwen3-8B; half with `--kv-type q8`). 16 GiB holds about 116k tokens of Qwen3-8B prefixes. The memory is allocated and locked at startup.

| Qwen3-8B Q4_K_M, RTX 6000 Ada, 4096-token KV pool | no host tier | `--host-cache-gb 4` |
|---|---:|---:|
| Repeat of a 2546-token prompt after two others displaced it: cached tokens | 0 | 2544 |
| Time to first token | 381 ms | 39 ms |

The 39 ms includes copying 375 MB back (159 blocks). Metrics:
`rllm_kv_host_blocks_total`, `rllm_kv_host_blocks_free`,
`rllm_prefix_cache_host_entries`, `rllm_kv_offloaded_blocks_total`,
`rllm_kv_restored_blocks_total`.

## KV cache type

`--kv-type q8` stores keys and values as 8-bit integers with one f16 scale
per 32 values (GGUF's q8_0 layout): about half the memory of f16, so twice
the context or concurrent requests in the same `--kv-tokens` budget.
- **Keys are rotated.** Each head's key vector is multiplied by an orthonormal Walsh–Hadamard matrix before quantizing, and queries get the same rotation, so q·k is unchanged. The rotation spreads a few large channels across the whole group of 32. Without it, Qwen3-0.6B's perplexity rose 1.4%. llama.cpp rotates the same way. Values aren't rotated (quantizing them costs about 0.02%).
- **Quality.** Perplexity changes are within noise on the larger models (wikitext-2, 20 chunks; Qwen3-0.6B on the laptop GPU, the others on the Ada):

| Perplexity, f16 → q8 | ctx 512 | ctx 2048 |
|---|---:|---:|
| Qwen3-0.6B Q8_0 | 23.676 → 23.721 (+0.19%) | 17.271 → 17.277 (+0.03%) |
| Qwen3-8B Q4_K_M | 11.003 → 10.997 | 8.344 → 8.340 |
| Nemotron E4B Q8_K_XL | 150.76 → 150.48 | 96.82 → 96.81 |

  llama.cpp's q8_0 cache on the same Qwen3-0.6B chunks: +0.21%.
- **Speed.** Decode reads half the bytes, which pays off at long contexts; prefill expands the blocks it reads to f16 first (the tensor-core kernels need f16), which costs a few percent:

| Qwen3-8B Q4_K_M, RTX 6000 Ada | f16 | q8 | llama.cpp f16 | llama.cpp q8_0 |
|---|---:|---:|---:|---:|
| Decode, empty context | 143.1 tok/s | 141.0 tok/s | 153.3 tok/s | 146.9 tok/s |
| Decode at 8k context | 110.4 tok/s | 121.8 tok/s | 125.5 tok/s | 124.9 tok/s |
| Decode at 32k context | 71.4 tok/s | 88.0 tok/s | 82.3 tok/s | 79.5 tok/s |
| Prefill (512) at 32k context | 2593 tok/s | 2477 tok/s | 3261 tok/s | 3180 tok/s |

  On Nemotron (mostly sliding-window layers) q8 decodes 7% faster at 32k context.
- **How.** One f16 scale per 32 values; decode-sized steps (up to 8 tokens) read the int8 cache directly in the decode attention kernel; larger prefill steps expand the blocks they attend to into an f16 staging buffer and run the f16 tensor-core kernels on a remapped block table. `rllm run`, `bench` and `perplexity` take `--kv-type` too.

## Metrics

`GET /metrics` serves Prometheus text, every series labelled with the model:
- **Engine gauges:** `rllm_requests_running`, `rllm_requests_waiting`, `rllm_kv_blocks_total`, `rllm_kv_blocks_free`, `rllm_prefix_cache_entries`, `rllm_prefix_cache_blocks`.
- **Engine counters:** `rllm_requests_finished_total`, `rllm_prompt_tokens_total`, `rllm_prompt_tokens_cached_total`, `rllm_generation_tokens_total`, `rllm_embedding_tokens_total`, `rllm_preemptions_total`, `rllm_engine_steps_total`, `rllm_engine_step_tokens_total`, `rllm_engine_busy_seconds_total`.
- **Speculative decoding:** `rllm_draft_tokens_total` (proposed) and `rllm_draft_tokens_accepted_total`; for prompt lookup, `rllm_lookup_tokens_total` and `rllm_lookup_tokens_accepted_total`.
- **Host tier:** `rllm_kv_host_blocks_total`, `rllm_kv_host_blocks_free`, `rllm_prefix_cache_host_entries`, `rllm_kv_offloaded_blocks_total`, `rllm_kv_restored_blocks_total`.
- **HTTP:** `rllm_http_requests_total` and `rllm_http_request_seconds_total`, by method, route pattern (`/v1/responses/{id}`) and status.

## Performance

Measured against llama.cpp's `llama-server` (same build as the
`compare.md` runs, `-ngl 99 -fa on -np 8 -c 32768`) on the RTX 6000 Ada,
with rllm at `--parallel 8 --ctx 4096`:
- **Method:** streaming chat completions, greedy, with end-of-turn tokens biased away so every request generates its full length. Results in `compare.md`.
- **Scenarios:** one request at a time (a 540–670-token prompt, 256 tokens out), then 1, 4 and 8 concurrent clients (2 requests each, about 120-token prompts, 192 tokens out), and a 4-turn chat on a 2–2.6k-token system prompt.

| | Qwen3-0.6B Q8_0, rllm | Qwen3-0.6B Q8_0, llama-server | Nemotron E4B Q8_0, rllm | Nemotron E4B Q8_0, llama-server |
|---|---:|---:|---:|---:|
| Single request: time to first token | 13 ms | 103 ms | 49 ms | 130 ms |
| Single request: decode | 547 tok/s | 491 tok/s | 119.8 tok/s | 118.5 tok/s |
| 4 clients: total output | 1874 tok/s | 545 tok/s | 450 tok/s | 233 tok/s |
| 8 clients: total output | 2760 tok/s | 1176 tok/s | 744 tok/s | 485 tok/s |
| 8 clients: mean time to first token | 14 ms | 262 ms | 58 ms | 305 ms |
| Chat turn 1 / later turns, time to first token | 50 / 12 ms | 338 / 16 ms | 170 / 20 ms | 385 / 102 ms |

On hybrid Intel CPUs (P- and E-cores) the engine thread pins itself to the
performance cores (`/sys/devices/cpu_core/cpus`). Each step's host work
(launching kernels, sampling a 262k-entry vocabulary) ran several times
slower on an E-core. Pinning took a Nemotron decode step from 7.92 to
7.64 ms on the Ada box's i9-14900K. `RLLM_NO_PIN=1` turns it off.
Sampling is vectorized (argmax, a fast `exp` with an AVX2 path, top-k
without a full sort) and works in place on the logits buffer.

## Testing

The server was exercised with the official `openai` Python SDK (v3.41) on
Qwen3-0.6B (Hermes format, laptop GPU) and Nemotron 3.5 Bolt E4B (Gemma 4
format, RTX 6000 Ada). All 47 checks passed on both models. They cover:
- models, including 404s
- chat:
  - basic and streaming with usage
  - `n`, logprobs, seed reproducibility, stop sequences, length limits
  - tool calls: non-streaming, streaming, named and required
  - a tool-result round trip
  - `json_object` and `json_schema`
  - the stored-completion endpoints
- legacy completions: `echo`, logprobs, batch prompts, streaming, `best_of`
- Responses:
  - basic and streaming (event order and sequence numbers)
  - `previous_response_id`, retrieve, input items
  - a function-call round trip, JSON format, input token counts
  - background and cancel, delete, the `incomplete` status
- embeddings (`dimensions`, base64)
- error shapes (`param`, out-of-range values, unsupported endpoints)
- API-key and strict-model errors

On Nemotron, reasoning output was also checked: `reasoning_content` in chat,
and `reasoning` items with `reasoning_text` events in Responses. So were
concurrent requests, and freeing the engine when a client disconnects
mid-stream (the next request completed in 0.06 s).

A second suite (17 checks, both models) covers the newer features:
- **Prefix cache:** a hit on the next chat turn (640 of 664 prompt tokens cached), identical output from the cached prefix, and `cached_tokens` in usage.
- **Batching:** concurrent requests, batched logprobs close to single-request ones, and a speed-up over sequential requests (2.8× on Qwen at 4 clients).
- **`n` batched.**
- **Schema enforcement:** keys, integer types, enums and `maxItems` under `json_schema`; `required` and named tool calls matching their tools' schemas.
- **Ops endpoints:** `/tokenize`, `/detokenize`, `/apply-template`, `/metrics`.

Both suites also passed (47/47, 17/17) on Qwen3-8B with a Qwen3-0.6B draft
model (laptop GPU). With the draft, greedy output on six mixed prompts was
identical to the run without one, sampling at temperature 0.8 worked
(1.37× faster), and the preemption test below passed again (2 preemptions,
outputs identical to the requests run alone).

Both suites passed again (47/47, 17/17) with `--prompt-lookup --host-cache-gb 2`
on Qwen3-0.6B (RTX 6000 Ada), with the default pool and with a 4096-token
pool. The preemption test below, on a 1024-token pool with a host tier,
moved 148 blocks to the host and copied 294 back over 3 preemptions; every
output matched the request run alone.

Separately:
- **Preemption:** a 1024-token pool with 4 concurrent 380-token completions preempted twice. Every output was identical to the same request run alone.
- **Persistence:** stored completions survived a restart with `--store-file`.
- **Gemma tool calls:** on Nemotron, 5/5 forced calls matched their schemas, nested objects and enums included. Before constraining, it had answered `"unit": "celsius"` for an `["c", "f"]` enum.

Unit tests cover:
- **The JSON validator:** schemas, `$ref` recursion, `anyOf`, the Gemma dialect, tool-call alternatives.
- **The prefix cache:** block sharing, eviction, superseded entries.
- **The sampler:** argmax and top-k against naive versions, `exp` accuracy, sampled frequencies against softmax for temperature, top-k and top-p, and speculative verification reproducing the target distribution (greedy, temperature, top-k, top-p).
- **The KV cache:** truncation after rejected drafts; the host tier (offload on eviction, full and partial restores, LRU eviction on the host, device hits preferred).
- **Prompt lookup:** latest match wins, shorter tails as fallback, proposals stop at the end of the sequence.

GPU tests (`crates/rllm-model/tests/gpu.rs`) need a model file:
`RLLM_TEST_MODEL=models/Qwen3-0.6B-Q8_0.gguf cargo test --release -p rllm-model --test gpu`
(add `RLLM_MMQ=1` to send a small model's prefill through the int8 kernels).
They check that the same tokens give bit-identical logits on every run,
that a verification step matches one-token steps, that a q8 cache tracks
f16 on real text (97–99% same top token, mean KL ≤ 4e-3), and that blocks
moved to the host tier and back give bit-identical logits. They passed on
Qwen3-0.6B, Qwen3-8B and Nemotron on both GPUs. The CUDA parity tests
(`crates/rllm-cuda/tests/parity.rs`) run every attention path on a q8 cache
against the CPU backend, which emulates q8's rounding.
- **The output parser:** reasoning and tool-call markers split at every possible point, Gemma argument syntax, stop sequences spanning chunks.
