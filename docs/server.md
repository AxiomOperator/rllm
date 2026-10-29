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
| `--api-key` | `$RLLM_API_KEY`, else none | Require `Authorization: Bearer <key>`. |
| `--model-name` | file name without `.gguf` | Model id reported by `/v1/models` and in responses. |
| `--strict-model` | off | Answer requests naming another model with 404 `model_not_found`, as OpenAI does. Off by default: any model name is served by the loaded model, so clients that hard-code e.g. `gpt-4o` still work. |
| `--prefill-chunk` | 512 | Prompt tokens per forward pass (long prompts are split, so running requests keep decoding in between). |
| `--store-file` | none | Save stored chat completions and responses to this JSON file (every 2 s when changed, and at shutdown) and load it at start. |
| `-q`, `--quiet` | off | Don't log a line per finished generation. |

Environment: `RLLM_API_KEY` (as `--api-key`), `RLLM_ENGINE_TIMING=1` (print
where engine step time goes every 256 steps), `RLLM_NO_PIN=1` (see
[Performance](#performance)).

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
  - Batched decode steps of up to 8 requests replay CUDA graphs. Steps that mix in prompt chunks run directly.
  - HTTP handling, tokenization, chat templating, stop sequences and output parsing run on the async runtime.
- **Prefix cache.** When a request finishes, its KV blocks stay cached. A later request whose prompt starts with the same tokens reuses those blocks instead of recomputing them. That covers the next turn of a chat, a repeated system prompt, `n` choices or `previous_response_id`.
  - Only whole 16-token blocks are shared, so the last few tokens before the difference are recomputed.
  - Least recently used entries are evicted when blocks run out.
- **When the KV cache is full.** Cached prefixes are evicted first. If that isn't enough, the newest running request is paused, and its blocks go into the prefix cache. It resumes later by recomputing (often reusing) its tokens. Clients only see a short stall.
- **Sampling.** Each request's logits are processed on the host: logprobs, `logit_bias`, penalties, the JSON constraint, then sampling. With several requests in a step, this runs on one thread per request.
- **Numerics.** A request batched with others runs through the multi-row int8 kernels (the ones prefill uses), and a lone request through the single-row decode kernels.
  - Logprobs agree to about 0.01–0.02 on average, but greedy text can drift apart after a few dozen tokens.
  - Results are reproducible for the same mix of concurrent requests.
- **Cancellation.** A client that disconnects (a closed stream, a dropped connection) cancels its generation at the next step and frees its slot. Stop sequences cancel the rest of the generation too.
- **`n` and `best_of`.** The choices run concurrently, so they share batches. They use seeds `seed`, `seed + 1`, …. Streaming with `n > 1` still sends the choices one after another.
- **Storage.** Stored chat completions and responses live in memory.
  - With `--store-file` they are written to that JSON file (atomically, at most every 2 s, and at shutdown) and loaded at start.
  - A background response still running at shutdown loads as `failed`.
- **Logging.** One line per finished generation: prompt tokens (cached), generated tokens, time to first token, decode rate, total time and finish reason.

## Metrics

`GET /metrics` serves Prometheus text, every series labelled with the model:
- **Engine gauges:** `rllm_requests_running`, `rllm_requests_waiting`, `rllm_kv_blocks_total`, `rllm_kv_blocks_free`, `rllm_prefix_cache_entries`, `rllm_prefix_cache_blocks`.
- **Engine counters:** `rllm_requests_finished_total`, `rllm_prompt_tokens_total`, `rllm_prompt_tokens_cached_total`, `rllm_generation_tokens_total`, `rllm_embedding_tokens_total`, `rllm_preemptions_total`, `rllm_engine_steps_total`, `rllm_engine_step_tokens_total`, `rllm_engine_busy_seconds_total`.
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

Separately:
- **Preemption:** a 1024-token pool with 4 concurrent 380-token completions preempted twice. Every output was identical to the same request run alone.
- **Persistence:** stored completions survived a restart with `--store-file`.
- **Gemma tool calls:** on Nemotron, 5/5 forced calls matched their schemas, nested objects and enums included. Before constraining, it had answered `"unit": "celsius"` for an `["c", "f"]` enum.

Unit tests cover:
- **The JSON validator:** schemas, `$ref` recursion, `anyOf`, the Gemma dialect, tool-call alternatives.
- **The prefix cache:** block sharing, eviction, superseded entries.
- **The sampler:** argmax and top-k against naive versions, `exp` accuracy, sampled frequencies against softmax for temperature, top-k and top-p.
- **The output parser:** reasoning and tool-call markers split at every possible point, Gemma argument syntax, stop sequences spanning chunks.
