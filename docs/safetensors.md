# Safetensors support: design and plan

Status (2026-11-17): **Qwen3, Qwen3.5/3.6 (dense, MoE, MTP), Clef, Llama 2/3, Mistral (7B with SentencePiece; Nemo and Small 3 with Tekken), TinyLlama and Gemma 4 (dense and the 26B-A4B MoE) checkpoints load directly, optionally quantized as they load to Q8_0 or llama.cpp's Q4_K_M mix** (`rllm serve DIR --quantize q8_0|q4_k_m`, or `RLLM_QUANTIZE` for the other commands). `rllm serve`, `bench`, `perplexity` and `tune` take a checkpoint directory (or a `.safetensors` file next to its `config.json`) wherever they take a `.gguf`. This document is the plan for loading Hugging Face checkpoints (`config.json`, `*.safetensors`, `tokenizer.json`, `tokenizer_config.json`) directly, without converting them to GGUF first. What is still to come is listed under Plan.

Each family is checked against llama.cpp's conversion of the same checkpoint with `crates/rllm-safetensors/tests/against_gguf.rs`. That test compares every tensor (dims, then bytes, or values within the reference's quantization) and every metadata key rllm reads. `examples/st_diff.rs` shows where they differ.

**Qwen/Qwen3-0.6B (BF16)**, against llama.cpp's BF16 GGUF:
- every tensor identical and every key equal;
- the chat template is the checkpoint's own (the published GGUF ships a rewritten one), and rllm renders it;
- served on the CPU: the same 38-token prompt, text and per-token logprobs as the GGUF.

**Qwen/Qwen3.5-0.8B (BF16, with MTP)**, against a Q8_0 GGUF of it:
- every matrix within Q8 rounding;
- every F32 tensor identical, except `ssm_a`'s −exp(A_log), within 1 ulp;
- every key equal, after taking added tokens from `tokenizer_config.json`'s `added_tokens_decoder` too (the audio tokens are only there);
- served on the CPU: coherent output, MTP accepting 21 of 37 drafts, and greedy output identical with and without MTP.

**unsloth/Llama-3.2-1B-Instruct (BF16, Llama 3.1+ RoPE scaling)**, against the Q4_K_M GGUF of it:
- every matrix within Q4/Q6 rounding, Q/K permuted as llama.cpp does them (rllm's llama loader undoes it);
- the generated `rope_freqs` within 2 ulp;
- every key equal (`llama-bpe`, BOS added from the template post-processor);
- served on the CPU: the same 61-token prompt and the same text as the GGUF.

**Quantizing at load** (ports of ggml's reference `quantize_row_q8_0`, `quantize_row_q4_K` and `quantize_row_q6_K`, which `llama-quantize` uses without an importance matrix), on Qwen3-0.6B against `llama-quantize` run on the BF16 GGUF:
- Q8_0: all 310 tensors byte-identical;
- Q4_K_M: every tensor gets llama.cpp's type (Q6_K for the output and for `attn_v` / `ffn_down` on its `use_more_bits` layers, Q4_K otherwise, Q8_0 for rows that aren't whole 256-value blocks), and 307 of 310 are byte-identical; the other three differ in 4–18 bytes of more than half a million, likely the C build fusing multiply-adds in the scale search;
- wikitext-2 perplexity (8 chunks of 512): Q4_K_M 23.5408 against llama.cpp's 23.5402; Q8_0 21.72 against BF16 21.73;
- the published Q8_0 and Q4_K_M GGUFs differ more (scales about ±0.5% apart): they weren't made from these BF16 weights by the reference quantizer;
- the decision head of Clef stays at Q8_0 under `q4_k_m` (it runs once per request, on the CPU).

**Qwen/Qwen3.6-35B-A3B (BF16, 256 experts, 32 value heads over 16 key heads, MTP)**, against its Q4_K_M GGUF, on the Ada box:
- all 1,045 tensors within Q4/Q6 rounding, including the value-head tiling and the MTP layer, and every key equal;
- the fused `experts.gate_up_proj` is kept as `ffn_gate_up_exps` (llama.cpp's `--fuse-gate-up-exps` layout, no copy); the test compares each half with the reference's split `ffn_gate_exps` / `ffn_up_exps`;
- served on CUDA with `--quantize q8_0` (37.6 GB on the GPU, ready in 55 s with quantization on all cores): the same 38-token prompt and the same answer as the GGUF;
- with `--quantize q4_k_m`: 22.5 GB on the GPU (the published Q4_K_M GGUF: 23.1 GB), ready in 187 s (Q4_K's scale search is the slow part), the same answer, greedy decode 245 tok/s (the GGUF: 218);
- its chat template uses transformers' `{% generation %}` blocks, which rllm now renders as plain blocks.

**google/gemma-4-E2B-it (BF16, revision 4adf157)**, against ggml-org's BF16 GGUF made from that revision:
- every tensor identical and every key equal, including the generated `rope_freqs` (1 for the 64 rotated pairs of a 512-wide global head, 1e30 for the rest), the per-layer FFN widths of its double-wide MLP and the `gemma4` vocabulary (CONTROL for transformers' special tokens, which include every named `*_token` in `tokenizer_config.json`; added tokens replacing base entries of the same id);
- unsloth's copy has the same weights but an older tokenizer, so its GGUF would differ from ggml-org's in merges and token types;
- E2B didn't load in rllm from either format before: the Gemma 4 loader now takes `feed_forward_length` per layer (its KV-sharing layers are twice as wide);
- served: on the CPU, the same 35-token prompt, text and logprobs as the GGUF; on CUDA (Ada), the same first logprobs within 0.005, and with `--quantize q4_k_m` 245 tok/s greedy (BF16: 135).

**SentencePiece vocabularies** (`tokenizer.model`, read with a small protobuf reader; llama.cpp's `_set_vocab_sentencepiece`), against GGUFs made by llama.cpp's `convert_hf_to_gguf.py` from the same checkpoints:
- TinyLlama-1.1B-Chat-v1.0 (Llama 2's tokenizer) and mistralai/Mistral-7B-Instruct-v0.3 (32,768 pieces): every tensor identical and every key equal;
- served on the CPU: the same prompt, text and logprobs as the GGUF for both.

**google/gemma-4-26B-A4B-it (BF16, 128 experts, top 8, 8 KV heads on sliding-window layers and 2 on global ones, K equal to V on global layers)**, against ggml-org's Q8_0 GGUF:
- every tensor within Q8 rounding and every key equal: the router (`router.proj`), its input scale (`ffn_gate_inp.scale`), the per-expert output scale (`ffn_down_exps.scale`), the fused `gate_up_proj` kept as `ffn_gate_up_exps`, and the three extra norms;
- rllm runs it from either form (new: experts with GELU and a per-expert scale, KV heads per kind of layer). On CUDA, quantized to Q8_0 at load, it gives the same answer and 39-token prompt as the GGUF.
- Against llama.cpp (`tools/llama-ref`, 29-token chat prompt and 16 generated tokens), rllm on CUDA is closer to llama.cpp's default numerics (KL mean 0.05, max 0.5) than llama.cpp's own default and exact-attention runs are to each other (KL mean 0.16, max 4.9), and every generated token agrees. The model is unusually sensitive to rounding: the same text prefilled in 512- and in 16-token chunks (two correct kernel paths) moves its predictions by mean KL 0.20 (Qwen3.6-35B-A3B and Nemotron E4B: 0.003-0.004). The GPU model tests now measure that noise floor and hold such a model to it; every new op also matches the CPU at the 26B's exact shapes (attention 16 heads over 8 and 2 KV heads, experts 128 × 704 on 2816).

**Mistral Tekken** (Nemo, Small 3, Ministral): byte-level BPE whose pre-tokenizer regex is detected from `tokenizer.json` (`tekken`, or `llama-bpe` for Small 3, whose file uses Llama 3's split). rllm's tokenizer gained the Tekken split:
- `tests/bpe_tokenizer.rs` wraps a tokenizer.json in a one-layer checkpoint and requires the same ids as Hugging Face's `tokenizers` on ten texts (case changes, digits, code, URLs, CJK, emoji, Arabic, whitespace runs): identical for Nemo, Small 3, Ministral 8B and Llama 3;
- mistralai/Mistral-Nemo-Instruct-2407 against the GGUF llama.cpp's converter makes from it: every tensor identical and every key equal; served on CUDA, the same 22-token prompt, text and logprobs.

**Cloudflare/clef (BF16, a 64-layer Qwen3.5 plus the decision head in `joint_head.safetensors`)**, against llama.cpp's `Clef-Q8_0.gguf`, on the Ada box:
- every tensor within Q8 rounding and every key equal, including the head's 125M parameters and the generated `systemone` template (a port of llama.cpp's `ClefModel`): routing blocks first, `in_proj` split into q/k/v, the three learned scalars stored as `decision.scales` (exp, capped at 100, and a sigmoid);
- served on CUDA with `--quantize q8_0` (ready in 52 s): the same 319-token prompt and the same decisions on the docs' example as the GGUF, probabilities within 0.02.

## Approach: a synthetic GGUF

Every loader (`qwen3.rs`, `qwen35.rs`, `gemma4.rs`, `clef.rs`, the tokenizer, `ModelProfile`, `rllm tune`, the agent) reads a `Gguf`: metadata keys plus tensors by GGUF name. The converter will build exactly the `Gguf` that llama.cpp's `convert_hf_to_gguf.py` would have written, in memory:

```
Checkpoint::open(dir)                       rllm-safetensors (done)
  → convert::to_gguf(&checkpoint)            new: rllm-safetensors::convert
       metadata  ← config.json (+ transformers defaults), tokenizer.json, tokenizer_config.json
       tensors   ← HF names → GGUF names; untransformed: views into the shard maps
                    transformed (stacked, reordered, +1, -exp): owned buffers
  → Gguf::from_parts(metadata, tensors, buffers)   rllm-gguf (done)
  → AnyModel::load, Tokenizer::from_gguf, ... unchanged
```

**Why not native safetensors loaders?** Every model file would need a second path, and the tokenizer, `ModelProfile`, `tune` and the agent would all need a second metadata source. The synthetic container keeps one path, and that path is tested against real GGUFs. The cost is that converted tensors are copied. Those are only the tensors with real transforms (see below).

**Zero copy where possible.** F32, F16 and BF16 tensors whose bytes are already what GGUF stores are described as views into the shard maps (`TensorInfo { buffer: shard, offset, .. }`). GGUF dims are the safetensors shape reversed. So a pure BF16 Llama or Qwen3 checkpoint loads with no extra copy, exactly like a GGUF.

**Detection.** `rllm serve PATH` takes a GGUF file, a `.safetensors` file, or a directory. A directory with `config.json` and safetensors weights goes through the converter. Everything else is unchanged.

## Per-family work (from the spec)

| Family | Copying tensors (transforms) | Other work | Effort |
|---|---|---|---|
| **Qwen3** dense | none | metadata from config; GPT-2 BPE tokenizer keys; `tokenizer.ggml.pre` = `qwen2` | 2–3 days |
| **Llama 2/3, Mistral 7B** (**done**) | none (see below); `rope_freqs` generated from `rope_scaling` (llama3) | **Q/K permutation:** the converter permutes Q/K and rllm undoes it on load for `llama`. On HF weights both steps must be skipped: a metadata flag (`rllm.qk_unpermuted`) that the loader honors. Mistral v0.1–v0.3 need SentencePiece scores from `tokenizer.model` (protobuf); Tekken (Nemo and later) is byte-level BPE with its own split (done) | 3–5 days |
| **Qwen3.5 / 3.6** dense, MoE, MTP (**done**) | value heads reordered from grouped to tiled in 7 tensors (`in_proj_qkv` value rows, `in_proj_z`, `in_proj_a/b`, `A_log`, `dt_bias`, conv1d value channels, `out_proj` columns); every RMSNorm +1 except `linear_attn.norm`; `A_log` → `ssm_a = -exp(A_log)`; conv1d squeezed to [C, K] as F32; experts stacked into `*_exps` (fused `gate_up_proj` maps directly) | MTP layer renamed to `blk.{n_layers}` (`fc` → `eh_proj`, `pre_fc_norm_*` → `enorm`/`hnorm`, `norm` → `shared_head_norm`), `block_count` includes it; `ffn_gate_inp_shexp` stored as [d] (real GGUFs load with rllm's strict [d] check) | 1–2 weeks |
| **Gemma 4** (dense and MoE **done**) | `rope_freqs` = [1.0 × n_rot, 1e30 …] generated | `key_length` is the global head size and `key_length_swa` the local one; no norm +1 (unlike Gemma 1–3); token types from `all_special_ids` with 7 tool/channel tokens forced to USER_DEFINED; configs with array-valued `feed_forward_length` / `head_count_kv` are rejected (as their GGUFs are) | 1 week |
| **Clef** (**done**) | as Qwen3.5 (it is a Qwen3.5 core) plus the decision head's tensors | the systemone template from `tokenizer_config.json` | 2–3 days after Qwen3.5 |

**Config defaults.** The converter reads `config.json` through transformers' `AutoConfig`, which fills defaults (`head_dim = hidden / heads`, `rope_theta`, `rms_norm_eps`, …). The Rust converter has to apply the same per-class defaults, listed in the spec's section 0.1 and each family's "Metadata" table.

**Tokenizer.** rllm's tokenizer is built from `tokenizer.ggml.*` keys: tokens, types, merges, and a pre-tokenizer chosen by name. The converter synthesizes them from `tokenizer.json` and `tokenizer_config.json`:
- vocabulary in id order, padded with `[PAD{id}]` up to `vocab_size`;
- merges;
- token types: added tokens marked `special` → CONTROL, other added tokens → USER_DEFINED;
- `bos`/`eos`/`pad` ids;
- `add_bos`;
- the chat template.

llama.cpp picks `tokenizer.ggml.pre` by hashing the tokenizer's output on a fixed text. Two options for the converter: reproduce that hash with the `tokenizers` crate (already a dependency), or map the pre-tokenizer regex in `tokenizer.json` to rllm's names (`qwen2`, `llama-bpe`, …). The regex mapping is simpler, and it fails loudly on unknown tokenizers.

## Beyond BF16: quantizing at load

A BF16 checkpoint takes about 2 bytes per parameter: 70 GB for Qwen3.6-35B-A3B, against 21 GB for its Q4_K_M GGUF. To make safetensors useful on the 48 GB Ada and to keep decode speed (decode is bandwidth-bound), the loader quantizes as it loads (**done**, `--quantize q8_0|q4_k_m`):
- **CPU quantizers** for Q8_0, Q4_K and Q6_K in `rllm_core::quant`, ports of ggml's reference functions, run on all cores;
- **per-tensor policy** as llama.cpp's `Q4_K_M` (see the results above);
- the quantized tensors are owned buffers in the container. Upload is unchanged.

**Pre-quantized formats.** These are later work, only if demand appears: FP8 block-scaled checkpoints (Qwen3 FP8, dequantized to BF16 or requantized at load), and GPTQ/AWQ int4 (needs a W4A16 kernel; see the performance review).

## Plan

1. **Done:** **Converter skeleton plus Qwen3 BF16**, end to end:
   - metadata, tensor views and tokenizer keys;
   - `rllm serve DIR`;
   - a parity test: greedy outputs and logits match the GGUF made by llama.cpp from the same checkpoint (BF16 → BF16 GGUF: identical bytes, so identical logits);
   - perplexity on wikitext-2 against the GGUF.
2. **Done:** **Llama 2/3 and Mistral 7B**, with generated `rope_freqs` and SentencePiece vocabularies. The converter permutes Q/K as llama.cpp does, so no loader flag is needed. **Done:** Tekken tokenizers (Mistral Nemo, Small 3, Ministral).
3. **Done:** **Q8_0 and Q4_K_M at load** (`--quantize q8_0|q4_k_m`), checked against `llama-quantize` byte for byte and by perplexity.
4. **Done:** **Qwen3.5 / 3.6**: transforms, expert stacking, MTP, checked on the real Qwen3.6-35B-A3B.
5. **Done:** **Clef and Gemma 4**, dense and MoE (26B-A4B).
6. **Integration:**
   - `ModelProfile`, `rllm tune` and the agent's model list read directories;
   - the agent fetches whole HF repos: the safetensors files plus the JSON files;
   - docs.

**Test data.** Small public checkpoints: Qwen3-0.6B, Llama-3.2-1B, Qwen3.5-0.8B (hybrid and MTP), and Gemma-4 E2B. llama.cpp's `convert_hf_to_gguf.py` (in `third_party`) makes the reference GGUFs.
