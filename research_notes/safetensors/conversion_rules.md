# HF safetensors -> "synthetic GGUF": llama.cpp conversion rules

Source: `third_party/llama.cpp` at commit `2e0d86b` (Hexagon IM2COL, #31050).
All `file:line` references are relative to `third_party/llama.cpp/` unless they start with `crates/`.
Converter entry point: `convert_hf_to_gguf.py:179-312` → `ModelBase.write()` (`conversion/base.py:1400`) =
`prepare_tensors()` (tensors) then `prepare_metadata()` (`base.py:1354`, which calls `set_gguf_parameters()` and,
via `TextModel.prepare_metadata` `base.py:1578-1604`, `set_vocab()`).

Purpose: describe, precisely enough to implement in Rust, what the converter writes for the four architecture
families rllm loads (`crates/rllm-model/src/{qwen3,qwen35,gemma4}.rs`), so a loader can build the same metadata
keys and tensors from `config.json`, `*.safetensors`, `tokenizer.json` and `tokenizer_config.json`.

---------------------------------------------------------------------------------------------------------------

## 0. Generic pipeline (applies to every architecture)

### 0.1 Reading config.json (`base.py:1438-1474`, `base.py:1514-1546`)

1. `hparams = AutoConfig.from_pretrained(dir, trust_remote_code=False).to_dict()`; only if that throws does it
   fall back to the raw `config.json` (`base.py:1450-1458`). **Non-obvious:** the converter therefore sees the
   *transformers-normalized* dict: class defaults filled in (e.g. `LlamaConfig.head_dim = hidden/heads`,
   `rope_theta` defaults), and with transformers v5, `rope_scaling` + `rope_theta` folded into
   `rope_parameters`. A Rust loader reading raw JSON must reproduce the defaults it relies on (listed per arch).
2. `architectures[0]` selects the class; if `text_config.architectures` exists it wins (`base.py:3134-3162`).
3. `text_config` is merged over the root: `hparams = {**hparams, **hparams["text_config"]}` (`base.py:1521-1523`).
   (Root keys survive unless the text config overrides them.)
4. `block_count = first of [n_layers, num_hidden_layers, n_layer, num_layers]` (`base.py:1525`).
5. `rope_parameters = hparams.get("rope_parameters", hparams.get("rope_scaling")) or {}` (`base.py:1528`). If it has
   neither a `full_attention` nor a `sliding_attention` sub-dict, top-level values are mirrored into it
   (`base.py:1536-1546`): `rope_theta` (from `global_rope_theta|rope_global_theta|rope_theta_global|rope_theta|rotary_emb_base`),
   `sliding_attention.rope_theta` (from `local_rope_theta|…|rope_local_base_freq`), `rope_type` (from `type`),
   `partial_rotary_factor` (from `partial_rotary_factor|rope_pct|rope_percent`), `original_max_position_embeddings`.

### 0.2 Base metadata — `TextModel.set_gguf_parameters` (`base.py:1606-1720`)

Every key is `{arch}.…` (`arch` = `general.architecture`). "first of" = first key present in hparams.

| GGUF key | Source | Line |
|---|---|---|
| `block_count` (u32) | `block_count` above (Qwen3.5: + MTP layers, see §2) | 1607 |
| `context_length` | first of `max_position_embeddings, n_ctx, n_positions, max_length, max_sequence_length, model_max_length` | 1609 |
| `embedding_length` | first of `hidden_size, n_embd, dim` | 1613 |
| `feed_forward_length` | first of `prefix_dense_intermediate_size, dense_intermediate_size, intermediate_size, n_inner, hidden_dim` | 1617 |
| `attention.head_count` | first of `num_attention_heads, n_head, n_heads` | 1621 |
| `attention.head_count_kv` | first of `num_key_value_heads, n_kv_heads` | 1625 |
| `attention.causal` = false | only if `is_causal == False` or `use_bidirectional_attention == True` | 1629 |
| `rope.scaling.*` | from `rope_parameters.get("full_attention", rope_parameters)`: `rope_type=="linear"`+`factor` → `rope.scaling.type="linear"`, `rope.scaling.factor`; `"yarn"` → type `yarn`, `factor`, `rope.scaling.original_context_length` (= `original_max_position_embeddings`), optional `rope.scaling.yarn_*` (`truncate`, `extrapolation_factor`, `attention_factor|attn_factor`, `beta_fast`, `beta_slow`); `"su"/"longrope"` → type `longrope`; `"llama3"`, `"dynamic"` → nothing (llama3 handled by a `rope_freqs` tensor, §3); anything else (`"default"`, `"proportional"`) → warning only, nothing written | 1634-1669 |
| `rope.dimension_sections` (u32[4]) | `rope_parameters["mrope_section"]` padded with 0 to 4 entries | 1671-1677 |
| `rope.freq_base` (f32) | `rope_params["rope_theta"]` (the `full_attention` sub-dict if present) | 1679 |
| `rope.freq_base_swa` (f32) | `rope_parameters["sliding_attention"]["rope_theta"]` | 1682 |
| `attention.layer_norm_rms_epsilon` (f32) | first of `rms_norm_eps, norm_eps` | 1685 |
| `attention.layer_norm_epsilon` | first of `layer_norm_eps, layer_norm_epsilon, norm_epsilon` | 1688 |
| `expert_count` | first of `num_local_experts, num_experts, n_routed_experts` | 1691 |
| `expert_used_count` | first of `num_experts_per_tok, num_experts_per_token, top_k_experts` | 1694 |
| `expert_group_count` / `expert_group_used_count` | `n_group` / `topk_group` | 1697-1702 |
| `expert_gating_func` | from `score_function|scoring_func|…` (`sigmoid`=2, `softmax`=1, `sqrtsoftplus`) | 1704-1713 |
| `attention.key_length`, `attention.value_length` | `head_dim` if present in hparams | 1715-1717 |
| `general.file_type` | output ftype (§0.4) | 1719 |

Also always: `general.architecture`, `general.type="model"`, `general.quantization_version=2`, `general.name` etc.
from `gguf.Metadata` (`base.py:1354-1387`). rllm reads only `general.architecture`, `general.name`, `general.alignment`.

`GGUFWriter.add_key_value` overwrites duplicate keys with a warning (`gguf-py/gguf/gguf_writer.py:278-282`):
**the last writer wins**, which matters for Gemma 4 (§4).

### 0.3 Tensor pipeline (`base.py:1130-1349`)

For every `(name, tensor)` in the safetensors (after `filter_tensors`), in this order:

1. **`filter_tensors`** (at index time, `base.py:291`):
   - `TextModel.filter_tensors` (`base.py:1556-1573`): drop multimodal tensors. Strip a leading `model.` for the
     test only, then drop if the name starts with `mlp, vit., vpm., siglip2., conformer., connector., merger.,
     resampler., sound_encoder., sound_projection., speech_embeddings.` or contains `visual., vision., audio., talker.,
     vision_, audio_, token2wav., code2wav., projector., pre_mm_projector_norm, image_newline, view_seperator,
     patch_embed, patch_embedding, patch_merger., patch_merge_mlp.`.
   - `ModelBase.filter_tensors` (`base.py:585-594`): `…e_score_correction_bias` → `…e_score_correction.bias`;
     **remove every occurrence of `language_model.`** (so `model.language_model.layers.3.x` → `model.layers.3.x`).
2. Skip names ending in `.attention.masked_bias`, `.attention.bias`, `.rotary_emb.inv_freq` (`base.py:1224`).
3. **dtype**: anything not f16/f32 (bf16, fp8 after dequant, ints) is cast to **f32** before any transform
   (`base.py:1230-1231`). All arithmetic transforms (+1, −exp) happen in f32.
4. `bid` = first dot-separated component that is all digits (`base.py:1234-1238`).
5. `modify_tensors(data, name, bid)` — arch-specific, ends in `ModelBase.modify_tensors` (`base.py:845-904`) which
   maps the HF name through `TensorNameMap` (`gguf-py/gguf/tensor_mapping.py:3117-3149`): exact key lookup, else
   strip a `.weight` / `.bias` suffix, look up the stem, re-append the suffix. Only `MODEL_TENSOR`s listed for the
   arch in `constants.py MODEL_TENSORS` participate, so ambiguous HF names resolve per arch (e.g.
   `post_attention_layernorm` → `ffn_norm` for llama/qwen3, `post_attention_norm` for qwen35/gemma4).
6. **Output type** (`base.py:1245-1321`):
   - F32 if `n_dims <= 1` or the GGUF name ends with `_norm.weight`;
   - F32 if the tensor is `FFN_GATE_INP` (router), `FFN_GATE_INP_SHEXP`, `SSM_CONV1D` (and a few others), or the GGUF
     name does not end with `.weight` (e.g. `blk.N.ssm_a`, `blk.N.ssm_dt.bias`, `*.scale`);
   - otherwise the file type: `--outtype auto` = **BF16 if the first ≥2-D safetensors tensor is bf16, F16 if f16,
     else F16** (`base.py:182-197`).
7. Shape: GGUF dims are the numpy shape reversed (`gguf_writer.py:266-269`): torch `[out, in]` → GGUF `[in, out]`.
   No squeezing: a torch `[1, d]` tensor becomes GGUF dims `[d, 1]`.

### 0.4 GGUF token types (`constants.py:6372-6378`)
`NORMAL=1, UNKNOWN=2, CONTROL=3, USER_DEFINED=4, UNUSED=5, BYTE=6`.

---------------------------------------------------------------------------------------------------------------

## 1. Qwen3 dense

### 1a. Architecture
| HF `architectures[0]` | Class | `general.architecture` |
|---|---|---|
| `Qwen3ForCausalLM`, `Qwen3Model` | `Qwen3Model(Qwen2Model)` `conversion/qwen.py:164-265` | `qwen3` |

MRO: `Qwen3Model → Qwen2Model → TextModel`. (A checkpoint whose README/name says "qwen3-reranker" or whose arch
contains "SequenceClassification" is converted as a reranker, `qwen.py:185-206` — ignore for causal LMs.)

### 1b. Metadata
Only the generic table §0.2 (`Qwen2Model.set_gguf_parameters` = base + `_try_set_pooling_type`, which writes
nothing without a sentence-transformers `modules.json`, `qwen.py:71-73`, `base.py:2513`).

| GGUF key | config.json | Notes |
|---|---|---|
| `qwen3.block_count` | `num_hidden_layers` | |
| `qwen3.context_length` | `max_position_embeddings` | 40960 for Qwen3 |
| `qwen3.embedding_length` | `hidden_size` | |
| `qwen3.feed_forward_length` | `intermediate_size` | |
| `qwen3.attention.head_count` / `head_count_kv` | `num_attention_heads` / `num_key_value_heads` | |
| `qwen3.attention.key_length`, `value_length` | `head_dim` (128; Qwen3 sets it explicitly, ≠ hidden/heads for some sizes) | rllm: required if ≠ hidden/heads |
| `qwen3.rope.freq_base` | `rope_theta` (1e6) | |
| `qwen3.attention.layer_norm_rms_epsilon` | `rms_norm_eps` | |
| `qwen3.rope.scaling.*` | only if `rope_scaling` is set (yarn) | rllm rejects anything but none/llama3 |
| no `rope.dimension_count`, no `vocab_size`, no `sliding_window` | | `use_sliding_window`/`sliding_window` ignored |

### 1c. Tensor names (`tensor_mapping.py` lines in brackets)
| HF name | GGUF name |
|---|---|
| `model.embed_tokens.weight` [16] | `token_embd.weight` |
| `model.norm.weight` [103] | `output_norm.weight` |
| `lm_head.weight` [81] (absent when `tie_word_embeddings`) | `output.weight` |
| `model.layers.N.input_layernorm.weight` [234] | `blk.N.attn_norm.weight` |
| `model.layers.N.self_attn.q_proj.weight` [301] | `blk.N.attn_q.weight` |
| `model.layers.N.self_attn.k_proj.weight` [322] | `blk.N.attn_k.weight` |
| `model.layers.N.self_attn.v_proj.weight` [344] | `blk.N.attn_v.weight` |
| `model.layers.N.self_attn.o_proj.weight` [371] | `blk.N.attn_output.weight` |
| `model.layers.N.self_attn.q_norm.weight` [785] | `blk.N.attn_q_norm.weight` |
| `model.layers.N.self_attn.k_norm.weight` [802] | `blk.N.attn_k_norm.weight` |
| `model.layers.N.post_attention_layernorm.weight` [458] | `blk.N.ffn_norm.weight` |
| `model.layers.N.mlp.gate_proj.weight` [642] | `blk.N.ffn_gate.weight` |
| `model.layers.N.mlp.up_proj.weight` [570] | `blk.N.ffn_up.weight` |
| `model.layers.N.mlp.down_proj.weight` [718] | `blk.N.ffn_down.weight` |

(`hf_arch == "Qwen2Model"` prefixes `model.`, `qwen.py:76-77`; not used for Qwen3.)

### 1d. Data transforms
None. No Q/K permutation (`Qwen2Model` is a plain `TextModel`), no norm offset (Qwen3 RMSNorm is `x*w`).
Types: norms (incl. q/k norms) F32; matrices BF16/F16 per §0.3. **Qwen3 tensors can be mapped zero-copy** (a
bf16 source stays bf16 bit-for-bit: bf16→f32→bf16 is lossless).

### 1e. Tokenizer
`Qwen3Model.set_vocab` → `Qwen2Model.set_vocab` (`qwen.py:61-69`): `_set_vocab_sentencepiece()` raises
`FileNotFoundError` (no `tokenizer.model`) → **`_set_vocab_gpt2()`** (`base.py:2159-2167`). See §5 for the
shared GPT-2 path. Specific values:

| Key | Value / rule |
|---|---|
| `tokenizer.ggml.model` | `gpt2` |
| `tokenizer.ggml.pre` | `qwen2` (hash `e636dc30…`, ref Qwen1.5-7B, `base.py:1948-1950`; Qwen3-Embedding hash `d4540891…` also → `qwen2`, `base.py:1867-1869`) |
| `tokenizer.ggml.tokens` | length = `config.vocab_size` (151936) — ids with no token become `[PAD{i}]` type UNUSED |
| `bos_token_id` | `tokenizer_config.bos_token` is `null` → falls to `config.json bos_token_id` (151643 `<|endoftext|>`) |
| `eos_token_id` | `tokenizer_config.eos_token` (`<|im_end|>` 151645 for instruct/hybrid, `<|endoftext|>` for base), id looked up in `tokenizer.json added_tokens` |
| `padding_token_id` | `tokenizer_config.pad_token` (`<|endoftext|>`) |
| `add_bos_token` / `add_eos_token` | `false` / `false` (ByteLevel post-processor + `add_bos_token:false`) |
| `<think>`, `</think>`, `<tool_call>`… | added tokens with `special:false` → **USER_DEFINED**; `<|im_start|>` etc. CONTROL |
| `tokenizer.ggml.normalizer.nfc` | `true` (tokenizer.json normalizer `NFC`) |
| `tokenizer.ggml.clean_spaces` | `tokenizer_config.clean_up_tokenization_spaces` (false) |

---------------------------------------------------------------------------------------------------------------

## 2. Qwen3.5 / Qwen3.6 hybrids (dense `qwen35`, MoE `qwen35moe`) incl. MTP

### 2a. Architecture
| HF `architectures[0]` | Class | `general.architecture` |
|---|---|---|
| `Qwen3_5ForConditionalGeneration`, `Qwen3_5ForCausalLM`, `Qwen3_5TextModel` | `Qwen3_5TextModel` `qwen.py:674-690` | `qwen35` |
| `Qwen3_5MoeForConditionalGeneration`, `Qwen3_5MoeForCausalLM` | `Qwen3_5MoeTextModel` `qwen.py:795-798` | `qwen35moe` |

Qwen3.6 checkpoints reuse these HF class names (verify `architectures` in the actual config; nothing in the
converter is 3.6-specific). `Qwen3_5TextModel` (HF arch) forces `no_mtp = True` (`qwen.py:679-686`).

MRO (both): `Qwen3_5[Moe]TextModel → _Qwen35MRopeMixin → _LinearAttentionVReorderBase → Qwen3NextModel →
_QwenMtpMixin → Qwen2MoeModel → TextModel`. So `modify_tensors` runs, in order:
`_LinearAttentionVReorderBase` (`qwen.py:602-655`) → `Qwen3NextModel` (`qwen.py:404-444`) →
`Qwen2MoeModel` (`qwen.py:97-152`) → `ModelBase` (`base.py:845`). The dense model goes through the MoE class's
expert code too; it is a no-op without `experts` in the name.

The ConditionalGeneration config nests text params under `text_config`; §0.1 merges them to the root. Vision
tensors (`model.visual.*`) are dropped by the `visual.` filter; `model.language_model.` → `model.`.

### 2b. Metadata

Writers in call order (later wins): `TextModel` (§0.2) → `Qwen2MoeModel` (`qwen.py:86-93`) → `_QwenMtpMixin`
(`qwen.py:359-364`) → `Qwen3NextModel` (`qwen.py:385-402`) → `_Qwen35MRopeMixin` (`qwen.py:668-671`) →
`Qwen3_5TextModel` (`_try_set_pooling_type`, no-op).

| GGUF key (`{arch}.`…) | Source / computation | Line | rllm reads |
|---|---|---|---|
| `block_count` | **`num_hidden_layers + n_mtp`** where `n_mtp = mtp_num_hidden_layers` (if absent/0: 1 + max index seen in `mtp.layers.<i>.*`; 0 when `no_mtp`) | `qwen.py:306-316`, `base.py:1607` | yes; `n_layers = block_count - nextn_predict_layers` |
| `nextn_predict_layers` | `block_count - num_hidden_layers`, written only if > 0 | `qwen.py:359-364` | yes |
| `context_length` | `max_position_embeddings` (262144) | base | yes |
| `embedding_length` | `hidden_size` | base | yes |
| `feed_forward_length` | `intermediate_size` (dense); MoE configs: whatever `intermediate_size` says, if present | base | dense only (MoE uses `expert_shared_feed_forward_length`) |
| `attention.head_count`, `head_count_kv` | `num_attention_heads`, `num_key_value_heads` | base | yes |
| `attention.key_length`, `value_length` | `head_dim` (256) | base | yes (required) |
| `attention.layer_norm_rms_epsilon` | `rms_norm_eps` | base | yes |
| `rope.freq_base` | `rope_parameters.rope_theta` (1e7) | base | yes (default 1e7) |
| `rope.dimension_count` | **`int(head_dim × rope_parameters.partial_rotary_factor)`**, factor default **0.25**; `head_dim` default `hidden_size // num_attention_heads` (→ 64 for head_dim 256) | `qwen.py:400-402` | yes (`n_rot`) |
| `rope.dimension_sections` (u32[4]) | `rope_parameters.mrope_section` padded to 4 (`[11,11,10,0]`); if absent, default `[11,11,10,0]` | `base.py:1671-1677`, `qwen.py:663-671` | no |
| `rope.scaling.*` | none (`rope_type:"default"` → warning only) | base | rejects any type ≠ none |
| `ssm.conv_kernel` | `linear_conv_kernel_dim` (4) | `qwen.py:387` | yes |
| `ssm.state_size` | **`linear_key_head_dim`** | `qwen.py:388` | yes (= GDN head dim) |
| `ssm.group_count` | **`linear_num_key_heads`** | `qwen.py:389` | yes (`n_k_heads`) |
| `ssm.time_step_rank` | **`linear_num_value_heads`** | `qwen.py:390` | yes (`n_v_heads`) |
| `ssm.inner_size` | `linear_value_head_dim × linear_num_value_heads` | `qwen.py:391` | yes (rllm requires = time_step_rank × state_size, i.e. key dim = value dim) |
| `attention.recurrent_layers` (bool[block_count]) | `[t == "linear_attention" for t in layer_types]` + `[False] × n_mtp`; error if `len(layer_types) != num_hidden_layers`; **omitted if `layer_types` absent** | `qwen.py:392-398` | yes (else falls back to interval) |
| `full_attention_interval` (u32) | `full_attention_interval`, default 4 | `qwen.py:399` | yes (fallback only) |
| `expert_count` | `num_experts` (MoE) | base | yes |
| `expert_used_count` | `num_experts_per_tok` | base | yes |
| `expert_feed_forward_length` | `moe_intermediate_size` | `qwen.py:88-90` | yes |
| `expert_shared_feed_forward_length` | `shared_expert_intermediate_size` | `qwen.py:91-93` | yes (0 = no shared expert) |
| (not written) `expert_weights_norm`, `expert_weights_scale`, `expert_gating_func` | `norm_topk_prob` is ignored by the converter | | rllm assumes softmax + renorm, scale 1 |

Not used by the converter: `attn_output_gate`, `mrope_interleaved`, `linear_value_head_dim` beyond inner size,
`tie_word_embeddings` (detected through the missing `lm_head`).

### 2c. Tensor names

HF prefix is `model.language_model.` (ConditionalGeneration) or `model.` (CausalLM); after §0.3 step 1 both are
`model.`. `tensor_mapping.py` lines in brackets. "(+1)" / "(V)" mark transforms from §2d.

**Global**
| HF | GGUF |
|---|---|
| `model.embed_tokens.weight` | `token_embd.weight` |
| `model.norm.weight` (+1) | `output_norm.weight` |
| `lm_head.weight` (absent if tied) | `output.weight` |

**Every layer** (`model.layers.N.`)
| HF | GGUF |
|---|---|
| `input_layernorm.weight` (+1) | `blk.N.attn_norm.weight` |
| `post_attention_layernorm.weight` (+1) [417] | `blk.N.post_attention_norm.weight` (qwen35 has no FFN_NORM) |

**Full-attention layers** (`layer_types[N] == "full_attention"`)
| HF | GGUF | Shape (torch) |
|---|---|---|
| `self_attn.q_proj.weight` | `blk.N.attn_q.weight` | `[2·n_head·head_dim, d]`: per head `[query(head_dim) ‖ gate(head_dim)]`, left as is |
| `self_attn.k_proj.weight` | `blk.N.attn_k.weight` | |
| `self_attn.v_proj.weight` | `blk.N.attn_v.weight` | |
| `self_attn.o_proj.weight` | `blk.N.attn_output.weight` | |
| `self_attn.q_norm.weight` (+1) | `blk.N.attn_q_norm.weight` | `[head_dim]` |
| `self_attn.k_norm.weight` (+1) | `blk.N.attn_k_norm.weight` | `[head_dim]` |

**Linear-attention (Gated DeltaNet) layers**
| HF | GGUF | Notes |
|---|---|---|
| `linear_attn.in_proj_qkv.weight` (V) [294] | `blk.N.attn_qkv.weight` | rows `[q: nk·dk ‖ k: nk·dk ‖ v: nv·dv]` |
| `linear_attn.in_proj_z.weight` (V) [442] | `blk.N.attn_gate.weight` | rows `nv·dv` |
| `linear_attn.in_proj_a.weight` (V) [993] | `blk.N.ssm_alpha.weight` | rows `nv` |
| `linear_attn.in_proj_b.weight` (V) [1021] | `blk.N.ssm_beta.weight` | rows `nv` |
| `linear_attn.conv1d.weight` (V, squeeze) [916] | `blk.N.ssm_conv1d.weight` | `[C,1,K]` → `[C,K]`, GGUF dims `[K, C]`, `C = 2·nk·dk + nv·dv`; **F32** |
| `linear_attn.dt_bias` (V, rename) | `blk.N.ssm_dt.bias` [932] | renamed to `linear_attn.dt_proj.bias` first (`qwen.py:407-408`); F32 |
| `linear_attn.A_log` (V, −exp) [949] | `blk.N.ssm_a` | **no `.weight` suffix**; F32 |
| `linear_attn.norm.weight` (NOT +1) [976] | `blk.N.ssm_norm.weight` | `[dv]`, shared by heads |
| `linear_attn.out_proj.weight` (V on columns) [987] | `blk.N.ssm_out.weight` | `[d, nv·dv]` |

**Dense MLP (`qwen35`)**: `mlp.{gate,up,down}_proj.weight` → `blk.N.ffn_{gate,up,down}.weight`.

**MoE MLP (`qwen35moe`)**
| HF | GGUF | Shape / notes |
|---|---|---|
| `mlp.gate.weight` [523] | `blk.N.ffn_gate_inp.weight` | `[E, d]` → GGUF `[d, E]`, **F32** |
| fused `mlp.experts.gate_up_proj` (`[E, 2F, d]`, with or without `.weight`) | `blk.N.ffn_gate_exps.weight` + `blk.N.ffn_up_exps.weight` | split on dim −2: gate = rows `[0,F)`, up = rows `[F,2F)` (`qwen.py:109-122`); with `--fuse-gate-up-exps` instead `blk.N.ffn_gate_up_exps.weight` = `cat(gate, up, dim=1)` = the HF tensor unchanged (`base.py:849-868`) |
| fused `mlp.experts.down_proj` (`[E, d, F]`) | `blk.N.ffn_down_exps.weight` | GGUF dims `[F, d, E]` (`qwen.py:103-107`) |
| per-expert `mlp.experts.X.{gate,up,down}_proj.weight` | `blk.N.ffn_{gate,up,down}_exps.weight` | `torch.stack` over X = 0..E−1 (dim 0), emitted once all `3E` tensors of the layer were seen (`qwen.py:124-150`) |
| `mlp.shared_expert.{gate,up,down}_proj.weight` [673/620/765] | `blk.N.ffn_{gate,up,down}_shexp.weight` | |
| `mlp.shared_expert_gate.weight` [542] | `blk.N.ffn_gate_inp_shexp.weight` | torch `[1, d]` → GGUF dims **`[d, 1]`**, F32 (see §6, item 6) |

**MTP block** (filtered/renamed at index time, `qwen.py:325-357`; `L = num_hidden_layers`, read from hparams
incl. `text_config`, `qwen.py:318-323`). A leading `model.mtp.` is first rewritten to `mtp.`.

| HF | Renamed to | GGUF |
|---|---|---|
| `mtp.fc.weight` | `model.layers.L.eh_proj.weight` | `blk.L.nextn.eh_proj.weight` (torch `[d, 2d]`; input = `cat(enorm(embedding), hnorm(hidden))`) [3001] |
| `mtp.pre_fc_norm_embedding.weight` (+1) | `model.layers.L.enorm.weight` | `blk.L.nextn.enorm.weight` |
| `mtp.pre_fc_norm_hidden.weight` (+1) | `model.layers.L.hnorm.weight` | `blk.L.nextn.hnorm.weight` |
| `mtp.norm.weight` (+1) | `model.layers.L.shared_head.norm.weight` | `blk.L.nextn.shared_head_norm.weight` |
| `mtp.layers.i.<rest>` | `model.layers.(L+i).<rest>` | the normal per-layer names above (`attn_norm`, `attn_q` with gate, `attn_k/v/output`, `attn_q/k_norm`, `post_attention_norm`, dense or MoE FFN incl. experts/shexp) with `bid = L+i` |

Only `mtp.fc`, `pre_fc_norm_embedding`, `pre_fc_norm_hidden`, `norm` are remapped as 3-part names; any other
`mtp.<x>.<y>` passes through unchanged and would fail mapping. `nextn.embed_tokens` / `nextn.shared_head_head`
are never produced for Qwen3.5 (the MTP block shares the main embedding and head). With `--no-mtp` all `mtp.*` are
dropped; with `--mtp` only `mtp.*` plus embed/norm/lm_head are kept.

### 2d. Data transforms (exact)

Applied to the f32 tensor (§0.3 step 3), in this order:

1. **V-head reorder** (`_LinearAttentionVReorderBase`, `qwen.py:602-655`), only if
   `nk = linear_num_key_heads`, `nv = linear_num_value_heads` are both > 0, **`nk != nv`**, and `"linear_attn."` in
   the name. Let `r = nv / nk`, `dk = linear_key_head_dim`, `dv = linear_value_head_dim`.
   `reorder(t, dim, hd)` (`qwen.py:476-501`): view axis `dim` (length `nk·r·hd`) as `[nk, r, hd]`, swap the first
   two → `[r, nk, hd]`, flatten. I.e. new head index `h' = j·nk + k` takes old head `h = k·r + j`
   (grouped-by-K-head → tiled). rllm's GDN kernel assumes exactly this tiled order (`crates/rllm-core/src/lib.rs:212-214`:
   value head `h` reads key head `h % n_k_heads`).
   | Tensor | What is reordered |
   |---|---|
   | `in_proj_qkv` | rows `[2·nk·dk, end)` only, `hd = dv` |
   | `in_proj_z` | all rows, `hd = dv` |
   | `in_proj_a`, `in_proj_b` | all rows, `hd = 1` |
   | `A_log`, `dt_bias` (1-D, `nv`) | the vector, `hd = 1` |
   | `conv1d` | after `squeeze()`, rows (channels) `[2·nk·dk, end)`, `hd = dv` |
   | `out_proj` | **columns** (dim 1), `hd = dv` (skipped only for Hadamard-folded checkpoints) |
   | `linear_attn.norm` | not touched (shared `[dv]`) |
2. **Qwen3-Next transforms** (`qwen.py:405-412`, an if/elif chain on the HF name):
   - name ends with `.A_log` → `data = -exp(data)` (stored as `ssm_a`);
   - elif ends with `.dt_bias` → rename to `….dt_proj.bias` (data unchanged);
   - elif `"conv1d"` in name → `data.squeeze()` (`[C,1,K]` → `[C,K]`);
   - elif name ends with `norm.weight` and not with `linear_attn.norm.weight` → **`data + 1`**. This hits
     `input_layernorm`, `post_attention_layernorm`, `model.norm`, `self_attn.q_norm`, `self_attn.k_norm`, and the
     MTP `enorm`, `hnorm`, `shared_head.norm` (after renaming, all end in `norm.weight`). Qwen3.5's RMSNorm is
     `x·(1+w)` (zero-centred weight); the gated `linear_attn.norm` is `x·w` and keeps its weight.
   - The `in_proj_qkvz` de-interleave (`qwen.py:414-442`) is Qwen3-Next only; Qwen3.5 ships split
     `in_proj_qkv` / `in_proj_z`.
3. **Experts** (`Qwen2MoeModel`, §2c): split fused gate_up, or stack per-expert matrices.
4. Output types: all norms, `ssm_a`, `ssm_dt.bias`, `ssm_conv1d`, `ffn_gate_inp`, `ffn_gate_inp_shexp` → F32;
   other matrices BF16/F16. rllm **requires** `ssm_conv1d` to be F32 (`crates/rllm-model/src/qwen35.rs:511-516`).

No Q/K permutation (Qwen3.5 attention is NeoX-style; rllm reorders dims for partial rotary itself,
`qwen35.rs:323-343`). Attention Q+gate interleave is kept; rllm splits it by rows (`qwen35.rs:518-523`).

### 2e. Tokenizer
`Qwen2MoeModel` has no `set_vocab` → `TextModel.set_vocab` → `_set_vocab_gpt2` (§5).

| Key | Value |
|---|---|
| `tokenizer.ggml.model` | `gpt2` |
| `tokenizer.ggml.pre` | **`qwen35`** (hash `d30d75d9…`, ref Qwen3.5-9B-Instruct, `base.py:2092-2094`; MiniCPM-V-4_6 hash `1444df51…` also `qwen35`). Regex differs from qwen2 (letters include `\p{M}`) |
| token list length | `vocab_size` from (merged) config (248320) |
| bos | `tokenizer_config.bos_token` null → `config.json bos_token_id`, or `text_config.bos_token_id` (`vocab.py:407-412`) |
| eos / pad | `tokenizer_config.eos_token` / `pad_token` looked up in `tokenizer.json added_tokens` |
| add_bos / add_eos | false / false |
| chat template | `tokenizer_config.chat_template`, else `chat_template.jinja`, else `chat_template.json` |

---------------------------------------------------------------------------------------------------------------

## 3. Llama 3 and Mistral (`llama`)

### 3a. Architecture
| HF | Class | `general.architecture` |
|---|---|---|
| `LlamaForCausalLM`, `MistralForCausalLM` (also `LLaMAForCausalLM`, `MixtralForCausalLM`, `LlamaModel`, …) | `LlamaModel` `conversion/llama.py:17-361` | `llama` |

(If `draft_vocab_size` is in config and `num_hidden_layers == 1` it becomes an EAGLE-3 draft, `llama.py:50-104`.
Mistral *native* format, `params.json`/`consolidated.safetensors`, goes through `MistralModel`, `mistral.py`; out of scope.)

### 3b. Metadata
Generic §0.2 plus `LlamaModel.set_gguf_parameters` (`llama.py:165-174`):

| GGUF key | Source |
|---|---|
| `llama.block_count`, `context_length`, `embedding_length`, `feed_forward_length`, `attention.head_count`, `attention.head_count_kv`, `attention.layer_norm_rms_epsilon` | as §0.2 |
| `llama.rope.freq_base` | `rope_theta` (Llama 3: 500000; Mistral: 1e6). AutoConfig default 10000 if missing |
| `llama.attention.key_length`/`value_length` | `head_dim` — present after `AutoConfig.to_dict()` for current transformers (= hidden/heads unless set) |
| `llama.vocab_size` | `vocab_size` (`llama.py:170`) |
| `llama.rope.dimension_count` | `head_dim` or `hidden_size // num_attention_heads` (`llama.py:172-174`) |
| `llama.rope.scaling.*` | **not written for `rope_type: "llama3"`** (handled by `rope_freqs`); linear/yarn as §0.2 |
| **not written**: `attention.sliding_window` | Mistral v0.1's `sliding_window: 4096` is ignored by the converter (rllm's optional `llama.attention.sliding_window` is therefore never set by llama.cpp GGUFs) |

### 3c. Tensor names
Same HF→GGUF table as Qwen3 (§1c) minus `q_norm`/`k_norm` (absent in Llama). `post_attention_layernorm` →
`ffn_norm`. `self_attn.rotary_emb.inv_freq` is skipped. Mixtral experts (`block_sparse_moe.experts.X.w{1,2,3}`)
are stacked into `ffn_{gate,down,up}_exps` (`llama.py:268-295`); rllm has no Mixtral support.

Extra generated tensor (`llama.py:299-327`), only if `rope_parameters(.full_attention).rope_type.lower() == "llama3"`:
`rope_freqs.weight`, F32, length `dim/2` with `dim = head_dim or hidden/heads`:
```
base = rope_theta (default 10000); factor = factor (8.0); lf = low_freq_factor (1.0); hf = high_freq_factor (4.0)
old = original_max_position_embeddings (8192)
low_wl = old / lf; high_wl = old / hf
for i in 0..dim/2: freq = 1 / base^(2i/dim); wl = 2π / freq
    wl < high_wl  → 1
    wl > low_wl   → factor
    else          → s = (old/wl − lf) / (hf − lf);  1 / ((1 − s)/factor + s)
```
(computed in f32 via torch for `freqs`, Python floats for the rest). rllm uses it as per-frequency divisors.

### 3d. Data transforms
- **Q/K permutation** (`LlamaModel.permute`, `llama.py:176-182`, applied at `llama.py:261-265` because
  `undo_permute = True`), to every name ending in `q_proj.weight|q_proj.bias` (with `n_head, n_head`) or
  `k_proj.weight|k_proj.bias` (with `n_head, n_kv_head`); `n_head = n_heads|num_attention_heads`,
  `n_kv_head = n_kv_heads|num_key_value_heads`:
  ```
  h = n_kv_head if (n_kv_head is not None and n_head != n_kv_head) else n_head
  w.reshape(h, 2, w.shape[0] // h // 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)
  ```
  Row `hd·head + 2j + t` of the output = HF row `hd·head + t·hd/2 + j` (NeoX half-split → interleaved pairs).
  It applies to every class inheriting `LlamaModel` with `undo_permute` true: Llama 1-3, Mistral, Mixtral,
  Llava/SmolVLM text, Arcee, SmolLM3, Apertus, Llama 4 (`Llama4Model`), ….
  **rllm undoes it on load for `arch == "llama"`** (`crates/rllm-model/src/loader.rs:97-123`,
  `qwen3.rs:209-210`): `out[h·hd+i] = stored[h·hd + 2i]` for `i < hd/2`, else `stored[h·hd + 2(i−hd/2)+1]`,
  which is the exact inverse. For HF weights the composition is the identity, so a synthetic loader should
  **skip both** (needs a flag telling rllm the Q/K rows are already NeoX-ordered) rather than permute and
  un-permute.
- No norm offsets, no casts beyond §0.3.

### 3e. Tokenizer (`llama.py:106-163`)
Decision chain:
1. `tekken.json` present and no `tokenizer.json` → `_set_vocab_mistral` (needs `mistral_common`).
2. If `tokenizer_config.add_prefix_space` is a bool → `tokenizer.ggml.add_space_prefix` = it.
3. `tokenizer.model` exists → **SentencePiece path** `_set_vocab_sentencepiece` (`base.py:2299-2399`) — Llama 2,
   Mistral 7B v0.1-v0.3:
   - `tokenizer.ggml.model = "llama"`, `pre = "default"`;
   - `tokens/scores/token_type` from the SentencePiece model: piece text (with `▁`), `GetScore`, type UNKNOWN/
     CONTROL/UNUSED/BYTE/NORMAL; list length `vocab_size` (padded `[PAD{i}]`, score −10000 / −1000, UNUSED);
   - overrides from `added_tokens.json` (USER_DEFINED, score −1000) and `tokenizer_config.added_tokens_decoder`
     (CONTROL if `special` or looks special, else USER_DEFINED with `▁`→space; score −1000);
   - `SpecialVocab(n_vocab=len(tokens))` (no merges).
4. else try `_set_vocab_llama_hf` (`base.py:2401-2421`, `LlamaHfVocab` `vocab.py:591-711`): requires
   `tokenizer.json` model BPE with `byte_fallback: true` and a `Sequence` decoder; raises `TypeError` for Llama 3
   (BPE, `ignore_merges: true`, no byte_fallback) → model `llama`, pre `default`, all scores −1000.
5. else (**Llama 3**) → `_set_vocab_gpt2` (§5): model `gpt2`, **pre `llama-bpe`** (hash `0ef9807a…`, ref
   Meta-Llama-3-8B, `base.py:1900-1902`; the hash includes the BOS that `encode` adds). Mistral-Nemo style
   tokenizer.json gives `tekken` (`base.py:1993-1995`, unsupported by rllm), Smaug `smaug-bpe`.
6. `vocab_size == 32016` (CodeLlama) adds FIM ids; `49152` sets `add_bos_token=false`.

Llama 3 specials: `bos` = `<|begin_of_text|>` 128000 (tokenizer_config), `eos` = tokenizer_config `eos_token`
(`<|eot_id|>` 128009 for 3.1/3.2/3.3 instruct, `<|end_of_text|>` 128001 for base); `config.json eos_token_id`
lists are ignored (non-int). `add_bos_token = true` from the `TemplateProcessing` post-processor
(`vocab.py:286-300`), `add_eos_token = false`. All 256 reserved/added tokens are `special` → CONTROL. Token list
length 128256.

---------------------------------------------------------------------------------------------------------------

## 4. Gemma 4 (`gemma4`)

### 4a. Architecture
| HF | Class | `general.architecture` |
|---|---|---|
| `Gemma4ForConditionalGeneration`, `Gemma4ForCausalLM` | `Gemma4Model(Gemma3Model)` `conversion/gemma.py:632-810` | `gemma4` |

(`Gemma4UnifiedForConditionalGeneration`, `Gemma4Assistant*`, `EmbeddingGemma2Model`, `Gemma4DSparkModel` are
other variants.) rllm supports the dense models only (it bails on `blk.0.ffn_gate_inp.weight`, `gemma4.rs:81-83`).

### 4b. Metadata
Writers in order: `TextModel` §0.2 → `Gemma3Model.set_gguf_parameters` (`gemma.py:141-158`) →
`Gemma4Model.set_gguf_parameters` (`gemma.py:671-733`). Last write wins.

| GGUF key (`gemma4.`…) | Final value | Line | rllm |
|---|---|---|---|
| `block_count` | `num_hidden_layers` | base | yes |
| `context_length` | `max_position_embeddings` (default 131072) | 146 | yes |
| `embedding_length` | `hidden_size` | base | yes |
| `feed_forward_length` | `intermediate_size`; **if `use_double_wide_mlp`: u32 array**, `[F if il < block_count − num_kv_shared_layers else 2F]` | base, 707-713 | scalar only (array → error) |
| `attention.head_count` | `num_attention_heads` (default 8) | 147 | yes |
| `attention.head_count_kv` | `num_key_value_heads` (default 4); **u32 array** `[kv_swa if swa else kv_full]` when `num_global_key_value_heads` (or a `per_layer_config` full-attention entry) exists | 158, 715-726 | scalar only |
| `attention.layer_norm_rms_epsilon` | `rms_norm_eps` (default 1e-6) | 148 | yes |
| `attention.key_length` / `value_length` | **`global_head_dim`** (else from `per_layer_config[i].head_dim` of the first full-attention layer) — overwrites the `head_dim` written earlier | 686-699 | yes (global head) |
| `attention.key_length_swa` / `value_length_swa` | `head_dim` | 700-701 | yes |
| `rope.freq_base` | `rope_parameters.full_attention.rope_theta` (default 1e6) | 151 | yes |
| `rope.freq_base_swa` | `rope_parameters.sliding_attention.rope_theta` | base 1682 | yes |
| `rope.dimension_count` | `int(global_head_dim)` (partial rotary is encoded in `rope_freqs`) | 730-732 | must equal key_length |
| `rope.dimension_count_swa` | `int(head_dim × rope_parameters.get("partial_rotary_factor", 1.0))` — top-level key of the nested dict, so normally **= head_dim** | 729-733 | must equal key_length_swa |
| `attention.sliding_window` | `sliding_window` (written because `sliding_window_pattern != 1`) | 156-157 | yes |
| `attention.sliding_window_pattern` | bool array `[t == "sliding_attention" for t in layer_types]` | 681-682 | yes |
| `attention.shared_kv_layers` | `num_kv_shared_layers` (required key) | 674-675 | yes |
| `embedding_length_per_layer_input` | `hidden_size_per_layer_input or 0` | 678-679 | yes |
| `final_logit_softcapping` | `final_logit_softcapping` if truthy | 154-155 | yes |
| `expert_feed_forward_length`, `expert_count`, `expert_used_count` | MoE only (`expert_intermediate_size|moe_intermediate_size`, base keys) | 703-705 | — |
| (not written) attn softcap, query scale, `attention_k_eq_v` | | | |

`rope_parameters` is nested (`full_attention: {rope_type: "proportional", partial_rotary_factor, rope_theta}`,
`sliding_attention: {rope_type: "default", rope_theta}`), so §0.1 mirroring does not happen.

### 4c. Tensor names
HF prefix `model.language_model.` → `model.` (vision/audio towers `model.vision_tower.*`, `model.embed_vision.*`,
`model.audio_tower.*`, `model.embed_audio.*` are dropped by the `vision.`/`audio.` filters).

| HF | GGUF | Notes |
|---|---|---|
| `model.embed_tokens.weight` | `token_embd.weight` | rows truncated to `len(tokenizer.json model.vocab) + len(added_tokens)` (`gemma.py:161-171`); for Gemma the added tokens are also in `model.vocab`, so this is ≥ rows and a no-op in practice |
| `model.embed_tokens_per_layer.weight` [835] | `per_layer_token_embd.weight` | `[vocab, L·P]` → GGUF `[L·P, vocab]`; not truncated |
| `model.per_layer_model_projection.weight` [839] | `per_layer_model_proj.weight` | |
| `model.per_layer_projection_norm.weight` [844] | `per_layer_proj_norm.weight` | |
| `model.norm.weight` | `output_norm.weight` | |
| `lm_head.weight` (normally tied/absent) | `output.weight` | |
| `model.layers.N.input_layernorm.weight` | `blk.N.attn_norm.weight` | |
| `….self_attn.{q,k,v,o}_proj.weight` | `blk.N.attn_{q,k,v,output}.weight` | `v_proj` absent on `attention_k_eq_v` layers (31B); KV-shared layers may lack k/v |
| `….self_attn.q_norm.weight` / `k_norm.weight` | `blk.N.attn_q_norm.weight` / `attn_k_norm.weight` | |
| `….post_attention_layernorm.weight` [417] | `blk.N.post_attention_norm.weight` | |
| `….pre_feedforward_layernorm.weight` [482] | `blk.N.ffn_norm.weight` | (FFN_PRE_NORM shares the name `ffn_norm`) |
| `….mlp.{gate,up,down}_proj.weight` | `blk.N.ffn_{gate,up,down}.weight` | |
| `….post_feedforward_layernorm.weight` [502] | `blk.N.post_ffw_norm.weight` | |
| `….per_layer_input_gate.weight` [857] | `blk.N.inp_gate.weight` | |
| `….per_layer_projection.weight` [862] | `blk.N.proj.weight` | |
| `….post_per_layer_input_norm.weight` [867] | `blk.N.post_norm.weight` | |
| `….layer_scalar` | `blk.N.layer_output_scale.weight` | filter appends `.weight` (`gemma.py:792-793`); shape `[1]`, F32 |
| MoE (26B-A4B): `router.proj` → `ffn_gate_inp`; `router.scale` → `blk.N.ffn_gate_inp.scale`; `router.per_expert_scale` → `blk.N.ffn_down_exps.scale`; `experts.gate_up_proj` → `ffn_gate_up_exps` (kept fused); `experts.down_proj` → `ffn_down_exps`; `pre_feedforward_layernorm_2` → `pre_ffw_norm_2`; `post_feedforward_layernorm_{1,2}` → `post_ffw_norm_{1,2}` | | `gemma.py:799-810`, mapping lines 489/513/517/538/688/761 |

Generated tensor (`gemma.py:735-762`): `rope_freqs.weight`, F32, length `global_head_dim/2`:
`n_rot = int(global_head_dim × full_attention.partial_rotary_factor / 2)`, values `[1.0]×n_rot + [1e30]×(global_head_dim/2 − n_rot)`.
Asserts `full_attention.rope_type == "proportional"`. Used only by full-attention layers.

### 4d. Data transforms
- **No norm offset**: `Gemma4Model.norm_shift` returns 0 (`gemma.py:637-639`), unlike Gemma 1/2/3 which add +1
  to every `*norm.weight` (`gemma.py:65-68`, `117-120`, `131-132`, `176-178`).
- No Q/K permutation, no embedding scaling (√d is applied at runtime).
- `embed_tokens` row truncation as above.
- Types: norms, `layer_output_scale`, `rope_freqs` F32; matrices per ftype.

### 4e. Tokenizer (`gemma.py:641-669`)
Uses `LlamaHfVocab` (`vocab.py:591-711`), not the GPT-2 path:
| Key | Value |
|---|---|
| `tokenizer.ggml.model` | **`gemma4`** |
| `tokenizer.ggml.pre` | **not written** (rllm defaults to `"default"`; it dispatches on model `gemma4`) |
| `tokens` | for id in `0..tokenizer.vocab_size` (base vocab; Gemma's added tokens are inside it): raw `tokenizer.json` vocab string (SPM `▁` form), then any added token with id ≥ base size |
| `scores` | all **−1000.0** |
| `token_type` | `BYTE` if text matches `<0x[0-9A-Fa-f]{2}>`; **`CONTROL` if id ∈ `tokenizer.all_special_ids`** (transformers: bos/eos/unk/sep/pad/cls/mask + additional/extra special tokens from tokenizer_config / special_tokens_map — *not* the `special` flag of `added_tokens`); else `NORMAL`; added-beyond-base tokens not special → `USER_DEFINED`. Then forced **USER_DEFINED** for `<|channel>`, `<channel|>`, `<|tool_call>`, `<tool_call|>`, `<|tool_response>`, `<tool_response|>`, `<|"|>` |
| `merges` | `tokenizer.json model.merges` (`SpecialVocab(load_merges=True)`) |
| bos/eos/unk/pad/mask ids | `SpecialVocab` (§5.3) |
| `add_bos_token` | `true` (forced, line 669) |
| `add_space_prefix` | `false` (line 668) |
| chat template | `SpecialVocab` (tokenizer_config → `chat_template.jinja` → `chat_template.json`) |

The pre-tokenizer hash table has only one `gemma4` entry (DFM-Mimir, `base.py:1891-1893`), used by other
classes that go through `get_vocab_base_pre`; Gemma 4 itself never hashes.

---------------------------------------------------------------------------------------------------------------

## 5. Tokenizer details common to the GPT-2/BPE path

### 5.1 `get_vocab_base` (`base.py:1755-1797`)
```
tokenizer = AutoTokenizer.from_pretrained(dir)
vocab_size = hparams.get("vocab_size", len(tokenizer.vocab))      # assert max id < vocab_size
reverse = {id: tok for tok, id in tokenizer.vocab.items()}        # includes added tokens
added = tokenizer.get_added_vocab(); dec = tokenizer.added_tokens_decoder
for i in 0..vocab_size:
  missing            → "[PAD{i}]", UNUSED
  tok in added       → if not dec[i].normalized: tok = decode(encode(tok, add_special_tokens=False))
                       CONTROL if dec[i].special or looks_special(tok)
                       else USER_DEFINED (with "▁" replaced by " ")
  else               → NORMAL (byte-level string as stored, e.g. "Ġthe")
```
`looks_special` (`base.py:1731-1752`): `<pad>`, `<mask>`, `<2mass>`, `[@BOS@]`, `<|…|>`, `<｜…｜>`, `<unused…>`.
Then `_set_vocab_gpt2` (`base.py:2159-2167`) writes `tokenizer.ggml.model="gpt2"`, `pre`, `tokens`, `token_type`,
and `SpecialVocab(dir, load_merges=True)`.

### 5.2 `tokenizer.ggml.pre` selection (`base.py:1803-2154`)
```
chktxt = <fixed string at base.py:1809>
chktok = tokenizer.encode(chktxt)          # HF AutoTokenizer, add_special_tokens defaults to True
chkhsh = sha256(str(chktok).encode()).hexdigest()   # str(list) = "[1, 2, 3]"
```
then an `if chkhsh == …` table; unknown hash → `NotImplementedError`. Relevant entries:

| Hash | Name | Ref model | Line |
|---|---|---|---|
| `0ef9807a4087ebef797fc749390439009c3b9eda9ad1a097abbe738f486c01e5` | `llama-bpe` | Meta-Llama-3-8B (also 3.1/3.2) | 1900 |
| `e636dc30a262dcc0d8c323492e32ae2b70728f4df7dfe9737d9f920a282b8aea` | `qwen2` | Qwen1.5-7B (= Qwen2/2.5/3) | 1948 |
| `d4540891389ea895b53b399da6ac824becc30f2fba0e9ddbb98f92e55ca0e97c` | `qwen2` | Qwen3-Embedding-0.6B | 1867 |
| `d30d75d9059f1aa2c19359de71047b3ae408c70875e8a3ccf8c5fba56c9d8af4` | `qwen35` | Qwen3.5-9B-Instruct | 2092 |
| `1444df51289cfa8063b96f0e62b1125440111bc79a52003ea14b6eac7016fd5f` | `qwen35` | MiniCPM-V-4_6 | 1870 |
| `c136ed14…` | `smaug-bpe` | Smaug-Llama-3-70B | 1972 |
| `63b97e42…` | `tekken` | Mistral-Nemo | 1993 |
| `846deafc…` | `gemma4` | DFM-Mimir | 1891 |
| (SPM path) | `default` | Llama 2 / Mistral 7B | 2303 |

Implementation note: rllm already links the HF `tokenizers` crate; `Tokenizer::from_file("tokenizer.json")`
+ `encode(chktxt, add_special_tokens=true)` reproduces `chktok` as long as the slow/fast `AutoTokenizer` would
use the same `tokenizer.json` (true for all four families). Alternatively match the `pre_tokenizer` regex in
`tokenizer.json` against the three regexes rllm supports (`crates/rllm-tokenizer/src/lib.rs:27-32`).

### 5.3 `SpecialVocab` (`gguf-py/gguf/vocab.py:50-413`)
Load order: `tokenizer.json` (+ `tokenizer_config.json`), then `config.json`; **first value set wins**
(`_set_special_token`, `vocab.py:167-177`; ids ≥ `n_vocab` are skipped when `n_vocab` is given).

| GGUF key | Source |
|---|---|
| `tokenizer.ggml.merges` | `tokenizer.json model.merges` if `load_merges`: list of `"a b"` strings as is, or list of pairs joined with `" "` where spaces *inside* a part become `chr(ord(' ')+256)` = `Ġ` (`vocab.py:219-244`); fallback `merges.txt` |
| `tokenizer.ggml.{bos,eos,unknown,seperator,padding,cls,mask}_token_id` | for typ in `bos, eos, unk, sep, pad, cls, mask`: `tokenizer_config["{typ}_token"]` (string or `{content}`) → the `id` of the first `tokenizer.json added_tokens` entry with that `content` (**added_tokens only**, not `model.vocab`); then `config.json["{typ}_token_id"]` (or `text_config[…]`) if int. Missing bos/eos fall back to cls/sep (`vocab.py:259-266`) |
| `tokenizer.ggml.add_bos_token` / `add_eos_token` / `add_sep_token` | post-processor first: `ByteLevel` → setdefault false/false; `RobertaProcessing` → true/true/true; `TemplateProcessing` → `add_bos = single[0] is bos/cls`, `add_eos = single[-1] is eos` (`vocab.py:267-350`); then `tokenizer_config["add_{typ}_token"]` only if not already set (`vocab.py:376-382`) |
| `tokenizer.chat_template` | `tokenizer_config["chat_template"]`, else `chat_template.jinja` (+ `additional_chat_templates/*.jinja` → list), else `chat_template.json["chat_template"]` (`vocab.py:353-371`). A list of `{name, template}` writes `name=="default"` to `tokenizer.chat_template`, others to `tokenizer.chat_template.<name>` plus `tokenizer.chat_templates` (`gguf_writer.py:1360-1390`) |
| `tokenizer.ggml.clean_spaces` | `tokenizer_config.clean_up_tokenization_spaces` (bool) |
| `tokenizer.ggml.normalizer.{nfc,lowercase,strip_accents}` | from `tokenizer.json normalizer` |
| `tokenizer.ggml.metaspace.prepend_scheme` | from a `Metaspace` pre-tokenizer |

rllm reads: `tokenizer.ggml.{model, pre, tokens, token_type, merges, scores (SPM), add_space_prefix, bos_token_id,
eos_token_id, eot_token_id, add_bos_token}`, `tokenizer.chat_template` (+ `.systemone`)
(`crates/rllm-tokenizer/src/lib.rs:71-222`). CONTROL/USER_DEFINED types decide which strings are matched whole.

---------------------------------------------------------------------------------------------------------------

## 6. Keys/tensors rllm reads that are derived non-obviously (checklist)

1. **`qwen35*.block_count` includes the MTP layer(s)**, and `nextn_predict_layers` is the difference; MTP tensors
   live at `blk.{num_hidden_layers}.*` (renamed from `mtp.*`, §2c). rllm sets `mtp` only if
   `blk.{n_layers}.nextn.eh_proj.weight` exists.
2. **`qwen35*.ssm.*` naming**: `state_size` = key head dim, `group_count` = K heads, `time_step_rank` = V heads,
   `inner_size` = V heads × V head dim.
3. **`qwen35*.attention.recurrent_layers`** from `layer_types` (+ `false` for MTP layers); absent → rllm uses
   `full_attention_interval`.
4. **`qwen35*.rope.dimension_count` = head_dim × partial_rotary_factor (default 0.25)**, not head_dim.
5. **Qwen3.5 norms +1** (every RMSNorm incl. q/k norms and MTP norms, except `linear_attn.norm`), **`ssm_a = −exp(A_log)`**,
   `dt_bias → ssm_dt.bias`, conv squeeze to `[C,K]` F32, and the **V-head grouped→tiled reorder** of 7 tensors when
   `nk != nv` (true for all published Qwen3.5/3.6 sizes with 16 K heads and 32/48/64 V heads — check config).
6. **`ffn_gate_inp_shexp` shape**: the converter writes torch `[1, d]` as GGUF dims `[d, 1]` (no squeeze;
   llama.cpp's loader tolerates trailing 1s), but `crates/rllm-model/src/qwen35.rs:210` checks dims `[d]`
   strictly (`loader.rs:45-51`). A synthetic GGUF should emit `[d]`; also worth checking against a real
   llama.cpp `qwen35moe` GGUF, since by the code above rllm should reject one.
7. **Llama `rope_freqs.weight`** is generated from `rope_scaling` (llama3), and Q/K rows are permuted (rllm
   un-permutes) — see §3d.
8. **Gemma 4 `attention.key_length` is the *global* head dim**, `key_length_swa` the local one;
   `sliding_window_pattern` is a bool array from `layer_types`; `rope.dimension_count = global_head_dim`; partial
   rotary lives in the generated `rope_freqs.weight` (`1e30` for unrotated pairs);
   `layer_output_scale.weight` comes from `layer_scalar`.
9. **Token types**: Gemma 4 CONTROL = transformers' `all_special_ids` (+ 7 forced USER_DEFINED); GPT-2 path
   CONTROL = `added_tokens[].special` or "looks special"; non-special added tokens USER_DEFINED.
10. **Token list length = `config.vocab_size`** (GPT-2 path, padded with `[PAD{i}]`), which also equals the
    `token_embd` row count rllm uses as vocab size.
11. **bos id** for Qwen comes from `config.json bos_token_id` because `tokenizer_config.bos_token` is null;
    eos ids only resolve through `tokenizer.json added_tokens`.
12. **AutoConfig defaults** (§0.1): values like `head_dim`, `rope_theta`, `rms_norm_eps`, `max_position_embeddings`
    come from the transformers config class when absent from `config.json`.
