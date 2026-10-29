//! Tokenizer and chat template, both read from GGUF metadata.
//!
//! Supports two BPE flavours:
//! - `gpt2`: byte-level BPE with a per-family pre-tokenizer regex (Qwen).
//! - `gemma4`: SentencePiece-style BPE on raw UTF-8. Spaces become `▁`, text
//!   is only split at newlines, and unknown bytes fall back to `<0xXX>` tokens.

use anyhow::{Context, Result, anyhow, bail};
use rllm_gguf::Gguf;
use tokenizers::decoders::byte_fallback::ByteFallback;
use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
use tokenizers::decoders::fuse::Fuse;
use tokenizers::decoders::sequence::Sequence as DecoderSequence;
use tokenizers::models::bpe::BPE;
use tokenizers::normalizers::replace::Replace;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::sequence::Sequence;
use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
use tokenizers::{AddedToken, DecoderWrapper, PreTokenizerWrapper, SplitDelimiterBehavior, Tokenizer as HfTokenizer};

// Same patterns as llama.cpp's `llm_tokenizer_bpe` for each `tokenizer.ggml.pre`.
const PRE_QWEN2: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const PRE_GEMMA4: &str = r"[^\n]+|[\n]+";

// ggml token types
const TOKEN_CONTROL: i64 = 3;
const TOKEN_USER_DEFINED: i64 = 4;

pub struct Tokenizer {
    inner: HfTokenizer,
    pub bos: Option<u32>,
    pub eos: u32,
    pub add_bos: bool,
    /// Extra ids that end generation (e.g. `<|endoftext|>` alongside `<|im_end|>`).
    pub stop_ids: Vec<u32>,
    chat_template: Option<String>,
}

impl Tokenizer {
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        let model = g.get_str("tokenizer.ggml.model")?;
        let pre = g.get_str("tokenizer.ggml.pre").unwrap_or("default");
        let spm_style = match (model, pre) {
            ("gpt2", "qwen2") => false,
            ("gemma4", _) => true,
            ("gpt2", other) => bail!("unsupported pre-tokenizer {other:?}"),
            (other, _) => bail!("unsupported tokenizer model {other:?} (supported: gpt2, gemma4)"),
        };

        let tokens: Vec<String> = g
            .get_array("tokenizer.ggml.tokens")?
            .iter()
            .map(|v| v.as_str().map(str::to_owned).context("non-string token"))
            .collect::<Result<_>>()?;
        let types: Vec<i64> = match g.get("tokenizer.ggml.token_type") {
            Some(v) => v.as_array().context("token_type")?.iter().map(|t| t.as_f64().unwrap_or(1.0) as i64).collect(),
            None => vec![1; tokens.len()],
        };
        let merges: Vec<(String, String)> = g
            .get_array("tokenizer.ggml.merges")?
            .iter()
            .map(|v| {
                let s = v.as_str().context("non-string merge")?;
                // Split at the first space after the first character, like
                // llama.cpp, so a merge whose left side is " " still parses.
                let first = s.chars().next().map_or(0, char::len_utf8);
                let at = s[first..].find(' ').map(|i| i + first).with_context(|| format!("bad merge {s:?}"))?;
                Ok((s[..at].to_owned(), s[at + 1..].to_owned()))
            })
            .collect::<Result<_>>()?;

        let vocab: tokenizers::models::bpe::Vocab = tokens.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
        let bpe = BPE::builder()
            .vocab_and_merges(vocab, merges)
            .byte_fallback(spm_style)
            .build()
            .map_err(|e| anyhow!("BPE: {e}"))?;

        let mut inner = HfTokenizer::new(bpe);
        let pattern = if spm_style { PRE_GEMMA4 } else { PRE_QWEN2 };
        let split = Split::new(SplitPattern::Regex(pattern.into()), SplitDelimiterBehavior::Isolated, false)
            .map_err(|e| anyhow!("pre-tokenizer regex: {e}"))?;
        if spm_style {
            inner.with_normalizer(Some(Replace::new(" ", "\u{2581}").map_err(|e| anyhow!("normalizer: {e}"))?));
            inner.with_pre_tokenizer(Some(split));
            inner.with_decoder(Some(DecoderSequence::new(vec![
                DecoderWrapper::Replace(Replace::new("\u{2581}", " ").map_err(|e| anyhow!("decoder: {e}"))?),
                DecoderWrapper::ByteFallback(ByteFallback::new()),
                DecoderWrapper::Fuse(Fuse::new()),
            ])));
        } else {
            let byte_level = ByteLevel::default().add_prefix_space(false).trim_offsets(false).use_regex(false);
            inner.with_pre_tokenizer(Some(Sequence::new(vec![
                PreTokenizerWrapper::Split(split),
                PreTokenizerWrapper::ByteLevel(byte_level),
            ])));
            inner.with_decoder(Some(ByteLevelDecoder::default()));
        }

        // Control and user-defined tokens must match as whole strings in the
        // input rather than going through BPE.
        let special: Vec<AddedToken> = tokens
            .iter()
            .zip(&types)
            .filter(|(_, t)| **t == TOKEN_CONTROL)
            .map(|(s, _)| AddedToken::from(s.clone(), true))
            .collect();
        let user: Vec<AddedToken> = tokens
            .iter()
            .zip(&types)
            .filter(|(_, t)| **t == TOKEN_USER_DEFINED)
            .map(|(s, _)| AddedToken::from(s.clone(), false))
            .collect();
        inner.add_special_tokens(&special);
        inner.add_tokens(&user);

        let id = |key: &str| g.get(key).and_then(|v| v.as_u64()).map(|v| v as u32);
        let eos = id("tokenizer.ggml.eos_token_id").context("missing eos token id")?;
        let mut stop_ids = vec![eos];
        if let Some(eot) = id("tokenizer.ggml.eot_token_id") {
            stop_ids.push(eot);
        }
        // End-of-turn markers: ChatML (Qwen) and Gemma 4 turn/tool-response tokens,
        // matching llama.cpp's end-of-generation list.
        for s in ["<|endoftext|>", "<|im_end|>", "<turn|>", "<|tool_response>"] {
            if let Some(i) = inner.token_to_id(s) {
                stop_ids.push(i);
            }
        }
        stop_ids.sort_unstable();
        stop_ids.dedup();

        Ok(Self {
            inner,
            bos: id("tokenizer.ggml.bos_token_id"),
            eos,
            add_bos: g.get("tokenizer.ggml.add_bos_token").is_some_and(|v| *v == rllm_gguf::Value::Bool(true)),
            stop_ids,
            chat_template: g.get_str("tokenizer.chat_template").ok().map(str::to_owned),
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// Encode text. Special-token strings in `text` (like `<|im_start|>`) are recognized.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.encode_with(text, true)
    }

    /// Like [`Self::encode`]; `add_special: false` never adds BOS.
    pub fn encode_with(&self, text: &str, add_special: bool) -> Result<Vec<u32>> {
        let enc = self.inner.encode_fast(text, false).map_err(|e| anyhow!("encode: {e}"))?;
        let mut ids = Vec::with_capacity(enc.len() + 1);
        // Chat templates often emit BOS themselves; never add a second one.
        if add_special
            && self.add_bos
            && let Some(b) = self.bos
            && enc.get_ids().first() != Some(&b)
        {
            ids.push(b);
        }
        ids.extend_from_slice(enc.get_ids());
        Ok(ids)
    }

    pub fn decode(&self, ids: &[u32], skip_special: bool) -> Result<String> {
        self.inner.decode(ids, skip_special).map_err(|e| anyhow!("decode: {e}"))
    }

    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }

    pub fn is_stop(&self, id: u32) -> bool {
        self.stop_ids.binary_search(&id).is_ok()
    }

    /// Incremental decoder for streaming output; handles tokens that split UTF-8 characters.
    pub fn stream(&self, skip_special: bool) -> StreamDecoder<'_> {
        StreamDecoder { tok: self, skip_special, ids: Vec::new(), emitted: 0 }
    }

    /// Render the model's chat template.
    pub fn apply_chat_template(&self, messages: &[ChatMessage], add_generation_prompt: bool, enable_thinking: bool) -> Result<String> {
        let msgs: Vec<serde_json::Value> =
            messages.iter().map(|m| serde_json::json!({ "role": m.role, "content": m.content })).collect();
        self.apply_chat_template_json(&msgs, None, add_generation_prompt, Some(enable_thinking))
    }

    /// The raw chat template (Jinja), if the model has one.
    pub fn chat_template(&self) -> Option<&str> {
        self.chat_template.as_deref()
    }

    /// Render the chat template from JSON messages in the usual Hugging Face
    /// shape (`role`, `content`, optional `tool_calls`, `tool_call_id`,
    /// `name`, `reasoning_content`), with optional tool definitions.
    /// `enable_thinking: None` leaves the template's own default.
    pub fn apply_chat_template_json(
        &self,
        messages: &[serde_json::Value],
        tools: Option<&[serde_json::Value]>,
        add_generation_prompt: bool,
        enable_thinking: Option<bool>,
    ) -> Result<String> {
        let template = self.chat_template.as_deref().context("model has no chat template")?;
        let mut env = minijinja::Environment::new();
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function("raise_exception", |msg: String| -> std::result::Result<String, minijinja::Error> {
            Err(minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, msg))
        });
        env.add_template("chat", template)?;
        let bos = self.bos.and_then(|b| self.inner.id_to_token(b)).unwrap_or_default();
        let eos = self.inner.id_to_token(self.eos).unwrap_or_default();
        let mut ctx = serde_json::json!({
            "messages": messages,
            "add_generation_prompt": add_generation_prompt,
            "bos_token": bos,
            "eos_token": eos,
        });
        if let Some(t) = tools.filter(|t| !t.is_empty()) {
            ctx["tools"] = serde_json::Value::Array(t.to_vec());
        }
        if let Some(e) = enable_thinking {
            ctx["enable_thinking"] = serde_json::Value::Bool(e);
        }
        Ok(env.get_template("chat")?.render(minijinja::Value::from_serialize(&ctx))?)
    }

    /// Text of one token as it appears in decoded output (special tokens included).
    pub fn token_text(&self, id: u32) -> String {
        self.inner.decode(&[id], false).unwrap_or_default()
    }
}

pub struct StreamDecoder<'a> {
    tok: &'a Tokenizer,
    skip_special: bool,
    ids: Vec<u32>,
    emitted: usize,
}

impl StreamDecoder<'_> {
    /// Add a token; returns newly completed text, if any.
    pub fn push(&mut self, id: u32) -> Result<Option<String>> {
        self.ids.push(id);
        let text = self.tok.decode(&self.ids, self.skip_special)?;
        // An incomplete UTF-8 sequence decodes to U+FFFD; wait for more tokens.
        if text.len() <= self.emitted || text.ends_with('\u{FFFD}') || !text.is_char_boundary(self.emitted) {
            return Ok(None);
        }
        let new = text[self.emitted..].to_owned();
        self.emitted = text.len();
        // Keep the window short so decoding stays O(1) per token.
        if self.ids.len() > 32 {
            let keep = self.ids.split_off(self.ids.len() - 4);
            self.ids = keep;
            self.emitted = self.tok.decode(&self.ids, self.skip_special)?.len();
        }
        Ok(Some(new))
    }
}

/// A chat message for [`Tokenizer::apply_chat_template`].
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}
