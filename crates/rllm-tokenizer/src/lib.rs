//! Tokenizer and chat template, both read from GGUF metadata.
//!
//! Supports three BPE flavours:
//! - `gpt2`: byte-level BPE with a per-family pre-tokenizer regex (Qwen,
//!   Llama 3).
//! - `gemma4`: SentencePiece-style BPE on raw UTF-8. Spaces become `▁`, text
//!   is only split at newlines, and unknown bytes fall back to `<0xXX>` tokens.
//! - `llama`: SentencePiece BPE with scores instead of merges (Llama 2,
//!   Mistral): merges are derived from the scores, text gets a leading `▁`.

use anyhow::{Context, Result, anyhow, bail};
use rllm_gguf::Gguf;
use tokenizers::decoders::byte_fallback::ByteFallback;
use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
use tokenizers::decoders::fuse::Fuse;
use tokenizers::decoders::sequence::Sequence as DecoderSequence;
use tokenizers::decoders::strip::Strip;
use tokenizers::normalizers::prepend::Prepend;
use tokenizers::normalizers::utils::Sequence as NormalizerSequence;
use tokenizers::models::bpe::BPE;
use tokenizers::normalizers::replace::Replace;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::sequence::Sequence;
use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
use tokenizers::{AddedToken, DecoderWrapper, PreTokenizerWrapper, SplitDelimiterBehavior, Tokenizer as HfTokenizer};

// Same patterns as llama.cpp's `llm_tokenizer_bpe` for each `tokenizer.ggml.pre`.
const PRE_QWEN2: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const PRE_LLAMA3: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const PRE_GEMMA4: &str = r"[^\n]+|[\n]+";

// ggml token types
const TOKEN_NORMAL: i64 = 1;
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
        #[derive(PartialEq)]
        enum Kind {
            Gpt2(&'static str),
            Gemma4,
            Spm,
        }
        let kind = match (model, pre) {
            ("gpt2", "qwen2") => Kind::Gpt2(PRE_QWEN2),
            ("gpt2", "llama-bpe") => Kind::Gpt2(PRE_LLAMA3),
            ("gemma4", _) => Kind::Gemma4,
            ("llama", _) => Kind::Spm,
            ("gpt2", other) => bail!("unsupported pre-tokenizer {other:?}"),
            (other, _) => bail!("unsupported tokenizer model {other:?} (supported: gpt2, gemma4, llama)"),
        };
        let spm_style = !matches!(kind, Kind::Gpt2(_));

        let tokens: Vec<String> = g
            .get_array("tokenizer.ggml.tokens")?
            .iter()
            .map(|v| v.as_str().map(str::to_owned).context("non-string token"))
            .collect::<Result<_>>()?;
        let types: Vec<i64> = match g.get("tokenizer.ggml.token_type") {
            Some(v) => v.as_array().context("token_type")?.iter().map(|t| t.as_f64().unwrap_or(1.0) as i64).collect(),
            None => vec![1; tokens.len()],
        };
        let merges: Vec<(String, String)> = if kind == Kind::Spm {
            spm_merges(&tokens, &types, g)?
        } else {
            g.get_array("tokenizer.ggml.merges")?
            .iter()
            .map(|v| {
                let s = v.as_str().context("non-string merge")?;
                // Split at the first space after the first character, like
                // llama.cpp, so a merge whose left side is " " still parses.
                let first = s.chars().next().map_or(0, char::len_utf8);
                let at = s[first..].find(' ').map(|i| i + first).with_context(|| format!("bad merge {s:?}"))?;
                Ok((s[..at].to_owned(), s[at + 1..].to_owned()))
            })
            .collect::<Result<_>>()?
        };

        let vocab: tokenizers::models::bpe::Vocab = tokens.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
        let bpe = BPE::builder()
            .vocab_and_merges(vocab, merges)
            .byte_fallback(spm_style)
            // Llama 3: a pre-token that is itself in the vocabulary is used
            // whole (llama.cpp's ignore_merges for llama-bpe).
            .ignore_merges(pre == "llama-bpe")
            .build()
            .map_err(|e| anyhow!("BPE: {e}"))?;

        let mut inner = HfTokenizer::new(bpe);
        let pattern = match kind {
            Kind::Gpt2(p) => p,
            _ => PRE_GEMMA4,
        };
        let split = Split::new(SplitPattern::Regex(pattern.into()), SplitDelimiterBehavior::Isolated, false)
            .map_err(|e| anyhow!("pre-tokenizer regex: {e}"))?;
        if kind == Kind::Spm {
            // SentencePiece: a leading space (unless the model says not to),
            // spaces as `▁`, no splitting; decoding drops the leading space.
            let prefix = g.get("tokenizer.ggml.add_space_prefix").is_none_or(|v| *v == rllm_gguf::Value::Bool(true));
            let mut norms = Vec::new();
            if prefix {
                norms.push(Prepend::new("\u{2581}".into()).into());
            }
            norms.push(Replace::new(" ", "\u{2581}").map_err(|e| anyhow!("normalizer: {e}"))?.into());
            inner.with_normalizer(Some(NormalizerSequence::new(norms)));
            let mut decoders = vec![
                DecoderWrapper::Replace(Replace::new("\u{2581}", " ").map_err(|e| anyhow!("decoder: {e}"))?),
                DecoderWrapper::ByteFallback(ByteFallback::new()),
                DecoderWrapper::Fuse(Fuse::new()),
            ];
            if prefix {
                decoders.push(DecoderWrapper::Strip(Strip::new(' ', 1, 0)));
            }
            inner.with_decoder(Some(DecoderSequence::new(decoders)));
        } else if spm_style {
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
        for s in ["<|endoftext|>", "<|im_end|>", "<turn|>", "<|tool_response>", "<|eot_id|>", "<|eom_id|>", "<|end_of_text|>"] {
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
        // Llama 3 templates put today's date in the system prompt.
        env.add_function("strftime_now", |fmt: String| strftime_utc(&fmt, std::time::SystemTime::now()));
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

/// Merges for a SentencePiece BPE vocabulary that has scores but no merge
/// list, as Hugging Face's converter derives them: every split of a normal
/// token into two vocabulary tokens is a merge, ranked by the merged
/// token's score (higher first).
fn spm_merges(tokens: &[String], types: &[i64], g: &Gguf) -> Result<Vec<(String, String)>> {
    let scores: Vec<f64> = g
        .get_array("tokenizer.ggml.scores")?
        .iter()
        .map(|v| v.as_f64().context("non-numeric score"))
        .collect::<Result<_>>()?;
    let ids: std::collections::HashMap<&str, usize> = tokens.iter().enumerate().map(|(i, t)| (t.as_str(), i)).collect();
    let mut merges: Vec<(f64, usize, usize, usize)> = Vec::new();
    for (i, t) in tokens.iter().enumerate() {
        if types.get(i).copied().unwrap_or(TOKEN_NORMAL) != TOKEN_NORMAL {
            continue;
        }
        for (at, _) in t.char_indices().skip(1) {
            if let (Some(&l), Some(&r)) = (ids.get(&t[..at]), ids.get(&t[at..])) {
                merges.push((scores.get(i).copied().unwrap_or(0.0), l, r, i));
            }
        }
    }
    // Score descending; ties by the parts' ids, as the converter sorts them.
    merges.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    Ok(merges.into_iter().map(|(_, l, r, _)| (tokens[l].clone(), tokens[r].clone())).collect())
}

/// `strftime` for the formats chat templates use (%d %e %m %b %B %y %Y %H %M %S %A %a %j %%), UTC.
fn strftime_utc(fmt: &str, now: std::time::SystemTime) -> String {
    const MONTHS: [&str; 12] = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
    const DAYS: [&str; 7] = ["Thursday", "Friday", "Saturday", "Sunday", "Monday", "Tuesday", "Wednesday"];
    let secs = now.duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs()) as i64;
    let (days, rem) = (secs.div_euclid(86400), secs.rem_euclid(86400));
    // Civil date from days since 1970-01-01 (Howard Hinnant's algorithm).
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    let leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
    let cum = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    let yday = cum[(month - 1) as usize] + day + i64::from(leap && month > 2);
    let weekday = DAYS[days.rem_euclid(7) as usize];
    let mut out = String::new();
    let mut chars = fmt.chars();
    while let Some(c) = chars.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('d') => out += &format!("{day:02}"),
            Some('e') => out += &format!("{day:2}"),
            Some('m') => out += &format!("{month:02}"),
            Some('b') => out += &MONTHS[(month - 1) as usize][..3],
            Some('B') => out += MONTHS[(month - 1) as usize],
            Some('y') => out += &format!("{:02}", year % 100),
            Some('Y') => out += &year.to_string(),
            Some('H') => out += &format!("{:02}", rem / 3600),
            Some('M') => out += &format!("{:02}", rem / 60 % 60),
            Some('S') => out += &format!("{:02}", rem % 60),
            Some('A') => out += weekday,
            Some('a') => out += &weekday[..3],
            Some('j') => out += &format!("{yday:03}"),
            Some('%') => out.push('%'),
            Some(o) => {
                out.push('%');
                out.push(o);
            }
            None => out.push('%'),
        }
    }
    out
}

#[cfg(test)]
mod strftime_tests {
    #[test]
    fn formats_a_known_date() {
        // 2024-07-26 13:25:09 UTC, a Friday.
        let t = std::time::UNIX_EPOCH + std::time::Duration::from_secs(1_722_000_309);
        assert_eq!(super::strftime_utc("%d %b %Y", t), "26 Jul 2024");
        assert_eq!(super::strftime_utc("%A %B %e, %H:%M:%S (%j) %%", t), "Friday July 26, 13:25:09 (208) %");
    }
}
