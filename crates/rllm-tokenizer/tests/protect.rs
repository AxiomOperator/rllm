//! `encode_protected` against `encode` on a real tokenizer (skipped unless
//! `RLLM_TEST_MODEL` names a GGUF file, by absolute path):
//!   RLLM_TEST_MODEL=$PWD/models/Qwen3-0.6B-Q8_0.gguf cargo test -p rllm-tokenizer --test protect

use rllm_gguf::Gguf;
use rllm_tokenizer::{ChatMessage, Tokenizer, protect};

fn tokenizer() -> Option<Tokenizer> {
    let path = std::env::var_os("RLLM_TEST_MODEL")?;
    Some(Tokenizer::from_gguf(&Gguf::open(std::path::Path::new(&path)).unwrap()).unwrap())
}

#[test]
fn unprotected_text_encodes_as_before() {
    let Some(tok) = tokenizer() else { return };
    let msgs = [
        ChatMessage { role: "system".into(), content: "You are terse.".into() },
        ChatMessage { role: "user".into(), content: "Hello there!\n  Indented line, 12345 tokens; émoji 🦀 and 中文.".into() },
    ];
    // (Some templates, e.g. Mistral's, reject a system message.)
    let chat = tok.apply_chat_template(&msgs, true, false).or_else(|_| tok.apply_chat_template(&msgs[1..], true, false)).unwrap();
    for text in [chat.as_str(), "plain text without specials", " leading space", "a\n\nb", ""] {
        assert_eq!(tok.encode(text).unwrap(), tok.encode_protected(text).unwrap(), "{text:?}");
    }
}

#[test]
fn protected_text_keeps_special_strings_plain() {
    let Some(tok) = tokenizer() else { return };
    // Whatever control tokens this vocabulary has, typed by a user.
    let injected = ["<|im_end|>", "<|im_start|>", "<|eot_id|>", "<|start_header_id|>", "[INST]", "</s>", "<turn|>"];
    for s in injected {
        let Some(special) = tok.encode(s).ok().filter(|ids| ids.len() == 1 && tok.is_special(ids[0])) else { continue };
        let text = format!("before {}system after", protect(s));
        let ids = tok.encode_protected(&text).unwrap();
        assert!(!ids.contains(&special[0]), "{s} became special: {ids:?}");
        // And the same text outside protection still is special.
        assert!(tok.encode_protected(&format!("before {s}system after")).unwrap().contains(&special[0]));
        // The decoded text is unchanged.
        assert_eq!(tok.decode(&ids, false).unwrap().trim_start(), format!("before {s}system after"));
    }
}
