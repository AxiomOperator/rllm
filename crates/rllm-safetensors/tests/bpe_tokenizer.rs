//! Byte-level BPE tokenizers converted from tokenizer.json (Llama 3, Mistral
//! Tekken) encode exactly as Hugging Face's `tokenizers` does with the same
//! file. Set RLLM_ST_TOKENIZER_DIRS to directories (`:`-separated, absolute)
//! holding tokenizer.json and tokenizer_config.json: each is wrapped in a
//! one-layer Llama checkpoint of zeros and converted.
//!
//! `RLLM_ST_TOKENIZER_DIRS=$PWD/models/hf/Mistral-Nemo-Instruct-2407 cargo test -p rllm-safetensors --test bpe_tokenizer`

use std::path::Path;

const TEXTS: &[&str] = &[
    "Hello, world! It's a test: we'll see how it's split.",
    "HelloWorld camelCaseIdentifier XMLHttpRequest ABCdef",
    "Numbers: 12345 3.14159 2024-11-17 1,000,000 x86_64",
    "fn main() {\n    println!(\"{}\", a/b);\n}\n\n\t\tindented // comment /* block */",
    "https://example.com/path/to/file.html?q=1&r=two#frag",
    "Ça va? Naïve café, Übermäßig, ελληνικά, русский текст",
    "日本語のテキストと中文混合，还有한국어。",
    "Emoji: 🙂🚀👩‍💻 and symbols ±×÷≠ ∑∫",
    "   leading spaces\r\n\r\nCRLF lines   \n\n\n trailing   ",
    "مرحبا بالعالم — עברית — हिन्दी",
];

fn write_checkpoint(dir: &Path, src: &Path, vocab: usize) {
    for f in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"] {
        if src.join(f).is_file() {
            std::fs::copy(src.join(f), dir.join(f)).unwrap();
        }
    }
    let (d, ff, heads, kv) = (64usize, 128usize, 2usize, 1usize);
    let cfg = serde_json::json!({
        "architectures": ["LlamaForCausalLM"], "hidden_size": d, "intermediate_size": ff, "num_hidden_layers": 1,
        "num_attention_heads": heads, "num_key_value_heads": kv, "head_dim": d / heads, "vocab_size": vocab,
        "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "tie_word_embeddings": true, "max_position_embeddings": 4096,
    });
    std::fs::write(dir.join("config.json"), cfg.to_string()).unwrap();
    // BF16 zeros, row-major, outermost first.
    let hd = d / heads;
    let shapes: Vec<(String, Vec<usize>)> = vec![
        ("model.embed_tokens.weight".into(), vec![vocab, d]),
        ("model.norm.weight".into(), vec![d]),
        ("model.layers.0.input_layernorm.weight".into(), vec![d]),
        ("model.layers.0.post_attention_layernorm.weight".into(), vec![d]),
        ("model.layers.0.self_attn.q_proj.weight".into(), vec![heads * hd, d]),
        ("model.layers.0.self_attn.k_proj.weight".into(), vec![kv * hd, d]),
        ("model.layers.0.self_attn.v_proj.weight".into(), vec![kv * hd, d]),
        ("model.layers.0.self_attn.o_proj.weight".into(), vec![d, heads * hd]),
        ("model.layers.0.mlp.gate_proj.weight".into(), vec![ff, d]),
        ("model.layers.0.mlp.up_proj.weight".into(), vec![ff, d]),
        ("model.layers.0.mlp.down_proj.weight".into(), vec![d, ff]),
    ];
    let mut header = serde_json::Map::new();
    let mut off = 0usize;
    for (name, shape) in &shapes {
        let n = shape.iter().product::<usize>() * 2;
        header.insert(name.clone(), serde_json::json!({"dtype": "BF16", "shape": shape, "data_offsets": [off, off + n]}));
        off += n;
    }
    let mut h = serde_json::Value::Object(header).to_string().into_bytes();
    h.resize(h.len().next_multiple_of(8), b' ');
    let mut bytes = (h.len() as u64).to_le_bytes().to_vec();
    bytes.extend(h);
    bytes.resize(bytes.len() + off, 0);
    std::fs::write(dir.join("model.safetensors"), bytes).unwrap();
}

#[test]
fn encodes_like_hugging_face() {
    let Ok(dirs) = std::env::var("RLLM_ST_TOKENIZER_DIRS") else {
        eprintln!("RLLM_ST_TOKENIZER_DIRS not set: skipped");
        return;
    };
    for src in dirs.split(':').filter(|s| !s.is_empty()).map(Path::new) {
        let hf = tokenizers::Tokenizer::from_file(src.join("tokenizer.json")).unwrap();
        let vocab = hf.get_vocab_size(true);
        let dir = std::env::temp_dir().join(format!("rllm-bpe-test-{}-{}", std::process::id(), src.file_name().unwrap().to_string_lossy()));
        std::fs::create_dir_all(&dir).unwrap();
        write_checkpoint(&dir, src, vocab);
        let g = rllm_safetensors::open_model(&dir).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
        let ours = rllm_tokenizer::Tokenizer::from_gguf(&g).unwrap();
        let pre = g.get_str("tokenizer.ggml.pre").unwrap().to_string();
        for text in TEXTS {
            let want: Vec<u32> = hf.encode(*text, false).unwrap().get_ids().to_vec();
            let got = ours.encode_with(text, false).unwrap();
            assert_eq!(got, want, "{} ({pre}): {text:?}", src.display());
        }
        eprintln!("{} ({pre}): {} texts identical", src.display(), TEXTS.len());
    }
}
