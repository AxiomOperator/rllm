//! Where a checkpoint's tokenizer keys differ from a GGUF's (debug aid):
//! `tok_diff DIR GGUF`.
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let ours = rllm_safetensors::open_model(&a[1]).unwrap();
    let theirs = rllm_gguf::Gguf::open(&a[2]).unwrap();
    for key in ["tokenizer.ggml.tokens", "tokenizer.ggml.token_type", "tokenizer.ggml.merges"] {
        let (x, y) = (ours.get_array(key).unwrap(), theirs.get_array(key).unwrap());
        let diffs: Vec<usize> = (0..x.len().max(y.len())).filter(|&i| format!("{:?}", x.get(i)) != format!("{:?}", y.get(i))).collect();
        println!("{key}: {} vs {} entries, {} differ", x.len(), y.len(), diffs.len());
        for &i in diffs.iter().take(24) {
            let tok = |g: &rllm_gguf::Gguf| g.get_array("tokenizer.ggml.tokens").ok().and_then(|t| t.get(i).map(|v| format!("{v:?}"))).unwrap_or_default();
            println!("  [{i}] ours {:?} theirs {:?}   (token ours {} theirs {})", x.get(i), y.get(i), tok(&ours), tok(&theirs));
        }
    }
}
