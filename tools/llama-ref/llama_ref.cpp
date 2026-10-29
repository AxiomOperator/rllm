// Reference dump from llama.cpp, for `rllm compare-ref`.
//
// Tokenizes a prompt with llama.cpp, runs it, greedily generates N more
// tokens, and writes every token id plus the full logits at every position:
//
//   "RLLMREF1" | u32 n_vocab | u32 n_prompt | u32 n_total
//   | u32 tokens[n_total] | f32 logits[n_total][n_vocab]
//
// Row i holds the logits after consuming tokens[0..=i].
//
// usage: llama_ref <model.gguf> <prompt.txt> <out.bin> [n_gen=16] [add_special=1] [exact_attn=0] [n_gpu_layers=0]
//
// exact_attn=1 uses an f32 KV cache and disables flash attention, so the only
// reduced-precision step left is llama.cpp's activation quantization in
// matmuls (which rllm can mimic with RLLM_CPU_LLAMACPP_NUMERICS=1).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "llama.h"

static void die(const char * msg) {
    fprintf(stderr, "llama_ref: %s\n", msg);
    exit(1);
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <model.gguf> <prompt.txt> <out.bin> [n_gen=16] [add_special=1] [exact_attn=0] [n_gpu_layers=0]\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    const int n_gen = argc > 4 ? atoi(argv[4]) : 16;
    const bool add_special = argc > 5 ? atoi(argv[5]) != 0 : true;
    const bool exact_attn = argc > 6 ? atoi(argv[6]) != 0 : false;
    const int n_gpu_layers = argc > 7 ? atoi(argv[7]) : 0;

    std::ifstream in(argv[2], std::ios::binary);
    if (!in) die("cannot read prompt file");
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string prompt = ss.str();

    llama_log_set([](ggml_log_level level, const char * text, void *) {
        if (level >= GGML_LOG_LEVEL_ERROR) fputs(text, stderr);
    }, nullptr);
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;  // 0 = CPU only (needs a GPU build of llama.cpp otherwise)
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) die("failed to load model");
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    const int n_need = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, add_special, true);
    std::vector<llama_token> tokens(n_need);
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), tokens.data(), n_need, add_special, true) != n_need) {
        die("tokenize failed");
    }
    const int n_prompt = (int) tokens.size();

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = n_prompt + n_gen + 16;
    cparams.n_batch = n_prompt;
    cparams.n_ubatch = n_prompt;
    cparams.no_perf = true;
    if (exact_attn) {
        cparams.type_k = GGML_TYPE_F32;
        cparams.type_v = GGML_TYPE_F32;
        cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) die("failed to create context");

    std::vector<float> logits;
    logits.reserve((size_t) (n_prompt + n_gen) * n_vocab);

    llama_batch batch = llama_batch_init(n_prompt, 0, 1);
    for (int i = 0; i < n_prompt; i++) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 1;
    }
    batch.n_tokens = n_prompt;
    if (llama_decode(ctx, batch) != 0) die("decode (prompt) failed");
    for (int i = 0; i < n_prompt; i++) {
        const float * row = llama_get_logits_ith(ctx, i);
        logits.insert(logits.end(), row, row + n_vocab);
    }

    for (int k = 0; k < n_gen; k++) {
        const float * last = logits.data() + logits.size() - n_vocab;
        llama_token next = 0;
        for (int v = 1; v < n_vocab; v++) {
            if (last[v] > last[next]) next = v;
        }
        tokens.push_back(next);
        batch.n_tokens = 1;
        batch.token[0] = next;
        batch.pos[0] = (llama_pos) tokens.size() - 1;
        batch.logits[0] = 1;
        if (llama_decode(ctx, batch) != 0) die("decode (generate) failed");
        const float * row = llama_get_logits_ith(ctx, 0);
        logits.insert(logits.end(), row, row + n_vocab);
    }

    FILE * out = fopen(argv[3], "wb");
    if (!out) die("cannot open output");
    const uint32_t hdr[3] = { (uint32_t) n_vocab, (uint32_t) n_prompt, (uint32_t) tokens.size() };
    fwrite("RLLMREF1", 1, 8, out);
    fwrite(hdr, sizeof(uint32_t), 3, out);
    std::vector<uint32_t> ids(tokens.begin(), tokens.end());
    fwrite(ids.data(), sizeof(uint32_t), ids.size(), out);
    fwrite(logits.data(), sizeof(float), logits.size(), out);
    fclose(out);

    // Generated text, for eyeballing.
    std::string text;
    for (size_t i = n_prompt; i < tokens.size(); i++) {
        char buf[256];
        const int n = llama_token_to_piece(vocab, tokens[i], buf, sizeof(buf), 0, true);
        if (n > 0) text.append(buf, n);
    }
    fprintf(stderr, "llama_ref: %d prompt + %d generated tokens, vocab %d\n", n_prompt, n_gen, n_vocab);
    printf("%s\n", text.c_str());

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
