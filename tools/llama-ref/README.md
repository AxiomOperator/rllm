# llama-ref

Dumps llama.cpp's tokens and logits for a prompt, so `rllm compare-ref` can
check rllm against llama.cpp position by position.

```sh
# one-time: CPU-only llama.cpp build in third_party/ (git-ignored)
git clone --depth 1 https://github.com/ggml-org/llama.cpp third_party/llama.cpp
cmake -S third_party/llama.cpp -B third_party/llama.cpp/build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TESTS=OFF
cmake --build third_party/llama.cpp/build --target llama -j 8

L=third_party/llama.cpp
g++ -O2 -std=c++17 -I$L/include -I$L/ggml/include tools/llama-ref/llama_ref.cpp \
    -L$L/build/bin -lllama -lggml -lggml-base -Wl,-rpath,$PWD/$L/build/bin -o $L/build/bin/llama_ref

# dump: prompt + 16 greedy tokens. add_special=0 if the prompt already starts with BOS
# (e.g. rendered with `rllm run --print-prompt`).
$L/build/bin/llama_ref model.gguf prompt.txt ref.bin 16 [add_special=1] [exact_attn=0] [n_gpu_layers=0]
# For llama.cpp's GPU backend, link against a CUDA build (e.g. build-cuda/) and pass n_gpu_layers=99.

# compare (also checks tokenization when --prompt-file is given)
rllm compare-ref model.gguf ref.bin --prompt-file prompt.txt [-d cpu|cuda]
```

**Reading the results.** llama.cpp's CPU path trades precision for speed:
- It quantizes activations to 8 bits before quantized matmuls.
- It stores the KV cache in f16.
- Its CPU flash attention accumulates in f16.

rllm keeps all of these in f32, so the two never match exactly. Judge rllm's
difference against llama.cpp's own *noise floor*: dump twice, once with
`exact_attn=0` and once with `exact_attn=1`, and compare those two dumps.
`exact_attn=1` uses an f32 KV cache with flash attention off.

`RLLM_CPU_LLAMACPP_NUMERICS=1` makes rllm's CPU backend round activations the
way llama.cpp does. Use it to check that a remaining gap is only numerics.
