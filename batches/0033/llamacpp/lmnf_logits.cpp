// SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
// SPDX-License-Identifier: AGPL-3.0-or-later
// Batch 0033, cell 15: the last-position logits of one token sequence from a GGUF file, through the C API of the pinned llama.cpp release (b11450),
// with a float32 key-value cache and flash attention off, as the float32 mode of batch 0014's driver. One decode, nothing else.
// Usage: lmnf-logits MODEL.gguf IDS OUT.f32 [THREADS]      IDS are comma-separated token ids; OUT.f32 receives n_vocab little-endian float32 logits.
// Build: g++ -O2 -std=c++17 -I LLAMA/include -I LLAMA/ggml/include -I LLAMA/vendor lmnf_logits.cpp -o RELEASE/lmnf-logits -LRELEASE -lllama -lggml -lggml-base -Wl,-rpath,'$ORIGIN'
#include "llama.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: lmnf-logits MODEL.gguf IDS OUT.f32 [THREADS]\n");
        return 1;
    }
    std::vector<int> ids;
    {
        const std::string s = argv[2];
        size_t p = 0;
        while (p < s.size()) {
            size_t q = s.find(',', p);
            if (q == std::string::npos) q = s.size();
            ids.push_back(std::atoi(s.substr(p, q - p).c_str()));
            p = q + 1;
        }
    }
    const int threads = argc > 4 ? std::atoi(argv[4]) : 4;
    ggml_backend_load_all();
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        fprintf(stderr, "cannot load %s\n", argv[1]);
        return 3;
    }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 64;
    cp.n_batch = 64;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    cp.type_k = GGML_TYPE_F32;
    cp.type_v = GGML_TYPE_F32;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "cannot create the context\n");
        return 4;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    llama_batch batch = llama_batch_init(64, 0, 1);
    const int n = (int) ids.size();
    batch.n_tokens = n;
    for (int i = 0; i < n; ++i) {
        batch.token[i] = ids[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == n - 1) ? 1 : 0;
    }
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "llama_decode failed\n");
        return 2;
    }
    const float * l = llama_get_logits_ith(ctx, -1);
    FILE * f = fopen(argv[3], "wb");
    fwrite(l, sizeof(float), (size_t) n_vocab, f);
    fclose(f);
    int best = 0;
    for (int i = 1; i < n_vocab; ++i) {
        if (l[i] > l[best]) best = i;
    }
    printf("n_vocab %d argmax %d logit %.6f\n", n_vocab, best, (double) l[best]);
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
