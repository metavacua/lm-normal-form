// SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
// SPDX-License-Identifier: AGPL-3.0-or-later
// The llama.cpp runtime of batch 0014, a small program against the C API of the pinned release (llama.h of b11450) and linked to that release's own
// libraries. It loads a GGUF file and, given token ids in a JSON file (the file that every other runtime reads), writes
//   OUT.logits.f32   the logits of every position of every test sequence (little-endian float32, positions x vocabulary), from one decode of each sequence
//   OUT.json         load time, greedy generations (32 new tokens, stopping after a token of `eos`), the negative log-likelihood of the perplexity windows,
//                    prefill speed (512 tokens, five repeats), decode speed (64 tokens after a prompt of 16, three repeats), peak resident memory
// Usage: lmnf-llama MODEL.gguf INPUTS.json OUT FULL [THREADS]      FULL is 1 for everything, 0 for load time and speed alone.
//        lmnf-llama tokenize MODEL.gguf TEXTS.json OUT.json                  the token ids of texts (see tokenize_mode)
// Build: g++ -O2 -std=c++17 -I LLAMA/include -I LLAMA/ggml/include -I LLAMA/vendor lmnf_llama.cpp -o RELEASE/lmnf-llama -LRELEASE -lllama -lggml -lggml-base -Wl,-rpath,'$ORIGIN'
// with the program placed in the release directory, where ggml_backend_load_all() finds the CPU backends.
#include "llama.h"
#include "ggml-backend.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using json = nlohmann::json;
using clk = std::chrono::steady_clock;

struct Run {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    int n_vocab = 0;
    llama_batch batch{};
};

static double secs(clk::time_point a, clk::time_point b) { return std::chrono::duration<double>(b - a).count(); }

static void reset(Run & r) { llama_memory_clear(llama_get_memory(r.ctx), true); }

// Decode `tokens` at positions pos0, pos0 + 1, ...; all = the logits of every position, else of the last one only.
static void decode(Run & r, const std::vector<int> & tokens, int pos0, bool all) {
    const int n = (int) tokens.size();
    r.batch.n_tokens = n;
    for (int i = 0; i < n; ++i) {
        r.batch.token[i] = tokens[i];
        r.batch.pos[i] = pos0 + i;
        r.batch.n_seq_id[i] = 1;
        r.batch.seq_id[i][0] = 0;
        r.batch.logits[i] = (all || i == n - 1) ? 1 : 0;
    }
    if (llama_decode(r.ctx, r.batch) != 0) {
        fprintf(stderr, "llama_decode failed\n");
        exit(2);
    }
}

static int argmax(const float * l, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i) {
        if (l[i] > l[best]) best = i;
    }
    return best;
}

static double nll(const float * l, int n, int target) {
    double m = l[0];
    for (int i = 1; i < n; ++i) m = std::max(m, (double) l[i]);
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += std::exp((double) l[i] - m);
    return -((double) l[target] - m - std::log(s));
}

static double peak_rss_mb() {
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("VmHWM:", 0) == 0) return std::atof(line.c_str() + 6) / 1024.0;
    }
    return 0.0;
}

static std::vector<std::vector<int>> generate(Run & r, const std::vector<std::vector<int>> & prompts, const std::vector<int> & eos) {
    std::vector<std::vector<int>> all;
    for (const auto & p : prompts) {
        reset(r);
        decode(r, p, 0, false);
        int pos = (int) p.size();
        std::vector<int> out;
        for (int step = 0; step < 32; ++step) {
            const int nxt = argmax(llama_get_logits_ith(r.ctx, -1), r.n_vocab);
            out.push_back(nxt);
            if (std::find(eos.begin(), eos.end(), nxt) != eos.end()) break;
            decode(r, {nxt}, pos, false);
            pos += 1;
        }
        all.push_back(out);
    }
    return all;
}

// lmnf-llama tokenize MODEL.gguf TEXTS.json OUT.json: the token ids of every text of TEXTS.json["texts"] (the vocabulary only is loaded; the model's own BOS rule, special
// tokens in the text not parsed), as a list of lists. Used to compare llama.cpp's tokenizer with Hugging Face's.
static int tokenize_mode(char ** argv) {
    ggml_backend_load_all();
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model * model = llama_model_load_from_file(argv[2], mp);
    if (!model) {
        fprintf(stderr, "cannot load %s\n", argv[2]);
        return 3;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::ifstream fi(argv[3]);
    const json texts = json::parse(fi);
    json out = json::array();
    for (const auto & t : texts["texts"]) {
        const std::string s = t.get<std::string>();
        std::vector<llama_token> toks(s.size() + 16);
        int n = llama_tokenize(vocab, s.c_str(), (int) s.size(), toks.data(), (int) toks.size(), llama_vocab_get_add_bos(vocab), false);
        if (n < 0) {
            toks.resize(-n);
            n = llama_tokenize(vocab, s.c_str(), (int) s.size(), toks.data(), (int) toks.size(), llama_vocab_get_add_bos(vocab), false);
        }
        toks.resize(n);
        out.push_back(toks);
    }
    std::ofstream(argv[4]) << out.dump();
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

int main(int argc, char ** argv) {
    if (argc >= 5 && std::string(argv[1]) == "tokenize") return tokenize_mode(argv);
    if (argc < 5) {
        fprintf(stderr, "usage: lmnf-llama MODEL.gguf INPUTS.json OUT FULL [THREADS]\n");
        return 1;
    }
    const std::string model_path = argv[1], inputs_path = argv[2], out = argv[3];
    const bool full = std::string(argv[4]) == "1";
    const int threads = argc > 5 ? std::atoi(argv[5]) : 4;

    std::ifstream fi(inputs_path);
    const json inputs = json::parse(fi);
    const auto & ids = inputs["ids"];
    const std::vector<std::vector<int>> wins = inputs["wins"].get<std::vector<std::vector<int>>>();
    const std::vector<int> eos = ids.contains("eos") ? ids["eos"].get<std::vector<int>>() : std::vector<int>{};

    ggml_backend_load_all();
    llama_backend_init();
    const auto t0 = clk::now();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    Run r;
    r.model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!r.model) {
        fprintf(stderr, "cannot load %s\n", model_path.c_str());
        return 3;
    }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 1024;
    cp.n_batch = 1024;
    cp.n_ubatch = 512;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    r.ctx = llama_init_from_model(r.model, cp);
    if (!r.ctx) {
        fprintf(stderr, "cannot create a context\n");
        return 4;
    }
    r.n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(r.model));
    r.batch = llama_batch_init(1024, 0, 1);
    json res;
    res["load_s"] = secs(t0, clk::now());

    if (full) {
        std::ofstream bin(out + ".logits.f32", std::ios::binary);
        long positions = 0;
        for (const auto & seq : ids["test"]) {
            const std::vector<int> s = seq.get<std::vector<int>>();
            reset(r);
            decode(r, s, 0, true);
            for (int i = 0; i < (int) s.size(); ++i) {
                bin.write(reinterpret_cast<const char *>(llama_get_logits_ith(r.ctx, i)), sizeof(float) * r.n_vocab);
                positions += 1;
            }
        }
        bin.close();
        res["positions"] = positions;
        res["vocab"] = r.n_vocab;

        json gen;
        gen["plain"] = generate(r, ids["plain"].get<std::vector<std::vector<int>>>(), eos);
        if (ids.contains("chat") && !ids["chat"].empty()) gen["chat"] = generate(r, ids["chat"].get<std::vector<std::vector<int>>>(), eos);
        res["gen"] = gen;

        double nll_sum = 0.0;
        long ntok = 0;
        for (const auto & w : wins) {
            reset(r);
            decode(r, w, 0, true);
            for (int i = 0; i + 1 < (int) w.size(); ++i) {
                nll_sum += nll(llama_get_logits_ith(r.ctx, i), r.n_vocab, w[i + 1]);
                ntok += 1;
            }
        }
        res["ppl_nll"] = nll_sum;
        res["ppl_tokens"] = ntok;
    }

    const std::vector<int> & w0 = wins[0];
    reset(r);
    decode(r, w0, 0, false);                            // warm-up
    std::vector<double> pre, dec;
    for (int k = 0; k < 5; ++k) {
        reset(r);
        const auto a = clk::now();
        decode(r, w0, 0, false);
        pre.push_back((double) w0.size() / secs(a, clk::now()));
    }
    std::vector<int> p16 = ids["plain"][0].get<std::vector<int>>();
    p16.insert(p16.end(), w0.begin(), w0.end());
    p16.resize(16);
    for (int k = 0; k < 3; ++k) {
        reset(r);
        decode(r, p16, 0, false);
        int pos = (int) p16.size();
        const auto a = clk::now();
        for (int step = 0; step < 64; ++step) {
            const int nxt = argmax(llama_get_logits_ith(r.ctx, -1), r.n_vocab);
            decode(r, {nxt}, pos, false);
            pos += 1;
        }
        dec.push_back(64.0 / secs(a, clk::now()));
    }
    res["prefill_tps"] = pre;
    res["decode_tps"] = dec;
    res["peak_rss_mb"] = peak_rss_mb();
    std::ofstream(out + ".json") << res.dump(1);

    llama_batch_free(r.batch);
    llama_free(r.ctx);
    llama_model_free(r.model);
    llama_backend_free();
    return 0;
}
