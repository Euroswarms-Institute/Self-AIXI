// Dev-only oracle: exact bit-token logits from llama.cpp for an explicit
// token-id stream (no tokenizer on either side — both engines consume the
// same raw ids, so this is a pure forward-pass comparison).
//
//   oracle_probe <model.gguf> <id0,id1,...>
//
// Prints one line per position: "<i> <logit_id15> <logit_id16>".
// Built by scripts/oracle_check.sh; never a dependency of the Rust crate.
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s model.gguf id0,id1,... [bit0_id bit1_id]\n", argv[0]);
        return 2;
    }
    const int bit0 = argc > 4 ? atoi(argv[3]) : 15;
    const int bit1 = argc > 4 ? atoi(argv[4]) : 16;
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        fprintf(stderr, "model load failed\n");
        return 1;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512;
    cp.n_batch = 64;
    cp.n_threads = 4;
    cp.n_threads_batch = 4;
    // Our engine keeps K/V in f32; llama.cpp defaults to f16 — align them so
    // the comparison isolates real graph differences from cache rounding.
    cp.type_k = GGML_TYPE_F32;
    cp.type_v = GGML_TYPE_F32;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "context init failed\n");
        return 1;
    }

    std::vector<llama_token> tokens;
    for (char * tok = strtok(argv[2], ","); tok; tok = strtok(nullptr, ",")) {
        tokens.push_back((llama_token) atoi(tok));
    }

    for (size_t i = 0; i < tokens.size(); ++i) {
        llama_batch batch = llama_batch_get_one(&tokens[i], 1);
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "decode failed at %zu\n", i);
            return 1;
        }
        const float * logits = llama_get_logits_ith(ctx, -1);
        printf("%zu %.6f %.6f\n", i, (double) logits[bit0], (double) logits[bit1]);
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
