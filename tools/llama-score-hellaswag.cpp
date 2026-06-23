#include "llama.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

static void quiet_log(enum ggml_log_level, const char *, void *) {}

// Minimal JSON string helpers for the fixed JSONL shape emitted by LocalMaxxing
// eval shards: { question_id, input, choices:[...], gold:"A" }. This intentionally
// avoids adding a third-party dependency to the CLI repo.
static bool extract_json_string(const std::string & s, const std::string & key, std::string & out) {
    const std::string needle = "\"" + key + "\"";
    size_t p = s.find(needle);
    if (p == std::string::npos) return false;
    p = s.find(':', p + needle.size());
    if (p == std::string::npos) return false;
    p = s.find('"', p + 1);
    if (p == std::string::npos) return false;
    std::string r;
    bool esc = false;
    for (size_t i = p + 1; i < s.size(); ++i) {
        char c = s[i];
        if (esc) {
            switch (c) {
                case '"': r.push_back('"'); break;
                case '\\': r.push_back('\\'); break;
                case '/': r.push_back('/'); break;
                case 'b': r.push_back('\b'); break;
                case 'f': r.push_back('\f'); break;
                case 'n': r.push_back('\n'); break;
                case 'r': r.push_back('\r'); break;
                case 't': r.push_back('\t'); break;
                case 'u':
                    // Shards are UTF-8; if unicode escapes appear, preserve a placeholder
                    // rather than corrupting JSON parsing. HF rows used here do not use them.
                    r.push_back('?');
                    i += std::min<size_t>(4, s.size() - i - 1);
                    break;
                default: r.push_back(c); break;
            }
            esc = false;
        } else if (c == '\\') {
            esc = true;
        } else if (c == '"') {
            out = r;
            return true;
        } else {
            r.push_back(c);
        }
    }
    return false;
}

static bool extract_choices(const std::string & s, std::vector<std::string> & out) {
    const std::string needle = "\"choices\"";
    size_t p = s.find(needle);
    if (p == std::string::npos) return false;
    p = s.find('[', p + needle.size());
    if (p == std::string::npos) return false;
    out.clear();
    for (;;) {
        p = s.find_first_of("\"]", p + 1);
        if (p == std::string::npos) return false;
        if (s[p] == ']') return out.size() == 4;
        std::string r;
        bool esc = false;
        for (size_t i = p + 1; i < s.size(); ++i) {
            char c = s[i];
            if (esc) {
                switch (c) {
                    case '"': r.push_back('"'); break;
                    case '\\': r.push_back('\\'); break;
                    case '/': r.push_back('/'); break;
                    case 'b': r.push_back('\b'); break;
                    case 'f': r.push_back('\f'); break;
                    case 'n': r.push_back('\n'); break;
                    case 'r': r.push_back('\r'); break;
                    case 't': r.push_back('\t'); break;
                    case 'u': r.push_back('?'); i += std::min<size_t>(4, s.size() - i - 1); break;
                    default: r.push_back(c); break;
                }
                esc = false;
            } else if (c == '\\') {
                esc = true;
            } else if (c == '"') {
                out.push_back(r);
                p = i;
                break;
            } else {
                r.push_back(c);
            }
        }
        if (out.size() == 4) return true;
    }
}

static std::string json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 16);
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out.push_back((char)c);
                }
        }
    }
    return out;
}

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text, bool add_special) {
    int32_t n = llama_tokenize(vocab, text.c_str(), (int32_t)text.size(), nullptr, 0, add_special, false);
    if (n < 0) n = -n;
    std::vector<llama_token> tokens(n);
    int32_t got = llama_tokenize(vocab, text.c_str(), (int32_t)text.size(), tokens.data(), (int32_t)tokens.size(), add_special, false);
    if (got < 0) throw std::runtime_error("tokenize failed");
    tokens.resize(got);
    return tokens;
}

static double logprob_of(const float * logits, int n_vocab, llama_token tok) {
    float mx = logits[0];
    for (int i = 1; i < n_vocab; ++i) mx = std::max(mx, logits[i]);
    double sum = 0.0;
    for (int i = 0; i < n_vocab; ++i) sum += std::exp((double)logits[i] - mx);
    return (double)logits[tok] - mx - std::log(sum);
}

static double score_choice(llama_context * ctx, const llama_vocab * vocab, int n_vocab, const std::string & context, const std::string & choice, int n_batch) {
    std::string full = context + " " + choice;
    std::vector<llama_token> full_toks = tokenize(vocab, full, true);
    std::vector<llama_token> ctx_toks = tokenize(vocab, context, true);
    if (full_toks.size() < 2 || ctx_toks.empty() || full_toks.size() <= ctx_toks.size()) {
        throw std::runtime_error("bad tokenization for continuation");
    }
    if ((int)full_toks.size() > llama_n_ctx(ctx)) {
        throw std::runtime_error("context + continuation exceeds n_ctx");
    }

    llama_memory_clear(llama_get_memory(ctx), true);
    llama_batch batch = llama_batch_init((int32_t)full_toks.size(), 0, 1);
    for (int32_t i = 0; i < (int32_t)full_toks.size(); ++i) {
        batch.token[i] = full_toks[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        // Need logits after every token from the last context token through the
        // penultimate continuation token, because those predict continuation tokens.
        batch.logits[i] = (i >= (int32_t)ctx_toks.size() - 1 && i < (int32_t)full_toks.size() - 1) ? 1 : 0;
    }
    batch.n_tokens = (int32_t)full_toks.size();
    (void)n_batch;
    if (llama_decode(ctx, batch) != 0) {
        llama_batch_free(batch);
        throw std::runtime_error("llama_decode failed");
    }

    double sum = 0.0;
    int count = 0;
    for (int32_t pos = (int32_t)ctx_toks.size(); pos < (int32_t)full_toks.size(); ++pos) {
        const float * logits = llama_get_logits_ith(ctx, pos - 1);
        if (!logits) {
            llama_batch_free(batch);
            throw std::runtime_error("missing logits");
        }
        sum += logprob_of(logits, n_vocab, full_toks[pos]);
        count++;
    }
    llama_batch_free(batch);
    return count > 0 ? sum / (double)count : -INFINITY; // token-normalized, as llama-perplexity --hellaswag
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string input_path;
    int n_gpu_layers = 999;
    int n_ctx = 4096;
    int n_batch = 512;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char * name) -> const char * {
            if (i + 1 >= argc) { std::cerr << name << " requires a value\n"; std::exit(2); }
            return argv[++i];
        };
        if (a == "--model" || a == "-m") model_path = need(a.c_str());
        else if (a == "--input" || a == "-i") input_path = need(a.c_str());
        else if (a == "--ctx-size") n_ctx = std::atoi(need(a.c_str()));
        else if (a == "--batch-size") n_batch = std::atoi(need(a.c_str()));
        else if (a == "--gpu-layers" || a == "--n-gpu-layers") n_gpu_layers = std::atoi(need(a.c_str()));
        else if (a == "--help" || a == "-h") {
            std::cout << "usage: lmx-llama-score-hellaswag --model model.gguf --input shard.jsonl [--ctx-size 4096]\n";
            return 0;
        } else {
            std::cerr << "unknown arg: " << a << "\n";
            return 2;
        }
    }
    if (model_path.empty() || input_path.empty()) {
        std::cerr << "--model and --input are required\n";
        return 2;
    }

    llama_log_set(quiet_log, nullptr);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = n_gpu_layers;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { std::cerr << "failed to load model\n"; return 1; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = n_batch;
    cp.n_ubatch = n_batch;
    cp.no_perf = true;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { std::cerr << "failed to create context\n"; return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    std::ifstream in(input_path);
    if (!in) { std::cerr << "failed to open input\n"; return 2; }
    std::string line;
    int idx = 0;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::string qid, input, gold;
        std::vector<std::string> choices;
        bool ok = extract_json_string(line, "question_id", qid) && extract_json_string(line, "input", input) && extract_json_string(line, "gold", gold) && extract_choices(line, choices);
        std::vector<double> scores(4, -INFINITY);
        std::string err;
        int best = 0;
        if (ok) {
            try {
                for (int c = 0; c < 4; ++c) scores[c] = score_choice(ctx, vocab, n_vocab, input, choices[c], n_batch);
                for (int c = 1; c < 4; ++c) if (scores[c] > scores[best]) best = c;
            } catch (const std::exception & e) {
                err = e.what();
            }
        } else {
            err = "failed to parse row";
        }
        char pred = char('A' + best);
        bool pass = err.empty() && !gold.empty() && pred == gold[0];
        std::cout << "{\"question_id\":\"" << json_escape(qid) << "\",\"itemIndex\":" << idx
                  << ",\"predicted\":\"" << pred << "\",\"gold\":\"" << json_escape(gold) << "\",\"pass\":" << (pass ? "true" : "false")
                  << ",\"choices\":[\"" << json_escape(choices[0]) << "\",\"" << json_escape(choices[1]) << "\",\"" << json_escape(choices[2]) << "\",\"" << json_escape(choices[3]) << "\"]"
                  << ",\"scoreNormalization\":\"token_avg\""
                  << ",\"scores\":{\"A\":" << scores[0] << ",\"B\":" << scores[1] << ",\"C\":" << scores[2] << ",\"D\":" << scores[3] << "}";
        if (!err.empty()) std::cout << ",\"error\":\"" << json_escape(err) << "\"";
        std::cout << "}\n";
        idx++;
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
