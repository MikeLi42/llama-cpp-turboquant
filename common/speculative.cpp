#include "speculative.h"

#include "common.h"
#include "ggml.h"
#include "llama.h"
#include "../src/llama-ext.h"
#include "log.h"
#include "ngram-cache.h"
#include "ngram-map.h"
#include "ngram-mod.h"
#include "sampling.h"

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <map>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5


const std::vector<enum common_speculative_type> common_speculative_types = {
    COMMON_SPECULATIVE_TYPE_NONE,
    COMMON_SPECULATIVE_TYPE_DRAFT,
    COMMON_SPECULATIVE_TYPE_EAGLE3,
    COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE,
    COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K,
    COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V,
    COMMON_SPECULATIVE_TYPE_NGRAM_MOD,
    COMMON_SPECULATIVE_TYPE_NGRAM_CACHE,
    COMMON_SPECULATIVE_TYPE_MTP,
    COMMON_SPECULATIVE_TYPE_NEXTN
};

const std::map<std::string, enum common_speculative_type> common_speculative_type_from_name_map = {
    {"none",          COMMON_SPECULATIVE_TYPE_NONE},
    {"draft",         COMMON_SPECULATIVE_TYPE_DRAFT},
    {"eagle3",        COMMON_SPECULATIVE_TYPE_EAGLE3},
    {"ngram_simple",  COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE},
    {"ngram_map_k",   COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K},
    {"ngram_map_k4v", COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V},
    {"ngram_mod",     COMMON_SPECULATIVE_TYPE_NGRAM_MOD},
    {"ngram_cache",   COMMON_SPECULATIVE_TYPE_NGRAM_CACHE},
    {"mtp",           COMMON_SPECULATIVE_TYPE_MTP},
    {"nextn",         COMMON_SPECULATIVE_TYPE_NEXTN}
};

struct common_speculative_config {
    common_speculative_type type;
    common_params_speculative params;

    common_speculative_config(common_speculative_type t,
            const common_params_speculative & p = common_params_speculative{}) : type(t), params(p) {}
};

static bool common_speculative_mtp_arch_ok(const llama_model * model_tgt, const llama_model * model_dft) {
    return std::strcmp(llama_model_arch_str(model_tgt), "gemma4") == 0
        && std::strcmp(llama_model_arch_str(model_dft), "gemma4_assistant") == 0;
}

// MTP-specific vocab compatibility:
// the assistant (draft) only predicts the next token id from the same SentencePiece
// vocabulary; stop-condition / chat-template special tokens are owned by the target.
// Therefore we only require identical vocab_type, vocab size (within tolerance) and
// per-token text equality, and intentionally skip bos/eos id and add_bos/add_eos
// checks (target may be chat-tuned with eos=<end_of_turn>=106, draft with eos=<eos>=1).
static bool common_speculative_are_compatible_mtp(
    const llama_model * model_tgt,
    const llama_model * model_dft) {
    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);

    if (llama_vocab_type(vocab_tgt) != llama_vocab_type(vocab_dft)) {
        return false;
    }

    const int n_vocab_tgt = llama_vocab_n_tokens(vocab_tgt);
    const int n_vocab_dft = llama_vocab_n_tokens(vocab_dft);
    const int vocab_diff  = n_vocab_tgt > n_vocab_dft
        ? n_vocab_tgt - n_vocab_dft
        : n_vocab_dft - n_vocab_tgt;

    if (vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
        return false;
    }

    for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
        const char * t_tgt = llama_vocab_get_text(vocab_tgt, i);
        const char * t_dft = llama_vocab_get_text(vocab_dft, i);
        if (std::strcmp(t_tgt, t_dft) != 0) {
            return false;
        }
    }

    return true;
}

static bool common_speculative_are_compatible(
    const llama_model * model_tgt,
    const llama_model * model_dft) {
    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);

    const bool vocab_type_tgt = llama_vocab_type(vocab_tgt);
    LOG_DBG("%s: vocab_type tgt: %d\n", __func__, vocab_type_tgt);

    const bool vocab_type_dft = llama_vocab_type(vocab_dft);
    LOG_DBG("%s: vocab_type dft: %d\n", __func__, vocab_type_dft);

    if (vocab_type_tgt != vocab_type_dft) {
        LOG_WRN("%s: draft model vocab type must match target model to use speculation but "
                "vocab_type_dft = %d while vocab_type_tgt = %d\n", __func__, vocab_type_dft, vocab_type_tgt);
        return false;
    }

    if (llama_vocab_get_add_bos(vocab_tgt) != llama_vocab_get_add_bos(vocab_dft) ||
        (llama_vocab_get_add_bos(vocab_tgt) && llama_vocab_bos(vocab_tgt) != llama_vocab_bos(vocab_dft))) {
        LOG_WRN("%s: draft model bos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n",
                __func__,
                llama_vocab_get_add_bos(vocab_tgt), llama_vocab_get_add_bos(vocab_dft),
                llama_vocab_bos(vocab_tgt), llama_vocab_bos(vocab_dft));
        return false;
    }

    if (llama_vocab_get_add_eos(vocab_tgt) != llama_vocab_get_add_eos(vocab_dft) ||
        (llama_vocab_get_add_eos(vocab_tgt) && llama_vocab_eos(vocab_tgt) != llama_vocab_eos(vocab_dft))) {
        LOG_WRN("%s: draft model eos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n",
                __func__,
                llama_vocab_get_add_eos(vocab_tgt), llama_vocab_get_add_eos(vocab_dft),
                llama_vocab_eos(vocab_tgt), llama_vocab_eos(vocab_dft));
        return false;
    }

    {
        const int n_vocab_tgt = llama_vocab_n_tokens(vocab_tgt);
        const int n_vocab_dft = llama_vocab_n_tokens(vocab_dft);
        const int vocab_diff  = n_vocab_tgt > n_vocab_dft
            ? n_vocab_tgt - n_vocab_dft
            : n_vocab_dft - n_vocab_tgt;

        if (vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
            LOG_DBG("%s: draft model vocab must closely match target model to use speculation but ", __func__);
            LOG_DBG("target vocab size %d does not match draft vocab size %d - difference %d, max allowed %d\n",
                    n_vocab_tgt, llama_vocab_n_tokens(vocab_dft), vocab_diff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
            return false;
        }

        for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
            const char * token_text_tgt = llama_vocab_get_text(vocab_tgt, i);
            const char * token_text_dft = llama_vocab_get_text(vocab_dft, i);

            if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
                LOG_DBG("%s: draft model vocab must match target model to use speculation but ", __func__);
                LOG_DBG("token %d content differs - target '%s', draft '%s'\n", i,
                        common_token_to_piece(vocab_tgt, i).c_str(),
                        common_token_to_piece(vocab_dft, i).c_str());
                return false;
            }
        }
    }

    return true;
}

using common_speculative_draft_params_vec = std::vector<common_speculative_draft_params>;

// state of an implementation of speculative decoding
//
// each implementation has a unique type and a state that is implementation-specific
// in a subclass of common_speculative_impl
struct common_speculative_impl {
    const common_speculative_type type;

    uint32_t n_seq;

    size_t n_call_begin  = 0; // number of times this implementation was called for refresh.
    size_t n_call_draft  = 0; // number of times this implementation was called for generation.
    size_t n_call_accept = 0; // number of times this implementation was called for accumulation.

    size_t n_gen_drafts = 0; // number of times a draft or part was generated by this implementation.
    size_t n_acc_drafts = 0; // number of times a draft or part was accepted by the target model.
    size_t n_gen_tokens = 0; // number of tokens generated by this implementation.
    size_t n_acc_tokens = 0; // number of tokens accepted by the target model.

    // TODO: track performance of most recent calls
    const bool gen_perf = true; // whether to generate performance stats.

    int64_t t_begin_us  = 0; // total time spent in refresh of this implementation in microseconds.
    int64_t t_draft_us  = 0; // total time spent in generating drafts in this implementation in microseconds.
    int64_t t_accept_us = 0; // total time spent in accumulation of this implementation in microseconds.

    common_speculative_impl(common_speculative_type type, uint32_t n_seq) : type(type), n_seq(n_seq) {}

    virtual ~common_speculative_impl() = default;

    virtual void begin(llama_seq_id seq_id, const llama_tokens & prompt) = 0;

    virtual bool process(const llama_batch & batch) = 0;

    virtual void accept(uint16_t n_accepted) = 0;

    // Optional hook: MTP submits async draft work after accept for overlap with the next draft() wait.
    virtual void prepare_next(llama_token id_last) {
        GGML_UNUSED(id_last);
    }

    // Optional hook: drain any in-flight async work (prepare_next) and discard.
    virtual void cancel() {}
};

struct common_speculative_state_draft : public common_speculative_state {
    llama_context * ctx_tgt; // only used for retokenizing from ctx_dft
    llama_context * ctx_dft;

    common_sampler * smpl;

    llama_batch  batch;
    llama_tokens prompt_dft;

    bool vocab_cmpt = true; // whether retokenization is needed
    std::unordered_map<std::string, std::string> vocab_map;

    common_speculative_state_draft(
            enum common_speculative_type type,
            llama_context * ctx_tgt,
            llama_context * ctx_dft,
            const std::vector<std::pair<std::string, std::string>> & replacements)
        : common_speculative_state(type)
        , ctx_tgt(ctx_tgt)
        , ctx_dft(ctx_dft)
    {
        auto * ctx_dft = this->params.ctx_dft;
        auto * ctx_tgt = this->params.ctx_tgt;

        batch = llama_batch_init(llama_n_batch(ctx_dft), 0, 1);

        // TODO: optimize or pass from outside?
        // {
        //     common_params_sampling params;
        //     params.no_perf = false;
        //
        //     params.top_k = 40;
        //     params.top_p = 0.9;
        //
        //     params.samplers = {
        //         COMMON_SAMPLER_TYPE_TOP_K,
        //         COMMON_SAMPLER_TYPE_TOP_P,
        //         COMMON_SAMPLER_TYPE_INFILL,
        //     };
        //
        //     result->smpl = common_sampler_init(llama_get_model(ctx_dft), params);
        // }

        smpls.resize(n_seq);
        for (auto & smpl : smpls) {
            common_params_sampling params;
            params.no_perf = false;
            params.top_k = 10;
            params.samplers = {
                COMMON_SAMPLER_TYPE_TOP_K,
            };

            smpl.reset(common_sampler_init(llama_get_model(ctx_dft), params));
        }

        const bool vocab_cmpt = common_speculative_are_compatible(llama_get_model(ctx_tgt), llama_get_model(ctx_dft));
        LOG_DBG("%s: vocab_cmpt = %d\n", __func__, vocab_cmpt);

        if (!vocab_cmpt) {
            LOG_ERR("%s: the target and draft vocabs are not compatible\n", __func__);

            throw std::runtime_error("draft model vocab type must match target model to use speculation");
        }

        if (n_seq != llama_n_seq_max(ctx_dft)) {
            LOG_ERR("%s: n_seq mismatch: %d != %d\n", __func__, n_seq, llama_n_seq_max(ctx_dft));

            throw std::runtime_error("the draft model number of sequences is incompatible with the speculative n_seq");
        }
    }

    ~common_speculative_impl_draft_simple() override {
        llama_batch_free(batch);
    }

    void begin(const llama_tokens & prompt) override {
        GGML_UNUSED(prompt);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        auto * spec = this;

        auto & batch      = spec->batch;
        auto & ctx_tgt    = spec->ctx_tgt;
        auto & ctx_dft    = spec->ctx_dft;
        auto & smpl       = spec->smpl;
        auto & prompt_dft = spec->prompt_dft;

        auto * mem_dft = llama_get_memory(ctx_dft);

        int reuse_i = 0;
        int reuse_n = 0;

        const int n_ctx = llama_n_ctx(ctx_dft) - params.n_max;

        llama_tokens prompt_cnv;
        if (!spec->vocab_cmpt) {
            std::string text;

            text = common_detokenize(ctx_tgt, prompt_tgt, true);
            text = replace_to_dft(text);

            LOG_DBG("%s: main->draft detokenized string: '%s'\n", __func__, text.c_str());

            prompt_cnv = common_tokenize(ctx_dft, text, false, true);

            // convert id_last to draft vocab. llama_detokenize is called directly to avoid an allocation
            const auto * model_tgt = llama_get_model(ctx_tgt);
            const auto * vocab_tgt = llama_model_get_vocab(model_tgt);

            int32_t n_chars = llama_detokenize(vocab_tgt, &id_last, 1, nullptr, 0, false, false);
            GGML_ASSERT(n_chars < 0 && "failed to detokenize id_last");

            text.resize(-n_chars);
            llama_detokenize(vocab_tgt, &id_last, 1, text.data(), text.size(), false, false);
            text = replace_to_dft(text);

            LOG_DBG("main->draft detokenized id_last(%d): '%s'\n", id_last, text.c_str());
            id_last = common_tokenize(ctx_dft, text, false, true)[0];
        }

        const llama_tokens & prompt_cur = spec->vocab_cmpt ? prompt_tgt : prompt_cnv;

        const int i_start = std::max<int>(0, (int) prompt_cur.size() - n_ctx);

        // reuse as much as possible from the old draft context
        // ideally, the draft context should be as big as the target context and we will always reuse the entire prompt
        for (int i = 0; i < (int) prompt_dft.size(); ++i) {
            int cur = 0;
            while (i_start + cur < (int) prompt_cur.size() &&
                    i       + cur < (int) prompt_dft.size() &&
                    prompt_cur[i_start + cur] == prompt_dft[i + cur]) {
                cur++;
            }

            if ((cur >= 256 || n_ctx >= (int) prompt_cur.size()) && cur > reuse_n) {
                reuse_i = i;
                reuse_n = cur;
            }
        }

        LOG_DBG("%s: reuse_i = %d, reuse_n = %d, prompt = %d\n", __func__, reuse_i, reuse_n, (int) prompt_dft.size());

        result.clear();
        result.reserve(params.n_max);

        if (reuse_n == 0) {
            llama_memory_clear(mem_dft, false);
            prompt_dft.clear();
        } else {
            // this happens when a previous draft has been discarded (for example, due to being too small), but the
            // target model agreed with it. in this case, we simply pass back the previous results to save compute
            if (reuse_i + reuse_n < (int) prompt_dft.size() && prompt_dft[reuse_i + reuse_n] == id_last) {
                for (int i = reuse_i + reuse_n + 1; i < (int) prompt_dft.size(); ++i) {
                    result.push_back(prompt_dft[i]);

                    if (params.n_max <= (int) result.size()) {
                        break;
                    }
                }

                return;
            }

            if (reuse_i > 0) {
                llama_memory_seq_rm (mem_dft, 0, 0, reuse_i);
                llama_memory_seq_add(mem_dft, 0, reuse_i, -1, -reuse_i);

                prompt_dft.erase(prompt_dft.begin(), prompt_dft.begin() + reuse_i);
            }

            if (reuse_n < (int) prompt_dft.size()) {
                llama_memory_seq_rm (mem_dft, 0, reuse_n, -1);
                prompt_dft.erase(prompt_dft.begin() + reuse_n, prompt_dft.end());
            }
        }

        // prepare a batch to evaluate any new tokens in the prompt
        common_batch_clear(batch);

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::vector<bool> drafting(n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];

        // we should rarely end-up here during normal decoding
        if (batch.n_tokens > 0) {
            //LOG_DBG("%s: draft prompt batch: %s\n", __func__, string_from(ctx, batch).c_str());

            llama_decode(ctx_dft, batch);
        }

        const llama_pos n_past = prompt_dft.size();

        LOG_DBG("%s: n_past = %d\n", __func__, n_past);

        common_batch_clear(batch);
        common_batch_add  (batch, id_last, n_past, { 0 }, true);

        prompt_dft.push_back(id_last);

        LOG_DBG("%s: draft prompt: %s\n", __func__, string_from(ctx_dft, prompt_dft).c_str());

        llama_decode(ctx_dft, batch);

        int i = 0;

        while (n_drafting > 0) {
            int i_batch = 0;

            common_batch_clear(batch);

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (!drafting[seq_id]) {
                    continue;
                }

                auto * smpl = smpls[seq_id].get();

                common_sampler_sample(smpl, ctx_dft, i_batch, true);
                ++i_batch;

                const auto * cur_p = common_sampler_get_candidates(smpl, true);

                for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                    LOG_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                            common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                // add drafted token for each sequence
                const llama_token id = cur_p->data[0].id;

                // only collect very high-confidence draft tokens
                if (cur_p->data[0].p < params.p_min) {
                    drafting[seq_id] = false;
                    n_drafting--;

                    continue;
                }

                common_sampler_accept(smpl, id, true);

                auto & dp = dparams.at(seq_id);
                auto & result = *dp.result;

                result.push_back(id);

                if ((params.n_max <= (int) result.size()) ||
                    (dp.n_max > 0 && dp.n_max <= (int) result.size())) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                common_batch_add(batch, id, dp.n_past + i + 1, { seq_id }, true);
            }

            if (batch.n_tokens == 0) {
                break;
            }

            // evaluate the drafted tokens on the draft model
            llama_decode(ctx_dft, batch);

            ++i;
        }

        for (auto & dp : dparams) {
            if (!dp.drafting) {
                continue;
            }

            if (dp.result->size() < (size_t) params.n_min) {
                dp.result->clear();
            }
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/) override {
        // noop
    }

    bool need_embd() const override {
        return false;
    }
};

struct common_speculative_impl_draft_eagle3 : public common_speculative_impl {
    //common_params_speculative_eagle3 params;

    common_speculative_impl_draft_eagle3(const common_params_speculative & /*params*/, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3, n_seq) {}

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & /*dparams*/) override {
        // TODO: implement
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/) override {
        // noop
    }

    bool need_embd() const override {
        return false;
    }
};

struct common_speculative_state_draft_mtp : public common_speculative_impl {
    common_params_speculative_draft params; // reuses the draft-model params slot (ctx_tgt/ctx_dft)

    llama_batch batch;

    std::vector<common_sampler_ptr> smpls;

    int32_t n_embd = 0;

    // Per-sequence cross-batch carryover: pair (h_p, x_{p+1}) at MTP pos p+1.
    // The last h-row of one process() call needs the first token of the NEXT
    // call to pair with, so it's stashed here until that next call fires.
    std::vector<std::vector<float>> pending_h;   // [n_seq][n_embd]

    std::vector<int32_t> i_batch_beg;
    std::vector<int32_t> i_batch_end;

    // Hidden rows from the most recent target verification batch, grouped by seq.
    // Row 0 corresponds to the sampled token, row N to the Nth accepted draft token.
    std::vector<std::vector<float>> verify_h;
    std::vector<int32_t> verify_h_rows;

    // Per-seq draft length from the last draft() call, used in accept() to
    // roll back ctx_dft's recurrent state past the AR draft's redundant
    // pre-advancement before process() mirrored the verify batch.
    std::vector<uint16_t> last_n_drafted;

    common_speculative_state_draft_mtp(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_MTP, n_seq)
        , params(params.draft)
    {
        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;
        GGML_ASSERT(ctx_tgt && ctx_dft && "MTP requires ctx_tgt and ctx_dft to be set");

        n_embd = llama_model_n_embd(llama_get_model(ctx_dft));

        const int32_t n_b = (int32_t) llama_n_batch(ctx_dft);
        batch = llama_batch_init(/*n_tokens=*/ n_b, /*embd=*/ n_embd, /*n_seq_max=*/ 1);
        // llama_batch_init allocates only one of token/embd; MTP needs both.
        // TODO: fix, how to call without malloc
        batch.token = (llama_token *) malloc(sizeof(llama_token) * n_b);

        smpls.resize(n_seq);
        for (auto & s : smpls) {
            common_params_sampling sparams;
            sparams.no_perf  = false;
            sparams.top_k    = 1; // TODO: re-enable top_k == 10 and utilize `p_min` spec param
            sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
            s.reset(common_sampler_init(llama_get_model(ctx_dft), sparams));
        }

        llama_set_embeddings_pre_norm(ctx_tgt, true);
        llama_set_embeddings_pre_norm(ctx_dft, true);

        pending_h.assign(n_seq, std::vector<float>(n_embd, 0.0f));

        i_batch_beg.assign(n_seq, -1);
        i_batch_end.assign(n_seq, -1);

        verify_h.assign(n_seq, {});
        verify_h_rows.assign(n_seq, 0);

        last_n_drafted.assign(n_seq, 0);
    }

    ~common_speculative_state_draft_mtp() override {
        if (batch.token != nullptr) {
            free(batch.token);
            batch.token = nullptr;
        }
        llama_batch_free(batch);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        const int32_t N = (int32_t) prompt.size();
        if (N <= 0) {
            return;
        }
        auto * ctx_dft = this->params.ctx_dft;
        const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id);
        if (pos_max < N - 1) {
            LOG_WRN("%s: ctx_dft pos_max=%d < N-1=%d — "
                    "process() hook may not have run on every prefill ubatch "
                    "(need_embd / logits=1 on every prompt position?). "
                    "Drafts may degrade.\n",
                    __func__, (int) pos_max, N - 1);
        }
    }

    bool process(const llama_batch & batch_in) override {
        if (batch_in.n_tokens <= 0) {
            return true;
        }

        // TODO: how to make it work with vision tokens?
        if (batch_in.token == nullptr || batch_in.embd != nullptr) {
            return true;
        }

        const int32_t n_tokens = batch_in.n_tokens;

        // remember the frist and last batch index for each sequence
        std::fill(i_batch_beg.begin(), i_batch_beg.end(), -1);
        std::fill(i_batch_end.begin(), i_batch_end.end(), -1);

        for (int k = 0; k < n_tokens; ++k) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                GGML_ASSERT(batch_in.n_seq_id[k] == 1);

                if (batch_in.seq_id[k][0] == seq_id) {
                    i_batch_end[seq_id] = k;
                    if (i_batch_beg[seq_id] < 0) {
                        i_batch_beg[seq_id] = k;
                    }
                }
            }
        }

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;

        const size_t row_bytes = (size_t) n_embd * sizeof(float);

        common_batch_clear(batch);

        for (int k = 0; k < n_tokens; ++k) {
            common_batch_add(batch, batch_in.token[k], batch_in.pos[k], { batch_in.seq_id[k][0] }, 0);
        }

        // shift the tgt embeddings to the right by one position
        // assumes that the tokens in the batch are sequential for each sequence
        // i.e. we cannot have seq_id like this: [0, 0, 0, 1, 1, 0, 1, 1]
        //                                                       ^--- this is a problem
        // TODO:this is generally true, but would be nice to assert it
        {
            const float * h_tgt = llama_get_embeddings_pre_norm(ctx_tgt);
            std::memcpy(batch.embd + (size_t) 1 * n_embd, h_tgt, row_bytes * (n_tokens-1));

            //{
            //    // string with seq_ids in the batch
            //    std::stringstream ss;
            //    for (int i = 0; i < n_tokens; ++i) {
            //        ss << batch_in.seq_id[i][0] << ",";
            //    }
            //    LOG_WRN("%s: batch_in.seq_id = %s\n", __func__, ss.str().c_str());
            //}
        }

        // fill the pending embeddings from a previous run
        auto set_h = [&](int idx, const float * h_row) {
            std::memcpy(batch.embd + (size_t) idx * n_embd, h_row, row_bytes);
        };

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_batch_beg[seq_id] < 0) {
                continue;
            }

            set_h(i_batch_beg[seq_id], pending_h[seq_id].data());
        }

        const int32_t rc = llama_decode(ctx_dft, batch);
        if (rc != 0) {
            LOG_ERR("%s: llama_decode(ctx_dft) failed rc=%d (pos=%d)\n", __func__, (int) rc, (int) batch_in.pos[0]);
            return false;
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_batch_end[seq_id] < 0) {
                continue;
            }

            const int32_t n_rows = i_batch_end[seq_id] - i_batch_beg[seq_id] + 1;
            verify_h_rows[seq_id] = n_rows;
            verify_h[seq_id].resize((size_t) n_rows * n_embd);

            for (int32_t i = 0; i < n_rows; ++i) {
                const float * h = llama_get_embeddings_pre_norm_ith(ctx_tgt, i_batch_beg[seq_id] + i);
                std::memcpy(verify_h[seq_id].data() + (size_t) i * n_embd, h, row_bytes);
            }

            std::memcpy(pending_h[seq_id].data(),
                    verify_h[seq_id].data() + (size_t) (n_rows - 1) * n_embd, row_bytes);
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        common_batch_clear(batch);

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::vector<bool> drafting(n_seq);

        const float * h_row = nullptr;
        const size_t row_bytes = (size_t) n_embd * sizeof(float);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }

            n_drafting++;
            drafting[seq_id] = true;
            common_sampler_reset(smpls[seq_id].get());

            common_batch_add(batch, dp.id_last, dp.n_past, { seq_id }, true);

            h_row = pending_h[seq_id].data();
            std::memcpy(batch.embd + n_embd*(batch.n_tokens - 1), h_row, row_bytes);
        }

        int ret = llama_decode(ctx_dft, batch);
        if (ret != 0) {
            LOG_WRN("%s: llama_decode returned %d\n", __func__, ret);
            return;
        }

        int i = 0;

        while (n_drafting > 0) {
            int i_batch = 0;

            common_batch_clear(batch);

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (!drafting[seq_id]) {
                    continue;
                }

                auto * smpl = smpls[seq_id].get();

                common_sampler_sample(smpl, ctx_dft, i_batch, true);
                h_row = llama_get_embeddings_pre_norm_ith(ctx_dft, i_batch);
                ++i_batch;

                const auto * cur_p = common_sampler_get_candidates(smpl, true);

                for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                    LOG_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                            common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                // add drafted token for each sequence
                const llama_token id = cur_p->data[0].id;

                common_sampler_accept(smpl, id, true);

                auto & dp = dparams.at(seq_id);
                auto & result = *dp.result;

                result.push_back(id);

                if (params.n_max <= (int) result.size()) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                common_batch_add(batch, id, dp.n_past + i + 1, { seq_id }, true);
                std::memcpy(batch.embd + n_embd*(batch.n_tokens - 1), h_row, row_bytes);
            }

            if (batch.n_tokens == 0) {
                break;
            }

            // evaluate the drafted tokens on the draft model
            ret = llama_decode(ctx_dft, batch);
            if (ret != 0) {
                LOG_WRN("%s: llama_decode[%d] returned %d\n", __func__, i, ret);
                break;
            }

            ++i;
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            if (dp.result->size() < (size_t) params.n_min) {
                dp.result->clear();
            }

            last_n_drafted[seq_id] = (uint16_t) dp.result->size();
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        const int32_t n_rows = verify_h_rows[seq_id];
        if (n_rows <= 0) {
            return;
        }

        const int32_t i_h = std::min<int32_t>(n_accepted, n_rows - 1);
        const size_t row_bytes = (size_t) n_embd * sizeof(float);
        std::memcpy(pending_h[seq_id].data(), verify_h[seq_id].data() + (size_t) i_h * n_embd, row_bytes);
    }

    bool need_embd() const override {
        return true;
    }
};

// Optional NDJSON tracer for MTP draft/accept events, gated by env LLAMA_MTP_ACC_TRACE.
// Value semantics:
//   unset / "0" / ""           -> disabled
//   "1"                        -> stderr
//   anything else              -> treated as a file path (append mode)
// Each event is one JSON object per line. Disabled at zero overhead; otherwise computes
// h_prev L2 norm (n_bb floats) per draft, which is negligible vs the MTP step itself.
namespace {
struct mtp_acc_tracer {
    bool       enabled = false;
    FILE *     fp      = nullptr;
    std::mutex mu;

    mtp_acc_tracer() {
        const char * v = std::getenv("LLAMA_MTP_ACC_TRACE");
        if (!v || v[0] == '\0' || std::strcmp(v, "0") == 0) {
            return;
        }
        if (std::strcmp(v, "1") == 0) {
            fp = stderr;
        } else {
            fp = std::fopen(v, "a");
        }
        enabled = (fp != nullptr);
    }

    ~mtp_acc_tracer() {
        if (fp && fp != stderr) {
            std::fclose(fp);
        }
    }

    void writeln(const std::string & line) {
        if (!enabled) {
            return;
        }
        std::lock_guard<std::mutex> lk(mu);
        std::fputs(line.c_str(), fp);
        std::fputc('\n', fp);
        std::fflush(fp);
    }
};

mtp_acc_tracer & mtp_tracer() {
    static mtp_acc_tracer t;
    return t;
}
} // namespace

struct common_speculative_state_mtp : public common_speculative_state {
    llama_context * ctx_tgt;
    llama_seq_id    seq_id = 0; // target-side sequence id (set by host, e.g. server slot.id)
    int             h_idx  = -1; // output index in target's last decode for h_prev (-1 = last)
    // Adaptive skip after consecutive zero-accept batches: when MTP head consistently
    // mispredicts (e.g. on numbers/code/rare tokens during long generation), drafting
    // costs ~10ms but yields no accepted tokens. Detect this and fall back to plain
    // verify-only for one batch; reset skip on next non-empty accept.
    size_t          prev_n_acc_drafts   = 0;
    int             zero_accept_streak  = 0;
    // 0 = disabled (always draft). Set LLAMA_MTP_SKIP_STREAK_THRESHOLD to 1–32 to enable.
    int skip_streak_threshold = 0;
    // After a skip-streak verify-only batch, do not count the next draft() as another
    // zero-accept miss (otherwise threshold==1 skips every round forever — see debug logs).

    bool skip_streak_last_draft = false;

    // Pipeline depth-2: submit at end of iteration via prepare_next(); draft() waits first.
    bool                         has_pending      = false;
    int32_t                      pending_n_steps  = 0;
    common_params_speculative    last_spec_params;

    // ---- MTP acceptance tracing (LLAMA_MTP_ACC_TRACE), no behavior change ----
    int        trace_iter           = 0;
    int        trace_submit_id_last = -1;     // id_last passed to the most recent submit
    int        trace_submit_h_idx   = -1;     // h_idx used for h_prev at submit time
    llama_pos  trace_submit_attn_pos = -1;    // attn_pos used at submit time
    float      trace_submit_h_l2    = 0.0f;   // L2 norm of h_prev at submit time
    int32_t    trace_submit_n_steps = 0;
    int        trace_last_n_drafted = 0;      // drafts.size() returned to caller (for accept pairing)

    static float compute_h_l2(const float * h, int32_t n) {
        if (!h || n <= 0) {
            return 0.0f;
        }
        double s = 0.0;
        for (int32_t i = 0; i < n; ++i) {
            const float v = h[i];
            s += (double) v * (double) v;
        }
        return (float) std::sqrt(s);
    }

    void trace_emit_draft(const llama_tokens & drafts, const char * path) {
        trace_last_n_drafted = (int) drafts.size();
        if (!mtp_tracer().enabled) {
            return;
        }
        std::ostringstream oss;
        oss << "{\"evt\":\"mtp_draft\""
            << ",\"iter\":" << trace_iter
            << ",\"path\":\"" << path << "\""
            << ",\"seq_id\":" << (int) seq_id
            << ",\"id_last\":" << trace_submit_id_last
            << ",\"h_idx\":" << trace_submit_h_idx
            << ",\"attn_pos\":" << (int) trace_submit_attn_pos
            << ",\"n_steps\":" << trace_submit_n_steps
            << ",\"h_l2\":" << std::fixed << std::setprecision(4) << trace_submit_h_l2
            << ",\"drafts\":[";
        for (size_t i = 0; i < drafts.size(); ++i) {
            if (i) oss << ',';
            oss << (int) drafts[i];
        }
        oss << "]}";
        mtp_tracer().writeln(oss.str());
    }

    void trace_emit_accept(int n_accepted) {
        if (!mtp_tracer().enabled) {
            return;
        }
        std::ostringstream oss;
        oss << "{\"evt\":\"mtp_accept\""
            << ",\"iter\":" << trace_iter
            << ",\"n_accepted\":" << n_accepted
            << ",\"n_drafted_prev\":" << trace_last_n_drafted
            << "}";
        mtp_tracer().writeln(oss.str());
        ++trace_iter;
    }

    explicit common_speculative_state_mtp(enum common_speculative_type type, llama_context * ctx_tgt)
        : common_speculative_state(type), ctx_tgt(ctx_tgt) {
        // MTP reads last backbone hidden from the target; keep embeddings on across decodes.
        llama_set_embeddings(ctx_tgt, true);
        // Optional: after N consecutive zero-accept MTP batches, skip drafting for one verify-only batch.
        if (const char * e = std::getenv("LLAMA_MTP_SKIP_STREAK_THRESHOLD")) {
            const int v = std::atoi(e);
            if (v >= 1 && v <= 32) {
                skip_streak_threshold = v;
            }
        }
    }

    // If a prepare_next() is in flight, block and discard its output (keeps worker contract sane).
    void mtp_drain_pending_discard() {
        if (!has_pending) {
            return;
        }
        std::vector<llama_token> discard((size_t) pending_n_steps);
        const int32_t rc = llama_decode_mtp_wait(ctx_tgt, discard.data(), /*out_h_prev_last*/ nullptr);
        if (rc != 0) {
            LOG_ERR("%s: llama_decode_mtp_wait (drain) failed (%d)\n", __func__, (int) rc);
        }
        has_pending     = false;
        pending_n_steps = 0;
    }

    // Projected skip on the next draft() call, without mutating zero_accept_streak.
    bool mtp_would_skip_next_draft() const {
        if (skip_streak_threshold <= 0) {
            return false;
        }
        if (skip_streak_last_draft) {
            return false;
        }
        const int proj = (n_call_draft > 0)
                ? (n_acc_drafts == prev_n_acc_drafts ? zero_accept_streak + 1 : 0)
                : zero_accept_streak;
        return proj >= skip_streak_threshold;
    }

    static int32_t mtp_effective_n_steps(const common_params_speculative & p) {
        const int32_t n_steps_raw = p.draft_block_size > 1 ? p.draft_block_size - 1 : 0;
        return std::min(n_steps_raw, p.n_max);
    }

    void begin(const llama_tokens & prompt) override {
        GGML_UNUSED(prompt);
        llama_set_embeddings(ctx_tgt, true);
        skip_streak_last_draft = false;
        // New request / prompt: do not leak in-flight MTP from the previous generation.
        mtp_drain_pending_discard();
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & draft_tokens) override {

        draft_tokens.clear();
        GGML_UNUSED(prompt_tgt);

        const llama_model * model_tgt = llama_get_model(ctx_tgt);
        const uint32_t      n_bb_u   = llama_model_mtp_n_embd_backbone(model_tgt);
        if (n_bb_u == 0) {
            LOG_ERR("%s: no MTP assistant on target model\n", __func__);
            return;
        }
        const int32_t n_bb = (int32_t) n_bb_u;

        last_spec_params = params;

        // Detect zero-accept of previous draft batch: n_acc_drafts only increments when
        // common_speculative_accept is called with n_accepted>0. So if it didn't move
        // since our previous draft() return, the previous batch produced 0 accepted drafts.
        if (n_call_draft > 0) {
            if (skip_streak_last_draft) {
                skip_streak_last_draft = false;
            } else if (n_acc_drafts == prev_n_acc_drafts) {
                ++zero_accept_streak;
            } else {
                zero_accept_streak = 0;
            }
        }
        // After threshold consecutive misses, skip MTP draft for one batch — drafting
        // would cost ~10ms with no benefit; better to let server do a single-token
        // verify (baseline path).
        if (skip_streak_threshold > 0 && zero_accept_streak >= skip_streak_threshold) {
            // Reset streak after one skip; if next batch still misses (streak resumes), we'll skip again.
            zero_accept_streak = 0;
            skip_streak_last_draft = true;
            prev_n_acc_drafts  = n_acc_drafts;
            mtp_drain_pending_discard();
            trace_submit_id_last  = (int) id_last;
            trace_submit_n_steps  = 0;
            trace_emit_draft(draft_tokens, "skip-streak");
            return; // empty draft_tokens — server falls back to single-token verify
        }

        int32_t n_steps_raw = params.draft_block_size > 1 ? params.draft_block_size - 1 : 0;
        int32_t n_steps     = std::min(n_steps_raw, params.n_max);

        if (n_steps <= 0) {
            mtp_drain_pending_discard();
            trace_submit_id_last  = (int) id_last;
            trace_submit_n_steps  = 0;
            trace_emit_draft(draft_tokens, "skip-nsteps");
            return;
        }

        llama_set_embeddings(ctx_tgt, true);

        // Lazy wait: overlap MTP worker with server post-accept work from the previous iteration.
        if (has_pending) {
            if (pending_n_steps != n_steps) {
                mtp_drain_pending_discard();
                // Fall through to synchronous submit below.
            } else {
                draft_tokens.resize((size_t) n_steps);
                const int32_t rc = llama_decode_mtp_wait(
                        ctx_tgt, draft_tokens.data(), /*out_h_prev_last*/ nullptr);
                has_pending     = false;
                pending_n_steps = 0;
                if (rc != 0) {
                    LOG_ERR("%s: llama_decode_mtp_wait failed (%d)\n", __func__, (int) rc);
                    draft_tokens.clear();
                }
                prev_n_acc_drafts = n_acc_drafts;
                // submit-time fields were captured by prepare_next() of previous iter
                trace_emit_draft(draft_tokens, "lazy");
                return;
            }
        }

        std::vector<float> h_prev((size_t) n_bb, 0.0f);
        // Use the explicit h_idx pointing at the last accepted output (set by the host after
        // sample_and_accept_n). If unset, fall back to -1 (last output) which is correct only
        // when the previous decode was prefill or when ALL drafts of the previous batch were
        // accepted (otherwise -1 points at a rejected draft's hidden state).
        if (float * h_tgt = llama_get_embeddings_ith(ctx_tgt, h_idx)) {
            const int32_t n_out_tgt = llama_model_n_embd_out(model_tgt);
            const int32_t n_copy  = std::min(n_bb, n_out_tgt);
            std::memcpy(h_prev.data(), h_tgt, (size_t) n_copy * sizeof(float));
        }

        llama_memory_t mem = llama_get_memory(ctx_tgt);
        llama_pos attn_pos = mem ? llama_memory_seq_pos_max(mem, seq_id) : (llama_pos) 0;
        if (attn_pos < 0) {
            attn_pos = 0;
        }

        // Capture submit-time fields for tracing.
        trace_submit_id_last  = (int) id_last;
        trace_submit_h_idx    = h_idx;
        trace_submit_attn_pos = attn_pos;
        trace_submit_n_steps  = n_steps;
        trace_submit_h_l2     = mtp_tracer().enabled
                                    ? compute_h_l2(h_prev.data(), n_bb)
                                    : 0.0f;

        // Bootstrap path (first iter or after drain): submit and wait immediately on sched_mtp.
        draft_tokens.resize((size_t) n_steps);
        int32_t rc = llama_decode_mtp_async(
                ctx_tgt,
                seq_id,
                attn_pos,
                id_last,
                h_prev.data(),
                n_steps);
        if (rc != 0) {
            LOG_ERR("%s: llama_decode_mtp_async failed (%d)\n", __func__, (int) rc);
            draft_tokens.clear();
        } else {
            rc = llama_decode_mtp_wait(ctx_tgt, draft_tokens.data(), /*out_h_prev_last*/ nullptr);
            if (rc != 0) {
                LOG_ERR("%s: llama_decode_mtp_wait failed (%d)\n", __func__, (int) rc);
                draft_tokens.clear();
            }
        }
        // Snapshot accepted-draft counter for next call's zero-accept detection.
        prev_n_acc_drafts = n_acc_drafts;
        trace_emit_draft(draft_tokens, "sync");
    }

    void accept(uint16_t n_accepted) override {
        GGML_UNUSED(n_accepted);
    }

    void cancel() override {
        skip_streak_last_draft = false;
        mtp_drain_pending_discard();
    }

    void prepare_next(llama_token id_last) override {
        // Kill switch for A/B testing depth-2 vs sync.
        static const bool depth2_disabled = []() {
            const char * v = std::getenv("LLAMA_PIPELINE_DEPTH2");
            return v && std::strcmp(v, "0") == 0;
        }();
        if (depth2_disabled) {
            return;
        }
        const llama_model * model_tgt = llama_get_model(ctx_tgt);
        const uint32_t      n_bb_u   = llama_model_mtp_n_embd_backbone(model_tgt);
        if (n_bb_u == 0) {
            return;
        }
        if (has_pending) {
            LOG_WRN("%s: MTP prepare_next called while a draft is already pending; ignoring\n", __func__);
            return;
        }
        if (mtp_would_skip_next_draft()) {
            return;
        }
        const int32_t n_steps = mtp_effective_n_steps(last_spec_params);
        if (n_steps <= 0) {
            return;
        }
        const int32_t n_bb = (int32_t) n_bb_u;

        std::vector<float> h_prev((size_t) n_bb, 0.0f);
        if (float * h_tgt = llama_get_embeddings_ith(ctx_tgt, h_idx)) {
            const int32_t n_out_tgt = llama_model_n_embd_out(model_tgt);
            const int32_t n_copy  = std::min(n_bb, n_out_tgt);
            std::memcpy(h_prev.data(), h_tgt, (size_t) n_copy * sizeof(float));
        }

        llama_memory_t mem = llama_get_memory(ctx_tgt);
        llama_pos attn_pos = mem ? llama_memory_seq_pos_max(mem, seq_id) : (llama_pos) 0;
        if (attn_pos < 0) {
            attn_pos = 0;
        }

        // Capture submit-time fields for tracing of the upcoming lazy-wait.
        trace_submit_id_last  = (int) id_last;
        trace_submit_h_idx    = h_idx;
        trace_submit_attn_pos = attn_pos;
        trace_submit_n_steps  = n_steps;
        trace_submit_h_l2     = mtp_tracer().enabled
                                    ? compute_h_l2(h_prev.data(), n_bb)
                                    : 0.0f;

        const int32_t rc = llama_decode_mtp_async(
                ctx_tgt,
                seq_id,
                attn_pos,
                id_last,
                h_prev.data(),
                n_steps);
        if (rc != 0) {
            LOG_ERR("%s: llama_decode_mtp_async failed (%d)\n", __func__, (int) rc);
            return;
        }
        has_pending     = true;
        pending_n_steps = n_steps;
    }
};

// Qwen NextN compatibility check. Two valid configurations:
//   (a) Shared-model path: target arch in {qwen35, qwen35moe} AND draft model pointer
//       == target model pointer (single llama_model, single mmap). This is the default
//       since the combined *_MTP GGUF ships NextN tensors inside the target model's
//       layer table; the draft context just runs a different graph against them.
//   (b) Standalone NEXTN_ONLY GGUF: target arch in {qwen35, qwen35moe}, draft arch is
//       the matching '*_mtp' override (qwen35_mtp / qwen35moe_mtp). Vocab must match.
static bool common_speculative_are_compatible_nextn(
        const llama_model * model_tgt,
        const llama_model * model_dft) {
    const char * at = llama_model_arch_str(model_tgt);
    if (std::strcmp(at, "qwen35") != 0 && std::strcmp(at, "qwen35moe") != 0) {
        return false;
    }
    if (model_dft == model_tgt) {
        // Shared-model path: NextN-layer tensors must exist in target.
        return llama_model_has_nextn_layer(model_tgt);
    }
    const char * ad = llama_model_arch_str(model_dft);
    if (std::strcmp(at, "qwen35") == 0) {
        if (std::strcmp(ad, "qwen35_mtp") != 0) {
            return false;
        }
    } else { // qwen35moe
        if (std::strcmp(ad, "qwen35moe_mtp") != 0) {
            return false;
        }
    }
    return common_speculative_are_compatible_mtp(model_tgt, model_dft);
}

struct common_speculative_state_nextn : public common_speculative_state {
    llama_context * ctx_tgt   = nullptr;
    llama_context * ctx_nextn = nullptr;

    llama_batch batch;
    common_sampler * smpl = nullptr;
    int32_t n_embd = 0;

    /// Output index in the target's last decode for the first-draft hidden row (-1 = last output).
    int h_idx = -1;

    uint16_t last_n_drafted   = 0;
    int32_t  last_n_accepted  = -1;

    // Async pipeline depth-2 (gated by LLAMA_PIPELINE_DEPTH2=0 to disable).
    // Worker thread overlaps NextN draft compute on ctx_nextn with the
    // server's per-token CPU work (sampling, tokenization, response build)
    // between prepare_next() and the next draft() call.
    bool pipeline_enabled = true;

    std::thread             worker_thread;
    std::mutex              pipe_mu;
    std::condition_variable pipe_cv;
    bool pipe_stop          = false;
    bool pipe_req_pending   = false;
    bool pipe_res_ready     = false;
    bool pipe_has_pending   = false;     // submit-side flag (set by prepare_next, cleared by draft)
    // Request data populated by prepare_next, consumed by worker.
    llama_token         pipe_req_id_last  = -1;
    int32_t             pipe_req_n_steps  = 0;
    llama_pos           pipe_req_pos_start = 0;
    std::vector<float>  pipe_req_embd;
    // Result populated by worker, consumed by draft.
    llama_tokens        pipe_res_drafts;
    int32_t             pipe_res_actual_steps = 0; // how many decodes the worker actually completed

    common_speculative_state_nextn(enum common_speculative_type type, llama_context * ctx_tgt, llama_context * ctx_nextn_in)
            : common_speculative_state(type), ctx_tgt(ctx_tgt), ctx_nextn(ctx_nextn_in) {
        GGML_ASSERT(ctx_tgt && ctx_nextn);
        const llama_model * model_nextn = llama_get_model(ctx_nextn);
        n_embd = llama_model_n_embd(model_nextn);

        {
            common_params_sampling sparams;
            sparams.no_perf = false;
            sparams.top_k   = 1;
            sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
            smpl             = common_sampler_init(model_nextn, sparams);
        }

        batch = llama_batch_init(/*n_tokens=*/ 1, /*embd=*/ n_embd, /*n_seq_max=*/ 1);
        batch.token = (llama_token *) malloc(sizeof(llama_token));
        batch.n_tokens      = 1;
        batch.n_seq_id[0]   = 1;
        batch.seq_id[0][0]  = 0;
        batch.logits[0]     = 1;

        llama_set_embeddings_pre_norm(ctx_tgt, true);
        llama_set_embeddings_pre_norm(ctx_nextn, true);
        llama_set_nextn(ctx_tgt, ctx_nextn);

        if (const char * v = std::getenv("LLAMA_PIPELINE_DEPTH2")) {
            if (std::strcmp(v, "0") == 0) {
                pipeline_enabled = false;
            }
        }
        if (pipeline_enabled) {
            worker_thread = std::thread([this] { worker_loop(); });
        }
    }

    ~common_speculative_state_nextn() override {
        if (pipeline_enabled) {
            {
                std::lock_guard<std::mutex> lk(pipe_mu);
                pipe_stop = true;
                pipe_cv.notify_all();
            }
            if (worker_thread.joinable()) {
                worker_thread.join();
            }
        }
        llama_set_embeddings_pre_norm(ctx_tgt, false);
        llama_set_embeddings_pre_norm(ctx_nextn, false);
        llama_set_nextn(ctx_tgt, nullptr);
        llama_batch_free(batch);
        common_sampler_free(smpl);
        if (ctx_nextn) {
            llama_free(ctx_nextn);
            ctx_nextn = nullptr;
        }
    }

    // Worker loop: blocks on pipe_cv, runs draft chain on ctx_nextn, publishes result.
    void worker_loop() {
        const size_t row_bytes = (size_t) n_embd * sizeof(float);
        std::vector<float> local_embd;
        for (;;) {
            llama_token id_last  = 0;
            int32_t     n_steps  = 0;
            llama_pos   pos_start = 0;
            {
                std::unique_lock<std::mutex> lk(pipe_mu);
                pipe_cv.wait(lk, [this] { return pipe_req_pending || pipe_stop; });
                if (pipe_stop) {
                    return;
                }
                id_last   = pipe_req_id_last;
                n_steps   = pipe_req_n_steps;
                pos_start = pipe_req_pos_start;
                local_embd.swap(pipe_req_embd);
                pipe_req_pending = false;
            }

            llama_tokens drafts;
            int32_t actual_steps = 0;
            llama_token cond_tok = id_last;
            llama_pos   pos      = pos_start;

            for (int32_t k = 0; k < n_steps; ++k) {
                const float * src = nullptr;
                if (k == 0) {
                    src = local_embd.data();
                } else {
                    llama_synchronize(ctx_nextn);
                    src = llama_get_embeddings_pre_norm_ith(ctx_nextn, -1);
                }
                if (!src) {
                    break;
                }
                std::memcpy(batch.embd, src, row_bytes);
                batch.token[0] = cond_tok;
                batch.pos[0]   = pos;

                if (llama_decode(ctx_nextn, batch) != 0) {
                    break;
                }
                ++actual_steps;
                const llama_token best = common_sampler_sample(smpl, ctx_nextn, 0);
                common_sampler_accept(smpl, best, /*accept_grammar=*/ false);

                drafts.push_back(best);
                cond_tok = best;
                ++pos;
            }

            {
                std::lock_guard<std::mutex> lk(pipe_mu);
                pipe_res_drafts       = std::move(drafts);
                pipe_res_actual_steps = actual_steps;
                pipe_res_ready        = true;
                pipe_cv.notify_all();
            }
        }
    }

    // Drain in-flight worker request (block until completion) and discard result.
    // Rolls back ctx_nextn KV by the number of steps the worker actually committed.
    void drain_pending_and_rollback() {
        if (!pipe_has_pending) {
            return;
        }
        int32_t advanced = 0;
        {
            std::unique_lock<std::mutex> lk(pipe_mu);
            pipe_cv.wait(lk, [this] { return pipe_res_ready; });
            advanced = pipe_res_actual_steps;
            pipe_res_drafts.clear();
            pipe_res_actual_steps = 0;
            pipe_res_ready = false;
        }
        pipe_has_pending = false;
        if (advanced > 0) {
            const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_nextn), 0);
            if (pos_max >= 0) {
                const llama_pos drop_from = pos_max - advanced + 1;
                llama_memory_seq_rm(llama_get_memory(ctx_nextn), 0, drop_from, -1);
            }
        }
    }

    void begin(const llama_tokens & prompt) override {
        last_n_accepted = -1;
        last_n_drafted  = 0;

        const int32_t N = (int32_t) prompt.size();
        if (N <= 0) {
            return;
        }

        // If the draft KV cache is already aligned with the current prompt
        // (e.g. prompt-cache reuse across requests), skip the prime. Otherwise we
        // need to seed ctx_nextn token-by-token from the target's pre-norm hidden
        // states. Target prefill must have been driven with logits=true on every
        // prompt token (see server-context: nextn_prefill_all_outputs).
        const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_nextn), 0);
        if (pos_max >= N - 1) {
            return;
        }

        // Ensure target pre-norm rows are materialized in host memory.
        llama_synchronize(ctx_tgt);

        // Drop any stale draft KV state to make absolute positions match the prompt.
        llama_memory_clear(llama_get_memory(ctx_nextn), false);

        // Sanity: the precomputed batch.embd buffer was sized for n_embd floats.
        const size_t row_bytes = (size_t) n_embd * sizeof(float);

        // Prime path: we don't need draft logits during seeding (no sampling here),
        // so disable per-step lm_head compute for speed. Restore logits=1 afterwards
        // so the regular chain-draft path in draft() works unchanged.
        batch.logits[0] = 0;

        int32_t primed = 0;
        for (int32_t i = 0; i < N; ++i) {
            float * src_row = llama_get_embeddings_pre_norm_ith(ctx_tgt, i);
            if (!src_row) {
                LOG_WRN("%s: missing pre-norm row at prompt pos %d/%d; aborting prime — drafts may degrade.\n",
                        __func__, (int) i, (int) N);
                break;
            }
            std::memcpy(batch.embd, src_row, row_bytes);
            batch.token[0] = prompt[i];
            batch.pos[0]   = i;

            const int32_t rc = llama_decode(ctx_nextn, batch);
            if (rc != 0) {
                LOG_WRN("%s: llama_decode(ctx_nextn) prime rc=%d at pos %d; aborting prime\n",
                        __func__, (int) rc, (int) i);
                break;
            }
            ++primed;
        }

        // Restore default for draft() path which samples after every decode.
        batch.logits[0] = 1;

        LOG_DBG("%s: primed ctx_nextn with %d/%d prompt tokens\n",
                __func__, (int) primed, (int) N);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & draft_tokens) override {
        GGML_UNUSED(prompt_tgt);
        draft_tokens.clear();

        // Async fast path: if prepare_next() submitted a draft on the previous
        // iteration, return its result (lazy wait) — the worker has been
        // computing the draft chain while the server processed sampling /
        // tokenization / response building for the accepted tokens.
        if (pipeline_enabled && pipe_has_pending) {
            std::unique_lock<std::mutex> lk(pipe_mu);
            pipe_cv.wait(lk, [this] { return pipe_res_ready; });
            draft_tokens = std::move(pipe_res_drafts);
            pipe_res_drafts.clear();
            pipe_res_actual_steps = 0;
            pipe_res_ready = false;
            pipe_has_pending = false;
            lk.unlock();
            last_n_drafted = (uint16_t) draft_tokens.size();
            return;
        }

        if (last_n_drafted > 0) {
            const int32_t n_to_drop = (int32_t) last_n_drafted - 1;
            if (n_to_drop > 0) {
                const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_nextn), 0);
                if (pos_max >= 0) {
                    const llama_pos drop_from = pos_max - n_to_drop + 1;
                    llama_memory_seq_rm(llama_get_memory(ctx_nextn), 0, drop_from, -1);
                }
            }
            last_n_drafted  = 0;
            last_n_accepted = 0;
        }

        const int32_t n_max = std::max(1, params.n_max);
        const size_t row_bytes = (size_t) n_embd * sizeof(float);

        llama_token cond_tok = id_last;
        llama_pos   pos      = llama_memory_seq_pos_max(llama_get_memory(ctx_nextn), 0) + 1;

        for (int32_t k = 0; k < n_max; ++k) {
            float * src_row_ptr = nullptr;
            if (k == 0) {
                llama_synchronize(ctx_tgt);
                const int32_t tgt_ith = last_n_accepted < 0
                        ? (h_idx >= 0 ? (int32_t) h_idx : -1)
                        : last_n_accepted;
                src_row_ptr = llama_get_embeddings_pre_norm_ith(ctx_tgt, tgt_ith);
            } else {
                llama_synchronize(ctx_nextn);
                src_row_ptr = llama_get_embeddings_pre_norm_ith(ctx_nextn, -1);
            }
            if (!src_row_ptr) {
                LOG_WRN("%s: missing pre-norm embeddings at k=%d; stopping chain\n", __func__, (int) k);
                return;
            }
            std::memcpy(batch.embd, src_row_ptr, row_bytes);

            batch.token[0] = cond_tok;
            batch.pos[0]   = pos;

            const int32_t dec_rc = llama_decode(ctx_nextn, batch);
            if (dec_rc != 0) {
                LOG_DBG("%s: llama_decode(ctx_nextn) rc=%d at k=%d; stopping chain\n", __func__, (int) dec_rc, (int) k);
                return;
            }

            const llama_token best = common_sampler_sample(smpl, ctx_nextn, 0);
            common_sampler_accept(smpl, best, /*accept_grammar=*/ false);

            draft_tokens.push_back(best);
            cond_tok = best;
            ++pos;
        }

        last_n_drafted = (uint16_t) draft_tokens.size();
    }

    void accept(uint16_t n_accepted) override {
        const llama_pos pos_max       = llama_memory_seq_pos_max(llama_get_memory(ctx_nextn), 0);
        const int32_t   n_drafted_last = (int32_t) last_n_drafted;
        const int32_t   n_to_drop      = std::max(0, n_drafted_last - (int32_t) n_accepted - 1);

        if (pos_max < 0) {
            last_n_accepted = (int32_t) n_accepted;
            return;
        }
        if (n_to_drop > 0) {
            const llama_pos drop_from = pos_max - n_to_drop + 1;
            llama_memory_seq_rm(llama_get_memory(ctx_nextn), /*seq_id=*/ 0,
                    /*p0=*/ drop_from, /*p1=*/ -1);
        }
        last_n_drafted  = 0;
        last_n_accepted = (int32_t) n_accepted;
    }

    void prepare_next(llama_token id_last) override {
        if (!pipeline_enabled) {
            return;
        }
        if (pipe_has_pending) {
            // Should not happen: server contract is one submit per cycle.
            LOG_WRN("%s: NextN prepare_next called while a draft is already pending; ignoring\n", __func__);
            return;
        }

        // Zero-accept reconciliation. `common_speculative_accept` early-outs at
        // n_accepted==0 and never calls state->accept(0), so `last_n_drafted`
        // can remain >0 here with ctx_nextn KV still holding the *rejected*
        // drafts from the previous cycle. The sync draft() path handles this
        // with its own rollback block at entry; the async path must do the
        // equivalent before reading pos_max — otherwise pos_start points into
        // the rejected-draft range and the worker chain decodes from the wrong
        // KV state, which empirically tanks acceptance from ~82% to ~66% over
        // a long generation. Mirror the sync rollback here.
        if (last_n_drafted > 0) {
            const int32_t n_to_drop = (int32_t) last_n_drafted - 1;
            if (n_to_drop > 0) {
                const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_nextn), 0);
                if (pos_max >= 0) {
                    const llama_pos drop_from = pos_max - n_to_drop + 1;
                    llama_memory_seq_rm(llama_get_memory(ctx_nextn), 0, drop_from, -1);
                }
            }
            last_n_drafted  = 0;
            last_n_accepted = 0;
        }

        // Capture seed embedding row (target's pre-norm at last_n_accepted) and
        // submit the request to the worker. The actual GPU draft compute happens
        // on the worker thread while the server runs CPU-side post-accept work.
        llama_synchronize(ctx_tgt);
        const int32_t tgt_ith = last_n_accepted < 0
                ? (h_idx >= 0 ? (int32_t) h_idx : -1)
                : last_n_accepted;
        float * src = llama_get_embeddings_pre_norm_ith(ctx_tgt, tgt_ith);
        if (!src) {
            return;
        }
        // Pipeline depth-2 with single-step submit is the empirically best
        // knob on Apple Silicon Metal (matches Gemma MTP DRAFT_BLOCK_SIZE=2):
        // overlap one draft decode with the next target verify, no deeper
        // chain. Multi-step worker chains diverge from the sync chain's
        // acceptance rate and tank end-to-end throughput; revisit once tree
        // drafting / multi-seq KV are wired in.
        const int32_t n_steps = 1;
        const llama_pos pos_start = llama_memory_seq_pos_max(llama_get_memory(ctx_nextn), 0) + 1;

        {
            std::lock_guard<std::mutex> lk(pipe_mu);
            pipe_req_embd.assign(src, src + n_embd);
            pipe_req_id_last   = id_last;
            pipe_req_n_steps   = n_steps;
            pipe_req_pos_start = pos_start;
            pipe_res_drafts.clear();
            pipe_res_actual_steps = 0;
            pipe_res_ready    = false;
            pipe_req_pending  = true;
            pipe_cv.notify_all();
        }
        pipe_has_pending = true;
    }

    void cancel() override {
        if (pipeline_enabled) {
            drain_pending_and_rollback();
        }
    }
};

// state of self-speculation (simple implementation, not ngram-map)
struct common_speculative_impl_ngram_simple : public common_speculative_impl {
    common_params_speculative_ngram_map params;

    // shared across all sequences
    common_ngram_simple_config config;

    common_speculative_impl_ngram_simple(
            const common_params_speculative & params, uint32_t n_seq,
            common_ngram_simple_config config)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE, n_seq)
        , params(params.ngram_simple)
        , config(config) {}

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            *dp.result = common_ngram_simple_draft(config, *dp.prompt, dp.id_last);
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/) override {
        // noop
    }

    bool need_embd() const override {
        return false;
    }
};

struct common_speculative_impl_ngram_map_k : public common_speculative_impl {
    common_params_speculative_ngram_map params;

    // n_seq configs
    std::vector<common_ngram_map> config;

    common_speculative_impl_ngram_map_k(
            const common_params_speculative & params,
            const common_ngram_map & config,
            uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K, n_seq)
        , params(params.ngram_map_k) {
        for (uint32_t i = 0; i < n_seq; i++) {
            this->config.push_back(config);
        }
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        GGML_ASSERT(seq_id < (llama_seq_id) n_seq);

        common_ngram_map_begin(config[seq_id], prompt);
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            common_ngram_map_draft(config[seq_id], *dp.prompt, dp.id_last, *dp.result);
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted) override {
        GGML_ASSERT((seq_id < (llama_seq_id) config.size()));

        common_ngram_map_accept(config[seq_id], n_accepted);
    }

    bool need_embd() const override {
        return false;
    }
};

struct common_speculative_impl_ngram_mod : public common_speculative_impl {
    common_params_speculative_ngram_mod params;

    // shared across all sequences
    common_ngram_mod mod;

    // enable trace logging if LLAMA_TRACE is set
    const bool verbose;

    struct seq_info {
        // the last position in the prompt that was added to the ngram container
        size_t i_last = 0;

        // length of the last drafted n‑gram (number of tokens returned by draft)
        size_t n_draft_last = 0;

        // consecutive accept rounds with low acceptance fraction (< 0.5)
        int n_low = 0;
    };

    std::vector<seq_info> sinfos;

    common_speculative_impl_ngram_mod(
            const common_params_speculative & params,
            uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_MOD, n_seq)
        , params(params.ngram_mod)
        , mod(params.ngram_mod.n_match, 4*1024*1024)
        , verbose(std::getenv("LLAMA_TRACE") != nullptr) {
        static_assert(sizeof(llama_token) == sizeof(common_ngram_mod::entry_t));

        LOG_INF("%s: initialized ngram_mod with n_match=%d, size=%zu (%.3f MB)\n", __func__,
                this->params.n_match, mod.size(), (float)(mod.size_bytes())/1024/1024);

        if (this->params.n_match < 16) {
            LOG_WRN("%s: ngram_mod n_match=%d is too small - poor quality is possible, "
                    "see: https://github.com/ggml-org/llama.cpp/pull/19164\n", __func__, this->params.n_match);
        }

        sinfos.resize(n_seq);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        auto & sinfo = sinfos[seq_id];

        sinfo.i_last = 0;
        sinfo.n_draft_last = 0;

        const size_t n = mod.get_n();
        if (prompt.size() < n) {
            return;
        }

        for (size_t i = 0; i < prompt.size() - n; ++i) {
            mod.add(prompt.data() + i);
        }

        sinfo.i_last = prompt.size() - n;

        const double f = (double)mod.get_used() / (double)mod.size();
        LOG_INF("%s: ngram_mod occupancy = %zu/%zu (%.2f)\n", __func__, mod.get_used(), mod.size(), f);

        constexpr double f_thold = 0.25;
        if (f > f_thold) {
            LOG_WRN("%s: ngram_mod occupancy %.2f exceeds threshold (%.2f) - resetting\n", __func__, f, f_thold);

            mod.reset();
        }
    }

    void draft_one(
            llama_seq_id seq_id,
            common_speculative_draft_params & dparams) {
        auto & sinfo = sinfos[seq_id];
        auto & result = *dparams.result;

        const auto & prompt = *dparams.prompt;

        sinfo.n_draft_last = 0;

        const size_t cur_len = prompt.size();
        if (cur_len < mod.get_n()) {
            return;
        }

        const size_t n = mod.get_n();

        // add new ngrams in chunks
        if (sinfo.i_last + 32 < cur_len) {
            for (size_t i = sinfo.i_last; i < cur_len - n; ++i) {
                mod.add(prompt.data() + i);
            }

            sinfo.i_last = cur_len - n;
        }

        result.resize(n + params.n_max);
        for (size_t i = 0; i < n - 1; ++i) {
            result[i] = prompt.at(cur_len - n + 1 + i);
        }
        result[n - 1] = dparams.id_last;

        for (int i = 0; i < params.n_max; ++i) {
            const llama_token token = mod.get(result.data() + i);
            if (token == common_ngram_mod::EMPTY) {
                if (i < params.n_min) {
                    result.clear();
                    return;
                }

                result.resize(n + i);
                break;
            }
            result[n + i] = token;
        }

        // only return the m tokens that were drafted
        for (size_t i = 0; n + i < result.size(); ++i) {
            result[i] = result[n + i];
        }
        result.resize(result.size() - n);

        // store length of drafted n‑gram for later acceptance analysis
        sinfo.n_draft_last = result.size();
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            draft_one(seq_id, dp);
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted) override {
        auto & sinfo = sinfos[seq_id];

        // compute acceptance fraction if we have a recorded draft length
        if (sinfo.n_draft_last > 0) {
            const double f_acc = (double)n_accepted / (double)sinfo.n_draft_last;
            if (f_acc < 0.5) {
                sinfo.n_low++;
                if (sinfo.n_low >= 3) {
                    if (verbose) {
                        LOG_WRN("%s: low acceptance streak (%d) – resetting ngram_mod\n", __func__, sinfo.n_low);
                    }

                    mod.reset();
                    n_low = 0;
                }
            } else {
                sinfo.n_low = 0;
            }
        }
    }

    bool need_embd() const override {
        return false;
    }
};

struct common_speculative_impl_ngram_cache : public common_speculative_impl {
    common_params_speculative_ngram_cache params;

    uint16_t n_draft;

    bool save_dynamic;
    bool save_static;

    struct seq_info {
        size_t cache_size = 0; // number of tokens in n-gram cache

        common_ngram_cache ngram_cache_context;
        common_ngram_cache ngram_cache_dynamic;
        common_ngram_cache ngram_cache_static;
    };

    std::vector<seq_info> sinfos;

    common_speculative_impl_ngram_cache(
            const common_params_speculative & params,
            uint32_t n_seq,
            uint16_t n_draft,
            const std::string & path_static,
            const std::string & path_dynamic,
            bool save_dynamic,
            bool save_static)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE, n_seq)
        , params(params.ngram_cache)
        , n_draft(n_draft)
        , save_dynamic(save_dynamic)
        , save_static(save_static)
    {
        sinfos.resize(n_seq);

        if (!path_static.empty()) {
            try {
                auto ngram_cache_static = common_ngram_cache_load(path_static);

                for (auto & sinfo : sinfos) {
                    sinfo.ngram_cache_static = ngram_cache_static;
                }
            } catch (...) {
                LOG_ERR("failed to open static lookup cache: %s", path_static.c_str());
                GGML_ABORT("Couldn't read static lookup cache");
            }
        }

        if (!path_dynamic.empty()) {
            try {
                auto ngram_cache_dynamic = common_ngram_cache_load(path_dynamic);

                for (auto & sinfo : sinfos) {
                    sinfo.ngram_cache_dynamic = ngram_cache_dynamic;
                }
            } catch (...) {
                LOG_ERR("failed to open dynamic lookup cache: %s", path_dynamic.c_str());
                GGML_ABORT("Couldn't read dynamic lookup cache");
            }
        }
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    void draft_one(
            llama_seq_id seq_id,
            common_speculative_draft_params & dparams) {
        auto & sinfo = sinfos[seq_id];
        auto & result = *dparams.result;

        const auto & prompt = *dparams.prompt;

        if (sinfo.cache_size < prompt.size() + 1) {
            llama_tokens tokens_new;
            tokens_new.reserve(prompt.size() + 1 - sinfo.cache_size);
            for (size_t j = sinfo.cache_size; j < prompt.size(); ++j) {
                tokens_new.push_back(prompt[j]);
            }
            tokens_new.push_back(dparams.id_last); // add the last token

            // Update context ngram cache with new dparams.prompt:
            common_ngram_cache_update(
                    sinfo.ngram_cache_context,
                    LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                    tokens_new, tokens_new.size(), false);
            sinfo.cache_size = prompt.size() + 1;
        }

        llama_tokens inp;
        inp.reserve(prompt.size() + 1);
        for (size_t j = 0; j < prompt.size(); ++j) {
            inp.push_back(prompt[j]);
        }
        inp.push_back(dparams.id_last);

        result.push_back(dparams.id_last);

        common_ngram_cache_draft(
                inp, result, n_draft, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                sinfo.ngram_cache_context,
                sinfo.ngram_cache_dynamic,
                sinfo.ngram_cache_static);

        if (result.size() > 0) {
            // delete first token in result (which is the id_last token)
            result.erase(result.begin());
        }
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            draft_one(seq_id, dp);
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/) override {
        // noop
    }

    bool need_embd() const override {
        return false;
    }
};

struct common_speculative {
    std::vector<std::unique_ptr<common_speculative_state>> impls; // list of implementations to use and their states
    common_speculative_state * curr_impl = nullptr; // current implementation in use (for stats)
};

static common_ngram_map get_common_ngram_map(
        common_speculative_type type,
        const common_params_speculative_ngram_map & config) {
    uint16_t size_key   = config.size_n;
    uint16_t size_value = config.size_m;
    bool     key_only   = type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K;
    uint16_t min_hits   = config.min_hits;

    return common_ngram_map(size_key, size_value, key_only, min_hits);
}

static common_speculative_impl_ngram_cache create_state_ngram_cache(
        const common_speculative_config & config,
        uint32_t n_seq,
        const std::string & path_static,
        const std::string & path_dynamic) {
    uint16_t n_draft = 8; // TODO get from config?

    // TODO bool param in common/common.h to set save_static/save_dynamic?
    bool save_static = false;
    bool save_dynamic = false;

    common_speculative_impl_ngram_cache state(config.params, n_seq, n_draft, path_static, path_dynamic, save_static, save_dynamic);

    return state;
}

std::string common_speculative_type_name_str(const std::vector<common_speculative_type> & types) {
    std::string result;

    for (size_t i = 0; i < types.size(); i++) {
        if (i > 0) {
            result += ",";
        }
        result += common_speculative_type_to_str(types[i]);
    }
    return result;
}

const char * common_speculative_all_types_str() {
    static std::string all_types_str = []() {
        std::vector<common_speculative_type> types;
        types.reserve(COMMON_SPECULATIVE_TYPE_COUNT);
        for (int i = 0; i < COMMON_SPECULATIVE_TYPE_COUNT; i++) {
            types.push_back((common_speculative_type) i);
        }
        return common_speculative_type_name_str(types);
    }();
    return all_types_str.c_str();
}

std::string common_speculative_type_to_str(common_speculative_type type) {
    switch (type) {
        case COMMON_SPECULATIVE_TYPE_NONE:          return "none";
        case COMMON_SPECULATIVE_TYPE_DRAFT:         return "draft";
        case COMMON_SPECULATIVE_TYPE_EAGLE3:        return "eagle3";
        case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE:  return "ngram_simple";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:   return "ngram_map_k";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: return "ngram_map_k4v";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:     return "ngram_mod";
        case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE:   return "ngram_cache";
        case COMMON_SPECULATIVE_TYPE_MTP:           return "mtp";
        case COMMON_SPECULATIVE_TYPE_NEXTN:         return "nextn";
        default:                                    return "unknown";
    }
}

std::vector<common_speculative_type> common_speculative_types_from_names(const std::vector<std::string> & names) {
    std::vector<common_speculative_type> types;
    types.reserve(names.size());

    for (const auto & name : names) {
        auto type = common_speculative_type_from_name_map.find(name);
        if (type != common_speculative_type_from_name_map.end()) {
            if (type->second == COMMON_SPECULATIVE_TYPE_NONE) {
                return std::vector<common_speculative_type> { COMMON_SPECULATIVE_TYPE_NONE };
            }
            types.push_back(type->second);
            continue;
        }
        throw std::invalid_argument("unknown speculative type: " + name);
    }

    return types;
}

common_speculative_type common_speculative_type_from_name(const std::string & name) {
    const auto it = common_speculative_type_from_name_map.find(name);
    if (it == common_speculative_type_from_name_map.end()) {
        return COMMON_SPECULATIVE_TYPE_COUNT;
    }
    return it->second;
}

bool common_speculative_is_mtmd_safe(enum common_speculative_type type) {
    switch (type) {
        case COMMON_SPECULATIVE_TYPE_MTP:
        case COMMON_SPECULATIVE_TYPE_NEXTN:
        case COMMON_SPECULATIVE_TYPE_EAGLE3:
            return true;
        default:
            return false;
    }
}

bool common_speculative_all_impls_mtmd_safe(const common_speculative * spec) {
    if (!spec) {
        return true;
    }
    for (const auto & impl : spec->impls) {
        if (!common_speculative_is_mtmd_safe(impl->type)) {
            return false;
        }
    }
    return true;
}

bool common_speculative_is_compat(llama_context * ctx_tgt) {
    auto * mem = llama_get_memory(ctx_tgt);
    if (mem == nullptr) {
        return false;
    }

    bool res = true;

    llama_memory_clear(mem, true);

    // eval 2 tokens to check if the context is compatible
    std::vector<llama_token> tmp;
    tmp.push_back(0);
    tmp.push_back(0);

    int ret = llama_decode(ctx_tgt, llama_batch_get_one(tmp.data(), tmp.size()));
    if (ret != 0) {
        LOG_ERR("%s: llama_decode() failed: %d\n", __func__, ret);
        res = false;
        goto done;
    }

    // try to remove the last tokens
    if (!llama_memory_seq_rm(mem, 0, 1, -1)) {
        LOG_WRN("%s: the target context does not support partial sequence removal\n", __func__);
        res = false;
        goto done;
    }

done:
    llama_memory_clear(mem, true);
    llama_synchronize(ctx_tgt);

    return res;
}

// initialization of the speculative decoding system
//
common_speculative * common_speculative_init(
        common_params_speculative & params,
        llama_context             * ctx_tgt) {
    llama_context * ctx_dft = nullptr;
    // Gemma4 MTP loads the assistant into the target model (llama_model_load_mtp_from_file); no second context.
    if (params.model_dft && params.type != COMMON_SPECULATIVE_TYPE_MTP) {
        ctx_dft = llama_init_from_model(params.model_dft, params.cparams_dft);
        if (ctx_dft == nullptr) {
            LOG_ERR("%s", "failed to create draft context\n");
            return nullptr;
        }
    }

    if (params.type == COMMON_SPECULATIVE_TYPE_MTP) {
        const llama_model * model_tgt = llama_get_model(ctx_tgt);
        const llama_model * model_mtp = llama_model_get_mtp_assistant(model_tgt);

        if (!model_mtp) {
            LOG_ERR("%s: MTP requires the assistant GGUF loaded into the target (CLI: --spec-type mtp with --mtp-head or --model-draft)\n", __func__);
            return nullptr;
        }
        if (!common_speculative_mtp_arch_ok(model_tgt, model_mtp)) {
            LOG_ERR("%s: MTP requires target arch gemma4 and assistant arch gemma4_assistant\n", __func__);
            return nullptr;
        }
        if (!common_speculative_are_compatible_mtp(model_tgt, model_mtp)) {
            LOG_ERR("%s: MTP assistant failed vocab compatibility check\n", __func__);
            return nullptr;
        }
    }

    if (params.type == COMMON_SPECULATIVE_TYPE_NEXTN) {
        const llama_model * model_tgt = llama_get_model(ctx_tgt);
        if (!params.model_dft) {
            LOG_ERR("%s: NextN requires a second model load from the same GGUF (CLI: --spec-type nextn with --model-draft)\n", __func__);
            if (ctx_dft) {
                llama_free(ctx_dft);
            }
            return nullptr;
        }
        if (!common_speculative_are_compatible_nextn(model_tgt, params.model_dft)) {
            LOG_ERR("%s: NextN failed arch/vocab check (target qwen35/qwen35moe + draft qwen35_mtp/qwen35moe_mtp from same tokenizer)\n", __func__);
            if (ctx_dft) {
                llama_free(ctx_dft);
            }
            return nullptr;
        }
    }

    // Compute the implementations to use based on the config and their order of preference
    std::vector<common_speculative_config> configs = {}; // list of speculative configs to try
    {
        uint32_t enabled_configs = common_get_enabled_speculative_configs(params.types);

        bool has_draft_model_path = !params.draft.mparams.path.empty();

        bool has_draft_simple = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE));
        bool has_draft_eagle3 = false; // TODO PR-18039: if params.speculative.eagle3
        bool has_mtp = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_DRAFT_MTP)) && params.draft.ctx_dft != nullptr;

        bool has_ngram_cache   = (params.type == COMMON_SPECULATIVE_TYPE_NGRAM_CACHE);
        bool has_ngram_simple  = (params.type == COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE);
        bool has_ngram_map_k   = (params.type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K);
        bool has_ngram_map_k4v = (params.type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V);
        bool has_ngram_mod     = (params.type == COMMON_SPECULATIVE_TYPE_NGRAM_MOD);
        bool has_mtp           = (params.type == COMMON_SPECULATIVE_TYPE_MTP);
        bool has_nextn         = (params.type == COMMON_SPECULATIVE_TYPE_NEXTN);

        // when adding a new type - update here the logic above
        static_assert(COMMON_SPECULATIVE_TYPE_COUNT == 9);

        // this list here defines the priority of the speculators
        // the one with highest priority are listed first
        if (has_ngram_simple) {
            // This implementation can guess a lot of tokens without any draft model.
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE, params));
        }
        if (has_ngram_map_k) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K, params));
        }
        if (has_ngram_map_k4v) {
            // This implementation can guess tokens with high acceptance rate but is more expensive.
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V, params));
        }
        if (has_ngram_mod) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MOD, params));
        }
        if (has_ngram_cache) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE, params));
        }
        if (has_draft) {
            if (has_mtp) {
                configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_MTP, params));
            } else if (has_nextn) {
                configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NEXTN, params));
            } else {
                configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DRAFT, params));
            }
        }
        if (has_draft_eagle3) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3, params));
        }
        if (has_mtp) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DRAFT_MTP, params));
        }
    }

    std::vector<std::unique_ptr<common_speculative_impl>> impls = {};

    for (const common_speculative_config & config : configs) {
        LOG_INF("%s: adding speculative implementation '%s'\n", __func__, common_speculative_type_to_str(config.type).c_str());
        switch (config.type) {
            case COMMON_SPECULATIVE_TYPE_NONE:
                break;
            case COMMON_SPECULATIVE_TYPE_DRAFT: {
                impls.push_back(std::make_unique<common_speculative_state_draft>(config.type,
                    /* .ctx_tgt      = */ ctx_tgt,
                    /* .ctx_dft      = */ ctx_dft,
                    /* .replacements = */ params.replacements
                ));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_eagle3>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_MTP: {
                impls.push_back(std::make_unique<common_speculative_state_draft_mtp>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_MTP: {
                impls.push_back(std::make_unique<common_speculative_state_mtp>(config.type, ctx_tgt));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NEXTN: {
                impls.push_back(std::make_unique<common_speculative_state_nextn>(config.type, ctx_tgt, ctx_dft));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE: {
                common_ngram_map ngram_map = get_common_ngram_map(config.type, config.params.ngram_simple);

                uint16_t ngram_size_key   = ngram_map.size_key;
                uint16_t mgram_size_value = ngram_map.size_value;

                auto config_simple = common_ngram_simple_config {
                    /* .size_ngram = */ ngram_size_key,
                    /* .size_mgram = */ mgram_size_value
                };
                auto state = std::make_unique<common_speculative_impl_ngram_simple>(
                    /* .params = */ config.params,
                    /* .n_seq  = */ n_seq,
                    /* .state  = */ config_simple
                );
                impls.push_back(std::move(state));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_map_k>(
                            config.params, get_common_ngram_map(config.type, config.params.ngram_map_k), n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MOD: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_mod>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE: {
                auto state = create_state_ngram_cache(
                        config, n_seq,
                        params.ngram_cache.lookup_cache_static,
                        params.ngram_cache.lookup_cache_dynamic);
                impls.push_back(std::make_unique<common_speculative_impl_ngram_cache>(state));
                break;
            }
            default:
                break;
        }
    }

    if (impls.empty()) {
        LOG_WRN("%s: no implementations specified for speculative decoding\n", __func__);
        return nullptr;
    }

    auto * result = new common_speculative {
        /* .impls = */ std::move(impls)
    };

    return result;
}

void common_speculative_free(common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    delete spec;
}

void common_speculative_set_seq_id(common_speculative * spec, llama_seq_id seq_id) {
    if (spec == nullptr) {
        return;
    }
    for (auto & impl : spec->impls) {
        if (impl->type == COMMON_SPECULATIVE_TYPE_MTP) {
            static_cast<common_speculative_state_mtp *>(impl.get())->seq_id = seq_id;
        }
    }
}

void common_speculative_set_h_idx(common_speculative * spec, int batch_idx) {
    if (spec == nullptr) {
        return;
    }
    for (auto & impl : spec->impls) {
        if (impl->type == COMMON_SPECULATIVE_TYPE_MTP) {
            static_cast<common_speculative_state_mtp *>(impl.get())->h_idx = batch_idx;
        } else if (impl->type == COMMON_SPECULATIVE_TYPE_NEXTN) {
            static_cast<common_speculative_state_nextn *>(impl.get())->h_idx = batch_idx;
        }
    }
}

void common_speculative_begin(common_speculative * spec, const llama_tokens & prompt) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls) {
        common_time_meas tm(impl->t_begin_us, !impl->gen_perf);
        impl->begin(seq_id, prompt);
        impl->n_call_begin++;
    }
}

bool common_speculative_process(common_speculative * spec, const llama_batch & batch) {
    bool result = true;

    if (spec == nullptr) {
        return result;
    }

    for (auto & impl : spec->impls) {
        result = result && impl->process(batch);
    }

    return result;
}

void common_speculative_accept(common_speculative * spec, uint16_t n_accepted) {
    // Trace BEFORE the early-out: the n_accepted==0 case is a zero-accept event we want to record.
    if (mtp_tracer().enabled && spec && spec->curr_impl
            && spec->curr_impl->type == COMMON_SPECULATIVE_TYPE_MTP) {
        static_cast<common_speculative_state_mtp *>(spec->curr_impl)
            ->trace_emit_accept((int) n_accepted);
    }

    if (n_accepted == 0) {
        return;
    }

    auto & dparams = spec->dparams;

    {
        int n_drafting = 0;

        for (auto & dp : dparams) {
            GGML_ASSERT(!dp.drafting || dp.result->empty());

            if (dp.drafting) {
                n_drafting++;
            }
        }

        if (n_drafting == 0) {
            return;
        }
    }

    for (auto & impl : spec->impls) {
        {
            common_time_meas tm(impl->t_draft_us, !impl->gen_perf);
            impl->draft(dparams);
            impl->n_call_draft++;
        }

        int n_drafting = 0;

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) dparams.size(); ++seq_id) {
            auto & dp = dparams[seq_id];

            auto & result = *dp.result;

            // a new draft has been sampled
            if (dp.drafting && !result.empty()) {
                dp.drafting = false;

                if (dp.n_max > 0) {
                    if (!result.empty() && (int) result.size() > dp.n_max) {
                        LOG_DBG("%s: truncating draft to %d tokens\n", __func__, dp.n_max);
                        result.resize(dp.n_max);
                    }
                }

                if (!result.empty()) {
                    LOG_DBG("%s: called impl %s, hist size = %zu, call_count = %zu, gen = %zu\n", __func__,
                            common_speculative_type_to_str(impl.get()->type).c_str(), dp.prompt->size(),
                            impl.get()->n_call_draft, result.size());

                    // remember which implementation was used
                    spec->impl_last[seq_id] = impl.get();

                    impl->n_gen_drafts++;
                    impl->n_gen_tokens += result.size();
                }
            }

            if (dp.drafting) {
                n_drafting++;
            }
        }

        if (n_drafting == 0) {
            break;
        }
    }

    // these sequences failed to generate a draft
    for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) dparams.size(); ++seq_id) {
        auto & dp = dparams[seq_id];

        if (dp.drafting) {
            dp.drafting = false;
        }
    }
}

void common_speculative_accept(common_speculative * spec, llama_seq_id seq_id, uint16_t n_accepted) {
    common_speculative_impl * impl = spec->impl_last[seq_id];

    GGML_ASSERT(impl);

    // TODO: currently only the implementation that generated the draft is used to accept it
    //       however, some implementations (such as MTP) need to also "see" the accepted tokens
    //       extend `common_speculative_impl::accept()` with an extra argument `bool is_other` to
    //       inform the implementation if the accepted tokens are from another implementation and
    //       pass the accepted tokens to all remaining implementations using `is_other == true`
    {
        common_time_meas tm(impl->t_accept_us, !impl->gen_perf);
        if (n_accepted > 0) {
            impl->n_acc_drafts++;
            impl->n_acc_tokens += n_accepted;
        }

        impl->accept(seq_id, n_accepted);
        impl->n_call_accept++;
    }
}

void common_speculative_prepare_next(common_speculative * spec, llama_token id_last) {
    if (spec == nullptr) {
        return;
    }
    for (auto & impl : spec->impls) {
        impl->prepare_next(id_last);
    }
}

void common_speculative_cancel(common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }
    for (auto & impl : spec->impls) {
        impl->cancel();
    }
}

void common_speculative_print_stats(const common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    for (const auto & impl : spec->impls) {
        std::string str_perf;
        if (impl->gen_perf) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3) << impl->t_begin_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_draft_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_accept_us / 1000.0;
            str_perf = ", dur(b,g,a) = " + oss.str() + " ms";
        } else {
            str_perf = "";
        }

        LOG_INF("statistics %s: #calls(b,g,a) = %zu %zu %zu, #gen drafts = %zu, #acc drafts = %zu, #gen tokens = %zu, #acc tokens = %zu%s\n",
                common_speculative_type_to_str(impl->type).c_str(),
                impl->n_call_begin, impl->n_call_draft, impl->n_call_accept,
                impl->n_gen_drafts,
                impl->n_acc_drafts,
                impl->n_gen_tokens,
                impl->n_acc_tokens,
                str_perf.c_str());
    }
}
