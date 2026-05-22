#pragma once

#include "llama.h"
#include "common.h"

struct common_speculative;

// comma separated list the provided types
std::string common_speculative_type_name_str(const std::vector<enum common_speculative_type> & types);

// comma separated list of all types
const char * common_speculative_all_types_str();

// parse user provided types
std::vector<enum common_speculative_type> common_speculative_types_from_names(const std::vector<std::string> & names);

// convert string to type
enum common_speculative_type common_speculative_type_from_name(const std::string & name);

// convert type to string
std::string common_speculative_type_to_str(enum common_speculative_type type);

// check if the llama_context is compatible for speculative decoding
// note: clears the memory of the context
bool common_speculative_is_compat(llama_context * ctx_tgt);

// True for speculative modes that do not consume prompt_tgt in common_speculative_draft()
// (MTP / NextN / EAGLE3 read target KV / hidden states instead). Safe to combine with --mmproj.
bool common_speculative_is_mtmd_safe(enum common_speculative_type type);

// True iff every registered impl is mtmd-safe (rejects mixed chains e.g. ngram + draft model).
bool common_speculative_all_impls_mtmd_safe(const common_speculative * spec);

common_speculative * common_speculative_init(
        common_params_speculative & params,
        llama_context             * ctx_tgt);

void common_speculative_free(common_speculative * spec);

struct common_speculative_draft_params {
    // this flag is used to chain the drafts through all the available implementations
    // after the first successful draft from an implementation, we set it
    //   to false to prevent further drafts for that sequence
    // at the end of the draft() call, all drafting flags will be reset to false
    bool drafting = false;

    // overrides individual configurations (-1 disabled)
    // can be used to constraint the max draft based on the remaining context size
    int32_t n_max = -1;

    llama_pos   n_past;
    llama_token id_last;

    // TODO: remove in the future by keeping track of the prompt from the _begin() call and the consecutive accept calls
    const llama_tokens * prompt;

    // the generated draft from the last _draft() call
    llama_tokens * result;
};

common_speculative_draft_params & common_speculative_get_draft_params(common_speculative * spec, llama_seq_id seq_id);

// optionally call once at the beginning of a new generation
void common_speculative_begin(common_speculative * spec, llama_seq_id seq_id, const llama_tokens & prompt);

// set target-side sequence id used by implementations that read from the target's KV memory
// (currently only used by the MTP implementation; safe no-op for others)
void common_speculative_set_seq_id(common_speculative * spec, llama_seq_id seq_id);

// Set the output index in the target's most recent decode whose embeddings should be read
// as h_prev for the next MTP draft. -1 means "last output" (default).
// In speculative verification, after partial draft acceptance the last batch output corresponds
// to a rejected draft; the correct h_prev is at the last *accepted* batch index.
// Safe no-op for non-MTP implementations.
void common_speculative_set_h_idx(common_speculative * spec, int batch_idx);

// sample up to n_draft tokens and add them to the batch using the draft model
llama_tokens common_speculative_draft(
                     common_speculative * spec,
        const common_params_speculative & params,
                     const llama_tokens & prompt,
                            llama_token   id_last);

// true if any implementation requires target embeddings to be extracted
bool common_speculative_need_embd(common_speculative * spec);

// generate drafts for the sequences specified with `common_speculative_get_draft_params`
void common_speculative_draft(common_speculative * spec);

// informs the speculative context that n_accepted tokens were accepted by the target model
void common_speculative_accept(common_speculative * spec, llama_seq_id, uint16_t n_accepted);

// After target sample/accept, submit MTP work for the next iteration so it can overlap
// server bookkeeping until the next common_speculative_draft() (pipeline depth-2, no optimistic token).
// Safe no-op for non-MTP implementations.
void common_speculative_prepare_next(common_speculative * spec, llama_token id_last);

// Drain any pending async draft from a previous prepare_next() and discard the result.
// MUST be called before the host mutates target KV in a way that would invalidate the
// snapshot (e.g. slot stop / release / new request seq_rm). Safe no-op when nothing is pending.
void common_speculative_cancel(common_speculative * spec);

// print statistics about the speculative decoding
void common_speculative_print_stats(const common_speculative * spec);
