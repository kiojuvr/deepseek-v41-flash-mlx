#include "dsv41/execution_policy.hpp"
#include "dsv41/deferred_decoder_plan.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {

constexpr const char* kVariables[] = {
    "DSV41_RUNTIME_LAYER_FINITE_CHECKS",
    "DSV41_RUNTIME_PACKED_EXPERT_BANK",
    "DSV41_RUNTIME_GROUP_SELECTED_EXPERTS",
    "DSV41_RUNTIME_RESIDENT_EXPERT_ATLAS",
    "DSV41_RUNTIME_COMPACT_EXPERT_BANK",
    "DSV41_RUNTIME_GROUPED_EXPERT_PIPELINE",
    "DSV41_RUNTIME_DECODE_STACK_GRAPH",
    "DSV41_RUNTIME_ROUTE_DIAGNOSTICS",
    "DSV41_RUNTIME_INDEX_DIAGNOSTICS",
    "DSV41_RUNTIME_CHUNK_ATTENTION",
    "DSV41_RUNTIME_BATCHED_SPLITK_QK",
    "DSV41_RUNTIME_BATCHED_SPLITK_QK_DIAGNOSTICS",
    "DSV41_RUNTIME_PACKED_CHUNK_ATTENTION",
    "DSV41_RUNTIME_WIDE_ATTENTION",
    "DSV41_RUNTIME_FIXED_TILE_ATTENTION",
    "DSV41_RUNTIME_FIXED_TILE_ATTENTION_DIAGNOSTICS",
    "DSV41_RUNTIME_RAGGED_TAIL_QK",
    "DSV41_RUNTIME_RAGGED_TAIL_AV",
    "DSV41_RUNTIME_LAYER_SWEEP",
    "DSV41_RUNTIME_DEFERRED_DECODER",
    "DSV41_RUNTIME_DEFERRED_DECODER_CLEAR_CACHE",
    "DSV41_RUNTIME_BATCHED_DENSE_QMM",
    "DSV41_RUNTIME_MLX_CACHE_LIMIT_BYTES",
    "DSV41_RUNTIME_LONG_CONTEXT_CACHE_LIMIT_BYTES",
    "DSV41_RUNTIME_EXPERT_IO_THREADS",
    "DSV41_RUNTIME_EXPERT_ASSIGNMENT_CHUNK",
};

void require(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

void clear_policy_environment() {
  for (const char* variable : kVariables) {
    if (unsetenv(variable) != 0) {
      throw std::runtime_error("cannot clear execution-policy environment");
    }
  }
}

void set_policy(const char* name, const char* value) {
  if (setenv(name, value, 1) != 0) {
    throw std::runtime_error("cannot set execution-policy environment");
  }
}

void check_production_defaults() {
  require(!dsv41::runtime_layer_finite_checks_enabled(), "finite scans must be off");
  require(dsv41::runtime_packed_expert_bank_enabled(), "packed expert banks must be on");
  require(!dsv41::runtime_group_selected_experts_enabled(), "group-selected banks must be off");
  require(dsv41::runtime_resident_expert_atlas_enabled(), "resident expert atlas must be on");
  require(!dsv41::runtime_compact_expert_bank_enabled(), "compact expert bank must be off");
  require(!dsv41::runtime_grouped_expert_pipeline_enabled(), "grouped pipeline candidate must be off");
  require(!dsv41::runtime_route_diagnostics_enabled(), "route diagnostics must be off");
  require(!dsv41::runtime_index_diagnostics_enabled(), "index diagnostics must be off");
  require(!dsv41::runtime_chunk_attention_enabled(), "legacy chunk attention must be off");
  require(dsv41::runtime_batched_splitk_qk_enabled(), "batched split-K QK must be on");
  require(!dsv41::runtime_batched_splitk_qk_diagnostics_enabled(), "split-K diagnostics must be off");
  require(!dsv41::runtime_packed_chunk_attention_enabled(), "packed comparison path must be off");
  require(!dsv41::runtime_wide_attention_enabled(), "wide candidate must be off");
  require(dsv41::runtime_fixed_tile_attention_enabled(), "fixed-tile attention must be on");
  require(!dsv41::runtime_fixed_tile_attention_diagnostics_enabled(), "fixed-tile diagnostics must be off");
  require(dsv41::runtime_ragged_tail_qk_enabled(), "ragged QK must be on");
  require(dsv41::runtime_ragged_tail_av_enabled(), "ragged AV must be on");
  require(dsv41::runtime_layer_sweep_enabled(), "layer sweep must be on");
  require(dsv41::runtime_deferred_decoder_enabled(),
          "qualified pending deferred decoder must be on");
  require(dsv41::runtime_deferred_decoder_clear_cache_enabled(),
          "qualified deferred decoder cache normalization must be on");
  require(!dsv41::runtime_batched_dense_qmm_enabled(), "batched dense QMM must be off");
  require(dsv41::runtime_mlx_cache_limit_bytes() == 0, "MLX cache limit must remain automatic");
  require(dsv41::runtime_long_context_cache_limit_bytes() == 17179869184ull,
          "long-context cache production limit mismatch");
  require(dsv41::runtime_effective_mlx_cache_limit_bytes(8191) == 0,
          "short production context must retain automatic cache policy");
  require(dsv41::runtime_effective_mlx_cache_limit_bytes(8192) == 17179869184ull,
          "long-context cache production boundary mismatch");
  require(dsv41::runtime_expert_io_threads() == 4, "expert I/O thread default must be four");
  require(dsv41::runtime_expert_assignment_chunk() == 128, "expert assignment chunk must be 128");
}

void check_reference_overrides() {
  set_policy("DSV41_RUNTIME_LAYER_FINITE_CHECKS", "1");
  set_policy("DSV41_RUNTIME_PACKED_EXPERT_BANK", "0");
  set_policy("DSV41_RUNTIME_RESIDENT_EXPERT_ATLAS", "0");
  set_policy("DSV41_RUNTIME_GROUPED_EXPERT_PIPELINE", "1");
  set_policy("DSV41_RUNTIME_INDEX_DIAGNOSTICS", "1");
  set_policy("DSV41_RUNTIME_BATCHED_SPLITK_QK", "0");
  set_policy("DSV41_RUNTIME_FIXED_TILE_ATTENTION", "0");
  set_policy("DSV41_RUNTIME_RAGGED_TAIL_QK", "0");
  set_policy("DSV41_RUNTIME_RAGGED_TAIL_AV", "0");
  set_policy("DSV41_RUNTIME_LAYER_SWEEP", "0");
  set_policy("DSV41_RUNTIME_DEFERRED_DECODER", "0");
  set_policy("DSV41_RUNTIME_DEFERRED_DECODER_CLEAR_CACHE", "0");
  set_policy("DSV41_RUNTIME_EXPERT_IO_THREADS", "1");
  set_policy("DSV41_RUNTIME_LONG_CONTEXT_CACHE_LIMIT_BYTES", "17179869184");

  require(dsv41::runtime_layer_finite_checks_enabled(), "finite override failed");
  require(!dsv41::runtime_packed_expert_bank_enabled(), "packed-bank override failed");
  require(!dsv41::runtime_resident_expert_atlas_enabled(), "resident-atlas override failed");
  require(dsv41::runtime_grouped_expert_pipeline_enabled(), "grouped pipeline override failed");
  require(dsv41::runtime_index_diagnostics_enabled(), "index-diagnostics override failed");
  require(!dsv41::runtime_batched_splitk_qk_enabled(), "split-K override failed");
  require(!dsv41::runtime_fixed_tile_attention_enabled(), "fixed-tile override failed");
  require(!dsv41::runtime_ragged_tail_qk_enabled(), "ragged-QK override failed");
  require(!dsv41::runtime_ragged_tail_av_enabled(), "ragged-AV override failed");
  require(!dsv41::runtime_layer_sweep_enabled(), "layer-sweep override failed");
  require(!dsv41::runtime_deferred_decoder_enabled(), "deferred-decoder fallback override failed");
  require(!dsv41::runtime_deferred_decoder_clear_cache_enabled(),
          "deferred-decoder cache-normalization fallback failed");
  require(dsv41::runtime_expert_io_threads() == 1, "expert-I/O override failed");
  require(dsv41::runtime_effective_mlx_cache_limit_bytes(8191) == 0,
          "short context must retain automatic cache policy");
  require(dsv41::runtime_effective_mlx_cache_limit_bytes(8192) == 17179869184ull,
          "long-context cache candidate boundary mismatch");
  set_policy("DSV41_RUNTIME_MLX_CACHE_LIMIT_BYTES", "8589934592");
  require(dsv41::runtime_effective_mlx_cache_limit_bytes(32768) == 8589934592ull,
          "explicit cache limit must override long-context policy");
  set_policy("DSV41_RUNTIME_MLX_CACHE_LIMIT_BYTES", "0");
  set_policy("DSV41_RUNTIME_LONG_CONTEXT_CACHE_LIMIT_BYTES", "0");
  require(dsv41::runtime_effective_mlx_cache_limit_bytes(32768) == 0,
          "explicit long-context cache fallback disable failed");
}

void check_deferred_decoder_geometry() {
  require(!dsv41::runtime_decode_stack_graph_enabled(), "decode graph default must be off");
  set_policy("DSV41_RUNTIME_DECODE_STACK_GRAPH", "1");
  set_policy("DSV41_RUNTIME_PACKED_EXPERT_BANK", "1");
  set_policy("DSV41_RUNTIME_RESIDENT_EXPERT_ATLAS", "1");
  require(dsv41::runtime_defer_decode_layer_eval(1), "resident decode must defer");
  require(!dsv41::runtime_defer_decode_layer_eval(2), "prefill must retain boundaries");
  set_policy("DSV41_RUNTIME_RESIDENT_EXPERT_ATLAS", "0");
  require(!dsv41::runtime_defer_decode_layer_eval(1), "nonresident must retain boundaries");
  set_policy("DSV41_RUNTIME_DECODE_STACK_GRAPH", "invalid");
  bool invalid_graph=false;
  try{(void)dsv41::runtime_decode_stack_graph_enabled();}catch(const std::runtime_error&){invalid_graph=true;}
  require(invalid_graph,"invalid graph policy must fail");
  set_policy("DSV41_RUNTIME_DECODE_STACK_GRAPH", "0");
  require(dsv41::deferred_decoder_suffix_rows(20) == 2414,
          "layer 20 decoder suffix must include all downstream raw dependencies");
  require(dsv41::deferred_decoder_suffix_rows(39) == 1,
          "final decoder layer must require one output row");
  const auto first = dsv41::deferred_decoder_span(16384, 20);
  require(first.first == 13970 && first.rows == 2414,
          "layer 20 suffix frontier mismatch");
  require(first.warm_first == 13843 && first.warm_rows == 127,
          "layer 20 raw-history warmup mismatch");
  for(std::size_t layer=20;layer<39;++layer) {
    require(dsv41::deferred_decoder_suffix_rows(layer) ==
              dsv41::deferred_decoder_suffix_rows(layer+1)+127,
            "decoder suffix must shrink by one raw window per layer");
  }
  require(!dsv41::should_defer_decoder(16383,8192), "short sweep must not defer");
  require(!dsv41::should_defer_decoder(16384,8191), "short remainder must not defer");
  require(dsv41::should_defer_decoder(16384,8192), "reviewed defer boundary mismatch");
  const auto short_step=dsv41::runtime_prefill_step(8191,true,true);
  require(short_step.tokens==4096&&short_step.action==dsv41::RuntimePrefillStep::Action::Full,
          "short production remainder must retain packed sweep");
  const auto boundary_step=dsv41::runtime_prefill_step(8192,true,true);
  require(boundary_step.tokens==4096&&boundary_step.action==dsv41::RuntimePrefillStep::Action::Full,
          "isolated 8K sweep must not create an unfinished decoder");
  const auto bounded_step=dsv41::runtime_prefill_step(32768,true,true);
  require(bounded_step.tokens==16384&&
          bounded_step.action==dsv41::RuntimePrefillStep::Action::BeginDeferredDecoder,
          "production CED begin boundary mismatch");
  const auto finish_step=dsv41::runtime_prefill_step(16384,true,true,true);
  require(finish_step.tokens==16384&&
          finish_step.action==dsv41::RuntimePrefillStep::Action::FinishDeferredDecoder,
          "production CED finish boundary mismatch");
  const auto fallback_step=dsv41::runtime_prefill_step(32768,true,false);
  require(fallback_step.tokens==4096&&fallback_step.action==dsv41::RuntimePrefillStep::Action::Full,
          "full-decoder fallback schedule mismatch");
  const auto oracle_step=dsv41::runtime_prefill_step(1000,false,true);
  require(oracle_step.tokens==128&&oracle_step.action==dsv41::RuntimePrefillStep::Action::Full,
          "reference schedule must ignore CED policy");
  bool rejected=false;
  try { (void)dsv41::deferred_decoder_span(2540,20); }
  catch(const std::runtime_error&) { rejected=true; }
  require(rejected, "insufficient suffix warmup must be rejected");
}

}  // namespace

int main() {
  try {
    clear_policy_environment();
    check_production_defaults();
    check_reference_overrides();
    check_deferred_decoder_geometry();
    std::cout << "PASS: production defaults, reference overrides, and deferred decoder geometry\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
