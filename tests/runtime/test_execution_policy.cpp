#include "dsv41/execution_policy.hpp"

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
    "DSV41_RUNTIME_BATCHED_DENSE_QMM",
    "DSV41_RUNTIME_MLX_CACHE_LIMIT_BYTES",
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
  require(!dsv41::runtime_batched_dense_qmm_enabled(), "batched dense QMM must be off");
  require(dsv41::runtime_mlx_cache_limit_bytes() == 0, "MLX cache limit must remain automatic");
  require(dsv41::runtime_expert_io_threads() == 4, "expert I/O thread default must be four");
  require(dsv41::runtime_expert_assignment_chunk() == 128, "expert assignment chunk must be 128");
}

void check_reference_overrides() {
  set_policy("DSV41_RUNTIME_LAYER_FINITE_CHECKS", "1");
  set_policy("DSV41_RUNTIME_PACKED_EXPERT_BANK", "0");
  set_policy("DSV41_RUNTIME_RESIDENT_EXPERT_ATLAS", "0");
  set_policy("DSV41_RUNTIME_INDEX_DIAGNOSTICS", "1");
  set_policy("DSV41_RUNTIME_BATCHED_SPLITK_QK", "0");
  set_policy("DSV41_RUNTIME_FIXED_TILE_ATTENTION", "0");
  set_policy("DSV41_RUNTIME_RAGGED_TAIL_QK", "0");
  set_policy("DSV41_RUNTIME_RAGGED_TAIL_AV", "0");
  set_policy("DSV41_RUNTIME_LAYER_SWEEP", "0");
  set_policy("DSV41_RUNTIME_EXPERT_IO_THREADS", "1");

  require(dsv41::runtime_layer_finite_checks_enabled(), "finite override failed");
  require(!dsv41::runtime_packed_expert_bank_enabled(), "packed-bank override failed");
  require(!dsv41::runtime_resident_expert_atlas_enabled(), "resident-atlas override failed");
  require(dsv41::runtime_index_diagnostics_enabled(), "index-diagnostics override failed");
  require(!dsv41::runtime_batched_splitk_qk_enabled(), "split-K override failed");
  require(!dsv41::runtime_fixed_tile_attention_enabled(), "fixed-tile override failed");
  require(!dsv41::runtime_ragged_tail_qk_enabled(), "ragged-QK override failed");
  require(!dsv41::runtime_ragged_tail_av_enabled(), "ragged-AV override failed");
  require(!dsv41::runtime_layer_sweep_enabled(), "layer-sweep override failed");
  require(dsv41::runtime_expert_io_threads() == 1, "expert-I/O override failed");
}

}  // namespace

int main() {
  try {
    clear_policy_environment();
    check_production_defaults();
    check_reference_overrides();
    std::cout << "PASS: production execution-policy defaults and reference overrides\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
