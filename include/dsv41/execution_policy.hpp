#pragma once
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>

namespace dsv41 {
// Canonical tests keep eager per-layer finite scans. A full-path measurement may
// disable only these redundant scans while retaining shape/state checks and its
// chunk-boundary finite check. This does not change arithmetic or routing policy.
inline bool runtime_layer_finite_checks_enabled() {
 const char* value=std::getenv("DSV41_RUNTIME_LAYER_FINITE_CHECKS");
 if(value==nullptr||std::string_view(value)=="1") return true;
 if(std::string_view(value)=="0") return false;
 throw std::runtime_error("DSV41_RUNTIME_LAYER_FINITE_CHECKS must be 0 or 1");
}

// Packed routed experts are an opt-in promotion stage until full-MoE and
// full-backbone parity/resource checks have been reviewed.
inline bool runtime_packed_expert_bank_enabled() {
 const char* value=std::getenv("DSV41_RUNTIME_PACKED_EXPERT_BANK");
 if(value==nullptr||std::string_view(value)=="0") return false;
 if(std::string_view(value)=="1") return true;
 throw std::runtime_error("DSV41_RUNTIME_PACKED_EXPERT_BANK must be 0 or 1");
}

// Stack only the six routed, on-demand expert tensors for one invocation.
// This is independently opt-in while its allocation/performance is measured.
inline bool runtime_group_selected_experts_enabled() {
 const char* value=std::getenv("DSV41_RUNTIME_GROUP_SELECTED_EXPERTS");
 if(value==nullptr||std::string_view(value)=="0") return false;
 if(std::string_view(value)=="1") return true;
 throw std::runtime_error("DSV41_RUNTIME_GROUP_SELECTED_EXPERTS must be 0 or 1");
}

// Keep each layer's final packed expert buffers after its first use. This is
// the production-residency candidate; it is valid only with packed banks and
// remains opt-in until full-backbone memory/load checks are reviewed.
inline bool runtime_resident_expert_atlas_enabled() {
 const char* value=std::getenv("DSV41_RUNTIME_RESIDENT_EXPERT_ATLAS");
 if(value==nullptr||std::string_view(value)=="0") return false;
 if(std::string_view(value)=="1") return true;
 throw std::runtime_error("DSV41_RUNTIME_RESIDENT_EXPERT_ATLAS must be 0 or 1");
}

// Build one route-first compact bank containing only the experts selected by
// the current tile. It is mutually exclusive with the full resident atlas.
inline bool runtime_compact_expert_bank_enabled() {
 const char* value=std::getenv("DSV41_RUNTIME_COMPACT_EXPERT_BANK");
 if(value==nullptr||std::string_view(value)=="0") return false;
 if(std::string_view(value)=="1") return true;
 throw std::runtime_error("DSV41_RUNTIME_COMPACT_EXPERT_BANK must be 0 or 1");
}

// Copy route IDs/boundaries to the host only for qualification diagnostics.
// Non-resident paths still require host IDs for compact-bank construction;
// the full resident production path leaves this disabled.
inline bool runtime_route_diagnostics_enabled() {
 const char* value=std::getenv("DSV41_RUNTIME_ROUTE_DIAGNOSTICS");
 if(value==nullptr||std::string_view(value)=="0") return false;
 if(std::string_view(value)=="1") return true;
 throw std::runtime_error("DSV41_RUNTIME_ROUTE_DIAGNOSTICS must be 0 or 1");
}

// Preserve full host-visible index/tie records for oracle runs. Optimized
// resident prefill keeps selected rows and candidate masks device-authoritative.
inline bool runtime_index_diagnostics_enabled() {
 const char* value=std::getenv("DSV41_RUNTIME_INDEX_DIAGNOSTICS");
 if(value==nullptr||std::string_view(value)=="1") return true;
 if(std::string_view(value)=="0") return false;
 throw std::runtime_error("DSV41_RUNTIME_INDEX_DIAGNOSTICS must be 0 or 1");
}

// Batch all reuse-layer attention rows in one padded GPU graph. The
// token-serial reference remains the default until full-backbone semantic and
// performance qualification has been reviewed.
inline bool runtime_chunk_attention_enabled() {
 const char* value=std::getenv("DSV41_RUNTIME_CHUNK_ATTENTION");
 if(value==nullptr||std::string_view(value)=="0") return false;
 if(std::string_view(value)=="1") return true;
 throw std::runtime_error("DSV41_RUNTIME_CHUNK_ATTENTION must be 0 or 1");
}

// Batch the exact scalar-oracle Steel split-K QK topology across tokens. The
// existing scalar MLX graph remains the oracle and default until full-backbone
// qualification; softmax and the already-qualified AV batch remain unchanged.
inline bool runtime_batched_splitk_qk_enabled() {
 const char* value=std::getenv("DSV41_RUNTIME_BATCHED_SPLITK_QK");
 if(value==nullptr||std::string_view(value)=="0") return false;
 if(std::string_view(value)=="1") return true;
 throw std::runtime_error("DSV41_RUNTIME_BATCHED_SPLITK_QK must be 0 or 1");
}

// Qualification-only comparison of batched split-K scores with the native
// token-scalar Steel result. This deliberately synchronizes every eligible
// complete QK block.
inline bool runtime_batched_splitk_qk_diagnostics_enabled() {
 const char* value=std::getenv("DSV41_RUNTIME_BATCHED_SPLITK_QK_DIAGNOSTICS");
 if(value==nullptr||std::string_view(value)=="0") return false;
 if(std::string_view(value)=="1") return true;
 throw std::runtime_error("DSV41_RUNTIME_BATCHED_SPLITK_QK_DIAGNOSTICS must be 0 or 1");
}

// Decode packed pooled rows on device while preserving the oracle's exact
// selected-count groups and qualified Steel split-K/AV reductions.
inline bool runtime_packed_chunk_attention_enabled() {
 const char* value=std::getenv("DSV41_RUNTIME_PACKED_CHUNK_ATTENTION");
 if(value==nullptr||std::string_view(value)=="0") return false;
 if(std::string_view(value)=="1") return true;
 throw std::runtime_error("DSV41_RUNTIME_PACKED_CHUNK_ATTENTION must be 0 or 1");
}

// Phase-4 architecture candidate: one DwarfStar-style fused attention
// operation owns a complete layer chunk. It is separate from the qualified
// exact-shape materializer until the fixed full-backbone gate passes.
inline bool runtime_wide_attention_enabled() {
 const char* value=std::getenv("DSV41_RUNTIME_WIDE_ATTENTION");
 if(value==nullptr||std::string_view(value)=="0") return false;
 if(std::string_view(value)=="1") return true;
 throw std::runtime_error("DSV41_RUNTIME_WIDE_ATTENTION must be 0 or 1");
}

// Phase-4 arithmetic bridge: one dense [tokens,512] device plan and ten
// fixed 64-row tiles replace request-dependent shape groups.  It deliberately
// keeps the already-qualified split-K QK/BF16-PV reductions while the rejected
// row-serial fused reduction remains diagnostic-only.
inline bool runtime_fixed_tile_attention_enabled() {
 const char* value=std::getenv("DSV41_RUNTIME_FIXED_TILE_ATTENTION");
 if(value==nullptr||std::string_view(value)=="0") return false;
 if(std::string_view(value)=="1") return true;
 throw std::runtime_error("DSV41_RUNTIME_FIXED_TILE_ATTENTION must be 0 or 1");
}

// Qualification-only synchronization of the dense work list against the
// exact-shape producer path. Never enable in a performance measurement.
inline bool runtime_fixed_tile_attention_diagnostics_enabled() {
 const char* value=std::getenv("DSV41_RUNTIME_FIXED_TILE_ATTENTION_DIAGNOSTICS");
 if(value==nullptr||std::string_view(value)=="0") return false;
 if(std::string_view(value)=="1") return true;
 throw std::runtime_error("DSV41_RUNTIME_FIXED_TILE_ATTENTION_DIAGNOSTICS must be 0 or 1");
}

// Phase-4 fixed-topology correction: preserve native Steel short-N QK
// reductions through three device-side width classes without host grouping.
inline bool runtime_ragged_tail_qk_enabled() {
 const char* value=std::getenv("DSV41_RUNTIME_RAGGED_TAIL_QK");
 if(value==nullptr||std::string_view(value)=="0") return false;
 if(std::string_view(value)=="1") return true;
 throw std::runtime_error("DSV41_RUNTIME_RAGGED_TAIL_QK must be 0 or 1");
}

// Own prefill at the request boundary and execute it as a transactional
// layer-major sweep.  Decode remains on the one-token reference schedule.
// This is independently opt-in until the 2K full-path result is reviewed.
inline bool runtime_layer_sweep_enabled() {
 const char* value=std::getenv("DSV41_RUNTIME_LAYER_SWEEP");
 if(value==nullptr||std::string_view(value)=="0") return false;
 if(std::string_view(value)=="1") return true;
 throw std::runtime_error("DSV41_RUNTIME_LAYER_SWEEP must be 0 or 1");
}

// The reference schedule fixes every packed projection at M=1.  Optimized
// prefill may submit the complete chunk to the same MLX QMM primitive; route,
// state, logits and generation gates decide promotion rather than intermediate
// reduction-order identity.
inline bool runtime_batched_dense_qmm_enabled() {
 const char* value=std::getenv("DSV41_RUNTIME_BATCHED_DENSE_QMM");
 if(value==nullptr||std::string_view(value)=="0") return false;
 if(std::string_view(value)=="1") return true;
 throw std::runtime_error("DSV41_RUNTIME_BATCHED_DENSE_QMM must be 0 or 1");
}

inline std::size_t runtime_mlx_cache_limit_bytes() {
 const char* value=std::getenv("DSV41_RUNTIME_MLX_CACHE_LIMIT_BYTES");
 if(value==nullptr||std::string_view(value).empty()||std::string_view(value)=="0") return 0;
 try {
  std::size_t consumed=0; auto parsed=std::stoull(value,&consumed);
  if(consumed!=std::string_view(value).size())throw std::invalid_argument("trailing");
  return parsed;
 } catch(...) { throw std::runtime_error("DSV41_RUNTIME_MLX_CACHE_LIMIT_BYTES must be an integer byte count or 0"); }
}

inline std::size_t runtime_expert_io_threads() {
 const char* value=std::getenv("DSV41_RUNTIME_EXPERT_IO_THREADS");
 if(value==nullptr||std::string_view(value).empty()||std::string_view(value)=="0") return 1;
 try {
  std::size_t consumed=0; auto parsed=std::stoull(value,&consumed);
  if(consumed!=std::string_view(value).size()||parsed<1||parsed>32)throw std::invalid_argument("range");
  return parsed;
 } catch(...) { throw std::runtime_error("DSV41_RUNTIME_EXPERT_IO_THREADS must be an integer in 1..32 or 0"); }
}

inline std::size_t runtime_expert_assignment_chunk() {
 const char* value=std::getenv("DSV41_RUNTIME_EXPERT_ASSIGNMENT_CHUNK");
 if(value==nullptr||std::string_view(value).empty()||std::string_view(value)=="0") return 128;
 try { std::size_t consumed=0; auto parsed=std::stoull(value,&consumed);
  if(consumed!=std::string_view(value).size()||parsed<128||parsed>768||(parsed%128)!=0) throw std::invalid_argument("range");
  return parsed;
 } catch(...) { throw std::runtime_error("DSV41_RUNTIME_EXPERT_ASSIGNMENT_CHUNK must be a multiple of 128 in 128..768 or 0"); }
}

}
