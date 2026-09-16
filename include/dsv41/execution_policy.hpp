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
