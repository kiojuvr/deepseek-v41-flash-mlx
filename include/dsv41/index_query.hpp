#pragma once
#include "dsv41/global_kv.hpp"
#include "dsv41/linear.hpp"
#include <cstddef>
#include <cstdint>
namespace dsv41 {
// Top-k reference synchronizes to CPU; output is re-sorted by position.
// strict rejects an exact boundary tie. Non-strict mode deterministically keeps
// the lowest row ID and records the cross-backend policy boundary.
struct IndexTieRecord {
 int layer=-1;
 std::uint64_t token=0;
 int selected_id=0,excluded_id=0;
 float selected_score=0,excluded_score=0;
 bool candidates=false,tied=false;
};
std::vector<std::int32_t> index_topk_reference(const mlx::core::array& scores,int offset,
 bool strict=true,IndexTieRecord* tie=nullptr);
std::size_t index_tie_count();
void reset_index_tie_count();
std::vector<IndexTieRecord> index_tie_records();
void reset_index_tie_records();
mlx::core::array restore_index_reference(const GlobalKVState& state);
// Level one of the decoder's two-level top-k: keep the top blocks by best position.
std::vector<std::uint8_t> select_candidate_blocks_reference(const std::vector<float>& logits,
 int compress_len,int topk_blocks,int block_size);
struct IndexSelection {
 std::vector<std::int32_t> rows;       // +offset, position-sorted
 std::vector<std::uint8_t> candidates; // bool mask over compressed positions; empty unless candidate source
 mlx::core::array device_rows{0};       // relative compressed row IDs, position-sorted
 mlx::core::array device_candidates{0}; // uint8 mask for candidate sources/consumers
};
class IndexQueryReference {
public:
 explicit IndexQueryReference(WeightCatalog& catalog,int layer,
   bool is_candidate_source=false,bool uses_candidates=false,
   int candidate_topk_blocks=2048,int candidate_block_size=8);
 // qr is attention's normalized wq_a output, x is the normalized attention input.
 IndexSelection forward(const mlx::core::array& x,const mlx::core::array& qr,
   const GlobalKVState& state,std::uint64_t position,int window_offset,
   const std::vector<std::uint8_t>* incoming_candidates=nullptr) const;
 // Computes query/cache scores, causal/candidate masks and stable lowest-ID Top-K
 // in one GPU graph. Host selection metadata is diagnostic-only; device rows
 // and candidate masks remain authoritative on the optimized path.
 std::vector<IndexSelection> forward_chunk(const mlx::core::array& x,const mlx::core::array& qr,
   const std::vector<GlobalKVState>& cache_prefixes,std::uint64_t start,
   const std::vector<std::vector<std::uint8_t>>* incoming_candidates=nullptr,
   const std::vector<mlx::core::array>* incoming_device_candidates=nullptr,
   bool force_diagnostics=false) const;
private:
 int layer_,ratio_,candidate_topk_blocks_,candidate_block_size_;
 bool is_candidate_source_,uses_candidates_;
 PackedLinearReference query_;
 mlx::core::array weights_;
};
}
