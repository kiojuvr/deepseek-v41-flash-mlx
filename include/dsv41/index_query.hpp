#pragma once
#include "dsv41/global_kv.hpp"
#include "dsv41/linear.hpp"
namespace dsv41 {
// Top-k reference synchronizes to CPU; output is re-sorted by position.
std::vector<std::int32_t> index_topk_reference(const mlx::core::array& scores,int offset);
mlx::core::array restore_index_reference(const GlobalKVState& state);
// Level one of the decoder's two-level top-k: keep the top blocks by best position.
std::vector<std::uint8_t> select_candidate_blocks_reference(const std::vector<float>& logits,
 int compress_len,int topk_blocks,int block_size);
struct IndexSelection {
 std::vector<std::int32_t> rows;       // +offset, position-sorted
 std::vector<std::uint8_t> candidates; // bool mask over compressed positions; empty unless candidate source
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
private:
 int layer_,ratio_,candidate_topk_blocks_,candidate_block_size_;
 bool is_candidate_source_,uses_candidates_;
 PackedLinearReference query_;
 mlx::core::array weights_;
};
}
