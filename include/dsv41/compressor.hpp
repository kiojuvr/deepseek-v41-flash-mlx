#pragma once
#include "dsv41/weights.hpp"
#include <mlx/mlx.h>
namespace dsv41 {
class CompressorReference;
class CompressorState {
public:
 CompressorState();
 void reset();
 std::uint64_t position() const{return position_;}
 const mlx::core::array& pending_kv() const{return kv_;}
 const mlx::core::array& pending_scores() const{return scores_;}
private:
 std::uint64_t position_=0;
 mlx::core::array kv_,scores_;
 friend class CompressorReference;
};
struct CompressedLatents {
 mlx::core::array values; // [completed groups,512] BF16, normalized, pre-RoPE
 std::vector<std::uint64_t> positions; // First token of each group
};
// Global KV producer for a kv_source layer. compress_ratio 1 is a plain projection with bf16
// weights; ratio > 1 pools the group with a learned softmax gate in fp32.
class CompressorReference {
public:
 explicit CompressorReference(WeightCatalog& catalog,int layer);
 CompressedLatents forward(const mlx::core::array& hidden,CompressorState& state,
                           std::uint64_t start,
                           std::vector<CompressorState>* prefixes=nullptr) const;
private:
 int layer_;
 int ratio_;
 mlx::core::array kv_weight_,gate_weight_,norm_;
};
}
