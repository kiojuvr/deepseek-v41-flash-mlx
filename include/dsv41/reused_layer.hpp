#pragma once
#include "dsv41/shared_attention.hpp"
#include "dsv41/linear.hpp"
#include "dsv41/index_query.hpp"
#include <memory>
namespace dsv41 {
class ReusedLayerReference;
class ReusedLayerState {
public:
 ReusedLayerState():window_(mlx::core::zeros({0,512},mlx::core::bfloat16)){}
 void reset(){*this=ReusedLayerState();}
 std::uint64_t position() const{return position_;}
 const mlx::core::array& window() const{return window_;}
private:
 std::uint64_t position_=0;
 mlx::core::array window_;
 friend class ReusedLayerReference;
};
// Reuse consumer attention (layers 3..39): owns window only, consumes its source publication.
// Decoder index sources (24/28/32/36) recompute top-k from the shared index K and republish.
class ReusedLayerReference {
public:
 explicit ReusedLayerReference(WeightCatalog& catalog,int layer);
 mlx::core::array forward(const mlx::core::array& hidden,ReusedLayerState& state,
                         SharedAttentionReference& publication,std::uint64_t position) const;
 mlx::core::array forward_chunk(const mlx::core::array& hidden,ReusedLayerState& state,
                         std::vector<SharedAttentionReference>& publications,
                         std::uint64_t start_position) const;
 // Construct the exact bounded raw window immediately before a deferred
 // suffix. Older rows cannot affect this layer's 128-row local attention.
 ReusedLayerState seed_window(const mlx::core::array& hidden,
                              std::uint64_t start_position) const;
private:
 int layer_,ratio_;
 bool is_index_source_,uses_candidates_;
 PackedLinearReference qa_,qb_,kv_,output_;
 mlx::core::array qnorm_,kvnorm_,grouped_,sink_;
 std::unique_ptr<IndexQueryReference> index_;
};
}
