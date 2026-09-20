#pragma once
#include "dsv41/global_kv.hpp"
#include "dsv41/index_query.hpp"
#include "dsv41/linear.hpp"
#include "dsv41/shared_attention.hpp"
#include <optional>
namespace dsv41 {
class CompressedLayerReference;
class CompressedLayerState {
public:
 CompressedLayerState():window_(mlx::core::zeros({0,512},mlx::core::bfloat16)){}
 void reset(){*this=CompressedLayerState();}
 std::uint64_t position() const{return global_.position();}
 const GlobalKVState& global() const{return global_;}
 const mlx::core::array& window() const{return window_;}
 const std::optional<SharedAttentionReference>& publication() const{return publication_;}
 std::optional<SharedAttentionReference>& publication(){return publication_;}
private:
 GlobalKVState global_;
 mlx::core::array window_;
 std::optional<SharedAttentionReference> publication_;
 friend class CompressedLayerReference;
};
// Producer attention for a kv_source layer; no mHC/FFN or cross-layer reuse here.
class CompressedLayerReference {
public:
 explicit CompressedLayerReference(WeightCatalog& catalog,int layer);
 mlx::core::array forward(const mlx::core::array& input,CompressedLayerState& state,std::uint64_t start) const;
 mlx::core::array forward_chunk(const mlx::core::array& input,CompressedLayerState& state,
                                std::uint64_t start,
                                std::vector<SharedAttentionReference>* publications) const;
 // CED preparation for a skipped producer prefix: append the exact global
 // cache and retain its raw window without evaluating query/output work.
 void prepare_chunk(const mlx::core::array& input,CompressedLayerState& state,
                    std::uint64_t start) const;
private:
 int layer_,ratio_;
 PackedLinearReference qa_,qb_,kv_,output_;
 mlx::core::array qnorm_,kvnorm_,grouped_,sink_;
 GlobalKVProducerReference producer_;
 IndexQueryReference index_;
};
}
