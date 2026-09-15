#pragma once
#include "dsv41/block.hpp"
#include "dsv41/reused_layer.hpp"
#include "dsv41/trace.hpp"
namespace dsv41 {
class ReusedBlockReference {
public:
 explicit ReusedBlockReference(WeightCatalog& catalog,int layer=3);
 BlockResult forward(const mlx::core::array& hidden,const mlx::core::array& pre_mix,
                     ReusedLayerState& state,SharedAttentionReference& publication,std::uint64_t start) const;
private:
 int layer_;
 HCReference attn_mix_,ffn_mix_;
 ReusedLayerReference attention_;
 MoEReference moe_;
 mlx::core::array attn_norm_,ffn_norm_;
};
}
