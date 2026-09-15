#pragma once
#include "dsv41/block.hpp"
#include "dsv41/compressed_layer.hpp"
namespace dsv41 {
class CompressedBlockReference {
public:
 explicit CompressedBlockReference(WeightCatalog& catalog,int layer);
 BlockResult forward(const mlx::core::array& hidden,const mlx::core::array& pre_mix,
                     CompressedLayerState& state,std::uint64_t start) const;
private:
 int layer_;
 HCReference attn_mix_,ffn_mix_;
 CompressedLayerReference attention_;
 MoEReference moe_;
 mlx::core::array attn_norm_,ffn_norm_;
};
}
