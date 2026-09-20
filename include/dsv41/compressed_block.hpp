#pragma once
#include "dsv41/block.hpp"
#include "dsv41/compressed_layer.hpp"
namespace dsv41 {
class CompressedBlockReference {
public:
 explicit CompressedBlockReference(WeightCatalog& catalog,int layer,
                                   std::shared_ptr<const ResidentExpertAtlas> atlas={});
 BlockResult forward(const mlx::core::array& hidden,const mlx::core::array& pre_mix,
                     CompressedLayerState& state,std::uint64_t start) const;
 BlockResult forward_packed_chunk(const mlx::core::array& hidden,const mlx::core::array& pre_mix,
                                  CompressedLayerState& state,std::uint64_t start,
                                  std::vector<SharedAttentionReference>* publications=nullptr) const;
 void prepare_packed_attention(const mlx::core::array& hidden,const mlx::core::array& pre_mix,
                               CompressedLayerState& state,std::uint64_t start) const;
 void release_packed_bank() const { moe_.release_packed_bank(); }
private:
 int layer_;
 HCReference attn_mix_,ffn_mix_;
 CompressedLayerReference attention_;
 MoEReference moe_;
 mlx::core::array attn_norm_,ffn_norm_;
};
}
