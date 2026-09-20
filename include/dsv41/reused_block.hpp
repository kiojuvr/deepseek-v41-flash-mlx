#pragma once
#include "dsv41/block.hpp"
#include "dsv41/reused_layer.hpp"
#include "dsv41/trace.hpp"
namespace dsv41 {
class ReusedBlockReference {
public:
 explicit ReusedBlockReference(WeightCatalog& catalog,int layer=3,
                               std::shared_ptr<const ResidentExpertAtlas> atlas={});
 BlockResult forward(const mlx::core::array& hidden,const mlx::core::array& pre_mix,
                     ReusedLayerState& state,SharedAttentionReference& publication,std::uint64_t start) const;
 BlockResult forward_packed_chunk(const mlx::core::array& hidden,const mlx::core::array& pre_mix,
                                  ReusedLayerState& state,
                                  std::vector<SharedAttentionReference>& publications,
                                  std::uint64_t start) const;
 ReusedLayerState seed_packed_attention(const mlx::core::array& hidden,
                                        const mlx::core::array& pre_mix,
                                        std::uint64_t start) const;
 void release_packed_bank() const { moe_.release_packed_bank(); }
private:
 int layer_;
 HCReference attn_mix_,ffn_mix_;
 ReusedLayerReference attention_;
 MoEReference moe_;
 mlx::core::array attn_norm_,ffn_norm_;
};
}
