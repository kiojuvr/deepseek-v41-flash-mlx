#pragma once
#include "dsv41/mhc.hpp"
#include "dsv41/moe.hpp"
#include "dsv41/swa_layer.hpp"
#include "dsv41/trace.hpp"
namespace dsv41 {
struct BlockResult { mlx::core::array hidden,pre_mix; };
// Pure SWA layers 0/1. Caller applies Engram before layer 1. All experts resident.
class BlockReference {
public:
 explicit BlockReference(WeightCatalog& catalog,int layer=0,
                         std::shared_ptr<const ResidentExpertAtlas> atlas={});
 BlockResult forward(const mlx::core::array& hidden,const mlx::core::array& pre_mix,
                     SwaLayerState& state,std::uint64_t start_position) const;
 // Layer-major chunk candidate: preserves token-serial attention/state while
 // batching the routed MoE. Requires DSV41_RUNTIME_PACKED_EXPERT_BANK=1.
 BlockResult forward_packed_chunk(const mlx::core::array& hidden,const mlx::core::array& pre_mix,
                                  SwaLayerState& state,std::uint64_t start_position) const;
 void release_packed_bank() const { moe_.release_packed_bank(); }
private:
 int layer_;
 HCReference attn_mix_,ffn_mix_;
 SwaLayerReference attention_;
 MoEReference moe_;
 mlx::core::array attn_norm_,ffn_norm_;
};
}
