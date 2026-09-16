#pragma once
#include "dsv41/text_pair.hpp"
#include "dsv41/compressed_block.hpp"
namespace dsv41 {
struct TextTripleState {
 explicit TextTripleState(std::shared_ptr<const EngramMetadata> m):pair(std::move(m)){}
 void reset(){pair.reset();third.reset();}
 TextPairState pair;
 CompressedLayerState third;
};
class TextTripleReference {
public:
 TextTripleReference(WeightCatalog& c,std::shared_ptr<const EngramMetadata> m):pair_(c,std::move(m)),third_(c,2){}
 BlockResult forward(std::span<const std::uint32_t> ids,TextTripleState& state,std::uint64_t start) const;
 BlockResult forward_packed_chunk(std::span<const std::uint32_t> ids,TextTripleState& state,
                                  std::uint64_t start,
                                  std::vector<SharedAttentionReference>* publications=nullptr) const;
private:
 TextPairReference pair_;
 CompressedBlockReference third_;
};
}
