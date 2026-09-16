#pragma once
#include "dsv41/text_front.hpp"
#include "dsv41/engram_layer.hpp"
namespace dsv41 {
struct TextPairState {
 explicit TextPairState(std::shared_ptr<const EngramMetadata> m):metadata(std::move(m)),hash(metadata){}
 void reset(){hash.reset();first.reset();second.reset();}
 std::shared_ptr<const EngramMetadata> metadata;
 EngramHashState hash;
 SwaLayerState first,second;
};
// Text IDs -> layer 0 -> Engram 1 -> layer 1. No global attention yet.
class TextPairReference {
public:
 TextPairReference(WeightCatalog& catalog,std::shared_ptr<const EngramMetadata> metadata);
 BlockResult forward(std::span<const std::uint32_t> ids,TextPairState& state,std::uint64_t start) const;
 BlockResult forward_packed_chunk(std::span<const std::uint32_t> ids,TextPairState& state,
                                  std::uint64_t start) const;
private:
 std::shared_ptr<const EngramMetadata> metadata_;
 TextFrontReference first_;
 EngramLayerReference engram_;
 BlockReference second_;
};
}
