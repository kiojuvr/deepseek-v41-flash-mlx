#pragma once
#include "dsv41/text_quad.hpp"
#include <array>
namespace dsv41 {
struct TextOctetState {
 explicit TextOctetState(std::shared_ptr<const EngramMetadata> m):quad(std::move(m)){}
 void reset(){quad.reset();for(auto& state:tail)state.reset();}
 TextQuadState quad;
 std::array<ReusedLayerState,4> tail; // layers 4..7
};
class TextOctetReference {
public:
 TextOctetReference(WeightCatalog& catalog,std::shared_ptr<const EngramMetadata> metadata);
 BlockResult forward(std::span<const std::uint32_t> ids,TextOctetState& state,std::uint64_t start) const;
 BlockResult forward_packed_chunk(std::span<const std::uint32_t> ids,TextOctetState& state,
                                  std::uint64_t start) const;
private:
 TextQuadReference quad_;
 std::array<std::unique_ptr<ReusedBlockReference>,4> tail_;
};
}
