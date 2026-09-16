#pragma once
#include "dsv41/text_triple.hpp"
#include "dsv41/reused_block.hpp"
namespace dsv41 {
struct TextQuadState {
 explicit TextQuadState(std::shared_ptr<const EngramMetadata> m):triple(std::move(m)){}
 void reset(){triple.reset();fourth.reset();}
 TextTripleState triple;
 ReusedLayerState fourth;
};
class TextQuadReference {
public:
 TextQuadReference(WeightCatalog& c,std::shared_ptr<const EngramMetadata> m):triple_(c,std::move(m)),fourth_(c){}
 BlockResult forward(std::span<const std::uint32_t> ids,TextQuadState& state,std::uint64_t start) const;
 BlockResult forward_packed_chunk(std::span<const std::uint32_t> ids,TextQuadState& state,
                                  std::uint64_t start,
                                  std::vector<SharedAttentionReference>* publications=nullptr) const;
private:
 TextTripleReference triple_;
 ReusedBlockReference fourth_;
};
}
