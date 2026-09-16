#pragma once
#include "dsv41/text_encoder.hpp"
#include "dsv41/text_decoder.hpp"
#include "dsv41/trace.hpp"
namespace dsv41 {
// Full text backbone: token IDs -> encoder 0..19 -> decoder 20..39 -> final collapse/norm/head.
struct TextBackboneState {
 explicit TextBackboneState(std::shared_ptr<const EngramMetadata> m):encoder(std::move(m)){}
 void reset(){encoder.reset();decoder.reset();}
 TextEncoderState encoder;
 TextDecoderState decoder;
};
class TextBackboneReference {
public:
 TextBackboneReference(WeightCatalog& catalog,std::shared_ptr<const EngramMetadata> metadata)
  :encoder_(catalog,std::move(metadata)),decoder_(catalog){}
 BlockResult forward(std::span<const std::uint32_t> ids,TextBackboneState& state,std::uint64_t start,TraceSink* trace=nullptr) const;
 BlockResult forward_packed_chunk(std::span<const std::uint32_t> ids,
                                  TextBackboneState& state,std::uint64_t start) const;
 mlx::core::array logits(const BlockResult& final_hidden,TraceSink* trace=nullptr) const{return decoder_.logits(final_hidden,trace);}
private:
 TextEncoderReference encoder_;
 TextDecoderReference decoder_;
};
}
