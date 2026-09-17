#include "dsv41/text_backbone.hpp"
#include <stdexcept>
namespace dsv41 {
TextBackboneReference::TextBackboneReference(WeightCatalog& catalog,
 std::shared_ptr<const EngramMetadata> metadata)
 :expert_atlas_(make_resident_expert_atlas(catalog)),
  encoder_(catalog,std::move(metadata),expert_atlas_),decoder_(catalog,expert_atlas_){}

BlockResult TextBackboneReference::forward_packed_chunk(std::span<const std::uint32_t> ids,
 TextBackboneState& state,std::uint64_t start) const{
 auto next=state;
 auto encoded=encoder_.forward_packed_chunk(ids,next.encoder,start);
 auto result=decoder_.forward_packed_chunk(encoded.hidden,encoded.pre_mix,next.decoder,start);
 state=std::move(next);return result;
}
BlockResult TextBackboneReference::forward_packed_sweep(std::span<const std::uint32_t> ids,
 TextBackboneState& state,std::uint64_t start) const{
 if(ids.empty()||ids.size()>4096)throw std::runtime_error("packed sweep requires 1..4096 tokens");
 auto next=state;
 auto encoded=encoder_.forward_packed_sweep(ids,next.encoder,start);
 auto result=decoder_.forward_packed_sweep(encoded.hidden,encoded.pre_mix,next.decoder,start);
 state=std::move(next);return result;
}
BlockResult TextBackboneReference::forward(std::span<const std::uint32_t> ids,TextBackboneState& state,std::uint64_t start,TraceSink* trace) const{
 auto next=state;
 auto encoder_out=encoder_.forward(ids,next.encoder,start,trace);
 if(trace){trace->record("encoder.out.hidden",encoder_out.hidden);trace->record("encoder.out.pre_mix",encoder_out.pre_mix);}
 auto result=decoder_.forward(encoder_out.hidden,encoder_out.pre_mix,next.decoder,start,trace);
 if(trace){trace->record("decoder.out.hidden",result.hidden);trace->record("decoder.out.pre_mix",result.pre_mix);}
 state=std::move(next);return result;
}
}
