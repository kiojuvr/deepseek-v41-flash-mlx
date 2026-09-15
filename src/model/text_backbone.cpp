#include "dsv41/text_backbone.hpp"
#include <stdexcept>
namespace dsv41 {
BlockResult TextBackboneReference::forward(std::span<const std::uint32_t> ids,TextBackboneState& state,std::uint64_t start,TraceSink* trace) const{
 auto next=state;
 auto encoder_out=encoder_.forward(ids,next.encoder,start,trace);
 if(trace){trace->record("encoder.out.hidden",encoder_out.hidden);trace->record("encoder.out.pre_mix",encoder_out.pre_mix);}
 auto result=decoder_.forward(encoder_out.hidden,encoder_out.pre_mix,next.decoder,start,trace);
 if(trace){trace->record("decoder.out.hidden",result.hidden);trace->record("decoder.out.pre_mix",result.pre_mix);}
 state=std::move(next);return result;
}
}
