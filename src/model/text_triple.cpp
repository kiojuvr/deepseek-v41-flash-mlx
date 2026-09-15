#include "dsv41/text_triple.hpp"
#include <stdexcept>
namespace dsv41 {
BlockResult TextTripleReference::forward(std::span<const std::uint32_t> ids,TextTripleState& state,std::uint64_t start) const{
 if(state.third.position()!=start)throw std::runtime_error("invalid third layer position");
 auto next=state;
 auto input=pair_.forward(ids,next.pair,start);
 auto result=third_.forward(input.hidden,input.pre_mix,next.third,start);
 // Pair and compressed Block have both evaluated successfully before publishing any state.
 state=std::move(next);return result;
}
}
