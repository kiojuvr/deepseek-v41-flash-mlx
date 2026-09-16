#include "dsv41/text_front.hpp"
#include <stdexcept>
namespace dsv41 {
BlockResult TextFrontReference::forward(std::span<const std::uint32_t> ids,SwaLayerState& state,
                                      std::uint64_t start) const{
 if(start!=state.position()||ids.empty()||ids.size()>128||start>=1048576||ids.size()>1048576-start)
  throw std::runtime_error("invalid text front position/token count");
 // Initial pre-mix is per input token, never recycled from a prior token's Block output.
 auto input=entry_.forward(ids);
 return block_.forward(input.hidden,input.pre_mix,state,start);
}
BlockResult TextFrontReference::forward_packed_chunk(std::span<const std::uint32_t> ids,
 SwaLayerState& state,std::uint64_t start) const{
 if(start!=state.position()||ids.empty()||ids.size()>128||start>=1048576||ids.size()>1048576-start)
  throw std::runtime_error("invalid packed text front position/token count");
 auto input=entry_.forward(ids);
 return block_.forward_packed_chunk(input.hidden,input.pre_mix,state,start);
}
}
