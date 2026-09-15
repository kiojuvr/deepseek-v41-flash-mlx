#include "dsv41/swa_state.hpp"
#include <algorithm>
#include <limits>
#include <stdexcept>
namespace dsv41 {
void SwaReferenceState::reset(){ring_.fill(0);position_=0;}
void SwaReferenceState::append(std::span<const std::uint16_t> rows,std::uint64_t start){
 if(start!=position_||rows.empty()||rows.size()%dim||rows.size()/dim>32768||
    rows.size()/dim>std::numeric_limits<std::uint64_t>::max()-position_)
  throw std::runtime_error("invalid SWA append position/shape");
 // Validate the full transaction before changing any state.
 for(auto bits:rows)if((bits&0x7f80)==0x7f80)throw std::runtime_error("nonfinite SWA row");
 const auto count=rows.size()/dim;
 // Stage only the retained tail before writes, including when rows aliases ring_.
 // The bounded scratch avoids corrupting unread source rows on a wrapped append.
 const auto retained=std::min(count,window),first=count-retained;
 std::array<std::uint16_t,window*dim> tail;
 std::copy_n(rows.begin()+first*dim,retained*dim,tail.begin());
 for(std::size_t t=0;t<retained;++t)
  std::copy_n(tail.begin()+t*dim,dim,ring_.begin()+((position_+first+t)%window)*dim);
 position_+=count;
}
std::vector<std::uint16_t> SwaReferenceState::chronological_rows() const {
 const auto count=std::min<std::uint64_t>(position_,window),first=position_-count;
 std::vector<std::uint16_t> result;result.reserve(count*dim);
 for(std::uint64_t t=first;t<position_;++t)
  result.insert(result.end(),ring_.begin()+(t%window)*dim,ring_.begin()+(t%window+1)*dim);
 return result;
}
std::vector<std::int32_t> SwaReferenceState::official_decode_indices() const {
 if(!position_)throw std::runtime_error("SWA decode requires a committed row");
 // Official start_pos = position-1; oldest = start_pos % window + 1.
 auto oldest=(position_-1)%window+1;
 std::vector<std::int32_t> result;result.reserve(window);
 for(std::size_t j=0;j<window;++j){auto slot=(oldest+j)%window;
  result.push_back(slot>position_-1?-1:static_cast<std::int32_t>(slot));}
 return result;
}
}
