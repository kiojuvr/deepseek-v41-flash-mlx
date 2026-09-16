#include "dsv41/text_quad.hpp"
#include <stdexcept>
namespace dsv41 {
namespace mx=mlx::core;
BlockResult TextQuadReference::forward(std::span<const std::uint32_t> ids,TextQuadState& state,std::uint64_t start) const{
 if(ids.empty()||ids.size()>128||start>=1048576||ids.size()>1048576-start||state.fourth.position()!=start)
  throw std::runtime_error("invalid four-layer input/state");
 for(auto id:ids)if(id>=129280||id==129264)throw std::runtime_error("four-layer reference requires text IDs");
 auto next=state;std::vector<mx::array> hidden,pre;
 for(std::size_t i=0;i<ids.size();++i){
  auto input=triple_.forward(ids.subspan(i,1),next.triple,start+i);
  auto& publication=next.triple.third.publication();
  if(!publication)throw std::runtime_error("missing layer 2 publication");
  auto out=fourth_.forward(input.hidden,input.pre_mix,next.fourth,*publication,start+i);
  hidden.push_back(out.hidden);pre.push_back(out.pre_mix);
 }
 BlockResult result{mx::concatenate(hidden,0),mx::concatenate(pre,0)};
 mx::eval(result.hidden,result.pre_mix);state=std::move(next);return result;
}
BlockResult TextQuadReference::forward_packed_chunk(std::span<const std::uint32_t> ids,
 TextQuadState& state,std::uint64_t start,std::vector<SharedAttentionReference>* output_publications) const{
 if(ids.empty()||ids.size()>128||start>=1048576||ids.size()>1048576-start||state.fourth.position()!=start)
  throw std::runtime_error("invalid packed four-layer input/state");
 auto next=state;std::vector<SharedAttentionReference> publications;
 auto input=triple_.forward_packed_chunk(ids,next.triple,start,&publications);
 auto result=fourth_.forward_packed_chunk(input.hidden,input.pre_mix,next.fourth,publications,start);
 fourth_.release_packed_bank();
 if(!publications.empty())next.triple.third.publication()=publications.back();
 if(output_publications)*output_publications=std::move(publications);
 state=std::move(next);return result;
}
}
