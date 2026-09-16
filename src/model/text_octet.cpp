#include "dsv41/text_octet.hpp"
#include <stdexcept>
namespace dsv41 {
namespace mx=mlx::core;
TextOctetReference::TextOctetReference(WeightCatalog& c,std::shared_ptr<const EngramMetadata> m):quad_(c,std::move(m)){
 for(int i=0;i<4;++i)tail_[i]=std::make_unique<ReusedBlockReference>(c,i+4);
}
BlockResult TextOctetReference::forward(std::span<const std::uint32_t> ids,TextOctetState& state,std::uint64_t start) const{
 if(ids.empty()||ids.size()>128||start>=1048576||ids.size()>1048576-start)throw std::runtime_error("invalid eight-layer token count/position");
 const auto& q=state.quad;
 if(q.triple.pair.hash.position()!=start||q.triple.pair.first.position()!=start||q.triple.pair.second.position()!=start||q.triple.third.position()!=start||q.fourth.position()!=start)throw std::runtime_error("inconsistent eight-layer front state");
 for(const auto& s:state.tail)if(s.position()!=start)throw std::runtime_error("inconsistent eight-layer tail state");
 for(auto id:ids)if(id>=129280||id==129264)throw std::runtime_error("eight-layer reference requires text IDs");
 auto next=state;std::vector<mx::array> hidden,pre;
 for(std::size_t t=0;t<ids.size();++t){
  auto out=quad_.forward(ids.subspan(t,1),next.quad,start+t);
  auto& publication=next.quad.triple.third.publication();
  if(!publication)throw std::runtime_error("missing layer 2 publication");
  for(int i=0;i<4;++i)out=tail_[i]->forward(out.hidden,out.pre_mix,next.tail[i],*publication,start+t);
  hidden.push_back(out.hidden);pre.push_back(out.pre_mix);
 }
 BlockResult result{mx::concatenate(hidden,0),mx::concatenate(pre,0)};
 mx::eval(result.hidden,result.pre_mix);state=std::move(next);return result;
}
BlockResult TextOctetReference::forward_packed_chunk(std::span<const std::uint32_t> ids,
 TextOctetState& state,std::uint64_t start) const{
 if(ids.empty()||ids.size()>128||start>=1048576||ids.size()>1048576-start)
  throw std::runtime_error("invalid packed eight-layer token count/position");
 auto next=state;std::vector<SharedAttentionReference> publications;
 auto out=quad_.forward_packed_chunk(ids,next.quad,start,&publications);
 for(int i=0;i<4;++i){
  out=tail_[i]->forward_packed_chunk(out.hidden,out.pre_mix,next.tail[i],publications,start);
  tail_[i]->release_packed_bank();
 }
 if(!publications.empty())next.quad.triple.third.publication()=publications.back();
 state=std::move(next);return out;
}
}
