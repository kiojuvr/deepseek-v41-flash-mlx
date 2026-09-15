#include "dsv41/text_pair.hpp"
#include <stdexcept>
namespace dsv41 {
namespace mx=mlx::core;
namespace {
std::shared_ptr<const EngramMetadata> validate(std::shared_ptr<const EngramMetadata> m){
 if(!m||m->layer_ids!=std::vector<std::uint64_t>{1,14})throw std::runtime_error("invalid text pair Engram metadata");
 return m;
}
}
TextPairReference::TextPairReference(WeightCatalog& c,std::shared_ptr<const EngramMetadata> m)
 :metadata_(validate(std::move(m))),first_(c),engram_(c,*metadata_,0,1e-20f,EngramReadMode::Mmap),second_(c,1){}
BlockResult TextPairReference::forward(std::span<const std::uint32_t> ids,TextPairState& state,std::uint64_t start) const{
 if(state.metadata!=metadata_||state.hash.position()!=start||state.first.position()!=start||state.second.position()!=start||
    ids.empty()||ids.size()>128||start>=1048576||ids.size()>1048576-start)throw std::runtime_error("invalid text pair request state");
 for(auto id:ids)if(id>=129280||id==129264)throw std::runtime_error("text pair requires legal text token IDs");
 auto next=state;std::vector<mx::array> hidden,pre;
 for(std::size_t i=0;i<ids.size();++i){
  auto token=ids.subspan(i,1);
  auto hashes=next.hash.append(token,{},start+i);
  auto h=first_.forward(token,next.first,start+i);
  // Hash layout is token-major, then layers [1,14], 24 rows per layer.
  auto e=engram_.forward(h.hidden,std::span(hashes).first(24));
  auto out=second_.forward(e.output,h.pre_mix,next.second,start+i);
  hidden.push_back(out.hidden);pre.push_back(out.pre_mix);
 }
 BlockResult result{mx::concatenate(hidden,0),mx::concatenate(pre,0)};
 mx::eval(result.hidden,result.pre_mix);
 state=std::move(next);return result;
}
}
