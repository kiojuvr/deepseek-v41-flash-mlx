#include "dsv41/compressed_block.hpp"
#include "dsv41/model_entry.hpp"
#include <stdexcept>
#include "dsv41/layer_owner.hpp"
namespace dsv41 {
namespace mx=mlx::core;
namespace {
mx::array norm(WeightCatalog& c,const std::string& name){
 auto t=c.tensor(name);
 if(t.dtype!="BF16"||t.shape!=std::vector<std::uint64_t>{5120}||t.size_bytes!=10240)throw std::runtime_error("invalid Block norm weight");
 std::vector<std::uint16_t> data(5120);t.read(0,{reinterpret_cast<std::byte*>(data.data()),10240});
 return mx::view(mx::array(data.begin(),{5120},mx::uint16),mx::bfloat16);
}
}
CompressedBlockReference::CompressedBlockReference(WeightCatalog& c,int layer):layer_(checked_producer_layer(layer)),attn_mix_(c,layer_,"attn"),ffn_mix_(c,layer_,"ffn"),attention_(c,layer_),moe_(c,layer_),
 attn_norm_(norm(c,("layers."+std::to_string(layer_))+".attn_norm.weight")),ffn_norm_(norm(c,("layers."+std::to_string(layer_))+".ffn_norm.weight")){}
BlockResult CompressedBlockReference::forward(const mx::array& h,const mx::array& pre,CompressedLayerState& state,std::uint64_t start) const{
 if(h.dtype()!=mx::bfloat16||h.ndim()!=3||h.shape(0)<1||h.shape(0)>128||h.shape(1)!=4||h.shape(2)!=5120||
    pre.dtype()!=mx::float32||pre.shape()!=mx::Shape({h.shape(0),4})||state.position()!=start||
    start>=1048576||std::uint64_t(h.shape(0))>1048576-start)throw std::runtime_error("invalid Block input/state");
 auto finite=mx::logical_and(mx::all(mx::isfinite(h)),mx::all(mx::isfinite(pre)));mx::eval(finite);
 if(!finite.item<bool>())throw std::runtime_error("nonfinite Block input");
 const std::string prefix=layer_<20?"encoder.layer":"decoder.layer";
 auto name=[&](const char* suffix){return prefix+std::to_string(layer_)+"."+suffix;};
 auto next=state;std::vector<mx::array> hidden,pre_mix;
 for(int i=0;i<h.shape(0);++i){
  auto residual=mx::slice(h,{i,0,0},{i+1,4,5120});
  auto a=attn_mix_.mixes(residual);
  auto attn_in=rms_norm_reference(hc_pre_reference(residual,mx::slice(pre,{i,0},{i+1,4})),attn_norm_,1e-20f);
  trace_record(name("attn_in"),attn_in);
  auto attn_out=attention_.forward(attn_in,next,start+i);
  trace_record(name("attn_out"),attn_out);
  auto x=hc_post_reference(attn_out,residual,a);
  trace_record(name("post_attn"),x);
  auto f=ffn_mix_.mixes(x);residual=x;
  auto ffn_in=rms_norm_reference(hc_pre_reference(x,a.pre),ffn_norm_,1e-20f);
  trace_record(name("ffn_in"),ffn_in);
  auto moe_out=moe_.forward(ffn_in);
  trace_record(name("moe_out"),moe_out);
  x=hc_post_reference(moe_out,residual,f);
  auto ok=mx::logical_and(mx::all(mx::isfinite(x)),mx::all(mx::isfinite(f.pre)));mx::eval(x,f.pre,ok);
  if(!ok.item<bool>())throw std::runtime_error("nonfinite Block output");
  hidden.push_back(x);pre_mix.push_back(f.pre);
 }
 BlockResult result{mx::concatenate(hidden,0),mx::concatenate(pre_mix,0)};
 mx::eval(result.hidden,result.pre_mix);state=std::move(next);return result;
}
}
