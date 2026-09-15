#include "dsv41/text_decoder.hpp"
#include "dsv41/model_entry.hpp"
#include "dsv41/layer_owner.hpp"
#include <stdexcept>
namespace dsv41 {
namespace mx=mlx::core;
namespace {
mx::array load_bf16(WeightCatalog& c,const std::string& name,mx::Shape shape){
 auto t=c.tensor(name);std::vector<std::uint64_t> expected(shape.begin(),shape.end());std::size_t n=1;for(auto d:shape)n*=d;
 if(t.dtype!="BF16"||t.shape!=expected||t.size_bytes!=2*n)throw std::runtime_error("unexpected decoder weight layout");
 std::vector<std::uint16_t> v(n);t.read(0,{reinterpret_cast<std::byte*>(v.data()),t.size_bytes});
 return mx::view(mx::array(v.begin(),shape,mx::uint16),mx::bfloat16);
}
}
TextDecoderReference::TextDecoderReference(WeightCatalog& c)
 :producer_(std::make_unique<CompressedBlockReference>(c,20)),
  norm_(load_bf16(c,"norm.weight",{5120})),head_(load_bf16(c,"head.weight",{129280,5120})){
 for(int layer=21;layer<40;++layer)reuse_[reuse_slot(layer)]=std::make_unique<ReusedBlockReference>(c,layer);
}
int TextDecoderReference::reuse_slot(int layer) const{
 if(layer<21||layer>=kBackboneLayers)throw std::runtime_error("invalid decoder reuse layer");
 return layer-21;
}
BlockResult TextDecoderReference::forward(const mx::array& h,const mx::array& pre,TextDecoderState& state,std::uint64_t start,TraceSink* trace) const{
 if(h.dtype()!=mx::bfloat16||h.ndim()!=3||h.shape(0)<1||h.shape(0)>128||h.shape(1)!=4||h.shape(2)!=5120||
    pre.dtype()!=mx::float32||pre.shape()!=mx::Shape({h.shape(0),4})||state.producer.position()!=start||
    start>=1048576||std::uint64_t(h.shape(0))>1048576-start)throw std::runtime_error("invalid decoder input/state");
 for(auto& s:state.reuse)if(s.position()!=start)throw std::runtime_error("invalid decoder reuse position");
 set_active_trace_sink(trace);
 auto next=state;std::vector<mx::array> hidden,pre_mix;
 std::array<std::vector<mx::array>,20> layer_hidden,layer_pre;
 auto capture=[&](int layer,const BlockResult& r){layer_hidden[layer-20].push_back(r.hidden);layer_pre[layer-20].push_back(r.pre_mix);};
 for(int i=0;i<h.shape(0);++i){
  std::uint64_t pos=start+i;
  set_route_trace_token(pos);
  auto token_hidden=mx::slice(h,{i,0,0},{i+1,4,5120});
  auto token_pre=mx::slice(pre,{i,0},{i+1,4});
  auto out=producer_->forward(token_hidden,token_pre,next.producer,pos);
  if(!next.producer.publication())throw std::runtime_error("missing decoder layer 20 publication");
  auto& publication=*next.producer.publication();
  capture(20,out);
  for(int layer=21;layer<40;++layer){
   out=reuse_[reuse_slot(layer)]->forward(out.hidden,out.pre_mix,next.reuse[reuse_slot(layer)],publication,pos);
   capture(layer,out);
  }
  hidden.push_back(out.hidden);pre_mix.push_back(out.pre_mix);
 }
 if(trace)for(int layer=20;layer<40;++layer){
  auto name=[&](const char* suffix){return "decoder.layer"+std::to_string(layer)+"."+suffix;};
  trace->record(name("hidden"),mx::concatenate(layer_hidden[layer-20],0));
  trace->record(name("pre_mix"),mx::concatenate(layer_pre[layer-20],0));
 }
 BlockResult result{mx::concatenate(hidden,0),mx::concatenate(pre_mix,0)};
 mx::eval(result.hidden,result.pre_mix);set_active_trace_sink(nullptr);state=std::move(next);return result;
}
mx::array TextDecoderReference::logits(const BlockResult& final_hidden,TraceSink* trace) const{
 set_active_trace_sink(trace);
 if(final_hidden.hidden.dtype()!=mx::bfloat16||final_hidden.hidden.ndim()!=3||final_hidden.hidden.shape(1)!=4||final_hidden.hidden.shape(2)!=5120)
  throw std::runtime_error("invalid decoder final hidden");
 auto collapsed=mx::sum(mx::multiply(mx::expand_dims(final_hidden.pre_mix,-1),mx::astype(final_hidden.hidden,mx::float32)),1);
 if(trace)trace->record("final.collapsed",mx::astype(collapsed,mx::bfloat16));
 auto x=rms_norm_reference(mx::astype(collapsed,mx::bfloat16),norm_,1e-20f);
 if(trace)trace->record("final.norm",x);
 auto out=mx::matmul(mx::astype(x,mx::float32),mx::transpose(mx::astype(head_,mx::float32)));
 if(trace)trace->record("logits",out);
 mx::eval(out);set_active_trace_sink(nullptr);return out;
}
}
