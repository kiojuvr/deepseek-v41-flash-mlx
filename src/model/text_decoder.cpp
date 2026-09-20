#include "dsv41/text_decoder.hpp"
#include "dsv41/sweep_telemetry.hpp"
#include "dsv41/execution_policy.hpp"
#include "dsv41/model_entry.hpp"
#include "dsv41/layer_owner.hpp"
#include "dsv41/runtime_profile.hpp"
#include "dsv41/deferred_decoder_plan.hpp"
#include <algorithm>
#include <iterator>
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
TextDecoderReference::TextDecoderReference(WeightCatalog& c,std::shared_ptr<const ResidentExpertAtlas> atlas)
 :producer_(std::make_unique<CompressedBlockReference>(c,20,atlas)),
  norm_(load_bf16(c,"norm.weight",{5120})),head_(load_bf16(c,"head.weight",{129280,5120})){
 for(int layer=21;layer<40;++layer)reuse_[reuse_slot(layer)]=std::make_unique<ReusedBlockReference>(c,layer,atlas);
}
int TextDecoderReference::reuse_slot(int layer) const{
 if(layer<21||layer>=kBackboneLayers)throw std::runtime_error("invalid decoder reuse layer");
 return layer-21;
}
BlockResult TextDecoderReference::forward_packed_chunk(const mx::array& h,const mx::array& pre,
 TextDecoderState& state,std::uint64_t start) const{
 if(h.dtype()!=mx::bfloat16||h.ndim()!=3||h.shape(0)<1||h.shape(0)>128||h.shape(1)!=4||h.shape(2)!=5120||
    pre.dtype()!=mx::float32||pre.shape()!=mx::Shape({h.shape(0),4})||state.producer.position()!=start||
    start>=1048576||std::uint64_t(h.shape(0))>1048576-start)throw std::runtime_error("invalid packed decoder input/state");
 for(const auto& s:state.reuse)if(s.position()!=start)throw std::runtime_error("invalid packed decoder reuse position");
 auto next=state;std::vector<SharedAttentionReference> publications;
 auto run=[](const auto& block,auto&& fn){
  try{auto result=fn();block.release_packed_bank();return result;}
  catch(...){block.release_packed_bank();throw;}
 };
 auto run_profiled=[&](int layer,const auto& block,auto&& fn){
  auto started=runtime_profile_start();auto result=run(block,fn);
  record_runtime_layer(layer,runtime_profile_elapsed(started));return result;
 };
 auto out=run_profiled(20,*producer_,[&]{return producer_->forward_packed_chunk(h,pre,next.producer,start,&publications);});
 for(int layer=21;layer<40;++layer){
  const int slot=reuse_slot(layer);
  out=run_profiled(layer,*reuse_[slot],[&]{return reuse_[slot]->forward_packed_chunk(out.hidden,out.pre_mix,next.reuse[slot],publications,start);});
 }
 next.producer.publication()=publications.back();
 mx::eval(out.hidden,out.pre_mix);state=std::move(next);return out;
}
BlockResult TextDecoderReference::forward_packed_sweep(const mx::array& h,const mx::array& pre,
 TextDecoderState& state,std::uint64_t start) const{
 if(h.dtype()!=mx::bfloat16||h.ndim()!=3||h.shape(0)<1||h.shape(1)!=4||h.shape(2)!=5120||
    pre.dtype()!=mx::float32||pre.shape()!=mx::Shape({h.shape(0),4})||state.producer.position()!=start||
    start>=1048576||std::uint64_t(h.shape(0))>1048576-start)
  throw std::runtime_error("invalid packed decoder sweep input/state");
 for(const auto& s:state.reuse)if(s.position()!=start)throw std::runtime_error("invalid packed decoder sweep reuse position");
 auto next=state;BlockResult out{h,pre};const std::size_t tokens=h.shape(0);
 const bool defer_layers=runtime_defer_decode_layer_eval(tokens);
 auto tile_input=[&](std::size_t offset,std::size_t count){
  return BlockResult{
   mx::slice(out.hidden,{int(offset),0,0},{int(offset+count),4,5120}),
   mx::slice(out.pre_mix,{int(offset),0},{int(offset+count),4})};
 };
 auto materialize=[&](std::vector<mx::array>& hidden,std::vector<mx::array>& pre_mix){
  out={mx::concatenate(hidden,0),mx::concatenate(pre_mix,0)};
  if(!defer_layers)mx::eval(out.hidden,out.pre_mix);
  record_sweep_layer(defer_layers);
 };
 auto run_producer=[&](int layer,const auto& block,auto& layer_state,
                       std::vector<SharedAttentionReference>& publications){
  auto started=runtime_profile_start();std::vector<mx::array> hidden,pre_mix;publications.clear();
  hidden.reserve((tokens+127)/128);pre_mix.reserve(hidden.capacity());publications.reserve(tokens);
  try{
   for(std::size_t offset=0;offset<tokens;offset+=128){
    const auto count=std::min<std::size_t>(128,tokens-offset);auto input=tile_input(offset,count);
    std::vector<SharedAttentionReference> tile_publications;
    auto result=block.forward_packed_chunk(input.hidden,input.pre_mix,layer_state,start+offset,&tile_publications);
    hidden.push_back(result.hidden);pre_mix.push_back(result.pre_mix);
    publications.insert(publications.end(),std::make_move_iterator(tile_publications.begin()),
                         std::make_move_iterator(tile_publications.end()));
   }
  }catch(...){block.release_packed_bank();throw;}
  block.release_packed_bank();materialize(hidden,pre_mix);
  record_runtime_layer(layer,runtime_profile_elapsed(started));
 };
 auto run_reuse=[&](int layer,const auto& block,auto& layer_state,
                    std::vector<SharedAttentionReference>& publications){
  auto started=runtime_profile_start();std::vector<mx::array> hidden,pre_mix;
  hidden.reserve((tokens+127)/128);pre_mix.reserve(hidden.capacity());
  try{
   for(std::size_t offset=0;offset<tokens;offset+=128){
    const auto count=std::min<std::size_t>(128,tokens-offset);auto input=tile_input(offset,count);
    std::vector<SharedAttentionReference> tile_publications(
     publications.begin()+offset,publications.begin()+offset+count);
    auto result=block.forward_packed_chunk(input.hidden,input.pre_mix,layer_state,
                                           tile_publications,start+offset);
    hidden.push_back(result.hidden);pre_mix.push_back(result.pre_mix);
    std::move(tile_publications.begin(),tile_publications.end(),publications.begin()+offset);
   }
  }catch(...){block.release_packed_bank();throw;}
  block.release_packed_bank();materialize(hidden,pre_mix);
  record_runtime_layer(layer,runtime_profile_elapsed(started));
 };
 std::vector<SharedAttentionReference> publications;
 run_producer(20,*producer_,next.producer,publications);
 for(int layer=21;layer<40;++layer){
  const int slot=reuse_slot(layer);run_reuse(layer,*reuse_[slot],next.reuse[slot],publications);
 }
 if(defer_layers){mx::eval(out.hidden,out.pre_mix);record_sweep_stack();}
 next.producer.publication()=publications.back();state=std::move(next);return out;
}
void TextDecoderReference::prepare_deferred_prefix(const mx::array& h,const mx::array& pre,
 TextDecoderState& state,std::uint64_t start) const{
 const std::size_t tokens=h.shape(0);
 if(h.dtype()!=mx::bfloat16||h.ndim()!=3||tokens<1||tokens>16384||
    h.shape(1)!=4||h.shape(2)!=5120||pre.dtype()!=mx::float32||
    pre.shape()!=mx::Shape({h.shape(0),4})||state.producer.position()!=start||
    start>=1048576||tokens>1048576-start)
  throw std::runtime_error("invalid deferred decoder prefix input/state");
 for(const auto& s:state.reuse)if(s.position()!=start)
  throw std::runtime_error("invalid deferred decoder prefix reuse position");
 auto next=state;
 for(std::size_t off=0;off<tokens;off+=128){
  const auto count=std::min<std::size_t>(128,tokens-off);
  auto hidden=mx::slice(h,{int(off),0,0},{int(off+count),4,5120});
  auto pre_mix=mx::slice(pre,{int(off),0},{int(off+count),4});
  producer_->prepare_packed_attention(hidden,pre_mix,next.producer,start+off);
 }
 state=std::move(next);
}
BlockResult TextDecoderReference::forward_deferred_suffix(const mx::array& h,const mx::array& pre,
 TextDecoderState& state,std::uint64_t start) const{
 return forward_deferred_suffix_impl(h,pre,state,start,false);
}
BlockResult TextDecoderReference::resume_deferred_suffix(const mx::array& h,const mx::array& pre,
 TextDecoderState& state,std::uint64_t start) const{
 return forward_deferred_suffix_impl(h,pre,state,start,true);
}
BlockResult TextDecoderReference::forward_deferred_suffix_impl(const mx::array& h,const mx::array& pre,
 TextDecoderState& state,std::uint64_t start,bool resume_prefix) const{
 const std::size_t tokens=h.shape(0);
 if(h.dtype()!=mx::bfloat16||h.ndim()!=3||tokens<deferred_decoder_suffix_rows(20)+kDecoderRawHistory||
    h.shape(1)!=4||h.shape(2)!=5120||pre.dtype()!=mx::float32||
    pre.shape()!=mx::Shape({h.shape(0),4})||state.producer.position()!=start||
    start>=1048576||tokens>1048576-start)
  throw std::runtime_error("invalid deferred decoder suffix input/state");
 for(const auto& s:state.reuse)if((!resume_prefix&&s.position()!=start)||
                                  (resume_prefix&&s.position()>=start))
  throw std::runtime_error("invalid deferred decoder suffix reuse position");
 auto next=state;
 auto slice=[&](const BlockResult& value,std::size_t first,std::size_t count){
  return BlockResult{
   mx::slice(value.hidden,{int(first),0,0},{int(first+count),4,5120}),
   mx::slice(value.pre_mix,{int(first),0},{int(first+count),4})};
 };
 auto materialize=[](std::vector<mx::array>& hidden,std::vector<mx::array>& pre_mix){
  BlockResult value{hidden.size()==1?hidden.front():mx::concatenate(hidden,0),
                    pre_mix.size()==1?pre_mix.front():mx::concatenate(pre_mix,0)};
  mx::eval(value.hidden,value.pre_mix);return value;
 };
 const BlockResult encoded{h,pre};
 const auto producer_span=deferred_decoder_span(tokens,20);
 for(std::size_t off=0;off<producer_span.first;off+=128){
  const auto count=std::min<std::size_t>(128,producer_span.first-off);
  auto input=slice(encoded,off,count);
  producer_->prepare_packed_attention(input.hidden,input.pre_mix,next.producer,start+off);
 }
 std::vector<SharedAttentionReference> publications;
 std::vector<mx::array> hidden,pre_mix;
 hidden.reserve((producer_span.rows+127)/128);pre_mix.reserve(hidden.capacity());
 publications.reserve(producer_span.rows);
 auto producer_started=runtime_profile_start();
 try{
  for(std::size_t off=producer_span.first;off<tokens;off+=128){
   const auto count=std::min<std::size_t>(128,tokens-off);auto input=slice(encoded,off,count);
   std::vector<SharedAttentionReference> part;
   auto result=producer_->forward_packed_chunk(input.hidden,input.pre_mix,next.producer,start+off,&part);
   hidden.push_back(result.hidden);pre_mix.push_back(result.pre_mix);
   publications.insert(publications.end(),std::make_move_iterator(part.begin()),
                        std::make_move_iterator(part.end()));
  }
 }catch(...){producer_->release_packed_bank();throw;}
 producer_->release_packed_bank();
 auto out=materialize(hidden,pre_mix);
 record_runtime_layer(20,runtime_profile_elapsed(producer_started));
 for(int layer=21;layer<40;++layer){
  const auto span=deferred_decoder_span(tokens,layer);
  if(out.hidden.shape(0)!=int(span.rows+kDecoderRawHistory)||
     publications.size()!=span.rows+kDecoderRawHistory)
   throw std::runtime_error("invalid deferred decoder dependency frontier");
  const int slot=reuse_slot(layer);
  auto warm=slice(out,0,kDecoderRawHistory);
  next.reuse[slot]=reuse_[slot]->seed_packed_attention(
   warm.hidden,warm.pre_mix,start+span.warm_first);
  auto active=slice(out,kDecoderRawHistory,span.rows);
  publications.erase(publications.begin(),publications.begin()+kDecoderRawHistory);
  hidden.clear();pre_mix.clear();hidden.reserve((span.rows+127)/128);pre_mix.reserve(hidden.capacity());
  auto layer_started=runtime_profile_start();
  try{
   for(std::size_t off=0;off<span.rows;off+=128){
    const auto count=std::min<std::size_t>(128,span.rows-off);auto input=slice(active,off,count);
    std::vector<SharedAttentionReference> part(publications.begin()+off,
                                               publications.begin()+off+count);
    auto result=reuse_[slot]->forward_packed_chunk(input.hidden,input.pre_mix,next.reuse[slot],
                                                    part,start+span.first+off);
    std::move(part.begin(),part.end(),publications.begin()+off);
    hidden.push_back(result.hidden);pre_mix.push_back(result.pre_mix);
   }
  }catch(...){reuse_[slot]->release_packed_bank();throw;}
  reuse_[slot]->release_packed_bank();out=materialize(hidden,pre_mix);
  record_runtime_layer(layer,runtime_profile_elapsed(layer_started));
 }
 if(out.hidden.shape(0)!=1||publications.size()!=1)
  throw std::runtime_error("deferred decoder did not collapse to one final row");
 next.producer.publication()=publications.back();state=std::move(next);return out;
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
