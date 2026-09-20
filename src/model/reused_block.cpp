#include "dsv41/reused_block.hpp"
#include "dsv41/model_entry.hpp"
#include "dsv41/execution_policy.hpp"
#include "dsv41/runtime_profile.hpp"
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
ReusedBlockReference::ReusedBlockReference(WeightCatalog& c,int layer,std::shared_ptr<const ResidentExpertAtlas> atlas):layer_(checked_reused_layer(layer)),attn_mix_(c,layer_,"attn"),ffn_mix_(c,layer_,"ffn"),attention_(c,layer_),moe_(c,layer_,std::move(atlas)),
 attn_norm_(norm(c,("layers."+std::to_string(layer_))+".attn_norm.weight")),ffn_norm_(norm(c,("layers."+std::to_string(layer_))+".ffn_norm.weight")){}
BlockResult ReusedBlockReference::forward(const mx::array& h,const mx::array& pre,ReusedLayerState& state,SharedAttentionReference& publication,std::uint64_t start) const{
 if(h.dtype()!=mx::bfloat16||h.ndim()!=3||h.shape(0)<1||h.shape(0)>1||h.shape(1)!=4||h.shape(2)!=5120||
    pre.dtype()!=mx::float32||pre.shape()!=mx::Shape({h.shape(0),4})||state.position()!=start||
    start>=1048576||std::uint64_t(h.shape(0))>1048576-start)throw std::runtime_error("invalid Block input/state");
 if(runtime_layer_finite_checks_enabled()){
  auto finite=mx::logical_and(mx::all(mx::isfinite(h)),mx::all(mx::isfinite(pre)));mx::eval(finite);
  if(!finite.item<bool>())throw std::runtime_error("nonfinite Block input");
 }
 const std::string prefix=layer_<20?"encoder.layer":"decoder.layer";
 auto name=[&](const char* suffix){return prefix+std::to_string(layer_)+"."+suffix;};
 auto next=state;std::vector<mx::array> hidden,pre_mix;
 for(int i=0;i<h.shape(0);++i){
  auto residual=mx::slice(h,{i,0,0},{i+1,4,5120});
  auto a=attn_mix_.mixes(residual);
  auto attn_in=rms_norm_reference(hc_pre_reference(residual,mx::slice(pre,{i,0},{i+1,4})),attn_norm_,1e-20f);
  trace_record(name("attn_in"),attn_in);
  auto attn_out=attention_.forward(attn_in,next,publication,start+i);
  trace_record(name("attn_out"),attn_out);
  auto x=hc_post_reference(attn_out,residual,a);
  trace_record(name("post_attn"),x);
  auto f=ffn_mix_.mixes(x);residual=x;
  auto ffn_in=rms_norm_reference(hc_pre_reference(x,a.pre),ffn_norm_,1e-20f);
  trace_record(name("ffn_in"),ffn_in);
  auto moe_out=moe_.forward(ffn_in);
  trace_record(name("moe_out"),moe_out);
  x=hc_post_reference(moe_out,residual,f);
  if(runtime_layer_finite_checks_enabled()){
   auto ok=mx::logical_and(mx::all(mx::isfinite(x)),mx::all(mx::isfinite(f.pre)));mx::eval(x,f.pre,ok);
   if(!ok.item<bool>())throw std::runtime_error("nonfinite Block output");
  }
  hidden.push_back(x);pre_mix.push_back(f.pre);
 }
 BlockResult result{mx::concatenate(hidden,0),mx::concatenate(pre_mix,0)};
 mx::eval(result.hidden,result.pre_mix);state=std::move(next);return result;
}
BlockResult ReusedBlockReference::forward_packed_chunk(const mx::array& h,const mx::array& pre,
 ReusedLayerState& state,std::vector<SharedAttentionReference>& publications,std::uint64_t start) const{
 if(h.dtype()!=mx::bfloat16||h.ndim()!=3||h.shape(0)<1||h.shape(0)>128||h.shape(1)!=4||h.shape(2)!=5120||
    pre.dtype()!=mx::float32||pre.shape()!=mx::Shape({h.shape(0),4})||state.position()!=start||
    publications.size()!=std::size_t(h.shape(0))||start>=1048576||std::uint64_t(h.shape(0))>1048576-start)
  throw std::runtime_error("invalid packed reused Block chunk input/state");
 if(runtime_layer_finite_checks_enabled()){
  auto finite=mx::logical_and(mx::all(mx::isfinite(h)),mx::all(mx::isfinite(pre)));mx::eval(finite);
  if(!finite.item<bool>())throw std::runtime_error("nonfinite packed reused Block chunk input");
 }
 auto next=state;auto pending_publications=publications;
 const std::string prefix=layer_<20?"encoder.layer":"decoder.layer";
 auto name=[&](const char* suffix){return prefix+std::to_string(layer_)+"."+suffix;};
 auto attn_mixes=attn_mix_.mixes(h);
 auto batched_attn_input=rms_norm_reference(hc_pre_reference(h,pre),attn_norm_,1e-20f);
 trace_record(name("attn_in"),batched_attn_input);
 auto attention_started=runtime_profile_start();
 auto batched_attn_output=attention_.forward_chunk(batched_attn_input,next,pending_publications,start);
 trace_record(name("attn_out"),batched_attn_output);
 finish_runtime_component(layer_,ProfileComponent::AttentionPath,attention_started,batched_attn_output);
 auto post_attention=hc_post_reference(batched_attn_output,h,attn_mixes);
 trace_record(name("post_attn"),post_attention);
 auto ffn_mixes=ffn_mix_.mixes(post_attention);
 auto batched_input=rms_norm_reference(hc_pre_reference(post_attention,attn_mixes.pre),ffn_norm_,1e-20f);
 trace_record(name("ffn_in"),batched_input);
 auto moe_started=runtime_profile_start();auto moe=moe_.forward_batch_components(batched_input,start);
 trace_record(name("moe_out"),moe.total);
 finish_runtime_component(layer_,ProfileComponent::MoEPath,moe_started,moe.total);
 BlockResult result{hc_post_reference(moe.total,post_attention,ffn_mixes),ffn_mixes.pre};
 auto post_started=runtime_profile_start();
 if(runtime_layer_finite_checks_enabled()){
  auto ok=mx::logical_and(mx::all(mx::isfinite(result.hidden)),mx::all(mx::isfinite(result.pre_mix)));
  mx::eval(result.hidden,result.pre_mix,ok);if(!ok.item<bool>())throw std::runtime_error("nonfinite packed reused Block chunk output");
 }
 finish_runtime_component(layer_,ProfileComponent::PostMoE,post_started,result.hidden);
 trace_record(name("hidden"),result.hidden);trace_record(name("pre_mix"),result.pre_mix);
 publications=std::move(pending_publications);
 state=std::move(next);return result;
}
ReusedLayerState ReusedBlockReference::seed_packed_attention(const mx::array& h,const mx::array& pre,
 std::uint64_t start) const{
 if(h.dtype()!=mx::bfloat16||h.ndim()!=3||h.shape(0)<1||h.shape(0)>128||h.shape(1)!=4||h.shape(2)!=5120||
    pre.dtype()!=mx::float32||pre.shape()!=mx::Shape({h.shape(0),4}))
  throw std::runtime_error("invalid packed reused Block seed input");
 auto input=rms_norm_reference(hc_pre_reference(h,pre),attn_norm_,1e-20f);
 return attention_.seed_window(input,start);
}
}
