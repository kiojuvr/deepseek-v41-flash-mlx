#include "dsv41/moe.hpp"
#include "dsv41/execution_policy.hpp"
#include "dsv41/runtime_profile.hpp"
#include "route_select.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include "dsv41/layer_owner.hpp"
namespace dsv41 {
namespace mx=mlx::core;
namespace {
std::size_t& tie_counter(){static std::size_t count=0;return count;}
std::vector<RouteTieRecord>& tie_records(){static std::vector<RouteTieRecord> records;return records;}
std::uint64_t& current_token(){static std::uint64_t token=0;return token;}
RouteUnionStats& union_stats(){static RouteUnionStats stats;return stats;}
RouteExecutionStats& execution_stats(){static RouteExecutionStats stats;return stats;}
}
std::size_t route_tie_count(){return tie_counter();}
void reset_route_tie_count(){tie_counter()=0;}
std::vector<RouteTieRecord> route_tie_records(){return tie_records();}
void reset_route_tie_records(){tie_records().clear();}
void set_route_trace_token(std::uint64_t token){current_token()=token;}
std::uint64_t route_trace_token(){return current_token();}
RouteUnionStats route_union_stats(){return union_stats();}
void reset_route_union_stats(){union_stats()={};}
RouteExecutionStats route_execution_stats(){return execution_stats();}
void reset_route_execution_stats(){execution_stats()={};}
namespace {
std::string prefix(int expert,int layer){
 if(expert < -1||expert>=384)throw std::runtime_error("invalid expert ID");
 return reference_moe_prefix(layer)+".ffn."+(expert==-1?std::string("shared_experts"):"experts."+std::to_string(expert));
}
mx::array load(WeightCatalog& c,const std::string& name,mx::Shape shape,bool bf){
 auto t=c.tensor(name);std::vector<std::uint64_t> expected(shape.begin(),shape.end());
 std::size_t count=1;for(auto d:shape)count*=d;
 if(t.shape!=expected||t.dtype!=(bf?"BF16":"F32")||t.size_bytes!=count*(bf?2:4))throw std::runtime_error("unexpected gate weight");
 if(bf){std::vector<std::uint16_t> v(count);t.read(0,{reinterpret_cast<std::byte*>(v.data()),t.size_bytes});return mx::view(mx::array(v.begin(),shape,mx::uint16),mx::bfloat16);}
 std::vector<float> v(count);t.read(0,{reinterpret_cast<std::byte*>(v.data()),t.size_bytes});return mx::array(v.begin(),shape,mx::float32);
}
std::array<int,7> select_top7(const float* raw,const float* corrected){
 std::array<int,7> order;order.fill(-1);
 for(int id=0;id<384;++id){
  if(!std::isfinite(raw[id])||raw[id]<0.0f||!std::isfinite(corrected[id]))
   throw std::runtime_error("invalid route score");
  int insert=7;
  for(int j=0;j<7;++j)
   if(order[j]<0||corrected[id]>corrected[order[j]]||
      (corrected[id]==corrected[order[j]]&&id<order[j])){insert=j;break;}
  if(insert<7){for(int j=6;j>insert;--j)order[j]=order[j-1];order[insert]=id;}
 }
 return order;
}
mx::array normalized_route_weights(const mx::array& scores,const std::array<int,6>& ids){
 auto weights=mx::take(scores,mx::array(ids.begin(),{6},mx::int32));
 return mx::multiply(mx::divide(weights,mx::add(mx::sum(weights),mx::array(1e-20f))),mx::array(1.5f));
}
}
RouteReference select_routes_reference(const mx::array& scores,const mx::array& bias,bool strict,RouteTieRecord* tie){
 if(scores.dtype()!=mx::float32||scores.shape()!=mx::Shape({384})||bias.dtype()!=mx::float32||bias.shape()!=mx::Shape({384}))throw std::runtime_error("invalid route scores");
 auto raw_cpu=mx::astype(scores,mx::float32,mx::Device::cpu);
 auto corrected=mx::add(raw_cpu,mx::astype(bias,mx::float32,mx::Device::cpu),mx::Device::cpu);
 mx::eval(raw_cpu,corrected);
 const float* raw=raw_cpu.data<float>();const float* p=corrected.data<float>();
 auto order=select_top7(raw,p);
 bool tied=p[order[5]]==p[order[6]];
 if(tie){tie->sixth_id=order[5];tie->seventh_id=order[6];tie->sixth_score=p[order[5]];tie->seventh_score=p[order[6]];tie->tied=tied;}
 if(tied){
  if(strict)throw std::runtime_error("ambiguous top-6 boundary requires official tie oracle");
  ++tie_counter();
 }
 std::array<int,6> ids;std::copy_n(order.begin(),6,ids.begin());
 return {ids,normalized_route_weights(scores,ids)};
}
RouteBatchReference select_routes_batch_reference(const mx::array& scores,const mx::array& bias,
 bool strict,int layer,std::uint64_t start){
 if(scores.dtype()!=mx::float32||scores.ndim()!=2||scores.shape(0)<1||scores.shape(0)>128||
    scores.shape(1)!=384||bias.dtype()!=mx::float32||bias.shape()!=mx::Shape({384}))
  throw std::runtime_error("invalid batched route scores");
 const int tokens=scores.shape(0);
 auto raw_cpu=mx::astype(scores,mx::float32,mx::Device::cpu);
 auto corrected=mx::add(raw_cpu,mx::astype(bias,mx::float32,mx::Device::cpu),mx::Device::cpu);
 mx::eval(raw_cpu,corrected);
 const float* raw=raw_cpu.data<float>();const float* p=corrected.data<float>();
 std::vector<std::array<int,6>> ids;ids.reserve(tokens);
 std::vector<mx::array> weights;weights.reserve(tokens);
 std::vector<RouteTieRecord> ties;
 for(int token=0;token<tokens;++token){
  const float* token_raw=raw+std::size_t(token)*384;
  const float* token_corrected=p+std::size_t(token)*384;
  auto order=select_top7(token_raw,token_corrected);
  const bool tied=token_corrected[order[5]]==token_corrected[order[6]];
  RouteTieRecord record{layer,start+std::uint64_t(token),order[5],order[6],
                        token_corrected[order[5]],token_corrected[order[6]],tied};
  if(tied){
   if(strict)throw std::runtime_error("ambiguous top-6 boundary requires official tie oracle");
   ++tie_counter();ties.push_back(record);
  }
  std::array<int,6> selected;std::copy_n(order.begin(),6,selected.begin());
  ids.push_back(selected);
  auto row=mx::reshape(mx::slice(scores,{token,0},{token+1,384}),{384});
  weights.push_back(normalized_route_weights(row,selected));
 }
 auto stacked=tokens==1?mx::reshape(weights.front(),{1,6}):mx::stack(weights,0);
 std::vector<std::uint32_t> flat_ids,lhs,slots;
 flat_ids.reserve(tokens*6);lhs.reserve(tokens*6);slots.reserve(tokens*6);
 for(int token=0;token<tokens;++token){
  std::array<std::uint32_t,6> order{0,1,2,3,4,5};
  std::sort(order.begin(),order.end(),[&](auto a,auto b){return ids[token][a]<ids[token][b];});
  for(int slot=0;slot<6;++slot){flat_ids.push_back(std::uint32_t(ids[token][slot]));lhs.push_back(std::uint32_t(token));}
  slots.insert(slots.end(),order.begin(),order.end());
 }
 return {std::move(ids),std::move(stacked),std::move(ties),
         mx::array(flat_ids.begin(),{tokens,6},mx::uint32),
         mx::array(lhs.begin(),{tokens,6},mx::uint32),
         mx::array(slots.begin(),{tokens,6},mx::uint32)};
}
RouteBatchReference select_routes_batch_device(const mx::array& scores,const mx::array& bias,
 bool strict,int layer,std::uint64_t start,bool diagnostics){
 if(scores.dtype()!=mx::float32||scores.ndim()!=2||scores.shape(0)<1||scores.shape(0)>128||
    scores.shape(1)!=384||bias.dtype()!=mx::float32||bias.shape()!=mx::Shape({384}))
  throw std::runtime_error("invalid device batched route scores");
 const int tokens=scores.shape(0);
 static auto kernel=mx::fast::metal_kernel("dsv41_route_select_batch",
  {"scores","bias","count"},{"ids","selected_raw","boundary","errors","lhs_ids","reduction_slots"},
  dsv41_route_select_source);
 auto out=kernel({scores,bias,mx::array({std::uint32_t(tokens)})},
  {{tokens,7},{tokens,6},{tokens,2},{tokens},{tokens,6},{tokens,6}},
  {mx::uint32,mx::float32,mx::float32,mx::uint8,mx::uint32,mx::uint32},
  {tokens,1,1},{32,1,1},{},std::nullopt,false,mx::Device::gpu);
 ++execution_stats().device_batches;
 if(!diagnostics){
  auto weights=mx::multiply(mx::divide(out[1],mx::add(mx::sum(out[1],1,true),mx::array(1e-20f))),mx::array(1.5f));
  return {{},std::move(weights),{},mx::slice(out[0],{0,0},{tokens,6}),out[4],out[5]};
 }
 ++execution_stats().diagnostic_readbacks;
 auto ids_cpu=mx::astype(out[0],mx::uint32,mx::Device::cpu);
 auto boundary_cpu=mx::astype(out[2],mx::float32,mx::Device::cpu);
 auto errors_cpu=mx::astype(out[3],mx::uint8,mx::Device::cpu);
 mx::eval(ids_cpu,boundary_cpu,errors_cpu);
 const auto* id_data=ids_cpu.data<std::uint32_t>();
 const auto* boundary_data=boundary_cpu.data<float>();
 const auto* error_data=errors_cpu.data<std::uint8_t>();
 std::vector<std::array<int,6>> ids;ids.reserve(tokens);
 std::vector<mx::array> weights;weights.reserve(tokens);
 std::vector<RouteTieRecord> ties;
 for(int token=0;token<tokens;++token){
  if(error_data[token])throw std::runtime_error("invalid route score");
  std::array<int,6> selected;
  for(int slot=0;slot<6;++slot)selected[slot]=int(id_data[token*7+slot]);
  const bool tied=boundary_data[token*2]==boundary_data[token*2+1];
  RouteTieRecord record{layer,start+std::uint64_t(token),selected[5],
                        int(id_data[token*7+6]),boundary_data[token*2],
                        boundary_data[token*2+1],tied};
  if(tied){
   if(strict)throw std::runtime_error("ambiguous top-6 boundary requires official tie oracle");
   ++tie_counter();ties.push_back(record);
  }
  ids.push_back(selected);
  auto raw=mx::reshape(mx::slice(out[1],{token,0},{token+1,6}),{6});
  weights.push_back(mx::multiply(mx::divide(raw,mx::add(mx::sum(raw),mx::array(1e-20f))),mx::array(1.5f)));
 }
 auto stacked=tokens==1?mx::reshape(weights.front(),{1,6}):mx::stack(weights,0);
 auto selected_ids=mx::slice(out[0],{0,0},{tokens,6});
 return {std::move(ids),std::move(stacked),std::move(ties),std::move(selected_ids),out[4],out[5]};
}
GateReference::GateReference(WeightCatalog& c,int layer):layer_(layer),weight_(load(c,reference_moe_prefix(layer)+".ffn.gate.weight",{384,5120},true)),bias_(load(c,reference_moe_prefix(layer)+".ffn.gate.bias",{384},false)){}
RouteReference GateReference::forward(const mx::array& x,RouteTieRecord* tie) const{
 if(x.dtype()!=mx::bfloat16||x.shape()!=mx::Shape({1,5120}))throw std::runtime_error("gate requires one BF16 token");
 auto z=mx::reshape(mx::matmul(mx::astype(x,mx::float32),mx::transpose(mx::astype(weight_,mx::float32))),{384});
 // PyTorch softplus beta=1, threshold=20. gate_temp=1 for this checkpoint.
 auto scores=mx::sqrt(mx::where(mx::greater(z,mx::array(20.0f)),z,mx::log1p(mx::exp(z))));
 return select_routes_reference(scores,bias_,false,tie);
}
RouteBatchReference GateReference::forward_batch(const mx::array& x,std::uint64_t start,bool diagnostics) const{
 if(x.dtype()!=mx::bfloat16||x.ndim()!=2||x.shape(0)<1||x.shape(0)>128||x.shape(1)!=5120)
  throw std::runtime_error("batched gate requires BF16 [1..128,5120]");
 // Keep the canonical one-row matmul reduction schedule while building all
 // rows into one lazy graph. A 2-D matmul changes score bits on the current
 // MLX backend and can change the routed sum even when expert IDs agree.
 std::vector<mx::array> rows;rows.reserve(x.shape(0));
 for(int token=0;token<x.shape(0);++token){
  auto row=mx::slice(x,{token,0},{token+1,5120});
  auto z=mx::reshape(mx::matmul(mx::astype(row,mx::float32),
                               mx::transpose(mx::astype(weight_,mx::float32))),{384});
  rows.push_back(mx::sqrt(mx::where(mx::greater(z,mx::array(20.0f)),z,mx::log1p(mx::exp(z)))));
 }
 auto scores=rows.size()==1?mx::reshape(rows.front(),{1,384}):mx::stack(rows,0);
 return select_routes_batch_device(scores,bias_,false,layer_,start,diagnostics);
}
GateDiagnostic GateReference::diagnose(const mx::array& x,RouteTieRecord* tie) const {
 if(x.dtype()!=mx::bfloat16||x.shape()!=mx::Shape({1,5120})) throw std::runtime_error("gate requires one BF16 token");
 auto z=mx::reshape(mx::matmul(mx::astype(x,mx::float32),mx::transpose(mx::astype(weight_,mx::float32))),{384});
 auto scores=mx::sqrt(mx::where(mx::greater(z,mx::array(20.0f)),z,mx::log1p(mx::exp(z))));
 auto corrected=mx::add(scores,bias_,mx::Device::cpu); mx::eval(scores,corrected);
 return {scores,corrected,select_routes_reference(scores,bias_,false,tie)};
}
mx::array expert_activation_reference(const mx::array& gate,const mx::array& up,const mx::array& weight){
 if(gate.dtype()!=mx::bfloat16||up.dtype()!=mx::bfloat16||gate.shape()!=up.shape()||gate.ndim()!=2||gate.shape(0)!=1||
    gate.shape(1)!=2304||weight.dtype()!=mx::float32||weight.size()!=1)throw std::runtime_error("invalid expert activation");
 auto g=mx::minimum(mx::astype(gate,mx::float32),mx::array(10.0f));
 auto u=mx::clip(mx::astype(up,mx::float32),mx::array(-10.0f),mx::array(10.0f));
 auto silu=mx::multiply(g,mx::sigmoid(g));
 return mx::astype(mx::multiply(mx::multiply(silu,u),mx::reshape(weight,{})),mx::bfloat16);
}
ExpertReference::ExpertReference(WeightCatalog& c,int e,int layer):w1_(c,prefix(e,layer)+".w1"),w2_(c,prefix(e,layer)+".w2"),w3_(c,prefix(e,layer)+".w3"){
 int bits=e==-1?8:4;
 if(w1_.input_dims()!=5120||w1_.output_dims()!=2304||w3_.input_dims()!=5120||w3_.output_dims()!=2304||w2_.input_dims()!=2304||w2_.output_dims()!=5120||w1_.bits()!=bits||w2_.bits()!=bits||w3_.bits()!=bits)throw std::runtime_error("unexpected expert geometry");
}
mx::array ExpertReference::forward(const mx::array& x,const mx::array& weight) const{
 if(x.dtype()!=mx::bfloat16||x.ndim()!=2||x.shape(0)<1||x.shape(0)>128||x.shape(1)!=5120||
    weight.dtype()!=mx::float32||weight.size()!=1)
  throw std::runtime_error("expert requires 1..128 BF16 tokens and one FP32 weight");
 // Official routing weights scale the expert output after w2, not the
 // intermediate SwiGLU activation. Keep this path identical to components().
 return components(x,weight).output;
}
ExpertComponents ExpertReference::components(const mx::array& x,const mx::array& weight) const {
 auto gate=w1_.forward(x), up=w3_.forward(x);
 auto g=mx::clip(mx::astype(gate,mx::float32),mx::array(-1e30f),mx::array(10.0f));
 auto u=mx::clip(mx::astype(up,mx::float32),mx::array(-10.0f),mx::array(10.0f));
 auto act=mx::astype(mx::multiply(mx::multiply(g,mx::sigmoid(g)),u),mx::bfloat16);
 auto out=mx::multiply(mx::astype(w2_.forward(act),mx::float32),weight);
 return {gate,up,act,mx::astype(out,mx::bfloat16)};
}
MoEReference::MoEReference(WeightCatalog& c,int layer,
 std::shared_ptr<const ResidentExpertAtlas> atlas)
 :catalog_(&c),layer_(layer),gate_(c,layer),shared_(c,-1,layer),
  packed_experts_(runtime_packed_expert_bank_enabled()),
  group_selected_experts_(runtime_group_selected_experts_enabled()),
  resident_expert_atlas_(runtime_resident_expert_atlas_enabled()),
  compact_expert_bank_(runtime_compact_expert_bank_enabled()),resident_atlas_(std::move(atlas)){
 if(packed_experts_&&group_selected_experts_)
  throw std::runtime_error("packed and selected expert grouping modes are mutually exclusive");
 if(resident_expert_atlas_&&!packed_experts_)
  throw std::runtime_error("resident expert atlas requires packed expert banks");
 if(compact_expert_bank_&&!packed_experts_)
  throw std::runtime_error("compact expert banks require packed expert banks");
 if(compact_expert_bank_&&resident_expert_atlas_)
  throw std::runtime_error("compact expert banks and resident expert atlas are mutually exclusive");
 if(resident_atlas_&&!resident_expert_atlas_)
  throw std::runtime_error("resident expert atlas supplied while residency is disabled");
}
void MoEReference::release_packed_bank() const{
 if(!resident_atlas_&&!resident_expert_atlas_)expert_bank_.reset();
}
bool MoEReference::packed_bank_loaded() const{return bool(resident_atlas_)||bool(expert_bank_);}
std::size_t MoEReference::packed_bank_expert_count() const{
 return resident_atlas_?resident_atlas_->bank(layer_).expert_count():(expert_bank_?expert_bank_->expert_count():0);
}
ExpertReference& MoEReference::expert(int id) const{
 auto it=experts_.find(id);
 if(it==experts_.end())it=experts_.emplace(id,std::make_unique<ExpertReference>(*catalog_,id,layer_)).first;
 return *it->second;
}
mx::array MoEReference::forward(const mx::array& x) const{
 return forward_components(x).total;
}
MoEComponents MoEReference::forward_components(const mx::array& x) const{
 RouteTieRecord tie{layer_,route_trace_token(),0,0,0.0f,0.0f,false};
 auto route=gate_.forward(x,&tie);
 if(tie.tied){++tie_count_;tie_records().push_back(tie);}
 mx::array y=mx::zeros({1,5120},mx::float32);
 if(packed_experts_){
  if(compact_expert_bank_){
   std::vector<int> selected(route.ids.begin(),route.ids.end());
   std::sort(selected.begin(),selected.end());
   expert_bank_=std::make_unique<PackedExpertBank>(*catalog_,layer_,selected);
   y=expert_bank_->forward_selected(x,route.ids,route.weights).accumulated;
  }else if(resident_atlas_){
   y=resident_atlas_->bank(layer_).forward_selected(x,route.ids,route.weights).accumulated;
  }else{
   if(!expert_bank_)expert_bank_=std::make_unique<PackedExpertBank>(*catalog_,layer_);
   y=expert_bank_->forward_selected(x,route.ids,route.weights).accumulated;
  }
 }else if(group_selected_experts_){
  std::array<const ExpertReference*,6> selected;
  for(int k=0;k<6;++k)selected[k]=&expert(route.ids[k]);
  y=forward_grouped_selected(x,route.ids,selected,route.weights).accumulated;
 }else{
  // Official sums in ascending expert ID, not top-k score order.
  for(int id=0;id<384;++id)for(int k=0;k<6;++k)if(route.ids[k]==id)
   y=mx::add(y,mx::astype(expert(id).forward(x,mx::take(route.weights,mx::array(k))),mx::float32));
 }
 auto shared=mx::astype(shared_.forward(x,mx::array(1.0f)),mx::float32);
 return {mx::astype(shared,mx::bfloat16),mx::astype(y,mx::bfloat16),mx::astype(mx::add(y,shared),mx::bfloat16)};
}
MoEComponents MoEReference::forward_batch_components(const mx::array& x,std::uint64_t start) const{
 if(!packed_experts_)throw std::runtime_error("batched MoE requires the packed expert bank");
 if(x.dtype()!=mx::bfloat16||x.ndim()!=2||x.shape(0)<1||x.shape(0)>128||x.shape(1)!=5120)
  throw std::runtime_error("batched MoE requires BF16 [1..128,5120]");
 const int tokens=x.shape(0);
 // Diagnostic-only MoE sub-phase boundaries. Each forces GPU completion and is
 // nested inside the caller's moe_path_seconds; production execution is unchanged
 // because the helpers return immediately when profiling is disabled.
 auto sub_started=runtime_profile_start();
 finish_runtime_subcomponent(layer_,ProfileSubcomponent::MoeInput,sub_started,x);
 // Diagnostic-only no-op: x is already materialized, so this measures pure
 // eval+synchronize overhead inside the full model (isolated probes: ~0.2 ms).
 sub_started=runtime_profile_start();
 finish_runtime_subcomponent(layer_,ProfileSubcomponent::MoeSyncNoop,sub_started,x);
 sub_started=runtime_profile_start();
 auto routes=gate_.forward_batch(x,start,!resident_atlas_||runtime_route_diagnostics_enabled());
 finish_runtime_subcomponent(layer_,ProfileSubcomponent::MoeRoute,sub_started,routes.weights);
 sub_started=runtime_profile_start();
 tie_count_+=routes.ties.size();
 tie_records().insert(tie_records().end(),routes.ties.begin(),routes.ties.end());
 set_route_trace_token(start+std::uint64_t(tokens-1));
 GroupedExpertBatchResult routed{mx::array(0),mx::array(0)};
 if(compact_expert_bank_){
  std::vector<int> selected;
  selected.reserve(tokens*6);
  for(const auto& token_ids:routes.ids)selected.insert(selected.end(),token_ids.begin(),token_ids.end());
  std::sort(selected.begin(),selected.end());
  selected.erase(std::unique(selected.begin(),selected.end()),selected.end());
  auto& stats=union_stats(); ++stats.batches; stats.selected_experts+=tokens*6; stats.unique_experts+=selected.size();
  if(!previous_compact_ids_.empty()){
   std::vector<int> common;
   std::set_intersection(previous_compact_ids_.begin(),previous_compact_ids_.end(),
     selected.begin(),selected.end(),std::back_inserter(common));
   std::size_t overlap=common.size();
   stats.overlap_experts+=overlap; if(previous_compact_ids_==selected)++stats.exact_reuses;
  }
  const bool exact_reuse=!previous_compact_ids_.empty() && previous_compact_ids_==selected && expert_bank_;
  previous_compact_ids_=selected;
  if(!exact_reuse) expert_bank_=std::make_unique<PackedExpertBank>(*catalog_,layer_,selected);
  routed=expert_bank_->forward_batch_selected(x,routes.ids,routes.weights);
 }else if(resident_atlas_){
  routed=resident_atlas_->bank(layer_).forward_batch_expert_major(x,routes.device_ids,routes.device_lhs,
                                                          routes.reduction_slots,routes.weights);
  // Diagnostic-only immediate repeat. The first call's weight pages are cold;
  // this second call reads the same experts while they are hot. Comparing the
  // two separates a memory-residency cost from per-call execution overhead.
  // Restricted to short decode-like batches so prefill is not doubled.
  if(runtime_component_profile_enabled()&&tokens<=8){
   auto warm_started=runtime_profile_start();
   auto warm=resident_atlas_->bank(layer_).forward_batch_expert_major(x,routes.device_ids,
    routes.device_lhs,routes.reduction_slots,routes.weights,true);
   finish_runtime_subcomponent(layer_,ProfileSubcomponent::MoeRoutedWarm,warm_started,warm.accumulated);
  }
 }else{
  if(!expert_bank_)expert_bank_=std::make_unique<PackedExpertBank>(*catalog_,layer_);
  routed=expert_bank_->forward_batch_selected(x,routes.device_ids,routes.device_lhs,
                                               routes.reduction_slots,routes.weights);
 }
 finish_runtime_subcomponent(layer_,ProfileSubcomponent::MoeRouted,sub_started,routed.accumulated);
 sub_started=runtime_profile_start();
 auto shared=mx::astype(shared_.forward(x,mx::array(1.0f)),mx::float32);
 finish_runtime_subcomponent(layer_,ProfileSubcomponent::MoeShared,sub_started,shared);
 sub_started=runtime_profile_start();
 auto total=mx::astype(mx::add(routed.accumulated,shared),mx::bfloat16);
 finish_runtime_subcomponent(layer_,ProfileSubcomponent::MoeCombine,sub_started,total);
 return {mx::astype(shared,mx::bfloat16),mx::astype(routed.accumulated,mx::bfloat16),total};
}
mx::array MoEReference::expert_contribution(const mx::array& x, int id, const mx::array& weight) const {
 if (id < 0 || id >= 384) throw std::runtime_error("invalid routed expert");
 return expert(id).forward(x, weight);
}
ExpertComponents MoEReference::expert_components(const mx::array& x, int id, const mx::array& weight) const {
 if (id < 0 || id >= 384) throw std::runtime_error("invalid routed expert");
 return expert(id).components(x,weight);
}
}
