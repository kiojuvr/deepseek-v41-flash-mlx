#include "dsv41/moe.hpp"
#include "dsv41/execution_policy.hpp"
#include "dsv41/layer_owner.hpp"
#include "dsv41/moe_pipeline.hpp"
#include "dsv41/runtime_profile.hpp"
#include "dsv41/expert_backing.hpp"
#include "route_reduce.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <mutex>
#include <utility>
#include <vector>
#include <chrono>

namespace dsv41 {
namespace mx=mlx::core;
namespace {
std::size_t& construction_counter(){static std::size_t value=0;return value;}
std::size_t& loaded_expert_counter(){static std::size_t value=0;return value;}
ExpertBankIoStats& io_stats(){static ExpertBankIoStats value;return value;}
std::mutex& io_stats_mutex(){static std::mutex value;return value;}
struct BankProjection {mx::array weight,scale;std::size_t bytes;};

class BufferOwner {
public:
 explicit BufferOwner(std::size_t bytes):buffer_(mx::allocator::malloc(bytes)){}
 BufferOwner(const BufferOwner&)=delete;
 BufferOwner& operator=(const BufferOwner&)=delete;
 ~BufferOwner(){if(owned_)mx::allocator::free(buffer_);}
 void* data(){return buffer_.raw_ptr();}
 mx::allocator::Buffer release(){owned_=false;return buffer_;}
private:
 mx::allocator::Buffer buffer_;
 bool owned_=true;
};

std::string expert_prefix(int layer,int expert,const char* name){
 return reference_moe_prefix(layer)+".ffn.experts."+std::to_string(expert)+"."+name;
}

BankProjection load_bank(WeightCatalog& catalog,int layer,const char* name,int n,int k,
                         const std::vector<int>& expert_ids){
 const std::size_t experts=expert_ids.size();
 const std::size_t bytes_per_expert=std::size_t(n)*std::size_t(k)/2;
 const std::size_t scales_per_expert=std::size_t(n)*std::size_t(k)/32;
 const std::size_t weight_bytes=std::size_t(experts)*bytes_per_expert;
 const std::size_t scale_bytes=std::size_t(experts)*scales_per_expert;
 BufferOwner weights(weight_bytes),scales(scale_bytes);
 auto* weight_data=static_cast<std::byte*>(weights.data());
 auto* scale_data=static_cast<std::byte*>(scales.data());
 struct Input { TensorFile weight,scale; };
 std::vector<Input> inputs;inputs.reserve(experts);
 for(const int expert:expert_ids)
  inputs.push_back({catalog.tensor(expert_prefix(layer,expert,name)+".weight"),
                    catalog.tensor(expert_prefix(layer,expert,name)+".scale")});
 const std::size_t workers=std::min<std::size_t>(runtime_expert_io_threads(),experts);
 std::atomic<std::size_t> next{0}; std::exception_ptr failure; std::mutex failure_mutex;
 const auto read_started=std::chrono::steady_clock::now();
 auto read_one=[&]{
  try {
   for(;;){
    const auto local=next.fetch_add(1); if(local>=experts)break;
    const auto& weight=inputs[local].weight;
    const auto& scale=inputs[local].scale;
    if((weight.dtype!="I8"&&weight.dtype!="U8")||
       weight.shape!=std::vector<std::uint64_t>{std::uint64_t(n),std::uint64_t(k/2)}||
       weight.size_bytes!=bytes_per_expert||scale.dtype!="F8_E8M0"||
       scale.shape!=std::vector<std::uint64_t>{std::uint64_t(n),std::uint64_t(k/32)}||
       scale.size_bytes!=scales_per_expert)throw std::runtime_error("unexpected routed expert bank layout");
    weight.read(0,{weight_data+local*bytes_per_expert,bytes_per_expert});
    scale.read(0,{scale_data+local*scales_per_expert,scales_per_expert});
   }
  } catch(...) { std::lock_guard lock(failure_mutex); if(!failure)failure=std::current_exception(); }
 };
 std::vector<std::thread> io_threads;io_threads.reserve(workers);
 for(std::size_t i=0;i<workers;++i)io_threads.emplace_back(read_one);
 for(auto& thread:io_threads)thread.join();
 const double read_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-read_started).count();
 if(failure)std::rethrow_exception(failure);
 auto* scale_codes=reinterpret_cast<const std::uint8_t*>(scale_data);
 if(std::find(scale_codes,scale_codes+scale_bytes,std::uint8_t(255))!=scale_codes+scale_bytes)
  throw std::runtime_error("NaN routed expert scale");
 auto weight=mx::array(weights.release(),{int(experts),n,k/8},mx::uint32);
 auto scale=mx::array(scales.release(),{int(experts),n,k/32},mx::uint8);
 {
  std::lock_guard lock(io_stats_mutex());
  io_stats().read_seconds+=read_seconds;
  io_stats().read_calls+=experts;
 }
 return {std::move(weight),std::move(scale),weight_bytes+scale_bytes};
}

GroupedExpertComponents grouped_forward(const mx::array& x,
 const std::array<int,6>& expert_ids,const mx::array& route_weights,
 const mx::array& w1,const mx::array& s1,const mx::array& w2,
 const mx::array& s2,const mx::array& w3,const mx::array& s3,
 const mx::array& rhs_ids){
 if(x.dtype()!=mx::bfloat16||x.shape()!=mx::Shape({1,5120})||
    route_weights.dtype()!=mx::float32||route_weights.shape()!=mx::Shape({6}))
  throw std::runtime_error("invalid grouped expert input");
 std::array<int,6> unique=expert_ids;std::sort(unique.begin(),unique.end());
 if(unique.front()<0||unique.back()>=384||std::adjacent_find(unique.begin(),unique.end())!=unique.end())
  throw std::runtime_error("invalid grouped expert IDs");
 auto first_lhs=mx::zeros({6},mx::uint32),down_lhs=mx::arange(6,mx::uint32);
 auto gather=[&](const mx::array& value,const mx::array& weight,const mx::array& scale,const mx::array& lhs){
  { std::lock_guard lock(io_stats_mutex()); ++io_stats().qmm_dispatches; io_stats().qmm_rows_total+=6; io_stats().qmm_rows_max=std::max<std::size_t>(io_stats().qmm_rows_max,6); }
  return mx::gather_qmm(value,weight,scale,std::nullopt,lhs,rhs_ids,true,32,4,"mxfp4",false,mx::Device::gpu);
 };
 auto first_input=linear_activation_reference(x).decoded;
 auto gate=gather(mx::reshape(first_input,{1,1,5120}),w1,s1,first_lhs);
 auto up=gather(mx::reshape(first_input,{1,1,5120}),w3,s3,first_lhs);
 auto g=mx::clip(mx::astype(gate,mx::float32),mx::array(-1e30f),mx::array(10.0f));
 auto u=mx::clip(mx::astype(up,mx::float32),mx::array(-10.0f),mx::array(10.0f));
 auto activation=mx::astype(mx::multiply(mx::multiply(g,mx::sigmoid(g)),u),mx::bfloat16);
 auto down_input=linear_activation_reference(mx::reshape(activation,{6,2304})).decoded;
 auto down=gather(mx::reshape(down_input,{6,1,2304}),w2,s2,down_lhs);
 auto weighted=mx::astype(mx::multiply(mx::astype(down,mx::float32),mx::reshape(route_weights,{6,1,1})),mx::bfloat16);
 auto routed=mx::zeros({1,5120},mx::float32);
 for(int id:unique){
  int slot=int(std::find(expert_ids.begin(),expert_ids.end(),id)-expert_ids.begin());
  routed=mx::add(routed,mx::astype(mx::reshape(mx::slice(weighted,{slot,0,0},{slot+1,1,5120}),{1,5120}),mx::float32));
 }
 return {gate,up,activation,down,weighted,routed,mx::astype(routed,mx::bfloat16)};
}
}

std::size_t packed_expert_bank_construction_count(){return construction_counter();}
std::size_t packed_expert_bank_loaded_expert_count(){return loaded_expert_counter();}
void reset_packed_expert_bank_construction_count(){construction_counter()=0;loaded_expert_counter()=0;}
ExpertBankIoStats expert_bank_io_stats(){std::lock_guard lock(io_stats_mutex());return io_stats();}
void reset_expert_bank_io_stats(){std::lock_guard lock(io_stats_mutex());io_stats()={};}

PackedExpertBank::PackedExpertBank(WeightCatalog& catalog,int layer)
 :PackedExpertBank(catalog,layer,[]{
   std::vector<int> ids(384);std::iota(ids.begin(),ids.end(),0);return ids;
  }()){}

PackedExpertBank::PackedExpertBank(WeightCatalog&,int layer,
 std::shared_ptr<const ExpertBackingStore> backing)
 :w1_(mx::array(0)),s1_(mx::array(0)),w2_(mx::array(0)),s2_(mx::array(0)),w3_(mx::array(0)),s3_(mx::array(0)){
 if(!backing)throw std::runtime_error("file-backed expert bank requires a backing store");
 const auto started=std::chrono::steady_clock::now();reference_moe_prefix(layer);layer_=layer;
 expert_ids_.resize(384);std::iota(expert_ids_.begin(),expert_ids_.end(),0);global_to_local_.fill(-1);
 for(int i=0;i<384;++i)global_to_local_[i]=i;
 auto w1=backing->projection(layer,"w1",2304,5120);
 auto w3=backing->projection(layer,"w3",2304,5120);
 auto w2=backing->projection(layer,"w2",5120,2304);
 packed_bytes_=w1.bytes+w2.bytes+w3.bytes;
 w1_=std::move(w1.weight);s1_=std::move(w1.scale);w2_=std::move(w2.weight);s2_=std::move(w2.scale);w3_=std::move(w3.weight);s3_=std::move(w3.scale);
 mx::eval(w1_,s1_,w2_,s2_,w3_,s3_);++construction_counter();loaded_expert_counter()+=384;
 {std::lock_guard lock(io_stats_mutex());io_stats().constructions++;io_stats().total_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();}
}

PackedExpertBank::PackedExpertBank(WeightCatalog& catalog,int layer,
 const std::vector<int>& expert_ids)
 :w1_(mx::array(0)),s1_(mx::array(0)),w2_(mx::array(0)),s2_(mx::array(0)),w3_(mx::array(0)),s3_(mx::array(0)){
 const auto started=std::chrono::steady_clock::now();
 reference_moe_prefix(layer);
 layer_=layer;
 if(expert_ids.empty()||expert_ids.size()>384||expert_ids.front()<0||expert_ids.back()>=384||
    !std::is_sorted(expert_ids.begin(),expert_ids.end())||
    std::adjacent_find(expert_ids.begin(),expert_ids.end())!=expert_ids.end())
  throw std::runtime_error("compact expert bank IDs must be sorted, unique, and in range");
 expert_ids_=expert_ids;global_to_local_.fill(-1);
 for(std::size_t local=0;local<expert_ids_.size();++local)
  global_to_local_[expert_ids_[local]]=int(local);
 auto w1=load_bank(catalog,layer,"w1",2304,5120,expert_ids_);
 auto w3=load_bank(catalog,layer,"w3",2304,5120,expert_ids_);
 auto w2=load_bank(catalog,layer,"w2",5120,2304,expert_ids_);
 packed_bytes_=w1.bytes+w2.bytes+w3.bytes;
 w1_=std::move(w1.weight);s1_=std::move(w1.scale);
 w2_=std::move(w2.weight);s2_=std::move(w2.scale);
 w3_=std::move(w3.weight);s3_=std::move(w3.scale);
 mx::eval(w1_,s1_,w2_,s2_,w3_,s3_);
 ++construction_counter();
 loaded_expert_counter()+=expert_ids_.size();
 {
  std::lock_guard lock(io_stats_mutex());
  io_stats().constructions++;
  io_stats().total_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
 }
}

ResidentExpertAtlas::ResidentExpertAtlas(WeightCatalog& catalog){
 // Populate the private object completely before make_shared publishes it.
 // If any layer fails, ordinary stack unwinding releases every completed bank.
 if(const char* root=std::getenv("DSV41_RUNTIME_EXPERT_BACKING_DIR");root&&*root)
  backing_=std::make_shared<ExpertBackingStore>(root,catalog);
 for(int layer=0;layer<40;++layer){
  auto bank=backing_?std::make_shared<PackedExpertBank>(catalog,layer,backing_):
                      std::make_shared<PackedExpertBank>(catalog,layer);
  packed_bytes_+=bank->packed_bytes();
  banks_[layer]=std::move(bank);
 }
}

const PackedExpertBank& ResidentExpertAtlas::bank(int layer) const{
 if(layer<0||layer>=int(banks_.size())||!banks_[layer])
  throw std::runtime_error("resident expert atlas layer is unavailable");
 return *banks_[layer];
}

std::shared_ptr<const ResidentExpertAtlas> make_resident_expert_atlas(WeightCatalog& catalog){
 if(!runtime_resident_expert_atlas_enabled())return {};
 if(!runtime_packed_expert_bank_enabled())
  throw std::runtime_error("resident expert atlas requires packed expert banks");
 if(runtime_compact_expert_bank_enabled())
  throw std::runtime_error("resident and compact expert banks are mutually exclusive");
 return std::make_shared<const ResidentExpertAtlas>(catalog);
}

std::uint32_t PackedExpertBank::local_expert_id(int global_id) const{
 if(global_id<0||global_id>=384||global_to_local_[global_id]<0)
  throw std::runtime_error("route references expert absent from compact bank");
 return std::uint32_t(global_to_local_[global_id]);
}

GroupedExpertComponents PackedExpertBank::forward_selected(const mx::array& x,
 const std::array<int,6>& expert_ids,const mx::array& route_weights) const{
 std::array<std::uint32_t,6> local_ids;
 for(int slot=0;slot<6;++slot)local_ids[slot]=local_expert_id(expert_ids[slot]);
 auto ids=mx::array(local_ids.begin(),{6},mx::uint32);
 return grouped_forward(x,expert_ids,route_weights,w1_,s1_,w2_,s2_,w3_,s3_,ids);
}

GroupedExpertBatchResult PackedExpertBank::forward_batch_selected(const mx::array& x,
 const std::vector<std::array<int,6>>& expert_ids,const mx::array& route_weights) const{
 const int tokens=x.ndim()==2?x.shape(0):0,assignments=tokens*6;
 if(x.dtype()!=mx::bfloat16||tokens<1||tokens>128||x.shape(1)!=5120||
    int(expert_ids.size())!=tokens||route_weights.dtype()!=mx::float32||
    route_weights.shape()!=mx::Shape({tokens,6}))
  throw std::runtime_error("invalid grouped expert batch input");
 std::vector<std::uint32_t> lhs_ids,rhs_ids,reduction_slots;
 lhs_ids.reserve(assignments);rhs_ids.reserve(assignments);reduction_slots.reserve(assignments);
 for(int token=0;token<tokens;++token){
  auto unique=expert_ids[token];std::sort(unique.begin(),unique.end());
  if(unique.front()<0||unique.back()>=384||std::adjacent_find(unique.begin(),unique.end())!=unique.end())
   throw std::runtime_error("invalid grouped expert batch IDs");
  for(int slot=0;slot<6;++slot){lhs_ids.push_back(std::uint32_t(token));rhs_ids.push_back(local_expert_id(expert_ids[token][slot]));}
  for(int id:unique)reduction_slots.push_back(std::uint32_t(
   std::find(expert_ids[token].begin(),expert_ids[token].end(),id)-expert_ids[token].begin()));
 }
 return forward_batch_selected(x,mx::array(rhs_ids.begin(),{tokens,6},mx::uint32),
  mx::array(lhs_ids.begin(),{tokens,6},mx::uint32),
  mx::array(reduction_slots.begin(),{tokens,6},mx::uint32),route_weights);
}

GroupedExpertBatchResult PackedExpertBank::forward_batch_selected(const mx::array& x,
 const mx::array& expert_ids,const mx::array& lhs_ids,const mx::array& reduction_slots,
 const mx::array& route_weights) const{
 const int tokens=x.ndim()==2?x.shape(0):0,assignments=tokens*6;
 if(x.dtype()!=mx::bfloat16||tokens<1||tokens>128||x.shape(1)!=5120||
    expert_ids.dtype()!=mx::uint32||expert_ids.shape()!=mx::Shape({tokens,6})||
    lhs_ids.dtype()!=mx::uint32||lhs_ids.shape()!=mx::Shape({tokens,6})||
    reduction_slots.dtype()!=mx::uint32||reduction_slots.shape()!=mx::Shape({tokens,6})||
    route_weights.dtype()!=mx::float32||route_weights.shape()!=mx::Shape({tokens,6}))
  throw std::runtime_error("invalid device grouped expert batch input");
 auto lhs=mx::reshape(lhs_ids,{assignments});
 auto rhs=mx::reshape(expert_ids,{assignments});
 auto gather=[&](const mx::array& value,const mx::array& weight,const mx::array& scale,const mx::array& left){
  { std::lock_guard lock(io_stats_mutex()); ++io_stats().qmm_dispatches; io_stats().qmm_rows_total+=assignments; io_stats().qmm_rows_max=std::max<std::size_t>(io_stats().qmm_rows_max,assignments); }
  return mx::gather_qmm(value,weight,scale,std::nullopt,left,rhs,true,32,4,"mxfp4",false,mx::Device::gpu);
 };
 auto first_input=linear_activation_reference(x).decoded;
 auto gate=gather(mx::reshape(first_input,{tokens,1,5120}),w1_,s1_,lhs);
 auto up=gather(mx::reshape(first_input,{tokens,1,5120}),w3_,s3_,lhs);
 auto g=mx::clip(mx::astype(gate,mx::float32),mx::array(-1e30f),mx::array(10.0f));
 auto u=mx::clip(mx::astype(up,mx::float32),mx::array(-10.0f),mx::array(10.0f));
 auto activation=mx::astype(mx::multiply(mx::multiply(g,mx::sigmoid(g)),u),mx::bfloat16);
 auto flat_activation=mx::reshape(activation,{assignments,2304});
 std::vector<mx::array> decoded_parts;
 const int assignment_chunk=int(runtime_expert_assignment_chunk());
 for(int offset=0;offset<assignments;offset+=assignment_chunk){
  int end=std::min(offset+assignment_chunk,assignments);
  decoded_parts.push_back(linear_activation_reference(mx::slice(flat_activation,{offset,0},{end,2304})).decoded);
 }
 auto down_input=decoded_parts.size()==1?decoded_parts.front():mx::concatenate(decoded_parts,0);
 auto assignment_ids=mx::arange(assignments,mx::uint32);
 auto down=gather(mx::reshape(down_input,{assignments,1,2304}),w2_,s2_,assignment_ids);
 auto weighted=mx::astype(mx::multiply(mx::astype(down,mx::float32),
  mx::reshape(route_weights,{assignments,1,1})),mx::bfloat16);
 auto token_base=mx::multiply(mx::expand_dims(mx::arange(tokens,mx::uint32),1),mx::array(std::uint32_t(6)));
 auto reduction_assignments=mx::add(reduction_slots,token_base);
 static auto reduce=mx::fast::metal_kernel("dsv41_route_reduce_batch",
  {"weighted","reduction_slots","params"},{"accumulated"},dsv41_route_reduce_source);
 auto result=reduce({weighted,reduction_assignments,mx::array({std::uint32_t(tokens)})},
  {{tokens,5120}},{mx::float32},{tokens*5120,1,1},{256,1,1},{},std::nullopt,false,mx::Device::gpu).front();
 return {result,mx::astype(result,mx::bfloat16)};
}

GroupedExpertBatchResult PackedExpertBank::forward_batch_expert_major(const mx::array& x,
 const mx::array& expert_ids,const mx::array& lhs_ids,const mx::array& reduction_slots,
 const mx::array& route_weights,bool diagnostic) const{
 const int tokens=x.ndim()==2?x.shape(0):0,assignments=tokens*6;
 if(x.dtype()!=mx::bfloat16||tokens<1||tokens>128||x.shape(1)!=5120||
    expert_ids.dtype()!=mx::uint32||expert_ids.shape()!=mx::Shape({tokens,6})||
    lhs_ids.dtype()!=mx::uint32||lhs_ids.shape()!=mx::Shape({tokens,6})||
    reduction_slots.dtype()!=mx::uint32||reduction_slots.shape()!=mx::Shape({tokens,6})||
    route_weights.dtype()!=mx::float32||route_weights.shape()!=mx::Shape({tokens,6}))
  throw std::runtime_error("invalid expert-major batch input");
 const bool grouped_pipeline=runtime_grouped_expert_pipeline_enabled()&&tokens<=8;
 if(!diagnostic){
  std::lock_guard lock(io_stats_mutex());
  ++io_stats().expert_major_batches;io_stats().expert_major_assignments+=assignments;
  if(grouped_pipeline){++io_stats().grouped_pipeline_batches;
   io_stats().grouped_pipeline_assignments+=assignments;}
 }
 auto routed_stage_started=runtime_profile_start();
 auto flat_rhs=mx::reshape(expert_ids,{assignments});
 auto flat_lhs=mx::reshape(lhs_ids,{assignments});
 auto order=mx::argsort(flat_rhs);
 auto rhs=mx::take(flat_rhs,order);
 auto lhs=mx::take(flat_lhs,order);
 auto sorted_weights=mx::take(mx::reshape(route_weights,{assignments}),order);
 if(!diagnostic)finish_runtime_routed_stage(layer_,0,routed_stage_started,rhs);
 routed_stage_started=runtime_profile_start();
 auto gather=[&](const mx::array& value,const mx::array& weight,const mx::array& scale,const mx::array& left){
  if(!diagnostic){ std::lock_guard lock(io_stats_mutex()); ++io_stats().qmm_dispatches; io_stats().qmm_rows_total+=assignments; io_stats().qmm_rows_max=std::max<std::size_t>(io_stats().qmm_rows_max,assignments); }
  return mx::gather_qmm(value,weight,scale,std::nullopt,left,rhs,true,32,4,"mxfp4",false,mx::Device::gpu);
 };
 auto first_input=linear_activation_reference(x).decoded;
 auto gate=gather(mx::reshape(first_input,{tokens,1,5120}),w1_,s1_,lhs);
 auto up=gather(mx::reshape(first_input,{tokens,1,5120}),w3_,s3_,lhs);
 if(!diagnostic)finish_runtime_routed_stage(layer_,1,routed_stage_started,up);
 routed_stage_started=runtime_profile_start();
 mx::array activation=grouped_pipeline?fused_swiglu_fp8_roundtrip(gate,up):[&]{
  auto g=mx::clip(mx::astype(gate,mx::float32),mx::array(-1e30f),mx::array(10.0f));
  auto u=mx::clip(mx::astype(up,mx::float32),mx::array(-10.0f),mx::array(10.0f));
  return mx::astype(mx::multiply(mx::multiply(g,mx::sigmoid(g)),u),mx::bfloat16);
 }();
 mx::array down_input=activation;
 if(!grouped_pipeline){
  auto flat_activation=mx::reshape(activation,{assignments,2304});
  std::vector<mx::array> decoded_parts;
  const int assignment_chunk=int(runtime_expert_assignment_chunk());
  for(int offset=0;offset<assignments;offset+=assignment_chunk){
   int end=std::min(offset+assignment_chunk,assignments);
   decoded_parts.push_back(linear_activation_reference(
    mx::slice(flat_activation,{offset,0},{end,2304})).decoded);
  }
  down_input=decoded_parts.size()==1?decoded_parts.front():mx::concatenate(decoded_parts,0);
 }
 if(!diagnostic)finish_runtime_routed_stage(layer_,2,routed_stage_started,down_input);
 routed_stage_started=runtime_profile_start();
 auto down=gather(mx::reshape(down_input,{assignments,1,2304}),w2_,s2_,mx::arange(assignments,mx::uint32));
 if(grouped_pipeline)down=grouped_expert_pipeline(gate,up,activation,down);
 auto weighted=mx::astype(mx::multiply(mx::astype(down,mx::float32),
  mx::reshape(sorted_weights,{assignments,1,1})),mx::bfloat16);
 if(!diagnostic)finish_runtime_routed_stage(layer_,3,routed_stage_started,weighted);
 routed_stage_started=runtime_profile_start();
 auto inverse=mx::argsort(order);
 auto token_base=mx::multiply(mx::expand_dims(mx::arange(tokens,mx::uint32),1),mx::array(std::uint32_t(6)));
 auto canonical=mx::reshape(mx::add(reduction_slots,token_base),{assignments});
 auto reduction_assignments=mx::reshape(mx::take(inverse,canonical),{tokens,6});
 static auto reduce=mx::fast::metal_kernel("dsv41_route_reduce_expert_major",
  {"weighted","reduction_slots","params"},{"accumulated"},dsv41_route_reduce_source);
 auto result=reduce({weighted,reduction_assignments,mx::array({std::uint32_t(tokens)})},
  {{tokens,5120}},{mx::float32},{tokens*5120,1,1},{256,1,1},{},std::nullopt,false,mx::Device::gpu).front();
 if(!diagnostic)finish_runtime_routed_stage(layer_,4,routed_stage_started,result);
 return {result,mx::astype(result,mx::bfloat16)};
}

GroupedExpertComponents forward_grouped_selected(const mx::array& x,
 const std::array<int,6>& expert_ids,const std::array<const ExpertReference*,6>& experts,
 const mx::array& route_weights){
 auto bank=[&](const PackedLinearReference& (ExpertReference::*projection)() const,bool scales){
  std::vector<mx::array> rows;rows.reserve(6);
  for(const auto* expert:experts){
   if(expert==nullptr)throw std::runtime_error("null selected expert");
   const auto& linear=(expert->*projection)();
   rows.push_back(scales?linear.scales():linear.packed_weight());
  }
  return mx::stack(rows,0);
 };
 auto w1=bank(&ExpertReference::gate_projection,false),s1=bank(&ExpertReference::gate_projection,true);
 auto w2=bank(&ExpertReference::down_projection,false),s2=bank(&ExpertReference::down_projection,true);
 auto w3=bank(&ExpertReference::up_projection,false),s3=bank(&ExpertReference::up_projection,true);
 return grouped_forward(x,expert_ids,route_weights,w1,s1,w2,s2,w3,s3,mx::arange(6,mx::uint32));
}
}
