#include "dsv41/checkpoint_atlas.hpp"
#include "dsv41/moe.hpp"
#include <mlx/mlx.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <set>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mx=mlx::core;
using Clock=std::chrono::steady_clock;
using J=nlohmann::json;

namespace {
void same(const mx::array& a,const mx::array& b,const char* name){
 if(a.shape()!=b.shape()||a.dtype()!=b.dtype())throw std::runtime_error(std::string(name)+" shape/dtype mismatch");
 auto dtype=a.dtype()==mx::bfloat16?mx::uint16:mx::uint32;
 auto ok=mx::all(mx::equal(mx::view(a,dtype),mx::view(b,dtype)));mx::eval(ok);
 if(!ok.item<bool>())throw std::runtime_error(std::string(name)+" bit mismatch");
}
double elapsed(Clock::time_point start){return std::chrono::duration<double>(Clock::now()-start).count();}
double median(std::vector<double> values){std::sort(values.begin(),values.end());return values[values.size()/2];}
}

int main(int argc,char** argv){try{
 if(argc!=4)throw std::runtime_error("usage: dsv41-expert-batch-probe checkpoint m1-summary output-json");
 mx::set_default_device(mx::Device::gpu);dsv41::WeightCatalog catalog(argv[1],argv[2]);
 auto full_started=Clock::now();dsv41::PackedExpertBank bank(catalog,0);mx::synchronize();
 const double full_construction_seconds=elapsed(full_started);
 constexpr int tokens=128;
 auto x=mx::astype(mx::reshape(mx::sin(mx::arange(tokens*5120,mx::float32)),{tokens,5120}),mx::bfloat16);
 std::vector<std::array<int,6>> ids(tokens);std::vector<float> raw_weights(tokens*6);
 for(int token=0;token<tokens;++token)for(int slot=0;slot<6;++slot){
  ids[token][slot]=(token%32)+slot*32;
  raw_weights[token*6+slot]=0.05f+0.025f*float((token+slot)%7);
 }
 auto weights=mx::array(raw_weights.begin(),{tokens,6},mx::float32);
 auto batched=[&]{return bank.forward_batch_selected(x,ids,weights);};
 std::vector<std::uint32_t> device_ids,device_lhs,reduction_slots;
 device_ids.reserve(tokens*6);device_lhs.reserve(tokens*6);reduction_slots.reserve(tokens*6);
 for(int token=0;token<tokens;++token){
  std::array<std::uint32_t,6> order{0,1,2,3,4,5};
  std::sort(order.begin(),order.end(),[&](auto a,auto b){return ids[token][a]<ids[token][b];});
  for(int slot=0;slot<6;++slot){device_ids.push_back(std::uint32_t(ids[token][slot]));device_lhs.push_back(std::uint32_t(token));}
  reduction_slots.insert(reduction_slots.end(),order.begin(),order.end());
 }
 auto expert_major=[&]{return bank.forward_batch_expert_major(x,
  mx::array(device_ids.begin(),{tokens,6},mx::uint32),
  mx::array(device_lhs.begin(),{tokens,6},mx::uint32),
  mx::array(reduction_slots.begin(),{tokens,6},mx::uint32),weights);};
 auto serial=[&]{
  std::vector<mx::array> accumulated,routed;accumulated.reserve(tokens);routed.reserve(tokens);
  for(int token=0;token<tokens;++token){
   auto value=bank.forward_selected(mx::slice(x,{token,0},{token+1,5120}),ids[token],
    mx::reshape(mx::slice(weights,{token,0},{token+1,6}),{6}));
   accumulated.push_back(value.accumulated);routed.push_back(value.routed);
  }
  return dsv41::GroupedExpertBatchResult{mx::concatenate(accumulated,0),mx::concatenate(routed,0)};
 };
 auto batch_result=batched(),expert_major_result=expert_major(),serial_result=serial();
 mx::eval(batch_result.accumulated,batch_result.routed,expert_major_result.accumulated,
          expert_major_result.routed,serial_result.accumulated,serial_result.routed);mx::synchronize();
 same(batch_result.accumulated,serial_result.accumulated,"batch accumulated");
 same(batch_result.routed,serial_result.routed,"batch routed");
 same(expert_major_result.accumulated,batch_result.accumulated,"expert-major accumulated");
 same(expert_major_result.routed,batch_result.routed,"expert-major routed");
 auto x1=mx::slice(x,{0,0},{1,5120});
 std::array<std::uint32_t,6> ids1_data{},slots1_data{};
 std::copy_n(device_ids.begin(),6,ids1_data.begin());
 std::copy_n(reduction_slots.begin(),6,slots1_data.begin());
 auto ids1=mx::array(ids1_data.begin(),{1,6},mx::uint32);
 auto lhs1=mx::zeros({1,6},mx::uint32);
 auto slots1=mx::array(slots1_data.begin(),{1,6},mx::uint32);
 auto weights1=mx::slice(weights,{0,0},{1,6});
 if(setenv("DSV41_RUNTIME_GROUPED_EXPERT_PIPELINE","0",1)!=0)
  throw std::runtime_error("cannot disable grouped expert pipeline");
 auto pipeline_reference=bank.forward_batch_expert_major(x1,ids1,lhs1,slots1,weights1);
 if(setenv("DSV41_RUNTIME_GROUPED_EXPERT_PIPELINE","1",1)!=0)
  throw std::runtime_error("cannot enable grouped expert pipeline");
 auto pipeline_candidate=bank.forward_batch_expert_major(x1,ids1,lhs1,slots1,weights1);
 if(unsetenv("DSV41_RUNTIME_GROUPED_EXPERT_PIPELINE")!=0)
  throw std::runtime_error("cannot clear grouped expert pipeline policy");
 mx::eval(pipeline_reference.accumulated,pipeline_reference.routed,
          pipeline_candidate.accumulated,pipeline_candidate.routed);mx::synchronize();
 same(pipeline_candidate.accumulated,pipeline_reference.accumulated,
      "grouped pipeline accumulated");
 same(pipeline_candidate.routed,pipeline_reference.routed,"grouped pipeline routed");
 std::set<int> selected_set;
 for(const auto& token_ids:ids)selected_set.insert(token_ids.begin(),token_ids.end());
 std::vector<int> selected(selected_set.begin(),selected_set.end());
 auto compact_started=Clock::now();dsv41::PackedExpertBank compact(catalog,0,selected);mx::synchronize();
 const double compact_construction_seconds=elapsed(compact_started);
 auto compact_result=compact.forward_batch_selected(x,ids,weights);
 mx::eval(compact_result.accumulated,compact_result.routed);mx::synchronize();
 same(compact_result.accumulated,batch_result.accumulated,"compact accumulated");
 same(compact_result.routed,batch_result.routed,"compact routed");
 auto timed=[](auto&& fn){auto start=Clock::now();auto value=fn();mx::eval(value.routed);mx::synchronize();return elapsed(start);};
 std::vector<double> batch_times,serial_times;
 for(int round=0;round<5;++round){
  if(round%2==0){serial_times.push_back(timed(serial));batch_times.push_back(timed(batched));}
  else{batch_times.push_back(timed(batched));serial_times.push_back(timed(serial));}
 }
 auto compact_run=[&]{return compact.forward_batch_selected(x,ids,weights);};
 std::vector<double> compact_times;
 for(int round=0;round<5;++round)compact_times.push_back(timed(compact_run));
 // One-token expert-major path: the exact decode shape on the real layer-0 bank.
 auto one_run=[&]{return bank.forward_batch_expert_major(x1,ids1,lhs1,slots1,weights1);};
 std::vector<double> one_times;
 for(int round=0;round<30;++round)one_times.push_back(timed(one_run));
 std::printf("tokens=1 expert_major median: %.4f ms (real layer-0 bank, 384 experts, 6 routes)\n",
  median(one_times)*1000.0);
 J report={{"schema_version",1},{"status","probe_completed_requires_review"},{"layer",0},{"tokens",tokens},
  {"routes",tokens*6},{"accumulated_and_routed_bits","exact"},{"expert_major_bits","exact"},
  {"grouped_pipeline_bits","exact"},
  {"expert_major_order","stable device argsort by expert; canonical expert-ID reduction order"},
  {"serial_median_seconds",median(serial_times)},
  {"batch_median_seconds",median(batch_times)},{"active_bytes",mx::get_active_memory()},
  {"cache_bytes",mx::get_cache_memory()},{"peak_bytes",mx::get_peak_memory()},
  {"full_bank_experts",bank.expert_count()},{"full_bank_bytes",bank.packed_bytes()},
  {"full_bank_construction_seconds",full_construction_seconds},
  {"compact_bank_experts",compact.expert_count()},{"compact_bank_bytes",compact.packed_bytes()},
  {"compact_bank_construction_seconds",compact_construction_seconds},
  {"compact_batch_median_seconds",median(compact_times)},
  {"tokens1_expert_major_median_seconds",median(one_times)},
  {"compact_vs_full_bits","exact"},
  {"scope","One-layer 128-token packed expert batch, including a 192-expert route-first compact bank; not full-backbone qualification."}};
 report["speedup"]=report["serial_median_seconds"].get<double>()/report["batch_median_seconds"].get<double>();
 std::ofstream out(argv[3]);if(!out)throw std::runtime_error("cannot create result JSON");out<<report.dump(2)<<'\n';
 std::cout<<"PASS: 128-token batched packed experts match token-serial accumulated/routed bits; review required\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
