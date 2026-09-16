#include "dsv41/checkpoint_atlas.hpp"
#include "dsv41/moe.hpp"
#include <mlx/mlx.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>
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
 if(argc!=4)throw std::runtime_error("usage: dsv41-moe-batch-probe checkpoint m1-summary output-json");
 mx::set_default_device(mx::Device::gpu);dsv41::WeightCatalog catalog(argv[1],argv[2]);
 constexpr int tokens=128;
 auto x=mx::astype(mx::reshape(mx::sin(mx::arange(tokens*5120,mx::float32)),{tokens,5120}),mx::bfloat16);
 if(setenv("DSV41_RUNTIME_PACKED_EXPERT_BANK","0",1)!=0||
    setenv("DSV41_RUNTIME_GROUP_SELECTED_EXPERTS","0",1)!=0||
    setenv("DSV41_RUNTIME_COMPACT_EXPERT_BANK","0",1)!=0)
  throw std::runtime_error("cannot select individual expert mode");
 dsv41::MoEReference individual(catalog,0);std::vector<mx::array> shared,routed,total;
 auto serial_started=Clock::now();
 for(int token=0;token<tokens;++token){
  dsv41::set_route_trace_token(std::uint64_t(token));
  auto value=individual.forward_components(mx::slice(x,{token,0},{token+1,5120}));
  shared.push_back(value.shared);routed.push_back(value.routed);total.push_back(value.total);
 }
 dsv41::MoEComponents expected{mx::concatenate(shared,0),mx::concatenate(routed,0),mx::concatenate(total,0)};
 mx::eval(expected.shared,expected.routed,expected.total);mx::synchronize();
 const double serial_seconds=elapsed(serial_started);
 if(setenv("DSV41_RUNTIME_PACKED_EXPERT_BANK","1",1)!=0||
    setenv("DSV41_RUNTIME_RESIDENT_EXPERT_ATLAS","1",1)!=0||
    setenv("DSV41_RUNTIME_COMPACT_EXPERT_BANK","0",1)!=0)
  throw std::runtime_error("cannot select packed expert mode");
 dsv41::MoEReference packed(catalog,0);auto batch_started=Clock::now();
 auto actual=packed.forward_batch_components(x,0);mx::eval(actual.shared,actual.routed,actual.total);mx::synchronize();
 if(!packed.packed_bank_loaded())throw std::runtime_error("packed expert bank was not retained");
 packed.release_packed_bank();
 if(!packed.packed_bank_loaded())throw std::runtime_error("resident expert bank was released");
 const double batch_seconds=elapsed(batch_started);
 same(actual.shared,expected.shared,"shared");same(actual.routed,expected.routed,"routed");same(actual.total,expected.total,"total");
 auto serial_run=[&]{
  std::vector<mx::array> values;values.reserve(tokens);
  for(int token=0;token<tokens;++token){dsv41::set_route_trace_token(std::uint64_t(token));
   values.push_back(individual.forward_components(mx::slice(x,{token,0},{token+1,5120})).total);}
  auto value=mx::concatenate(values,0);mx::eval(value);mx::synchronize();
 };
 auto batch_run=[&]{auto value=packed.forward_batch_components(x,0);mx::eval(value.total);mx::synchronize();};
 std::vector<double> serial_times,batch_times;
 for(int round=0;round<5;++round){
  auto timed=[](auto&& fn){auto start=Clock::now();fn();return elapsed(start);};
  if(round%2==0){serial_times.push_back(timed(serial_run));batch_times.push_back(timed(batch_run));}
  else{batch_times.push_back(timed(batch_run));serial_times.push_back(timed(serial_run));}
 }
 if(setenv("DSV41_RUNTIME_RESIDENT_EXPERT_ATLAS","0",1)!=0||
    setenv("DSV41_RUNTIME_COMPACT_EXPERT_BANK","1",1)!=0)
  throw std::runtime_error("cannot select compact expert mode");
 const auto loaded_before=dsv41::packed_expert_bank_loaded_expert_count();
 dsv41::MoEReference compact(catalog,0);auto compact_started=Clock::now();
 auto compact_value=compact.forward_batch_components(x,0);
 mx::eval(compact_value.shared,compact_value.routed,compact_value.total);mx::synchronize();
 const double compact_seconds=elapsed(compact_started);
 same(compact_value.shared,actual.shared,"compact shared");
 same(compact_value.routed,actual.routed,"compact routed");
 same(compact_value.total,actual.total,"compact total");
 J report={{"schema_version",1},{"status","probe_completed_requires_review"},{"layer",0},{"tokens",tokens},
  {"shared_routed_total_bits","exact"},{"serial_first_seconds",serial_seconds},{"batch_first_seconds",batch_seconds},
  {"active_bytes",mx::get_active_memory()},{"cache_bytes",mx::get_cache_memory()},{"peak_bytes",mx::get_peak_memory()},
  {"route_ties",dsv41::route_tie_count()},
  {"compact_experts",compact.packed_bank_expert_count()},
  {"compact_loaded_experts",dsv41::packed_expert_bank_loaded_expert_count()-loaded_before},
  {"compact_first_seconds",compact_seconds},{"compact_shared_routed_total_bits","exact"},
  {"routing_boundary","device top-7, assignment IDs and ascending-expert reduction slots; host sync only for tile tie/error audit; canonical one-row gate reductions"},
  {"scope","Layer-0 128-token full MoE batch versus token-serial individual experts; bit parity and bounded warm timing, not qualification."}};
 report["serial_warm_median_seconds"]=median(serial_times);
 report["batch_warm_median_seconds"]=median(batch_times);
 report["warm_speedup"]=median(serial_times)/median(batch_times);
 std::ofstream out(argv[3]);if(!out)throw std::runtime_error("cannot create result JSON");out<<report.dump(2)<<'\n';
 std::cout<<"PASS: 128-token full MoE batch matches token-serial individual shared/routed/total bits; review required\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
