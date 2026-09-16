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
 if(a.shape()!=b.shape()||a.dtype()!=b.dtype())
  throw std::runtime_error(std::string(name)+" shape/dtype mismatch");
 auto dtype=a.dtype()==mx::bfloat16?mx::uint16:mx::uint32;
 auto ok=mx::all(mx::equal(mx::view(a,dtype),mx::view(b,dtype)));mx::eval(ok);
 if(!ok.item<bool>())throw std::runtime_error(std::string(name)+" bit mismatch");
}
double elapsed(Clock::time_point start){
 return std::chrono::duration<double>(Clock::now()-start).count();
}
double median(std::vector<double> values){
 std::sort(values.begin(),values.end());return values[values.size()/2];
}
}

int main(int argc,char** argv){try{
 if(argc!=4)throw std::runtime_error("usage: dsv41-packed-moe-probe checkpoint m1-summary output-json");
 mx::set_default_device(mx::Device::gpu);
 dsv41::WeightCatalog catalog(argv[1],argv[2]);
 auto x=mx::astype(mx::reshape(mx::sin(mx::arange(5120,mx::float32)),{1,5120}),mx::bfloat16);

 if(setenv("DSV41_RUNTIME_PACKED_EXPERT_BANK","0",1)!=0)
  throw std::runtime_error("cannot disable packed expert bank");
 if(setenv("DSV41_RUNTIME_GROUP_SELECTED_EXPERTS","0",1)!=0)
  throw std::runtime_error("cannot disable selected expert grouping");
 dsv41::reset_route_tie_count();dsv41::reset_route_tie_records();
 dsv41::MoEReference individual(catalog,0);
 auto individual_started=Clock::now();
 auto expected=individual.forward_components(x);mx::eval(expected.shared,expected.routed,expected.total);mx::synchronize();
 const double individual_seconds=elapsed(individual_started);

 if(setenv("DSV41_RUNTIME_GROUP_SELECTED_EXPERTS","1",1)!=0)
  throw std::runtime_error("cannot enable selected expert grouping");
 dsv41::MoEReference selected(catalog,0);
 auto selected_started=Clock::now();
 auto selected_actual=selected.forward_components(x);
 mx::eval(selected_actual.shared,selected_actual.routed,selected_actual.total);mx::synchronize();
 const double selected_first_seconds=elapsed(selected_started);
 same(selected_actual.shared,expected.shared,"selected shared");
 same(selected_actual.routed,expected.routed,"selected routed");
 same(selected_actual.total,expected.total,"selected total");

 auto timed=[&](dsv41::MoEReference& moe){
  auto start=Clock::now();auto value=moe.forward_components(x);mx::eval(value.total);mx::synchronize();
  return elapsed(start);
 };
 std::vector<double> individual_times,selected_times;
 for(int round=0;round<9;++round){
  if(round%2==0){individual_times.push_back(timed(individual));selected_times.push_back(timed(selected));}
  else{selected_times.push_back(timed(selected));individual_times.push_back(timed(individual));}
 }

 if(setenv("DSV41_RUNTIME_GROUP_SELECTED_EXPERTS","0",1)!=0)
  throw std::runtime_error("cannot restore selected expert grouping");
 if(setenv("DSV41_RUNTIME_PACKED_EXPERT_BANK","1",1)!=0)
  throw std::runtime_error("cannot enable packed expert bank");
 dsv41::MoEReference packed(catalog,0);
 auto packed_started=Clock::now();
 auto actual=packed.forward_components(x);mx::eval(actual.shared,actual.routed,actual.total);mx::synchronize();
 const double packed_first_seconds=elapsed(packed_started);

 same(actual.shared,expected.shared,"shared");
 same(actual.routed,expected.routed,"routed");
 same(actual.total,expected.total,"total");
 J report={{"schema_version",1},{"status","probe_completed_requires_review"},{"layer",0},
  {"packed_shared_routed_total_bits","exact"},{"selected_shared_routed_total_bits","exact"},
  {"individual_first_seconds",individual_seconds},{"selected_first_seconds",selected_first_seconds},
  {"packed_first_seconds",packed_first_seconds},{"active_bytes",mx::get_active_memory()},
  {"cache_bytes",mx::get_cache_memory()},{"peak_bytes",mx::get_peak_memory()},
  {"route_ties",dsv41::route_tie_count()},
  {"scope","Layer-0 full MoE individual-versus-selected-group/full-bank bit parity and selected-group warm timing; not full-backbone qualification."}};
 report["individual_warm_median_seconds"]=median(individual_times);
 report["selected_warm_median_seconds"]=median(selected_times);
 report["selected_warm_speedup"]=median(individual_times)/median(selected_times);
 std::ofstream out(argv[3]);if(!out)throw std::runtime_error("cannot create result JSON");out<<report.dump(2)<<'\n';
 std::cout<<"PASS: layer-0 selected-group and full-bank MoE shared/routed/total match individual path bits\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
