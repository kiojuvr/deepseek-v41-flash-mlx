#include "dsv41/checkpoint_atlas.hpp"
#include "dsv41/moe.hpp"
#include <mlx/mlx.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace mx=mlx::core;
using Clock=std::chrono::steady_clock;
using J=nlohmann::json;

void same(const mx::array& a,const mx::array& b,const char* name){
 if(a.shape()!=b.shape()||a.dtype()!=b.dtype())throw std::runtime_error(std::string(name)+" shape/dtype mismatch");
 auto dtype=a.dtype()==mx::bfloat16?mx::uint16:mx::uint32;
 auto ok=mx::all(mx::equal(mx::view(a,dtype),mx::view(b,dtype)));mx::eval(ok);
 if(!ok.item<bool>())throw std::runtime_error(std::string(name)+" bit mismatch");
}
double elapsed(Clock::time_point start){return std::chrono::duration<double>(Clock::now()-start).count();}
double median(std::vector<double> values){std::sort(values.begin(),values.end());return values[values.size()/2];}

int main(int argc,char** argv){try{
 if(argc!=4)throw std::runtime_error("usage: dsv41-expert-bank-probe checkpoint m1-summary output-json");
 mx::set_default_device(mx::Device::gpu);
 dsv41::WeightCatalog catalog(argv[1],argv[2]);
 auto started=Clock::now();dsv41::PackedExpertBank bank(catalog,0);mx::synchronize();
 const double construction_seconds=elapsed(started);
 const std::array<int,6> ids{383,0,255,63,191,127};
 const std::array<float,6> raw_weights{0.05f,0.1f,0.15f,0.2f,0.25f,0.3f};
 auto weights=mx::array(raw_weights.begin(),{6},mx::float32);
 auto x=mx::astype(mx::reshape(mx::sin(mx::arange(5120,mx::float32)),{1,5120}),mx::bfloat16);
 auto grouped=bank.forward_selected(x,ids,weights);mx::eval(grouped.gate,grouped.up,grouped.activation,grouped.down,grouped.weighted,grouped.accumulated,grouped.routed);
 std::vector<std::unique_ptr<dsv41::ExpertReference>> individual;
 std::vector<mx::array> outputs;
 for(int slot=0;slot<6;++slot){
  individual.push_back(std::make_unique<dsv41::ExpertReference>(catalog,ids[slot],0));
  auto component=individual.back()->components(x,mx::array(raw_weights[slot]));
  same(mx::slice(grouped.gate,{slot,0,0},{slot+1,1,2304}),mx::reshape(component.gate,{1,1,2304}),"gate");
  same(mx::slice(grouped.up,{slot,0,0},{slot+1,1,2304}),mx::reshape(component.up,{1,1,2304}),"up");
  same(mx::slice(grouped.activation,{slot,0,0},{slot+1,1,2304}),mx::reshape(component.activation,{1,1,2304}),"activation");
  same(mx::slice(grouped.weighted,{slot,0,0},{slot+1,1,5120}),mx::reshape(component.output,{1,1,5120}),"weighted down");
  outputs.push_back(component.output);
 }
 auto order=ids;std::sort(order.begin(),order.end());auto expected=mx::zeros({1,5120},mx::float32);
 for(int id:order){int slot=int(std::find(ids.begin(),ids.end(),id)-ids.begin());expected=mx::add(expected,mx::astype(outputs[slot],mx::float32));}
 same(grouped.accumulated,expected,"routed FP32 accumulation");
 same(grouped.routed,mx::astype(expected,mx::bfloat16),"routed sum");

 auto grouped_run=[&]{auto value=bank.forward_selected(x,ids,weights);mx::eval(value.routed);mx::synchronize();};
 auto individual_run=[&]{
  auto value=mx::zeros({1,5120},mx::float32);
  for(int id:order){int slot=int(std::find(ids.begin(),ids.end(),id)-ids.begin());
   value=mx::add(value,mx::astype(individual[slot]->forward(x,mx::array(raw_weights[slot])),mx::float32));}
  auto out=mx::astype(value,mx::bfloat16);mx::eval(out);mx::synchronize();
 };
 grouped_run();individual_run();std::vector<double> grouped_times,individual_times;
 for(int round=0;round<9;++round){
  auto measure=[](auto&& fn){auto begin=Clock::now();fn();return elapsed(begin);};
  if(round%2==0){individual_times.push_back(measure(individual_run));grouped_times.push_back(measure(grouped_run));}
  else{grouped_times.push_back(measure(grouped_run));individual_times.push_back(measure(individual_run));}
 }
 J report={{"schema_version",1},{"status","probe_completed_requires_review"},{"layer",0},
  {"bank_experts",384},{"selected_expert_ids",ids},{"packed_bank_bytes",bank.packed_bytes()},
  {"construction_seconds",construction_seconds},{"active_bytes",mx::get_active_memory()},
  {"cache_bytes",mx::get_cache_memory()},{"peak_bytes",mx::get_peak_memory()},
  {"individual_median_seconds",median(individual_times)},{"grouped_median_seconds",median(grouped_times)},
  {"scope","One-layer packed expert bank: intermediate/output bit parity and bounded warm timing. Not full-model qualification."}};
 report["speedup"]=report["individual_median_seconds"].get<double>()/report["grouped_median_seconds"].get<double>();
 std::ofstream out(argv[3]);if(!out)throw std::runtime_error("cannot create result JSON");out<<report.dump(2)<<'\n';
 std::cout<<"PASS: 384-expert layer bank matches six selected individual experts and routed sum bits; review required\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
