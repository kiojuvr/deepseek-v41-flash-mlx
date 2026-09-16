#include "dsv41/block.hpp"
#include "dsv41/checkpoint_atlas.hpp"
#include "dsv41/model_entry.hpp"
#include <mlx/mlx.h>
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
}

int main(int argc,char** argv){try{
 if(argc!=4)throw std::runtime_error("usage: dsv41-block-batch-probe checkpoint m1-summary output-json");
 mx::set_default_device(mx::Device::gpu);dsv41::WeightCatalog catalog(argv[1],argv[2]);
 std::vector<std::uint32_t> ids(128);for(int i=0;i<128;++i)ids[i]=std::uint32_t((i*7919)%129263);
 dsv41::TextEntryReference entry(catalog);auto input=entry.forward(ids);
 if(setenv("DSV41_RUNTIME_LAYER_FINITE_CHECKS","0",1)!=0||
    setenv("DSV41_RUNTIME_PACKED_EXPERT_BANK","0",1)!=0||
    setenv("DSV41_RUNTIME_GROUP_SELECTED_EXPERTS","0",1)!=0)
  throw std::runtime_error("cannot select individual block mode");
 dsv41::BlockReference individual(catalog,0);dsv41::SwaLayerState individual_state;
 auto individual_first_started=Clock::now();auto expected=individual.forward(input.hidden,input.pre_mix,individual_state,0);
 const double individual_first=elapsed(individual_first_started);
 if(setenv("DSV41_RUNTIME_PACKED_EXPERT_BANK","1",1)!=0)throw std::runtime_error("cannot select packed block mode");
 dsv41::BlockReference packed(catalog,0);dsv41::SwaLayerState packed_state;
 auto packed_first_started=Clock::now();auto actual=packed.forward_packed_chunk(input.hidden,input.pre_mix,packed_state,0);
 const double packed_first=elapsed(packed_first_started);
 same(actual.hidden,expected.hidden,"first hidden");same(actual.pre_mix,expected.pre_mix,"first pre-mix");
 same(packed_state.rows(),individual_state.rows(),"first state rows");
 auto individual_warm_started=Clock::now();auto expected_next=individual.forward(input.hidden,input.pre_mix,individual_state,128);
 const double individual_warm=elapsed(individual_warm_started);
 auto packed_warm_started=Clock::now();auto actual_next=packed.forward_packed_chunk(input.hidden,input.pre_mix,packed_state,128);
 const double packed_warm=elapsed(packed_warm_started);
 same(actual_next.hidden,expected_next.hidden,"continuation hidden");same(actual_next.pre_mix,expected_next.pre_mix,"continuation pre-mix");
 same(packed_state.rows(),individual_state.rows(),"continuation state rows");
 if(packed_state.position()!=256||individual_state.position()!=256)throw std::runtime_error("continuation position mismatch");
 packed.release_packed_bank();
 J report={{"schema_version",1},{"status","probe_completed_requires_review"},{"layer",0},{"chunk_tokens",128},
  {"chunks",2},{"hidden_pre_mix_state_bits","exact"},{"individual_first_seconds",individual_first},
  {"packed_first_seconds",packed_first},{"individual_continuation_seconds",individual_warm},
  {"packed_continuation_seconds",packed_warm},{"active_bytes_after_release",mx::get_active_memory()},
  {"cache_bytes_after_release",mx::get_cache_memory()},{"peak_bytes",mx::get_peak_memory()},
  {"scope","Layer-0 pure-SWA Block token-serial attention plus batched packed MoE; two chunks, not full-backbone qualification."}};
 report["continuation_speedup"]=individual_warm/packed_warm;
 std::ofstream out(argv[3]);if(!out)throw std::runtime_error("cannot create result JSON");out<<report.dump(2)<<'\n';
 std::cout<<"PASS: layer-0 packed-MoE chunk Block matches token-serial hidden/pre-mix/state bits; review required\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
