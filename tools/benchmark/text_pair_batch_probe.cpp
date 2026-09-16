#include "dsv41/checkpoint_atlas.hpp"
#include "dsv41/text_pair.hpp"
#include <mlx/mlx.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mx=mlx::core;
using J=nlohmann::json;

namespace {
void same(const mx::array& a,const mx::array& b,const char* name){
 if(a.shape()!=b.shape()||a.dtype()!=b.dtype())throw std::runtime_error(std::string(name)+" shape/dtype mismatch");
 auto dtype=a.dtype()==mx::bfloat16?mx::uint16:mx::uint32;
 auto ok=mx::all(mx::equal(mx::view(a,dtype),mx::view(b,dtype)));mx::eval(ok);
 if(!ok.item<bool>())throw std::runtime_error(std::string(name)+" bit mismatch");
}
}

int main(int argc,char** argv){try{
 if(argc!=5)throw std::runtime_error("usage: dsv41-text-pair-batch-probe checkpoint m1-summary metadata output-json");
 mx::set_default_device(mx::Device::gpu);dsv41::WeightCatalog catalog(argv[1],argv[2]);
 auto metadata=dsv41::EngramMetadata::load(argv[3]);
 const std::vector<std::uint32_t> ids{0,42,1000,17,9999,128,65535,7};
 if(setenv("DSV41_RUNTIME_LAYER_FINITE_CHECKS","0",1)!=0||
    setenv("DSV41_RUNTIME_PACKED_EXPERT_BANK","0",1)!=0||
    setenv("DSV41_RUNTIME_GROUP_SELECTED_EXPERTS","0",1)!=0)
  throw std::runtime_error("cannot select individual pair mode");
 dsv41::TextPairReference individual(catalog,metadata);dsv41::TextPairState individual_state(metadata);
 auto expected=individual.forward(ids,individual_state,0);
 if(setenv("DSV41_RUNTIME_PACKED_EXPERT_BANK","1",1)!=0)throw std::runtime_error("cannot select packed pair mode");
 dsv41::TextPairReference packed(catalog,metadata);dsv41::TextPairState packed_state(metadata);
 auto actual=packed.forward_packed_chunk(ids,packed_state,0);
 same(actual.hidden,expected.hidden,"hidden");same(actual.pre_mix,expected.pre_mix,"pre-mix");
 same(packed_state.first.rows(),individual_state.first.rows(),"layer-0 state");
 same(packed_state.second.rows(),individual_state.second.rows(),"layer-1 state");
 if(packed_state.hash.position()!=ids.size()||packed_state.first.position()!=ids.size()||
    packed_state.second.position()!=ids.size())throw std::runtime_error("pair position mismatch");
 J report={{"schema_version",1},{"status","probe_completed_requires_review"},{"tokens",ids.size()},
  {"layers",2},{"engram_layer",1},{"hidden_pre_mix_state_bits","exact"},
  {"active_bytes",mx::get_active_memory()},{"cache_bytes",mx::get_cache_memory()},{"peak_bytes",mx::get_peak_memory()},
  {"scope","Eight-token layer-major layer-0/Engram/layer-1 packed-MoE correctness probe; not performance/full-backbone qualification."}};
 std::ofstream out(argv[4]);if(!out)throw std::runtime_error("cannot create result JSON");out<<report.dump(2)<<'\n';
 std::cout<<"PASS: two-layer packed chunk with Engram matches token-serial hidden/pre-mix/state bits; review required\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
