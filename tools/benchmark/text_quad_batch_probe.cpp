#include "dsv41/checkpoint_atlas.hpp"
#include "dsv41/text_quad.hpp"
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
 auto dtype=a.dtype()==mx::bfloat16?mx::uint16:mx::uint8;
 auto ok=mx::all(mx::equal(mx::view(a,dtype),mx::view(b,dtype)));mx::eval(ok);
 if(!ok.item<bool>())throw std::runtime_error(std::string(name)+" bit mismatch");
}
void compressed_same(const dsv41::CompressedLayerState& a,const dsv41::CompressedLayerState& b){
 if(a.position()!=b.position()||a.global().rows()!=b.global().rows())throw std::runtime_error("compressed position mismatch");
 same(a.window(),b.window(),"compressed window");same(a.global().main_bytes(),b.global().main_bytes(),"main bytes");
 same(a.global().main_scales(),b.global().main_scales(),"main scales");
 same(a.global().index_bytes(),b.global().index_bytes(),"index bytes");
 same(a.global().index_scales(),b.global().index_scales(),"index scales");
 same(a.global().compressor().pending_kv(),b.global().compressor().pending_kv(),"pending kv");
 same(a.global().compressor().pending_scores(),b.global().compressor().pending_scores(),"pending scores");
}
}

int main(int argc,char** argv){try{
 if(argc!=5)throw std::runtime_error("usage: dsv41-text-quad-batch-probe checkpoint m1-summary metadata output-json");
 mx::set_default_device(mx::Device::gpu);dsv41::WeightCatalog catalog(argv[1],argv[2]);
 auto metadata=dsv41::EngramMetadata::load(argv[3]);
 const std::vector<std::uint32_t> ids{0,42,1000,17,9999,128,65535,7};
 if(setenv("DSV41_RUNTIME_LAYER_FINITE_CHECKS","0",1)!=0||setenv("DSV41_RUNTIME_PACKED_EXPERT_BANK","0",1)!=0||
    setenv("DSV41_RUNTIME_GROUP_SELECTED_EXPERTS","0",1)!=0)throw std::runtime_error("cannot select individual quad mode");
 dsv41::TextQuadReference individual(catalog,metadata);dsv41::TextQuadState individual_state(metadata);
 auto expected=individual.forward(ids,individual_state,0);
 if(setenv("DSV41_RUNTIME_PACKED_EXPERT_BANK","1",1)!=0)throw std::runtime_error("cannot select packed quad mode");
 dsv41::TextQuadReference packed(catalog,metadata);dsv41::TextQuadState packed_state(metadata);
 auto actual=packed.forward_packed_chunk(ids,packed_state,0);
 same(actual.hidden,expected.hidden,"hidden");same(actual.pre_mix,expected.pre_mix,"pre-mix");
 same(packed_state.triple.pair.first.rows(),individual_state.triple.pair.first.rows(),"layer-0 state");
 same(packed_state.triple.pair.second.rows(),individual_state.triple.pair.second.rows(),"layer-1 state");
 compressed_same(packed_state.triple.third,individual_state.triple.third);
 same(packed_state.fourth.window(),individual_state.fourth.window(),"layer-3 window");
 if(packed_state.fourth.position()!=ids.size())throw std::runtime_error("layer-3 position mismatch");
 J report={{"schema_version",1},{"status","probe_completed_requires_review"},{"tokens",ids.size()},
  {"layers",4},{"reuse_layer",3},{"hidden_pre_mix_state_bits","exact"},
  {"active_bytes",mx::get_active_memory()},{"cache_bytes",mx::get_cache_memory()},{"peak_bytes",mx::get_peak_memory()},
  {"scope","Eight-token layer-major layers 0..3 with per-token publications into first reuse layer; not performance/full-backbone qualification."}};
 std::ofstream out(argv[4]);if(!out)throw std::runtime_error("cannot create result JSON");out<<report.dump(2)<<'\n';
 std::cout<<"PASS: four-layer packed chunk with first reuse consumer matches token-serial bits; review required\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
