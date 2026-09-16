#include "dsv41/checkpoint_atlas.hpp"
#include "dsv41/text_octet.hpp"
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
void state_same(const dsv41::TextOctetState& a,const dsv41::TextOctetState& b){
 const auto& ap=a.quad.triple.pair;const auto& bp=b.quad.triple.pair;
 if(ap.hash.position()!=bp.hash.position()||ap.first.position()!=bp.first.position()||
    ap.second.position()!=bp.second.position())throw std::runtime_error("pair positions differ");
 same(ap.first.rows(),bp.first.rows(),"SWA 0");same(ap.second.rows(),bp.second.rows(),"SWA 1");
 auto ah=ap.hash,bh=bp.hash;const std::array<std::uint32_t,4> suffix{42,17,1000,7};
 if(ah.append(suffix,{},ah.position())!=bh.append(suffix,{},bh.position()))
  throw std::runtime_error("hash continuation differs");
 const auto& ac=a.quad.triple.third;const auto& bc=b.quad.triple.third;
 if(ac.position()!=bc.position()||ac.global().rows()!=bc.global().rows())throw std::runtime_error("producer positions differ");
 same(ac.window(),bc.window(),"producer window");
 same(ac.global().main_bytes(),bc.global().main_bytes(),"main bytes");
 same(ac.global().main_scales(),bc.global().main_scales(),"main scales");
 same(ac.global().index_bytes(),bc.global().index_bytes(),"index bytes");
 same(ac.global().index_scales(),bc.global().index_scales(),"index scales");
 same(ac.global().compressor().pending_kv(),bc.global().compressor().pending_kv(),"pending kv");
 same(ac.global().compressor().pending_scores(),bc.global().compressor().pending_scores(),"pending scores");
 if(bool(ac.publication())!=bool(bc.publication()))throw std::runtime_error("publication presence differs");
 if(ac.publication()){
  const auto& x=*ac.publication();const auto& y=*bc.publication();const auto pos=ac.position()-1;
  if(x.source_layer()!=y.source_layer()||x.index_source_layer()!=y.index_source_layer()||
     x.indices(7,pos,pos?128:1)!=y.indices(7,pos,pos?128:1)||x.candidates()!=y.candidates())
   throw std::runtime_error("publication differs");
  same(x.cache().main_bytes(),y.cache().main_bytes(),"publication main");
  same(x.cache().index_bytes(),y.cache().index_bytes(),"publication index");
 }
 if(a.quad.fourth.position()!=b.quad.fourth.position())throw std::runtime_error("layer 3 position differs");
 same(a.quad.fourth.window(),b.quad.fourth.window(),"layer 3 window");
 for(int i=0;i<4;++i){
  if(a.tail[i].position()!=b.tail[i].position())throw std::runtime_error("tail position differs");
  same(a.tail[i].window(),b.tail[i].window(),"tail window");
 }
}
}
int main(int argc,char** argv){try{
 if(argc!=5)throw std::runtime_error("usage: dsv41-text-octet-batch-probe checkpoint m1-summary metadata output-json");
 mx::set_default_device(mx::Device::gpu);dsv41::WeightCatalog catalog(argv[1],argv[2]);
 auto metadata=dsv41::EngramMetadata::load(argv[3]);
 int count=4;if(const auto* v=std::getenv("DSV41_OCTET_CHUNK_TOKENS"))count=std::stoi(v);
 if(count<1||count>128)throw std::runtime_error("chunk tokens must be 1..128");
 std::vector<std::uint32_t> ids(count);for(int i=0;i<count;++i)ids[i]=std::uint32_t((i*7919)%129263);
 if(setenv("DSV41_RUNTIME_LAYER_FINITE_CHECKS","0",1)!=0||setenv("DSV41_RUNTIME_PACKED_EXPERT_BANK","0",1)!=0||
    setenv("DSV41_RUNTIME_GROUP_SELECTED_EXPERTS","0",1)!=0)throw std::runtime_error("cannot select individual octet mode");
 dsv41::TextOctetReference individual(catalog,metadata);dsv41::TextOctetState individual_state(metadata);
 auto expected=individual.forward(ids,individual_state,0);
 if(setenv("DSV41_RUNTIME_PACKED_EXPERT_BANK","1",1)!=0)throw std::runtime_error("cannot select packed octet mode");
 dsv41::TextOctetReference packed(catalog,metadata);dsv41::TextOctetState packed_state(metadata);
 auto actual=packed.forward_packed_chunk(ids,packed_state,0);
 state_same(packed_state,individual_state);
 same(actual.hidden,expected.hidden,"hidden");same(actual.pre_mix,expected.pre_mix,"pre-mix");
 same(packed_state.quad.triple.pair.first.rows(),individual_state.quad.triple.pair.first.rows(),"layer-0 state");
 same(packed_state.quad.triple.pair.second.rows(),individual_state.quad.triple.pair.second.rows(),"layer-1 state");
 same(packed_state.quad.triple.third.window(),individual_state.quad.triple.third.window(),"layer-2 window");
 for(int i=0;i<4;++i){same(packed_state.tail[i].window(),individual_state.tail[i].window(),"reuse window");
  if(packed_state.tail[i].position()!=ids.size())throw std::runtime_error("reuse position mismatch");}
 auto expected_next=individual.forward(ids,individual_state,ids.size());
 auto actual_next=packed.forward_packed_chunk(ids,packed_state,ids.size());
 same(actual_next.hidden,expected_next.hidden,"continuation hidden");
 same(actual_next.pre_mix,expected_next.pre_mix,"continuation pre-mix");
 state_same(packed_state,individual_state);
 auto saved=packed_state;const std::array<std::uint32_t,1> invalid{129264};bool rejected=false;
 try{packed.forward_packed_chunk(invalid,packed_state,2*ids.size());}catch(const std::exception&){rejected=true;}
 if(!rejected)throw std::runtime_error("invalid token accepted");state_same(saved,packed_state);
 J report={{"schema_version",2},{"status","probe_completed_requires_review"},{"tokens",ids.size()},{"chunks",2},{"layers",8},
  {"full_state_continuation_invalid_token","exact"},
  {"reuse_layers","3..7"},{"hidden_pre_mix_attention_state_bits","exact"},
  {"active_bytes",mx::get_active_memory()},{"cache_bytes",mx::get_cache_memory()},{"peak_bytes",mx::get_peak_memory()},
  {"scope","Two chunks through layers 0..7: output, state, publication, hash continuation and invalid-token atomicity; not performance/full-backbone qualification."}};
 std::ofstream out(argv[4]);if(!out)throw std::runtime_error("cannot create result JSON");out<<report.dump(2)<<'\n';
 std::cout<<"PASS: eight-layer packed chunk through producer/reuse group matches token-serial bits; review required\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
