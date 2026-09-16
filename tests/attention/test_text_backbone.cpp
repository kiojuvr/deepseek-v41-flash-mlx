#include "dsv41/text_backbone.hpp"
#include "dsv41/execution_policy.hpp"
#include "dsv41/generation_loop.hpp"
#include <bit>
#include <cstdlib>
#include <iostream>
#include "dsv41/checkpoint_atlas.hpp"
#include <fstream>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <tuple>
namespace mx=mlx::core;
void same(const mx::array& a,const mx::array& b,const char* what){
 if(a.shape()!=b.shape()||a.dtype()!=b.dtype())throw std::runtime_error(std::string(what)+": shape/dtype mismatch");
 auto dtype=a.dtype()==mx::bfloat16?mx::uint16:mx::uint32;
 auto ok=mx::all(mx::equal(mx::view(a,dtype),mx::view(b,dtype)));mx::eval(ok);
 if(!ok.item<bool>())throw std::runtime_error(std::string(what)+": bit mismatch");
}
void bytes(const mx::array& a,const mx::array& b,const char* what){
 if(a.shape()!=b.shape()||a.dtype()!=b.dtype())throw std::runtime_error(std::string(what)+": shape/dtype mismatch");
 auto ok=mx::all(mx::equal(a,b));mx::eval(ok);if(!ok.item<bool>())throw std::runtime_error(std::string(what)+": byte mismatch");
}
void producer_same(const dsv41::CompressedLayerState& a,const dsv41::CompressedLayerState& b,const char* what){
 if(a.position()!=b.position()||a.global().rows()!=b.global().rows())throw std::runtime_error(std::string(what)+": position mismatch");
 same(a.window(),b.window(),"producer window");
 bytes(a.global().main_bytes(),b.global().main_bytes(),"producer main bytes");
 bytes(a.global().main_scales(),b.global().main_scales(),"producer main scales");
 bytes(a.global().index_bytes(),b.global().index_bytes(),"producer index bytes");
 bytes(a.global().index_scales(),b.global().index_scales(),"producer index scales");
 same(a.global().compressor().pending_kv(),b.global().compressor().pending_kv(),"producer pending kv");
 same(a.global().compressor().pending_scores(),b.global().compressor().pending_scores(),"producer pending scores");
}
void state_same(const dsv41::TextBackboneState& a,const dsv41::TextBackboneState& b){
 if(a.encoder.hash.position()!=b.encoder.hash.position())throw std::runtime_error("hash position mismatch");
 for(int i=0;i<2;++i){if(a.encoder.swa[i].position()!=b.encoder.swa[i].position())throw std::runtime_error("swa position mismatch");same(a.encoder.swa[i].rows(),b.encoder.swa[i].rows(),"swa rows");}
 for(int i=0;i<3;++i)producer_same(a.encoder.producer[i],b.encoder.producer[i],"encoder producer");
 for(int i=0;i<15;++i){if(a.encoder.reuse[i].position()!=b.encoder.reuse[i].position())throw std::runtime_error("encoder reuse position mismatch");same(a.encoder.reuse[i].window(),b.encoder.reuse[i].window(),"encoder reuse window");}
 producer_same(a.decoder.producer,b.decoder.producer,"decoder producer");
 for(int i=0;i<19;++i){if(a.decoder.reuse[i].position()!=b.decoder.reuse[i].position())throw std::runtime_error("decoder reuse position mismatch");same(a.decoder.reuse[i].window(),b.decoder.reuse[i].window(),"decoder reuse window");}
}
void ties_same(const std::vector<dsv41::RouteTieRecord>& a,const std::vector<dsv41::RouteTieRecord>& b,const char* what){
 if(a.size()!=b.size())throw std::runtime_error(std::string(what)+": count mismatch");
 for(std::size_t i=0;i<a.size();++i){
  if(a[i].layer!=b[i].layer||a[i].token!=b[i].token||a[i].sixth_id!=b[i].sixth_id||
     a[i].seventh_id!=b[i].seventh_id||a[i].sixth_score!=b[i].sixth_score||
     a[i].seventh_score!=b[i].seventh_score||a[i].tied!=b[i].tied)
   throw std::runtime_error(std::string(what)+": record mismatch");
 }
}
std::vector<std::uint32_t> tokens(const char* path){
 std::ifstream file(path);if(!file)throw std::runtime_error("cannot open token file");
 std::vector<std::uint32_t> result;std::uint64_t token;
 while(file>>token){if(token>=129280||token==129264)throw std::runtime_error("invalid text token");result.push_back(std::uint32_t(token));}
 if(!file.eof()||result.empty()||result.size()>262144)throw std::runtime_error("invalid token file");
 return result;
}
int main(int argc,char** argv){try{
 if(argc!=4&&argc!=5)throw std::runtime_error("usage: dsv41-text-backbone-test checkpoint m1-summary metadata [tokens]");
 mx::set_default_device(mx::Device::gpu);dsv41::WeightCatalog c(argv[1],argv[2]);
 auto metadata=dsv41::EngramMetadata::load(argv[3]);
 auto proof=dsv41::read_json_file("artifacts/engram/fixture-provenance.json");
 std::ifstream file(argv[3],std::ios::binary);std::string raw{std::istreambuf_iterator<char>(file),{}};
 if(dsv41::sha256_text(raw)!=proof.at("fixture_sha256").at("metadata.json").get<std::string>())throw std::runtime_error("Engram metadata identity mismatch");
 std::cout<<"Loading full backbone layers 0..39 on-demand experts and Engram 1/14 mmap backing"<<std::endl;
 dsv41::TextBackboneReference model(c,metadata);
 if(std::getenv("DSV41_CHECK_LAYER_MAJOR_BACKBONE")){
  const bool compact_check=std::getenv("DSV41_CHECK_COMPACT_LAYER_MAJOR_BACKBONE")!=nullptr;
  if(dsv41::runtime_packed_expert_bank_enabled())throw std::runtime_error("comparison must start in individual mode");
  if(setenv("DSV41_RUNTIME_PACKED_EXPERT_BANK","1",1)!=0||
     setenv("DSV41_RUNTIME_COMPACT_EXPERT_BANK",compact_check?"1":"0",1)!=0)
   throw std::runtime_error("cannot select packed model");
  dsv41::TextBackboneReference packed(c,metadata);
  dsv41::reset_packed_expert_bank_construction_count();
  dsv41::TextBackboneState expected_state(metadata),actual_state(metadata);
  auto full_state_same=[&](const auto& a,const auto& b){
   state_same(a,b);
   auto ah=a.encoder.hash,bh=b.encoder.hash;const std::array<std::uint32_t,4> suffix{42,17,1000,7};
   if(ah.append(suffix,{},ah.position())!=bh.append(suffix,{},bh.position()))throw std::runtime_error("hash history mismatch");
   auto publication_same=[&](const auto& x,const auto& y,int consumer){
    if(bool(x.publication())!=bool(y.publication()))throw std::runtime_error("publication presence mismatch");
    if(!x.publication())return;
    const auto& p=*x.publication();const auto& q=*y.publication();const auto pos=x.position()-1;
    if(p.source_layer()!=q.source_layer()||p.index_source_layer()!=q.index_source_layer()||
       p.indices(consumer,pos,pos?128:1)!=q.indices(consumer,pos,pos?128:1)||p.candidates()!=q.candidates())
     throw std::runtime_error("publication routing mismatch");
    bytes(p.cache().main_bytes(),q.cache().main_bytes(),"publication main");
    bytes(p.cache().main_scales(),q.cache().main_scales(),"publication main scales");
    bytes(p.cache().index_bytes(),q.cache().index_bytes(),"publication index");
    bytes(p.cache().index_scales(),q.cache().index_scales(),"publication index scales");
   };
   for(int i=0;i<3;++i)publication_same(a.encoder.producer[i],b.encoder.producer[i],3+6*i);
   publication_same(a.decoder.producer,b.decoder.producer,39);
  };
  std::vector<std::uint32_t> input(128);for(int i=0;i<128;++i)input[i]=std::uint32_t((i*7919)%129263);
  for(int chunk_index=0;chunk_index<2;++chunk_index){
   const auto start=std::uint64_t(chunk_index*128);
   dsv41::reset_route_tie_records();auto expected=model.forward(input,expected_state,start);
   auto expected_ties=dsv41::route_tie_records();dsv41::reset_route_tie_records();
   auto actual=packed.forward_packed_chunk(input,actual_state,start);auto actual_ties=dsv41::route_tie_records();
   same(actual.hidden,expected.hidden,"layer-major hidden");same(actual.pre_mix,expected.pre_mix,"layer-major pre-mix");
   same(packed.logits(actual),model.logits(expected),"layer-major logits");full_state_same(actual_state,expected_state);
   auto order=[](const auto& x,const auto& y){return std::tie(x.token,x.layer)<std::tie(y.token,y.layer);};
   std::sort(actual_ties.begin(),actual_ties.end(),order);std::sort(expected_ties.begin(),expected_ties.end(),order);
   ties_same(actual_ties,expected_ties,"layer-major route ties");
   if(actual_state.decoder.producer.publication()->index_source_layer()!=36)throw std::runtime_error("decoder republishing did not reach layer 36");
   std::cout<<"PASS: 40-layer chunk "<<chunk_index<<" hidden/pre-mix/logits/state/publication/hash/route ties exact"<<std::endl;
  }
  auto saved=actual_state;
  for(std::uint32_t invalid:{129264u,129280u}){bool rejected=false;
   try{packed.forward_packed_chunk(std::span(&invalid,1),actual_state,256);}catch(const std::exception&){rejected=true;}
   if(!rejected)throw std::runtime_error("invalid token accepted");full_state_same(saved,actual_state);
  }
  std::cout<<"PASS: layer-major backbone 2x128 tokens and invalid-token atomicity; active_bytes="<<mx::get_active_memory()
   <<" cache_bytes="<<mx::get_cache_memory()<<" peak_bytes="<<mx::get_peak_memory()
   <<" bank_constructions="<<dsv41::packed_expert_bank_construction_count()
   <<" loaded_experts="<<dsv41::packed_expert_bank_loaded_expert_count()
   <<"; performance/32K unqualified"<<std::endl;
  return 0;
 }
 const auto ids=argc==5?tokens(argv[4]):std::vector<std::uint32_t>{0,42,1000};
 dsv41::TextBackboneState chunk(metadata),serial(metadata);
 dsv41::reset_route_tie_count();dsv41::reset_route_tie_records();
 std::vector<mx::array> batch_hidden,batch_pre;
 dsv41::run_prefill_chunks(ids.size(),128,[&](std::size_t offset,std::size_t count){
  auto part=model.forward(std::span(ids).subspan(offset,count),chunk,offset);
  batch_hidden.push_back(part.hidden);batch_pre.push_back(part.pre_mix);
 });
 dsv41::BlockResult batch{mx::concatenate(batch_hidden,0),mx::concatenate(batch_pre,0)};
 const auto chunk_ties=dsv41::route_tie_records();
 dsv41::reset_route_tie_count();dsv41::reset_route_tie_records();
 dsv41::BlockResult serial_last{mx::zeros({1,4,5120},mx::bfloat16),mx::zeros({1,4},mx::float32)};
 for(std::size_t i=0;i<ids.size();++i){auto one=model.forward(std::span(ids).subspan(i,1),serial,i);
  const int row=int(i);
  same(one.hidden,mx::slice(batch.hidden,{row,0,0},{row+1,4,5120}),"backbone hidden chunk bits");
  same(one.pre_mix,mx::slice(batch.pre_mix,{row,0},{row+1,4}),"backbone pre-mix chunk bits");
  serial_last=one;}
 const auto serial_ties=dsv41::route_tie_records();ties_same(chunk_ties,serial_ties,"backbone route ties chunk/token");
 state_same(chunk,serial);
 const int last=int(ids.size()-1);
 dsv41::BlockResult last_slice{mx::slice(batch.hidden,{last,0,0},{last+1,4,5120}),mx::slice(batch.pre_mix,{last,0},{last+1,4})};
 auto logits_chunk=model.logits(last_slice);auto logits_serial=model.logits(serial_last);
 same(logits_chunk,logits_serial,"backbone logits chunk bits");
 if(std::getenv("DSV41_CHECK_PACKED_EXPERT_BACKBONE_PARITY")||
    std::getenv("DSV41_CHECK_COMPACT_EXPERT_BACKBONE_PARITY")){
  const bool compact_check=std::getenv("DSV41_CHECK_COMPACT_EXPERT_BACKBONE_PARITY")!=nullptr;
  if(dsv41::runtime_packed_expert_bank_enabled())
   throw std::runtime_error("packed expert backbone parity must start from the individual path");
  dsv41::TextBackboneState individual_state(metadata);
  dsv41::reset_route_tie_count();dsv41::reset_route_tie_records();
  auto individual_result=model.forward(std::span(ids).subspan(0,1),individual_state,0);
  auto individual_logits=model.logits(individual_result);mx::eval(individual_logits);
  const auto individual_ties=dsv41::route_tie_records();
  if(setenv("DSV41_RUNTIME_PACKED_EXPERT_BANK","1",1)!=0||
     setenv("DSV41_RUNTIME_COMPACT_EXPERT_BANK",compact_check?"1":"0",1)!=0)
   throw std::runtime_error("cannot enable packed expert bank");
  {
   dsv41::TextBackboneReference packed_model(c,metadata);
   dsv41::TextBackboneState packed_state(metadata);
   dsv41::reset_route_tie_count();dsv41::reset_route_tie_records();
   auto packed_result=packed_model.forward(std::span(ids).subspan(0,1),packed_state,0);
   auto packed_logits=packed_model.logits(packed_result);
   same(packed_result.hidden,individual_result.hidden,"packed expert backbone hidden bits");
   same(packed_result.pre_mix,individual_result.pre_mix,"packed expert backbone pre-mix bits");
   same(packed_logits,individual_logits,"packed expert backbone logits bits");
   state_same(packed_state,individual_state);
   ties_same(dsv41::route_tie_records(),individual_ties,"packed expert backbone route ties");
  }
  if(setenv("DSV41_RUNTIME_PACKED_EXPERT_BANK","0",1)!=0||
     setenv("DSV41_RUNTIME_COMPACT_EXPERT_BANK","0",1)!=0)
   throw std::runtime_error("cannot restore packed expert bank policy");
  std::cout<<"PASS: one-token full-backbone individual/"<<(compact_check?"compact":"packed")
           <<" expert hidden, pre-mix, state, logits and route ties are bit-exact"<<std::endl;
 }
 if(std::getenv("DSV41_CHECK_LAYER_FINITE_POLICY_PARITY")){
  const bool original=dsv41::runtime_layer_finite_checks_enabled();
  if(setenv("DSV41_RUNTIME_LAYER_FINITE_CHECKS",original?"0":"1",1)!=0)
   throw std::runtime_error("cannot switch layer finite policy");
  dsv41::TextBackboneState alternate(metadata);
  dsv41::reset_route_tie_count();dsv41::reset_route_tie_records();
  std::vector<mx::array> alternate_hidden,alternate_pre;
  dsv41::run_prefill_chunks(ids.size(),128,[&](std::size_t offset,std::size_t count){
   auto part=model.forward(std::span(ids).subspan(offset,count),alternate,offset);
   alternate_hidden.push_back(part.hidden);alternate_pre.push_back(part.pre_mix);
  });
  dsv41::BlockResult alternate_result{mx::concatenate(alternate_hidden,0),mx::concatenate(alternate_pre,0)};
  same(alternate_result.hidden,batch.hidden,"layer finite policy hidden bits");
  same(alternate_result.pre_mix,batch.pre_mix,"layer finite policy pre-mix bits");
  state_same(alternate,chunk);
  dsv41::BlockResult alternate_last{
   mx::slice(alternate_result.hidden,{last,0,0},{last+1,4,5120}),
   mx::slice(alternate_result.pre_mix,{last,0},{last+1,4})};
  same(model.logits(alternate_last),logits_chunk,"layer finite policy logits bits");
  ties_same(dsv41::route_tie_records(),chunk_ties,"layer finite policy route ties");
  if(setenv("DSV41_RUNTIME_LAYER_FINITE_CHECKS",original?"1":"0",1)!=0)
   throw std::runtime_error("cannot restore layer finite policy");
  std::cout<<"PASS: layer finite check on/off hidden, pre-mix, state, logits and route ties are bit-exact"<<std::endl;
 }
 auto finite=mx::all(mx::isfinite(logits_chunk));mx::eval(finite);if(!finite.item<bool>())throw std::runtime_error("nonfinite backbone logits");
 auto saved=chunk;
 for(std::uint32_t invalid:{129264u,129280u}){
  bool failed=false;try{model.forward(std::span(&invalid,1),chunk,ids.size());}catch(const std::exception&){failed=true;}
  if(!failed||chunk.encoder.hash.position()!=ids.size())throw std::runtime_error("invalid token changed backbone state");state_same(saved,chunk);
 }
 auto fork=chunk;const std::array<std::uint32_t,1> next{42};
 auto resumed=model.forward(next,fork,ids.size());auto repeated=model.forward(next,serial,ids.size());
 same(resumed.hidden,repeated.hidden,"backbone continuation hidden");same(resumed.pre_mix,repeated.pre_mix,"backbone continuation pre-mix");
 state_same(fork,serial);
 if(chunk.encoder.hash.position()!=ids.size()||fork.encoder.hash.position()!=ids.size()+1)throw std::runtime_error("backbone fork position mismatch");state_same(saved,chunk);
 chunk.reset();dsv41::reset_route_tie_count();dsv41::reset_route_tie_records();std::vector<mx::array> reset_hidden,reset_pre;
 dsv41::run_prefill_chunks(ids.size(),128,[&](std::size_t offset,std::size_t count){
  auto part=model.forward(std::span(ids).subspan(offset,count),chunk,offset);
  reset_hidden.push_back(part.hidden);reset_pre.push_back(part.pre_mix);
 });
 dsv41::BlockResult reset{mx::concatenate(reset_hidden,0),mx::concatenate(reset_pre,0)};
 const auto reset_ties=dsv41::route_tie_records();ties_same(chunk_ties,reset_ties,"backbone route ties reset");
 same(reset.hidden,batch.hidden,"backbone reset hidden");same(reset.pre_mix,batch.pre_mix,"backbone reset pre-mix");state_same(saved,chunk);
 for(const auto& tie:chunk_ties)std::cout<<"route_tie: layer="<<tie.layer<<" token="<<tie.token
  <<" sixth_id="<<tie.sixth_id<<" seventh_id="<<tie.seventh_id
  <<" sixth_score_bits="<<std::bit_cast<std::uint32_t>(tie.sixth_score)
  <<" seventh_score_bits="<<std::bit_cast<std::uint32_t>(tie.seventh_score)<<std::endl;
 std::cout<<"PASS: "<<ids.size()<<" token IDs -> encoder 0..19 -> decoder 20..39 -> logits; <=128 chunk/token bits, state, continuation, fork, reset, invalid token rejection; route tie records exact across schedules = "<<chunk_ties.size()<<" (lowest-ID policy remains an external-oracle gap); external oracle/256K NOT qualified."<<std::endl;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
