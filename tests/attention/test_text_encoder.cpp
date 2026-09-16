#include "dsv41/text_encoder.hpp"
#include "dsv41/text_octet.hpp"
#include <iostream>
#include "dsv41/checkpoint_atlas.hpp"
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <cstdlib>
#include <tuple>
namespace mx=mlx::core;
void same(const mx::array& a,const mx::array& b,const char* what){
 if(a.shape()!=b.shape()||a.dtype()!=b.dtype())throw std::runtime_error(std::string(what)+": shape/dtype mismatch");
 auto dtype=a.dtype()==mx::bfloat16?mx::uint16:mx::uint32;
 auto ok=mx::all(mx::equal(mx::view(a,dtype),mx::view(b,dtype)));mx::eval(ok);
 if(!ok.item<bool>())throw std::runtime_error(std::string(what)+": bit mismatch");
}
void state_same(const dsv41::TextEncoderState& a,const dsv41::TextEncoderState& b){
 if(a.hash.position()!=b.hash.position())throw std::runtime_error("encoder hash position mismatch");
 auto ah=a.hash,bh=b.hash;const std::array<std::uint32_t,4> suffix{42,17,1000,7};
 if(ah.append(suffix,{},ah.position())!=bh.append(suffix,{},bh.position()))throw std::runtime_error("encoder hash history mismatch");
 for(int i=0;i<2;++i){if(a.swa[i].position()!=b.swa[i].position())throw std::runtime_error("encoder swa position mismatch");same(a.swa[i].rows(),b.swa[i].rows(),"encoder swa window");}
 for(int i=0;i<3;++i){
  if(a.producer[i].position()!=b.producer[i].position()||a.producer[i].global().rows()!=b.producer[i].global().rows())throw std::runtime_error("encoder producer position mismatch");
  same(a.producer[i].window(),b.producer[i].window(),"encoder producer window");
  auto bytes=[](const mx::array& x,const mx::array& y,const char* what){if(x.shape()!=y.shape())throw std::runtime_error(std::string(what)+": shape mismatch");auto ok=mx::all(mx::equal(x,y));mx::eval(ok);if(!ok.item<bool>())throw std::runtime_error(std::string(what)+": byte mismatch");};
  bytes(a.producer[i].global().main_bytes(),b.producer[i].global().main_bytes(),"producer main bytes");
  bytes(a.producer[i].global().main_scales(),b.producer[i].global().main_scales(),"producer main scales");
  bytes(a.producer[i].global().index_bytes(),b.producer[i].global().index_bytes(),"producer index bytes");
  bytes(a.producer[i].global().index_scales(),b.producer[i].global().index_scales(),"producer index scales");
  same(a.producer[i].global().compressor().pending_kv(),b.producer[i].global().compressor().pending_kv(),"producer pending kv");
  same(a.producer[i].global().compressor().pending_scores(),b.producer[i].global().compressor().pending_scores(),"producer pending scores");
  const auto& ap=a.producer[i].publication();const auto& bp=b.producer[i].publication();
  if(bool(ap)!=bool(bp))throw std::runtime_error("publication presence mismatch");
  if(ap){
   const int consumer=3+i*6;const auto pos=a.producer[i].position()-1;
   if(ap->source_layer()!=bp->source_layer()||ap->index_source_layer()!=bp->index_source_layer()||
      ap->indices(consumer,pos,pos?128:1)!=bp->indices(consumer,pos,pos?128:1)||ap->candidates()!=bp->candidates())
    throw std::runtime_error("encoder publication mismatch");
   bytes(ap->cache().main_bytes(),bp->cache().main_bytes(),"publication main");
   bytes(ap->cache().main_scales(),bp->cache().main_scales(),"publication main scales");
   bytes(ap->cache().index_bytes(),bp->cache().index_bytes(),"publication index");
   bytes(ap->cache().index_scales(),bp->cache().index_scales(),"publication index scales");
  }
 }
 for(int i=0;i<15;++i){if(a.reuse[i].position()!=b.reuse[i].position())throw std::runtime_error("encoder reuse position mismatch");same(a.reuse[i].window(),b.reuse[i].window(),"encoder reuse window");}
}
int main(int argc,char** argv){try{
 if(argc!=4)throw std::runtime_error("usage: dsv41-text-encoder-test checkpoint m1-summary metadata");
 mx::set_default_device(mx::Device::gpu);dsv41::WeightCatalog c(argv[1],argv[2]);
 auto metadata=dsv41::EngramMetadata::load(argv[3]);
 auto proof=dsv41::read_json_file("artifacts/engram/fixture-provenance.json");
 std::ifstream file(argv[3],std::ios::binary);std::string raw{std::istreambuf_iterator<char>(file),{}};
 if(dsv41::sha256_text(raw)!=proof.at("fixture_sha256").at("metadata.json").get<std::string>())throw std::runtime_error("Engram metadata identity mismatch");
 std::cout<<"Loading embedding, encoder layers 0..19 on-demand experts and Engram 1/14 mmap backing"<<std::endl;
 dsv41::TextEncoderReference encoder(c,metadata);
 if(std::getenv("DSV41_CHECK_ENCODER_PACKED_CHUNK")){
  if(setenv("DSV41_RUNTIME_PACKED_EXPERT_BANK","1",1)!=0)throw std::runtime_error("cannot select packed encoder");
  dsv41::TextEncoderReference packed(c,metadata);
  dsv41::TextEncoderState expected_state(metadata),actual_state(metadata);
  std::vector<std::uint32_t> input(128);for(int i=0;i<128;++i)input[i]=std::uint32_t((i*7919)%129263);
  for(int chunk_index=0;chunk_index<2;++chunk_index){
   const auto start=std::uint64_t(chunk_index*128);
   dsv41::reset_route_tie_records();auto expected=encoder.forward(input,expected_state,start);
   auto expected_ties=dsv41::route_tie_records();dsv41::reset_route_tie_records();
   auto actual=packed.forward_packed_chunk(input,actual_state,start);auto actual_ties=dsv41::route_tie_records();
   same(actual.hidden,expected.hidden,"packed encoder hidden");same(actual.pre_mix,expected.pre_mix,"packed encoder pre-mix");
   state_same(actual_state,expected_state);
   auto order=[](const auto& x,const auto& y){return std::tie(x.token,x.layer)<std::tie(y.token,y.layer);};
   std::sort(expected_ties.begin(),expected_ties.end(),order);std::sort(actual_ties.begin(),actual_ties.end(),order);
   if(expected_ties.size()!=actual_ties.size())throw std::runtime_error("packed encoder tie count mismatch");
   for(std::size_t i=0;i<expected_ties.size();++i){const auto& x=expected_ties[i];const auto& y=actual_ties[i];
    if(x.token!=y.token||x.layer!=y.layer||x.sixth_id!=y.sixth_id||x.seventh_id!=y.seventh_id||
       x.sixth_score!=y.sixth_score||x.seventh_score!=y.seventh_score||x.tied!=y.tied)
     throw std::runtime_error("packed encoder tie mismatch");
   }
   std::cout<<"PASS: encoder packed chunk "<<chunk_index<<" output/state/publication/hash/ties exact"<<std::endl;
  }
  auto saved=actual_state;
  for(std::uint32_t bad:{129264u,129280u}){bool rejected=false;
   try{packed.forward_packed_chunk(std::span(&bad,1),actual_state,256);}catch(const std::exception&){rejected=true;}
   if(!rejected)throw std::runtime_error("packed encoder invalid token accepted");state_same(saved,actual_state);
  }
  std::cout<<"PASS: 20-layer packed encoder 2x128 tokens and invalid-token atomicity; active_bytes="<<mx::get_active_memory()
   <<" cache_bytes="<<mx::get_cache_memory()<<" peak_bytes="<<mx::get_peak_memory()<<"; review required"<<std::endl;
  return 0;
 }
 const std::array<std::uint32_t,3> ids{0,42,1000};
 dsv41::TextEncoderState chunk(metadata),serial(metadata);
 auto batch=encoder.forward(ids,chunk,0);
 for(int i=0;i<3;++i){auto one=encoder.forward(std::span(ids).subspan(i,1),serial,i);
  same(one.hidden,mx::slice(batch.hidden,{i,0,0},{i+1,4,5120}),"encoder hidden chunk bits");
  same(one.pre_mix,mx::slice(batch.pre_mix,{i,0},{i+1,4}),"encoder pre-mix chunk bits");}
 state_same(chunk,serial);
 auto saved=chunk;
 for(std::uint32_t invalid:{129264u,129280u}){
  bool failed=false;try{encoder.forward(std::span(&invalid,1),chunk,3);}catch(const std::exception&){failed=true;}
  if(!failed||chunk.hash.position()!=3)throw std::runtime_error("invalid token changed encoder state");state_same(saved,chunk);
 }
 auto fork=chunk;const std::array<std::uint32_t,1> next{42};
 auto resumed=encoder.forward(next,fork,3);auto repeated=encoder.forward(next,serial,3);
 same(resumed.hidden,repeated.hidden,"encoder continuation hidden");same(resumed.pre_mix,repeated.pre_mix,"encoder continuation pre-mix");
 state_same(fork,serial);
 if(chunk.hash.position()!=3||fork.hash.position()!=4)throw std::runtime_error("encoder fork position mismatch");state_same(saved,chunk);
 chunk.reset();auto reset=encoder.forward(ids,chunk,0);same(reset.hidden,batch.hidden,"encoder reset hidden");same(reset.pre_mix,batch.pre_mix,"encoder reset pre-mix");state_same(saved,chunk);
 std::cout<<"PASS: real token IDs -> encoder layers 0..19 with producers 2/8/14, reuse 3-7/9-13/15-19, Engram 1/14; chunk/token bits, continuation, fork, reset, invalid token rejection; top-6 route ties broken by lowest ID = "<<dsv41::route_tie_count()<<" (unqualified oracle gap); oracle/full model NOT qualified."<<std::endl;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
