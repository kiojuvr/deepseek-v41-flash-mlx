#include "dsv41/text_encoder.hpp"
#include "dsv41/text_octet.hpp"
#include <iostream>
#include "dsv41/checkpoint_atlas.hpp"
#include <fstream>
#include <stdexcept>
namespace mx=mlx::core;
void same(const mx::array& a,const mx::array& b,const char* what){
 if(a.shape()!=b.shape()||a.dtype()!=b.dtype())throw std::runtime_error(std::string(what)+": shape/dtype mismatch");
 auto dtype=a.dtype()==mx::bfloat16?mx::uint16:mx::uint32;
 auto ok=mx::all(mx::equal(mx::view(a,dtype),mx::view(b,dtype)));mx::eval(ok);
 if(!ok.item<bool>())throw std::runtime_error(std::string(what)+": bit mismatch");
}
void state_same(const dsv41::TextEncoderState& a,const dsv41::TextEncoderState& b){
 if(a.hash.position()!=b.hash.position())throw std::runtime_error("encoder hash position mismatch");
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
