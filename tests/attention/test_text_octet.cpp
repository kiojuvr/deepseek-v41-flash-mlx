#include "dsv41/text_octet.hpp"
#include <iostream>
#include "dsv41/checkpoint_atlas.hpp"
#include <fstream>
#include <stdexcept>
namespace mx=mlx::core;
void same(const mx::array& a,const mx::array& b){
 if(a.shape()!=b.shape()||a.dtype()!=b.dtype())throw std::runtime_error("text front shape/dtype mismatch");
 auto dtype=a.dtype()==mx::bfloat16?mx::uint16:mx::uint32;
 auto ok=mx::all(mx::equal(mx::view(a,dtype),mx::view(b,dtype)));mx::eval(ok);
 if(!ok.item<bool>())throw std::runtime_error("text front bit mismatch");
}
void state_same(const dsv41::TextOctetState& a,const dsv41::TextOctetState& b){
 if(a.quad.triple.pair.first.position()!=b.quad.triple.pair.first.position()||a.quad.triple.pair.second.position()!=b.quad.triple.pair.second.position()||a.quad.triple.pair.hash.position()!=b.quad.triple.pair.hash.position())throw std::runtime_error("pair position mismatch");
 same(a.quad.triple.pair.first.rows(),b.quad.triple.pair.first.rows());same(a.quad.triple.pair.second.rows(),b.quad.triple.pair.second.rows());
 if(a.quad.triple.third.position()!=b.quad.triple.third.position())throw std::runtime_error("third layer position mismatch");
 same(a.quad.triple.third.window(),b.quad.triple.third.window());
 auto bytes=[](const mx::array& x,const mx::array& y){
  if(x.shape()!=y.shape())throw std::runtime_error("third state shape mismatch");
  auto ok=mx::all(mx::equal(x,y));mx::eval(ok);if(!ok.item<bool>())throw std::runtime_error("third state byte mismatch");};
 bytes(a.quad.triple.third.global().main_bytes(),b.quad.triple.third.global().main_bytes());bytes(a.quad.triple.third.global().main_scales(),b.quad.triple.third.global().main_scales());
 bytes(a.quad.triple.third.global().index_bytes(),b.quad.triple.third.global().index_bytes());bytes(a.quad.triple.third.global().index_scales(),b.quad.triple.third.global().index_scales());
 same(a.quad.triple.third.global().compressor().pending_kv(),b.quad.triple.third.global().compressor().pending_kv());
 same(a.quad.triple.third.global().compressor().pending_scores(),b.quad.triple.third.global().compressor().pending_scores());
 if(a.quad.fourth.position()!=b.quad.fourth.position())throw std::runtime_error("fourth position mismatch");
 same(a.quad.fourth.window(),b.quad.fourth.window());
 for(int i=0;i<4;++i){if(a.tail[i].position()!=b.tail[i].position())throw std::runtime_error("tail position mismatch");same(a.tail[i].window(),b.tail[i].window());}
 // Compare hidden hash history by probing copies with a full n-gram continuation.
 auto ah=a.quad.triple.pair.hash,bh=b.quad.triple.pair.hash;const std::array<std::uint32_t,4> probe{42,1000,42,0};
 if(ah.append(probe,{},ah.position())!=bh.append(probe,{},bh.position()))throw std::runtime_error("pair hash history mismatch");
}
int main(int argc,char** argv){try{
 if(argc!=4)throw std::runtime_error("usage: dsv41-text-octet-test checkpoint m1-summary metadata");
 mx::set_default_device(mx::Device::gpu);dsv41::WeightCatalog c(argv[1],argv[2]);
 auto metadata=dsv41::EngramMetadata::load(argv[3]);
 auto proof=dsv41::read_json_file("artifacts/engram/fixture-provenance.json");
 std::ifstream file(argv[3],std::ios::binary);std::string raw{std::istreambuf_iterator<char>(file),{}};
 if(dsv41::sha256_text(raw)!=proof.at("fixture_sha256").at("metadata.json").get<std::string>())throw std::runtime_error("Engram metadata identity mismatch");
 std::cout<<"Loading embedding, layers 0..7 on-demand experts and Engram 1 mmap backing"<<std::endl;
 dsv41::TextOctetReference front(c,metadata);
 // Fixed legal vocabulary IDs; this is a token-ID fixture, not a rendered chat prompt.
 const std::array<std::uint32_t,3> ids{0,42,1000};
 dsv41::TextOctetState chunk(metadata),serial(metadata);
 auto batch=front.forward(ids,chunk,0);
 if(chunk.quad.triple.third.global().rows()!=1)throw std::runtime_error("third layer incomplete-group publication mismatch");
 for(int i=0;i<3;++i){auto one=front.forward(std::span(ids).subspan(i,1),serial,i);
  same(one.hidden,mx::slice(batch.hidden,{i,0,0},{i+1,4,5120}));
  same(one.pre_mix,mx::slice(batch.pre_mix,{i,0},{i+1,4}));}
 state_same(chunk,serial);
 auto saved=chunk;
 for(std::uint32_t invalid:{129264u,129280u}){
  bool failed=false;try{front.forward(std::span(&invalid,1),chunk,3);}catch(const std::exception&){failed=true;}
  if(!failed||chunk.quad.triple.pair.first.position()!=3)throw std::runtime_error("invalid token changed state");state_same(saved,chunk);
 }
 auto fork=chunk;const std::array<std::uint32_t,1> next{42};
 auto resumed=front.forward(next,fork,3);auto repeated=front.forward(next,serial,3);
 if(fork.quad.triple.third.global().rows()!=2)throw std::runtime_error("third layer completed-group publication mismatch");
 same(resumed.hidden,repeated.hidden);same(resumed.pre_mix,repeated.pre_mix);state_same(fork,serial);
 if(chunk.quad.triple.pair.first.position()!=3||fork.quad.triple.pair.first.position()!=4)throw std::runtime_error("text front fork position mismatch");state_same(saved,chunk);
 auto inconsistent=chunk;inconsistent.tail[3].reset();auto inconsistent_saved=inconsistent;
 bool rejected=false;try{front.forward(next,inconsistent,3);}catch(const std::exception&){rejected=true;}
 if(!rejected)throw std::runtime_error("inconsistent pair positions accepted");state_same(inconsistent,inconsistent_saved);
 chunk.reset();auto reset=front.forward(ids,chunk,0);same(reset.hidden,batch.hidden);same(reset.pre_mix,batch.pre_mix);state_same(saved,chunk);
 std::cout<<"PASS: real token IDs -> layer 0 -> Engram 1 -> layer 1 -> layer 2 -> layer 3 -> layers 4..7; chunk/token bits, continuation, fork, reset, invalid/image token and inconsistent-state rejection; hash continuation agreement. Oracle/full model NOT qualified."<<std::endl;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
