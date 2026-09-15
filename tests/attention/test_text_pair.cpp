#include "dsv41/text_pair.hpp"
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
void state_same(const dsv41::TextPairState& a,const dsv41::TextPairState& b){
 if(a.first.position()!=b.first.position()||a.second.position()!=b.second.position()||a.hash.position()!=b.hash.position())throw std::runtime_error("pair position mismatch");
 same(a.first.rows(),b.first.rows());same(a.second.rows(),b.second.rows());
 // Compare hidden hash history by probing copies with a full n-gram continuation.
 auto ah=a.hash,bh=b.hash;const std::array<std::uint32_t,4> probe{42,1000,42,0};
 if(ah.append(probe,{},ah.position())!=bh.append(probe,{},bh.position()))throw std::runtime_error("pair hash history mismatch");
}
int main(int argc,char** argv){try{
 if(argc!=4)throw std::runtime_error("usage: dsv41-text-pair-test checkpoint m1-summary metadata");
 mx::set_default_device(mx::Device::gpu);dsv41::WeightCatalog c(argv[1],argv[2]);
 auto metadata=dsv41::EngramMetadata::load(argv[3]);
 auto proof=dsv41::read_json_file("artifacts/engram/fixture-provenance.json");
 std::ifstream file(argv[3],std::ios::binary);std::string raw{std::istreambuf_iterator<char>(file),{}};
 if(dsv41::sha256_text(raw)!=proof.at("fixture_sha256").at("metadata.json").get<std::string>())throw std::runtime_error("Engram metadata identity mismatch");
 std::cout<<"Loading embedding, layers 0/1 on-demand experts and Engram 1 mmap backing"<<std::endl;
 dsv41::TextPairReference front(c,metadata);
 // Fixed legal vocabulary IDs; this is a token-ID fixture, not a rendered chat prompt.
 const std::array<std::uint32_t,3> ids{0,42,1000};
 dsv41::TextPairState chunk(metadata),serial(metadata);
 auto batch=front.forward(ids,chunk,0);
 for(int i=0;i<3;++i){auto one=front.forward(std::span(ids).subspan(i,1),serial,i);
  same(one.hidden,mx::slice(batch.hidden,{i,0,0},{i+1,4,5120}));
  same(one.pre_mix,mx::slice(batch.pre_mix,{i,0},{i+1,4}));}
 state_same(chunk,serial);
 auto saved=chunk;
 for(std::uint32_t invalid:{129264u,129280u}){
  bool failed=false;try{front.forward(std::span(&invalid,1),chunk,3);}catch(const std::exception&){failed=true;}
  if(!failed||chunk.first.position()!=3)throw std::runtime_error("invalid token changed state");state_same(saved,chunk);
 }
 auto fork=chunk;const std::array<std::uint32_t,1> next{42};
 auto resumed=front.forward(next,fork,3);auto repeated=front.forward(next,serial,3);
 same(resumed.hidden,repeated.hidden);same(resumed.pre_mix,repeated.pre_mix);state_same(fork,serial);
 if(chunk.first.position()!=3||fork.first.position()!=4)throw std::runtime_error("text front fork position mismatch");state_same(saved,chunk);
 auto inconsistent=chunk;inconsistent.second.reset();auto inconsistent_saved=inconsistent;
 bool rejected=false;try{front.forward(next,inconsistent,3);}catch(const std::exception&){rejected=true;}
 if(!rejected)throw std::runtime_error("inconsistent pair positions accepted");state_same(inconsistent,inconsistent_saved);
 chunk.reset();auto reset=front.forward(ids,chunk,0);same(reset.hidden,batch.hidden);same(reset.pre_mix,batch.pre_mix);state_same(saved,chunk);
 std::cout<<"PASS: real token IDs -> layer 0 -> Engram 1 -> layer 1; chunk/token bits, continuation, fork, reset, invalid/image token and inconsistent-state rejection; hash continuation agreement. Oracle/full model NOT qualified."<<std::endl;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
