#include "dsv41/text_front.hpp"
#include <iostream>
#include <stdexcept>
namespace mx=mlx::core;
void same(const mx::array& a,const mx::array& b){
 if(a.shape()!=b.shape()||a.dtype()!=b.dtype())throw std::runtime_error("text front shape/dtype mismatch");
 auto dtype=a.dtype()==mx::bfloat16?mx::uint16:mx::uint32;
 auto ok=mx::all(mx::equal(mx::view(a,dtype),mx::view(b,dtype)));mx::eval(ok);
 if(!ok.item<bool>())throw std::runtime_error("text front bit mismatch");
}
int main(int argc,char** argv){try{
 if(argc!=3)throw std::runtime_error("usage: dsv41-text-front-test checkpoint m1-summary");
 mx::set_default_device(mx::Device::gpu);dsv41::WeightCatalog c(argv[1],argv[2]);
 std::cout<<"Loading embedding and resident layer 0"<<std::endl;
 dsv41::TextFrontReference front(c);
 // Fixed legal vocabulary IDs; this is a token-ID fixture, not a rendered chat prompt.
 const std::array<std::uint32_t,3> ids{0,42,1000};
 dsv41::SwaLayerState chunk,serial;
 auto batch=front.forward(ids,chunk,0);
 for(int i=0;i<3;++i){auto one=front.forward(std::span(ids).subspan(i,1),serial,i);
  same(one.hidden,mx::slice(batch.hidden,{i,0,0},{i+1,4,5120}));
  same(one.pre_mix,mx::slice(batch.pre_mix,{i,0},{i+1,4}));}
 same(chunk.rows(),serial.rows());
 auto saved=chunk.rows();
 for(std::uint32_t invalid:{129264u,129280u}){
  bool failed=false;try{front.forward(std::span(&invalid,1),chunk,3);}catch(const std::exception&){failed=true;}
  if(!failed||chunk.position()!=3)throw std::runtime_error("invalid token changed state");same(saved,chunk.rows());
 }
 auto fork=chunk;const std::array<std::uint32_t,1> next{42};
 auto resumed=front.forward(next,fork,3);auto repeated=front.forward(next,serial,3);
 same(resumed.hidden,repeated.hidden);same(resumed.pre_mix,repeated.pre_mix);same(fork.rows(),serial.rows());
 if(chunk.position()!=3||fork.position()!=4)throw std::runtime_error("text front fork position mismatch");same(saved,chunk.rows());
 chunk.reset();auto reset=front.forward(ids,chunk,0);same(reset.hidden,batch.hidden);same(reset.pre_mix,batch.pre_mix);
 std::cout<<"PASS: real token IDs -> embedding -> layer 0; chunk/token bits, continuation, fork, reset, invalid/image token rejection. Oracle/full model NOT qualified."<<std::endl;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
