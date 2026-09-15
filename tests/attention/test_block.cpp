#include "dsv41/block.hpp"
#include <iostream>
#include <stdexcept>
namespace mx=mlx::core;
void same(const mx::array& a,const mx::array& b){
 if(a.shape()!=b.shape()||a.dtype()!=b.dtype())throw std::runtime_error("shape/dtype mismatch");
 auto dtype=a.dtype()==mx::bfloat16?mx::uint16:mx::uint32;
 auto ok=mx::all(mx::equal(mx::view(a,dtype),mx::view(b,dtype)));mx::eval(ok);
 if(!ok.item<bool>())throw std::runtime_error("Block chunk/token bit mismatch");
}
int main(int argc,char** argv){try{
 if(argc!=3)throw std::runtime_error("usage: dsv41-block-test checkpoint m1-summary");
 mx::set_default_device(mx::Device::gpu);
 dsv41::WeightCatalog catalog(argv[1],argv[2]);
 std::cout<<"Loading resident layer 0 (384 routed experts)"<<std::endl;
 dsv41::BlockReference block(catalog);
 auto h=mx::astype(mx::reshape(mx::sin(mx::arange(2*4*5120,mx::float32)),{2,4,5120}),mx::bfloat16);
 auto pre=mx::broadcast_to(mx::array({1.0f,0.0f,0.0f,0.0f}),{2,4});
 dsv41::SwaLayerState chunk,serial;
 auto batch=block.forward(h,pre,chunk,0);
 for(int i=0;i<2;++i){auto one=block.forward(mx::slice(h,{i,0,0},{i+1,4,5120}),mx::slice(pre,{i,0},{i+1,4}),serial,i);
  same(one.hidden,mx::slice(batch.hidden,{i,0,0},{i+1,4,5120}));same(one.pre_mix,mx::slice(batch.pre_mix,{i,0},{i+1,4}));}
 same(chunk.rows(),serial.rows());
 if(chunk.position()!=2||serial.position()!=2)throw std::runtime_error("Block position mismatch");
 auto saved=chunk.rows();bool rejected=false;
 try{block.forward(h,pre,chunk,0);}catch(const std::exception&){rejected=true;}
 if(!rejected||chunk.position()!=2)throw std::runtime_error("Block rejected state mismatch");same(saved,chunk.rows());
 chunk.reset();auto reset=block.forward(h,pre,chunk,0);same(reset.hidden,batch.hidden);same(reset.pre_mix,batch.pre_mix);
 std::cout<<"PASS: layer 0 Block hidden/pre-mix/state chunk-token bits, reset and position rejection. Official oracle and full model NOT qualified."<<std::endl;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
