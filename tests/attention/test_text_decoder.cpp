#include "dsv41/text_decoder.hpp"
#include <iostream>
#include "dsv41/checkpoint_atlas.hpp"
#include <stdexcept>
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
void state_same(const dsv41::TextDecoderState& a,const dsv41::TextDecoderState& b){
 if(a.producer.position()!=b.producer.position()||a.producer.global().rows()!=b.producer.global().rows())throw std::runtime_error("decoder producer position mismatch");
 same(a.producer.window(),b.producer.window(),"decoder producer window");
 bytes(a.producer.global().main_bytes(),b.producer.global().main_bytes(),"producer main bytes");
 bytes(a.producer.global().main_scales(),b.producer.global().main_scales(),"producer main scales");
 bytes(a.producer.global().index_bytes(),b.producer.global().index_bytes(),"producer index bytes");
 bytes(a.producer.global().index_scales(),b.producer.global().index_scales(),"producer index scales");
 same(a.producer.global().compressor().pending_kv(),b.producer.global().compressor().pending_kv(),"producer pending kv");
 same(a.producer.global().compressor().pending_scores(),b.producer.global().compressor().pending_scores(),"producer pending scores");
 for(int i=0;i<19;++i){if(a.reuse[i].position()!=b.reuse[i].position())throw std::runtime_error("decoder reuse position mismatch");same(a.reuse[i].window(),b.reuse[i].window(),"decoder reuse window");}
}
int main(int argc,char** argv){try{
 if(argc!=3)throw std::runtime_error("usage: dsv41-text-decoder-test checkpoint m1-summary");
 mx::set_default_device(mx::Device::gpu);dsv41::WeightCatalog c(argv[1],argv[2]);
 std::cout<<"Loading decoder layers 20..39 on-demand experts"<<std::endl;
 dsv41::TextDecoderReference decoder(c);
 // Synthetic hidden/pre-mix fixture; this isolates the decoder from the encoder.
 auto hidden=mx::astype(mx::reshape(mx::sin(mx::arange(3*4*5120,mx::float32)),{3,4,5120}),mx::bfloat16);
 auto pre=mx::astype(mx::reshape(mx::cos(mx::arange(3*4,mx::float32)),{3,4}),mx::float32);
 dsv41::TextDecoderState chunk,serial;
 auto batch=decoder.forward(hidden,pre,chunk,0);
 dsv41::BlockResult serial_result{mx::zeros({1,4,5120},mx::bfloat16),mx::zeros({1,4},mx::float32)};
 for(int i=0;i<3;++i){auto one=decoder.forward(mx::slice(hidden,{i,0,0},{i+1,4,5120}),mx::slice(pre,{i,0},{i+1,4}),serial,i);
  same(one.hidden,mx::slice(batch.hidden,{i,0,0},{i+1,4,5120}),"decoder hidden chunk bits");
  same(one.pre_mix,mx::slice(batch.pre_mix,{i,0},{i+1,4}),"decoder pre-mix chunk bits");
  serial_result=one;}
 state_same(chunk,serial);
 auto saved=chunk;
 dsv41::BlockResult last_slice{mx::slice(batch.hidden,{2,0,0},{3,4,5120}),mx::slice(batch.pre_mix,{2,0},{3,4})};
 auto logits_batch=decoder.logits(last_slice);auto logits_serial=decoder.logits(serial_result);
 same(logits_batch,logits_serial,"decoder logits chunk bits");
 auto finite=mx::all(mx::isfinite(logits_batch));mx::eval(finite);if(!finite.item<bool>())throw std::runtime_error("nonfinite decoder logits");
 auto fork=chunk;
 auto resumed=decoder.forward(mx::slice(hidden,{0,0,0},{1,4,5120}),mx::slice(pre,{0,0},{1,4}),fork,3);
 auto repeated=decoder.forward(mx::slice(hidden,{0,0,0},{1,4,5120}),mx::slice(pre,{0,0},{1,4}),serial,3);
 same(resumed.hidden,repeated.hidden,"decoder continuation hidden");same(resumed.pre_mix,repeated.pre_mix,"decoder continuation pre-mix");
 state_same(fork,serial);
 if(chunk.producer.position()!=3||fork.producer.position()!=4)throw std::runtime_error("decoder fork position mismatch");state_same(saved,chunk);
 chunk.reset();auto reset=decoder.forward(hidden,pre,chunk,0);same(reset.hidden,batch.hidden,"decoder reset hidden");same(reset.pre_mix,batch.pre_mix,"decoder reset pre-mix");state_same(saved,chunk);
 bool rejected=false;try{decoder.forward(hidden,pre,chunk,1);}catch(const std::exception&){rejected=true;}
 if(!rejected)throw std::runtime_error("decoder invalid position accepted");state_same(saved,chunk);
 std::cout<<"PASS: decoder layers 20..39 with ratio-1 producer 20, index sources 24/28/32/36, final collapse/norm/head; chunk/token bits, continuation, fork, reset, invalid rejection; top-6 route ties broken by lowest ID = "<<dsv41::route_tie_count()<<" (unqualified oracle gap); oracle/full model NOT qualified."<<std::endl;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
