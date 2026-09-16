#include "dsv41/swa_attention.hpp"
#include "dsv41/swa_projection.hpp"
#include "dsv41/swa_layer.hpp"
#include "dsv41/moe.hpp"
#include "dsv41/compressor.hpp"
#include "dsv41/index_key.hpp"
#include "dsv41/kv_quant.hpp"
#include "dsv41/global_kv.hpp"
#include "dsv41/index_query.hpp"
#include "dsv41/compressed_layer.hpp"
#include "dsv41/execution_policy.hpp"
#include "dsv41/reused_layer.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
namespace mx=mlx::core;
void equal(const mx::array& a,const mx::array& b,const char* message,bool bits=true){
 if(a.shape()!=b.shape()||a.dtype()!=mx::bfloat16||b.dtype()!=mx::bfloat16)throw std::runtime_error(message);
 auto e=mx::all(bits?mx::equal(mx::view(a,mx::uint16),mx::view(b,mx::uint16)):mx::equal(a,b));mx::eval(e);if(!e.item<bool>())throw std::runtime_error(message);
}
void rms_close(const mx::array& candidate,const mx::array& reference,const char* message){
 if(candidate.shape()!=reference.shape()||candidate.dtype()!=reference.dtype())throw std::runtime_error(message);
 auto c=mx::astype(candidate,mx::float32),r=mx::astype(reference,mx::float32),d=mx::subtract(c,r);
 auto ratio=mx::sqrt(mx::divide(mx::sum(mx::multiply(d,d)),
  mx::maximum(mx::sum(mx::multiply(r,r)),mx::array(1e-30f))));
 auto finite=mx::all(mx::isfinite(c));mx::eval(ratio,finite);
 if(!finite.item<bool>()||ratio.item<float>()>=0.002f)
  throw std::runtime_error(std::string(message)+" relative_rms="+std::to_string(ratio.item<float>()));
}
int main(int argc,char** argv){try{
 mx::set_default_device(mx::Device::gpu);
 {
  auto top=dsv41::index_topk_reference(mx::arange(513,mx::float32),128);
  if(top.size()!=512||top.front()!=129||top.back()!=640)throw std::runtime_error("index top-k selection/position order mismatch");
  if(!dsv41::index_topk_reference(mx::zeros({0}),128).empty())throw std::runtime_error("empty index top-k failed");
  bool tie=false;try{dsv41::index_topk_reference(mx::zeros({513}),128);}catch(const std::exception&){tie=true;}
  if(!tie)throw std::runtime_error("index top-k boundary tie accepted");
 }
 for(auto format:{dsv41::KVQuantFormat::IndexE8M0,dsv41::KVQuantFormat::MainE4M3}){
  int k=format==dsv41::KVQuantFormat::IndexE8M0?128:512;
  std::array<float,8> midpoint{0.25f,0.75f,1.25f,1.75f,2.5f,3.5f,5.0f,6.0f};
  std::array<std::uint8_t,8> codes{0,2,2,4,4,6,6,7};
  std::vector<float> v(k);std::vector<std::uint8_t> expected(k/2);
  for(int j=0;j<k;++j){bool neg=(j%16)>=8;v[j]=(neg?-1:1)*midpoint[j%8];expected[j/2]|=(codes[j%8]|(neg?8:0))<<((j%2)*4);}
  auto input=mx::astype(mx::array(v.begin(),{1,k},mx::float32),mx::bfloat16);
  auto q=dsv41::kv_quant_reference(input,format);
  auto ok=mx::all(mx::equal(q.packed,mx::array(expected.begin(),{1,k/2},mx::uint8)));mx::eval(ok);if(!ok.item<bool>())throw std::runtime_error("FP4 midpoint/packing mismatch");
  ok=mx::all(mx::equal(q.scales,mx::array(format==dsv41::KVQuantFormat::IndexE8M0?127:56)));mx::eval(ok);if(!ok.item<bool>())throw std::runtime_error("FP4 unit scale mismatch");
  auto batch=dsv41::kv_quant_reference(mx::concatenate({input,input},0),format);
  equal(q.decoded,mx::slice(batch.decoded,{1,0},{2,k}),"FP4 chunk bits mismatch");
  auto zero=dsv41::kv_quant_reference(mx::zeros({1,k},mx::bfloat16),format);
  equal(zero.decoded,mx::zeros({1,k},mx::bfloat16),"FP4 zero restoration");
  ok=mx::all(mx::equal(zero.scales,mx::array(1)));mx::eval(ok);if(!ok.item<bool>())throw std::runtime_error("FP4 minimum scale mismatch");
  bool rejected=false;try{dsv41::kv_quant_reference(mx::full({1,k},NAN,mx::bfloat16),format);}catch(const std::exception&){rejected=true;}
  if(!rejected)throw std::runtime_error("FP4 nonfinite accepted");
 }
 {
  const std::array<std::uint64_t,3> positions{0,65536,262143};
  auto x=mx::astype(mx::reshape(mx::sin(mx::arange(3*128,mx::float32)),{3,128}),mx::bfloat16);
  auto full=dsv41::compressed_rope_reference(x,positions);
  for(int i=0;i<3;++i)equal(mx::slice(full,{i,0},{i+1,128}),dsv41::compressed_rope_reference(mx::slice(x,{i,0},{i+1,128}),std::span(positions).subspan(i,1)),"compressed RoPE chunk bits");
  equal(mx::slice(full,{0,0},{1,128}),mx::slice(x,{0,0},{1,128}),"compressed RoPE zero position",false);
  equal(mx::slice(full,{0,0},{3,64}),mx::slice(x,{0,0},{3,64}),"compressed RoPE touched non-rotary channels");
  bool bad=false;const std::array<std::uint64_t,1> invalid{1048576};try{dsv41::compressed_rope_reference(mx::slice(x,{0,0},{1,128}),invalid);}catch(const std::exception&){bad=true;}
  if(!bad)throw std::runtime_error("compressed RoPE position overflow accepted");
 }
 auto route_scores=mx::arange(1,385,mx::float32);
 std::vector<float> bias(384,0);for(int i=0;i<6;++i)bias[i]=1000;
 auto route=dsv41::select_routes_reference(route_scores,mx::array(bias.begin(),{384},mx::float32));
 for(int i=0;i<6;++i)if(route.ids[i]!=5-i)throw std::runtime_error("biased route selection mismatch");
 auto expected_route=mx::multiply(mx::divide(mx::arange(6,0,-1,mx::float32),mx::array(21.0f)),mx::array(1.5f));
 auto route_ok=mx::all(mx::equal(route.weights,expected_route));mx::eval(route_ok);if(!route_ok.item<bool>())throw std::runtime_error("bias contaminated route weights");
 bool tied=false;try{dsv41::select_routes_reference(mx::ones({384}),mx::zeros({384}));}catch(const std::exception&){tied=true;}
 if(!tied)throw std::runtime_error("ambiguous route boundary accepted");
 for(float gate:{-20.0f,0.0f,20.0f})for(float up:{-20.0f,20.0f}){
  float g=std::min(gate,10.0f),u=std::clamp(up,-10.0f,10.0f);
  float e=(g/(1+std::exp(-g))*u)*0.25f;
  equal(dsv41::expert_activation_reference(mx::full({1,2304},gate,mx::bfloat16),mx::full({1,2304},up,mx::bfloat16),mx::array(0.25f)),mx::full({1,2304},e,mx::bfloat16),"expert clamp/weight placement mismatch",false);
 }
 // Independent analytic check of the first complex pair (frequency exactly one).
 std::vector<float> values(512,0.0f);values[0]=3;values[448]=1;
 auto input=mx::astype(mx::array(values.begin(),{1,512},mx::float32),mx::bfloat16);
 for(int position:{0,1,127,128,262143,1048575})for(bool inverse:{false,true}){
  auto expected=values;expected[448]=std::cos(float(position));expected[449]=(inverse?-1:1)*std::sin(float(position));
  equal(dsv41::swa_rope_reference(input,position,inverse),
        mx::astype(mx::array(expected.begin(),{1,512},mx::float32),mx::bfloat16),"RoPE analytic mismatch",false);
 }
 auto varied=mx::astype(mx::reshape(mx::sin(mx::arange(2*64*512,mx::float32)),{2,64,512}),mx::bfloat16);
 auto rotated=dsv41::swa_rope_reference(varied,127);
 for(int i=0;i<2;++i)equal(mx::slice(rotated,{i,0,0},{i+1,64,512}),
  dsv41::swa_rope_reference(mx::slice(varied,{i,0,0},{i+1,64,512}),127+i),"RoPE chunk mismatch");
 bool invalid_position=false;try{dsv41::swa_rope_reference(varied,1048575);}catch(const std::exception&){invalid_position=true;}
 if(!invalid_position)throw std::runtime_error("RoPE accepted position overflow");
 for(int count:{1,63,64,65,127,128}){
  auto q=mx::zeros({64,512},mx::bfloat16),kv=mx::ones({count,512},mx::bfloat16),sink=mx::zeros({64},mx::float32);
  auto y=dsv41::swa_attention_reference(q,kv,sink);
  // Zero logits: count identical unit values and a zero-valued sink with equal mass.
  auto expected=mx::full({64,512},float(count)/float(count+1),mx::bfloat16);
  auto equal=mx::all(mx::equal(y,expected));mx::eval(equal);if(!equal.item<bool>())throw std::runtime_error("sink/block boundary mismatch");
 }
 bool rejected=false;try{dsv41::swa_attention_reference(mx::zeros({64,512},mx::bfloat16),mx::zeros({0,512},mx::bfloat16),mx::zeros({64}));}catch(const std::exception&){rejected=true;}
 if(!rejected)throw std::runtime_error("empty KV accepted");
 for(int live:{1,63,64,65,127,128}){
  auto mask=mx::greater_equal(mx::arange(128,mx::int32),mx::array(128-live));
  auto out=dsv41::swa_attention_masked_reference(mx::zeros({64,512},mx::bfloat16),mx::ones({128,512},mx::bfloat16),mx::zeros({64}),mask);
  equal(out,mx::full({64,512},float(live)/float(live+1),mx::bfloat16),"masked sink mismatch");
 }
 if(argc!=1&&argc!=3)throw std::runtime_error("usage: dsv41-swa-attention-test [checkpoint m1-summary]");
 if(argc==3){
  dsv41::WeightCatalog catalog(argv[1],argv[2]);
  {
   dsv41::CompressedLayerReference producer(catalog,2);dsv41::ReusedLayerReference consumer(catalog,3);
   dsv41::CompressedLayerState source;dsv41::ReusedLayerState target;
   auto inputs=mx::astype(mx::reshape(mx::sin(mx::arange(3*5120,mx::float32)),{3,5120}),mx::bfloat16);
   std::vector<dsv41::SharedAttentionReference> publications;std::vector<mx::array> outputs;
   for(int i=0;i<3;++i){auto x=mx::slice(inputs,{i,0},{i+1,5120});producer.forward(x,source,i);
    publications.push_back(*source.publication());auto snapshot=publications.back().cache().main_bytes();
    outputs.push_back(consumer.forward(x,target,publications.back(),i));
    auto intact=mx::all(mx::equal(snapshot,publications.back().cache().main_bytes()));mx::eval(intact);if(!intact.item<bool>())throw std::runtime_error("consumer changed producer bytes");}
   dsv41::ReusedLayerState chunk_target;auto chunk_publications=publications;
   auto chunk_output=consumer.forward_chunk(inputs,chunk_target,chunk_publications,0);
   if(dsv41::runtime_chunk_attention_enabled())
    rms_close(chunk_output,mx::concatenate(outputs,0),"consumer chunk output tolerance");
   else equal(chunk_output,mx::concatenate(outputs,0),"consumer chunk output bits");
   equal(chunk_target.window(),target.window(),"consumer chunk window bits");
   if(chunk_target.position()!=target.position())throw std::runtime_error("consumer chunk position mismatch");
   auto saved=target;target.reset();
   for(int i=0;i<3;++i)equal(consumer.forward(mx::slice(inputs,{i,0},{i+1,5120}),target,publications[i],i),outputs[i],"consumer snapshot replay bits");
   equal(target.window(),saved.window(),"consumer window replay bits");
   bool stale=false;auto x=mx::slice(inputs,{0,0},{1,5120});try{consumer.forward(x,target,publications[0],3);}catch(const std::exception&){stale=true;}
   if(!stale||target.position()!=3)throw std::runtime_error("consumer stale publication accepted");equal(target.window(),saved.window(),"consumer rejection state");
   producer.forward(x,source,3);auto fork=target;
   equal(consumer.forward(x,fork,*source.publication(),3),consumer.forward(x,target,*source.publication(),3),"consumer fork bits");
   if(saved.position()!=3||fork.position()!=4)throw std::runtime_error("consumer fork position");
   auto finite=mx::all(mx::isfinite(outputs.back()));mx::eval(finite);if(!finite.item<bool>())throw std::runtime_error("nonfinite consumer");
   std::cout<<"Layer 3 attention: real layer 2 snapshots, replay/reset/fork bits, stale rejection passed (not Block/oracle qualification)\n";
   for(int layer=4;layer<=7;++layer){
    dsv41::ReusedLayerReference reader(catalog,layer);dsv41::ReusedLayerState a,b;
    for(int i=0;i<3;++i){auto x=mx::slice(inputs,{i,0},{i+1,5120});
     equal(reader.forward(x,a,publications[i],i),reader.forward(x,b,publications[i],i),"consumer 4..7 repeat mismatch");}
    equal(a.window(),b.window(),"consumer 4..7 window mismatch");
   }
   bool invalid_owner=false;try{dsv41::ReusedLayerReference wrong(catalog,8);}catch(const std::exception&){invalid_owner=true;}
   if(!invalid_owner)throw std::runtime_error("layer 8 accepted as layer 2 consumer");
   std::cout<<"Layers 4..7 real consumer attention repeat/window bits and layer 8 rejection passed\n";
  }
  {
   dsv41::CompressedLayerReference attention(catalog,2);dsv41::CompressedLayerState chunk,serial;
   auto input=mx::astype(mx::reshape(mx::sin(mx::arange(5*5120,mx::float32)),{5,5120}),mx::bfloat16);
   std::vector<dsv41::SharedAttentionReference> production_publications;
   auto batch=attention.forward_chunk(input,chunk,0,&production_publications);
   if(!chunk.publication())throw std::runtime_error("layer 2 publication missing");
   auto published=*chunk.publication();
   for(int layer=3;layer<=7;++layer){
    if(dsv41::runtime_index_diagnostics_enabled()){
     if(published.indices(layer,4,128)!=std::vector<std::int32_t>{128,129})throw std::runtime_error("reuse consumer mismatch");
    }else{
     auto rows=published.device_indices(layer,4,128);mx::eval(rows);
     const auto* p=rows.data<std::int32_t>();
     if(rows.dtype()!=mx::int32||rows.shape()!=mx::Shape({2})||p[0]!=0||p[1]!=1)
      throw std::runtime_error("device reuse consumer mismatch");
    }
   }
   for(int layer:{2,8}){bool bad=false;try{
    if(dsv41::runtime_index_diagnostics_enabled())published.indices(layer,4,128);
    else published.device_indices(layer,4,128);
   }catch(const std::exception&){bad=true;}if(!bad)throw std::runtime_error("wrong reuse source accepted");}
   bool stale=false;try{
    if(dsv41::runtime_index_diagnostics_enabled())published.indices(3,5,128);
    else published.device_indices(3,5,128);
   }catch(const std::exception&){stale=true;}if(!stale)throw std::runtime_error("stale publication accepted");
   for(int i=0;i<5;++i)equal(attention.forward(mx::slice(input,{i,0},{i+1,5120}),serial,i),mx::slice(batch,{i,0},{i+1,5120}),"compressed attention chunk bits");
   auto same_state=[&](const dsv41::CompressedLayerState& a,const dsv41::CompressedLayerState& b){
    if(a.position()!=b.position()||a.global().rows()!=b.global().rows())throw std::runtime_error("compressed state position mismatch");
    equal(a.window(),b.window(),"compressed window bits");
    auto bytes=[](const mx::array& x,const mx::array& y){if(x.shape()!=y.shape())throw std::runtime_error("compressed state shape mismatch");auto ok=mx::all(mx::equal(x,y));mx::eval(ok);if(!ok.item<bool>())throw std::runtime_error("compressed global bytes mismatch");};
    bytes(a.global().main_bytes(),b.global().main_bytes());bytes(a.global().main_scales(),b.global().main_scales());
    bytes(a.global().index_bytes(),b.global().index_bytes());bytes(a.global().index_scales(),b.global().index_scales());
    bytes(mx::view(a.global().compressor().pending_kv(),mx::uint32),mx::view(b.global().compressor().pending_kv(),mx::uint32));
    bytes(mx::view(a.global().compressor().pending_scores(),mx::uint32),mx::view(b.global().compressor().pending_scores(),mx::uint32));
   };
   same_state(chunk,serial);auto saved=chunk,fork=chunk;auto token=mx::slice(input,{0,0},{1,5120});
   equal(attention.forward(token,fork,5),attention.forward(token,serial,5),"compressed continuation bits");same_state(fork,serial);same_state(chunk,saved);
   if(published.cache().position()!=5||published.cache().rows()!=2||fork.publication()->cache().rows()!=3)throw std::runtime_error("published snapshot mutated");
   bool rejected=false;try{attention.forward(token,chunk,0);}catch(const std::exception&){rejected=true;}
   if(!rejected)throw std::runtime_error("compressed invalid position accepted");same_state(chunk,saved);
   rejected=false;try{attention.forward(mx::full({1,5120},NAN,mx::bfloat16),chunk,5);}catch(const std::exception&){rejected=true;}
   if(!rejected)throw std::runtime_error("compressed NaN accepted");same_state(chunk,saved);
   chunk.reset();equal(attention.forward(input,chunk,0),batch,"compressed reset bits");same_state(chunk,saved);
   std::cout<<"Layer 2 attention: window/global output and state chunk bits, continuation/fork/reset/rejection passed (not oracle qualification)\n";
  }
  {
   dsv41::GlobalKVProducerReference producer(catalog,2);dsv41::GlobalKVState chunk,serial;
   auto input=mx::astype(mx::reshape(mx::sin(mx::arange(5*5120,mx::float32)),{5,5120}),mx::bfloat16);
   auto bytes=[](const mx::array& a,const mx::array& b){
    if(a.shape()!=b.shape()||a.dtype()!=b.dtype())throw std::runtime_error("global KV shape mismatch");
    auto ok=mx::all(mx::equal(a,b));mx::eval(ok);if(!ok.item<bool>())throw std::runtime_error("global KV bytes mismatch");};
   auto same_cache=[&](const dsv41::GlobalKVState& a,const dsv41::GlobalKVState& b){
    if(a.position()!=b.position()||a.rows()!=b.rows())throw std::runtime_error("global KV position mismatch");
    bytes(a.main_bytes(),b.main_bytes());bytes(a.main_scales(),b.main_scales());bytes(a.index_bytes(),b.index_bytes());bytes(a.index_scales(),b.index_scales());
    bytes(mx::view(a.compressor().pending_kv(),mx::uint32),mx::view(b.compressor().pending_kv(),mx::uint32));
    bytes(mx::view(a.compressor().pending_scores(),mx::uint32),mx::view(b.compressor().pending_scores(),mx::uint32));};
   producer.append(input,chunk,0);
   dsv41::IndexQueryReference query(catalog,2);
   auto x=mx::slice(input,{4,0},{5,5120});
   auto qr=mx::astype(mx::reshape(mx::sin(mx::arange(1280,mx::float32)),{1,1280}),mx::bfloat16);
   auto selected=query.forward(x,qr,chunk,4,128);
   if(selected.rows!=std::vector<std::int32_t>{128,129})throw std::runtime_error("real index reachable positions mismatch");
   auto restored=dsv41::restore_index_reference(chunk);
   auto finite=mx::all(mx::isfinite(restored));mx::eval(finite);if(!finite.item<bool>())throw std::runtime_error("nonfinite restored index cache");
   bool stale=false;try{query.forward(x,qr,chunk,3,128);}catch(const std::exception&){stale=true;}
   if(!stale)throw std::runtime_error("index accepted future cache");
   for(int i=0;i<5;++i){producer.append(mx::slice(input,{i,0},{i+1,5120}),serial,i);if(serial.rows()!=std::size_t((i+1)/2))throw std::runtime_error("premature global KV publication");}
   same_cache(chunk,serial);auto saved=chunk,fork=chunk;auto token=mx::slice(input,{0,0},{1,5120});
   producer.append(token,fork,5);producer.append(token,serial,5);same_cache(fork,serial);same_cache(chunk,saved);
   if(fork.rows()!=3||chunk.rows()!=2)throw std::runtime_error("global KV fork failed");
   bool rejected=false;try{producer.append(token,chunk,0);}catch(const std::exception&){rejected=true;}
   if(!rejected)throw std::runtime_error("global KV invalid position accepted");same_cache(chunk,saved);
   rejected=false;try{producer.append(mx::full({1,5120},NAN,mx::bfloat16),chunk,5);}catch(const std::exception&){rejected=true;}
   if(!rejected)throw std::runtime_error("global KV NaN accepted");same_cache(chunk,saved);
   chunk.reset();producer.append(input,chunk,0);same_cache(chunk,saved);
   std::cout<<"Layer 2 packed global KV publication: byte-exact chunk/token, partial groups, continuation, fork/reset/rejection passed\n";
  }
  {
   dsv41::CompressorReference compressor(catalog,2);dsv41::CompressorState chunk,serial;
   auto h=mx::astype(mx::reshape(mx::sin(mx::arange(3*5120,mx::float32)),{3,5120}),mx::bfloat16);
   auto batch=compressor.forward(h,chunk,0);
   dsv41::IndexKeyReference index(catalog,2);
   auto key=index.before_quantization(batch.values,batch.positions);
   auto repeated_key=index.before_quantization(batch.values,batch.positions);
   equal(key,repeated_key,"index key repeat bits");
   auto key_finite=mx::all(mx::isfinite(key));mx::eval(key_finite);if(!key_finite.item<bool>())throw std::runtime_error("nonfinite index key");
   if(batch.positions!=std::vector<std::uint64_t>{0}||chunk.position()!=3||chunk.pending_kv().shape()!=mx::Shape({1,512}))throw std::runtime_error("compressor group boundary mismatch");
   for(int i=0;i<3;++i){auto one=compressor.forward(mx::slice(h,{i,0},{i+1,5120}),serial,i);
    if(i==1)equal(one.values,batch.values,"compressor chunk bits mismatch");
    else if(one.values.shape()!=mx::Shape({0,512})||!one.positions.empty())throw std::runtime_error("incomplete compressor emitted latent");}
   auto state_bits=[&](const mx::array& a,const mx::array& b){auto ok=mx::all(mx::equal(mx::view(a,mx::uint32),mx::view(b,mx::uint32)));mx::eval(ok);if(!ok.item<bool>())throw std::runtime_error("compressor pending bits mismatch");};
   state_bits(chunk.pending_kv(),serial.pending_kv());state_bits(chunk.pending_scores(),serial.pending_scores());
   auto fork=chunk;auto token=mx::slice(h,{0,0},{1,5120});auto next=compressor.forward(token,fork,3);
   equal(next.values,compressor.forward(token,serial,3).values,"compressor continuation mismatch");
   if(next.positions!=std::vector<std::uint64_t>{2}||chunk.position()!=3||fork.position()!=4||fork.pending_kv().shape(0)!=0)throw std::runtime_error("compressor fork mismatch");
   auto saved=chunk;bool rejected=false;try{compressor.forward(token,chunk,0);}catch(const std::exception&){rejected=true;}
   if(!rejected||chunk.position()!=3)throw std::runtime_error("compressor invalid position accepted");
   rejected=false;try{compressor.forward(mx::full({1,5120},NAN,mx::bfloat16),chunk,3);}catch(const std::exception&){rejected=true;}
   if(!rejected)throw std::runtime_error("compressor NaN accepted");state_bits(saved.pending_kv(),chunk.pending_kv());state_bits(saved.pending_scores(),chunk.pending_scores());
   chunk.reset();equal(compressor.forward(h,chunk,0).values,batch.values,"compressor reset mismatch");
   chunk.reset();auto zero=compressor.forward(mx::zeros({2,5120},mx::bfloat16),chunk,0);
   equal(zero.values,mx::zeros({1,512},mx::bfloat16),"compressor zero analytic mismatch",false);
   std::cout<<"Layer 2 compressor: groups/positions, chunk bits, pending state, fork/reset/rejection and zero analytic passed (not oracle qualification)\n";
  }
  bool unsupported=false;try{dsv41::SwaProjectionReference bad(catalog,2);}catch(const std::exception&){unsupported=true;}
  if(!unsupported)throw std::runtime_error("compressed layer accepted as pure SWA");
  {
   dsv41::SwaProjectionReference layer1(catalog,1);
   auto input1=mx::astype(mx::reshape(mx::sin(mx::arange(2*5120,mx::float32)),{2,5120}),mx::bfloat16);
   auto qkv1=layer1.forward(input1,127);
   for(int i=0;i<2;++i){auto one=layer1.forward(mx::slice(input1,{i,0},{i+1,5120}),127+i);
    equal(one.query,mx::slice(qkv1.query,{i,0,0},{i+1,64,512}),"layer 1 query chunk mismatch");
    equal(one.kv,mx::slice(qkv1.kv,{i,0},{i+1,512}),"layer 1 KV chunk mismatch");}
   dsv41::GateReference gate1(catalog,1);auto token1=mx::slice(input1,{0,0},{1,5120});auto route1=gate1.forward(token1);
   dsv41::ExpertReference expert1(catalog,route1.ids[0],1);
   auto out1=expert1.forward(token1,mx::take(route1.weights,mx::array(0)));
   equal(out1,expert1.forward(token1,mx::take(route1.weights,mx::array(0))),"layer 1 expert repeat mismatch");
   auto valid1=mx::all(mx::isfinite(out1));mx::eval(valid1);if(!valid1.item<bool>())throw std::runtime_error("nonfinite layer 1 expert");
   std::cout<<"Layer 1 Q/KV chunk bits and selected expert repeat/finite passed; compressed-layer rejection passed\n";
  }
  dsv41::GateReference gate(catalog);
  auto expert_input=mx::astype(mx::reshape(mx::sin(mx::arange(5120,mx::float32)),{1,5120}),mx::bfloat16);
  auto selected=gate.forward(expert_input);auto repeated=gate.forward(expert_input);
  if(selected.ids!=repeated.ids)throw std::runtime_error("router repeat mismatch");
  for(int expert:{-1,selected.ids[0]}){
   dsv41::ExpertReference ffn(catalog,expert);
   auto weight=expert==-1?mx::array(1.0f):mx::take(selected.weights,mx::array(0));
   auto output=ffn.forward(expert_input,weight);
   equal(output,ffn.forward(expert_input,weight),"expert repeat mismatch");
   equal(ffn.forward(expert_input,mx::array(0.0f)),mx::zeros({1,5120},mx::bfloat16),"zero route expert output",false);
   auto finite_output=mx::all(mx::isfinite(output));mx::eval(finite_output);if(!finite_output.item<bool>())throw std::runtime_error("nonfinite expert output");
  }
  std::cout<<"MoE selection/activation analytic and real shared/selected-expert finite/repeat checks passed (not full MoE)\n";
  auto owner=std::make_unique<dsv41::SwaProjectionReference>(catalog);
  auto h=mx::astype(mx::reshape(mx::sin(mx::arange(2*5120,mx::float32)),{2,5120}),mx::bfloat16);
  auto batch=owner->forward(h,127);
  for(int i=0;i<2;++i){auto one=owner->forward(mx::slice(h,{i,0},{i+1,5120}),127+i);
   equal(one.query,mx::slice(batch.query,{i,0,0},{i+1,64,512}),"Q projection chunk mismatch");
   equal(one.kv,mx::slice(batch.kv,{i,0},{i+1,512}),"KV projection chunk mismatch");}
  auto deferred=owner->forward(h,127);owner.reset();
  equal(deferred.query,batch.query,"Q owner lifetime mismatch");equal(deferred.kv,batch.kv,"KV owner lifetime mismatch");
  auto finite=mx::all(mx::isfinite(batch.query));mx::eval(finite);if(!finite.item<bool>())throw std::runtime_error("nonfinite query");
  finite=mx::all(mx::isfinite(batch.kv));mx::eval(finite);if(!finite.item<bool>())throw std::runtime_error("nonfinite KV");
  std::cout<<"Real layer 0 Q/KV projection: finite, chunk/token exact, owner lifetime passed (not oracle qualification)\n";
  dsv41::SwaLayerReference layer(catalog);dsv41::SwaLayerState chunk,serial;
  auto y=layer.forward(h,chunk,0);
  for(int i=0;i<2;++i)equal(layer.forward(mx::slice(h,{i,0},{i+1,5120}),serial,i),mx::slice(y,{i,0},{i+1,5120}),"layer chunk mismatch");
  equal(chunk.rows(),serial.rows(),"layer state chunk mismatch");
  auto fork=chunk;auto saved=chunk.rows();auto token=mx::slice(h,{0,0},{1,5120});
  auto continuation=layer.forward(token,fork,2);(void)continuation;
  if(chunk.position()!=2||fork.position()!=3)throw std::runtime_error("layer fork position mismatch");
  equal(chunk.rows(),saved,"layer fork mutated source");
  bool wrong=false;try{layer.forward(token,chunk,0);}catch(const std::exception&){wrong=true;}
  if(!wrong||chunk.position()!=2)throw std::runtime_error("layer invalid position accepted");
  wrong=false;try{layer.forward(mx::full({1,5120},NAN,mx::bfloat16),chunk,2);}catch(const std::exception&){wrong=true;}
  if(!wrong||chunk.position()!=2)throw std::runtime_error("layer nonfinite input accepted");
  equal(chunk.rows(),saved,"layer rejection mutated state");chunk.reset();
  equal(layer.forward(h,chunk,0),y,"layer reset mismatch");
  dsv41::SwaLayerState wrapped;
  auto prefix=mx::broadcast_to(token,{128,5120});layer.forward(prefix,wrapped,0);
  if(wrapped.position()!=128||wrapped.rows().shape()!=mx::Shape({128,512}))throw std::runtime_error("layer window fill mismatch");
  auto before_wrap=wrapped.rows();auto wrap_copy=wrapped;
  auto after=layer.forward(token,wrapped,128);
  equal(after,layer.forward(token,wrap_copy,128),"layer wrapped fork mismatch");
  if(wrapped.position()!=129||wrapped.rows().shape()!=mx::Shape({128,512}))throw std::runtime_error("layer window overflow");
  equal(mx::slice(before_wrap,{1,0},{128,512}),mx::slice(wrapped.rows(),{0,0},{127,512}),"layer oldest row eviction mismatch");
  dsv41::SwaProjectionReference projection(catalog);
  equal(projection.forward(token,128).kv,mx::slice(wrapped.rows(),{127,0},{128,512}),"layer newest row mismatch");
  std::cout<<"Real layer 0 attention: chunk/token and state bitwise exact; continuation/fork/reset/rejection passed (not oracle qualification)\n";
 }
 std::cout<<"SWA RoPE analytic/chunk/position and sink/64-row boundary checks passed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
