#include "dsv41/compressor.hpp"
#include "dsv41/execution_policy.hpp"
#include "dsv41/global_kv.hpp"
#include "dsv41/index_key.hpp"
#include "dsv41/index_query.hpp"
#include "dsv41/shared_attention.hpp"
#include <iostream>
#include <stdexcept>

namespace mx=mlx::core;
namespace {
void same(const mx::array& a,const mx::array& b,const char* message){
 if(a.shape()!=b.shape()||a.dtype()!=b.dtype())throw std::runtime_error(message);
 auto ok=mx::all(mx::equal(a,b));mx::eval(ok);
 if(!ok.item<bool>())throw std::runtime_error(message);
}
void same_compressor(const dsv41::CompressorState& a,const dsv41::CompressorState& b){
 if(a.position()!=b.position())throw std::runtime_error("compressor position mismatch");
 same(mx::view(a.pending_kv(),mx::uint32),mx::view(b.pending_kv(),mx::uint32),"compressor pending KV mismatch");
 same(mx::view(a.pending_scores(),mx::uint32),mx::view(b.pending_scores(),mx::uint32),"compressor pending score mismatch");
}
void same_global(const dsv41::GlobalKVState& a,const dsv41::GlobalKVState& b){
 if(a.position()!=b.position()||a.rows()!=b.rows())throw std::runtime_error("global position mismatch");
 same(a.main_bytes(),b.main_bytes(),"global main bytes mismatch");
 same(a.main_scales(),b.main_scales(),"global main scales mismatch");
 same(a.index_bytes(),b.index_bytes(),"global index bytes mismatch");
 same(a.index_scales(),b.index_scales(),"global index scales mismatch");
 same_compressor(a.compressor(),b.compressor());
}
void same_device_selection(const dsv41::IndexSelection& batch,const dsv41::IndexSelection& serial,
 int offset,const std::vector<std::uint8_t>* expected_candidates,const char* message){
 std::vector<std::int32_t> relative;relative.reserve(serial.rows.size());
 for(auto row:serial.rows)relative.push_back(row-offset);
 if(relative.empty()){
  if(batch.device_rows.shape()!=mx::Shape({0})||batch.device_rows.dtype()!=mx::int32)
   throw std::runtime_error(message);
 }else same(batch.device_rows,mx::array(relative.begin(),{int(relative.size())},mx::int32),message);
 if(!expected_candidates||expected_candidates->empty()){
  if(batch.device_candidates.shape()!=mx::Shape({0})||batch.device_candidates.dtype()!=mx::uint8)
   throw std::runtime_error(message);
 }else same(batch.device_candidates,
  mx::array(expected_candidates->begin(),{int(expected_candidates->size())},mx::uint8),message);
}
}

int main(int argc,char** argv){try{
 if(argc!=3)throw std::runtime_error("usage: dsv41-producer-chunk-test checkpoint summary");
 mx::set_default_device(mx::Device::gpu);
 dsv41::WeightCatalog catalog(argv[1],argv[2]);
 auto h=mx::astype(mx::reshape(mx::sin(mx::arange(5*5120,mx::float32)),{5,5120}),mx::bfloat16);
 {
  dsv41::CompressorReference compressor(catalog,2);dsv41::CompressorState batch_state,serial_state;
  auto batch=compressor.forward(h,batch_state,0);std::vector<mx::array> rows;
  std::vector<std::uint64_t> positions;
  for(int i=0;i<5;++i){
   auto one=compressor.forward(mx::slice(h,{i,0},{i+1,5120}),serial_state,i);
   if(one.values.shape(0))rows.push_back(one.values);
   positions.insert(positions.end(),one.positions.begin(),one.positions.end());
  }
  auto serial=mx::concatenate(rows,0);same(batch.values,serial,"ratio-2 latent mismatch");
  if(batch.positions!=positions)throw std::runtime_error("ratio-2 positions mismatch");
  same_compressor(batch_state,serial_state);
  dsv41::IndexKeyReference index(catalog,2);
  auto keys=index.before_quantization(batch.values,batch.positions);std::vector<mx::array> key_rows;
  for(std::size_t i=0;i<batch.positions.size();++i)
   key_rows.push_back(index.before_quantization(mx::slice(batch.values,{int(i),0},{int(i+1),512}),
                                                std::span(batch.positions).subspan(i,1)));
  same(keys,mx::concatenate(key_rows,0),"index key chunk mismatch");
 }
 {
  dsv41::CompressorReference compressor(catalog,20);dsv41::CompressorState batch_state,serial_state;
  auto batch=compressor.forward(h,batch_state,0);std::vector<mx::array> rows;
  for(int i=0;i<5;++i)rows.push_back(compressor.forward(mx::slice(h,{i,0},{i+1,5120}),serial_state,i).values);
  same(batch.values,mx::concatenate(rows,0),"ratio-1 latent mismatch");
  same_compressor(batch_state,serial_state);
 }
 {
  dsv41::GlobalKVProducerReference producer(catalog,2);dsv41::GlobalKVState batch_state,serial_state;
  auto prefixes=producer.append_chunk(h,batch_state,0);std::vector<dsv41::GlobalKVState> serial_prefixes;
  for(int i=0;i<5;++i){producer.append(mx::slice(h,{i,0},{i+1,5120}),serial_state,i);serial_prefixes.push_back(serial_state);}
  same_global(batch_state,serial_state);
  for(int i=0;i<5;++i)same_global(prefixes[i],serial_prefixes[i]);
  auto qr=mx::astype(mx::reshape(mx::cos(mx::arange(5*1280,mx::float32)),{5,1280}),mx::bfloat16);
  dsv41::IndexQueryReference query(catalog,2);auto selected=query.forward_chunk(h,qr,prefixes,0);
  for(int i=0;i<5;++i){
   auto one=query.forward(mx::slice(h,{i,0},{i+1,5120}),mx::slice(qr,{i,0},{i+1,1280}),
                          serial_prefixes[i],i,i?128:1);
   same_device_selection(selected[i],one,i?128:1,nullptr,"index query device selection mismatch");
   if(dsv41::runtime_index_diagnostics_enabled()&&
      (selected[i].rows!=one.rows||selected[i].candidates!=one.candidates))
    throw std::runtime_error("index query chunk mismatch");
  }
  auto saved=batch_state;bool rejected=false;
  try{producer.append(mx::full({1,5120},NAN,mx::bfloat16),batch_state,5);}catch(const std::exception&){rejected=true;}
  if(!rejected)throw std::runtime_error("nonfinite producer input accepted");
  same_global(batch_state,saved);
 }
 {
  dsv41::GlobalKVProducerReference producer(catalog,20);dsv41::GlobalKVState state;
  auto prefixes=producer.append_chunk(h,state,0);
  auto qr=mx::astype(mx::reshape(mx::cos(mx::arange(5*1280,mx::float32)),{5,1280}),mx::bfloat16);
  dsv41::IndexQueryReference source(catalog,20,true,false);
  auto source_batch=source.forward_chunk(h,qr,prefixes,0);
  std::vector<std::vector<std::uint8_t>> candidates; candidates.reserve(source_batch.size());
  std::vector<mx::array> device_candidates;device_candidates.reserve(source_batch.size());
  for(int i=0;i<5;++i){
   auto one=source.forward(mx::slice(h,{i,0},{i+1,5120}),mx::slice(qr,{i,0},{i+1,1280}),
                           prefixes[i],i,i?128:1);
   same_device_selection(source_batch[i],one,i?128:1,&one.candidates,
                         "candidate source device selection mismatch");
   if(dsv41::runtime_index_diagnostics_enabled()&&
      (source_batch[i].rows!=one.rows||source_batch[i].candidates!=one.candidates))
    throw std::runtime_error("candidate source chunk mismatch token="+std::to_string(i)+
     " rows="+std::to_string(source_batch[i].rows.size())+"/"+std::to_string(one.rows.size())+
     " candidates="+std::to_string(source_batch[i].candidates.size())+"/"+std::to_string(one.candidates.size())+
     " first-row="+std::to_string(source_batch[i].rows.empty()?-1:source_batch[i].rows.front())+"/"+
     std::to_string(one.rows.empty()?-1:one.rows.front())+" last-row="+
     std::to_string(source_batch[i].rows.empty()?-1:source_batch[i].rows.back())+"/"+
     std::to_string(one.rows.empty()?-1:one.rows.back())+" candidate-bits="+
     std::to_string(source_batch[i].candidates.empty()?9:source_batch[i].candidates.front())+
     std::to_string(source_batch[i].candidates.size()<2?9:source_batch[i].candidates[1])+"/"+
     std::to_string(one.candidates.empty()?9:one.candidates.front())+
     std::to_string(one.candidates.size()<2?9:one.candidates[1]));
   candidates.push_back(one.candidates);device_candidates.push_back(source_batch[i].device_candidates);
  }
  dsv41::IndexQueryReference consumer(catalog,24,false,true);
  auto consumer_batch=dsv41::runtime_index_diagnostics_enabled()?
   consumer.forward_chunk(h,qr,prefixes,0,&candidates):
   consumer.forward_chunk(h,qr,prefixes,0,nullptr,&device_candidates);
  for(int i=0;i<5;++i){
   auto one=consumer.forward(mx::slice(h,{i,0},{i+1,5120}),mx::slice(qr,{i,0},{i+1,1280}),
                             prefixes[i],i,i?128:1,&candidates[i]);
   same_device_selection(consumer_batch[i],one,i?128:1,&candidates[i],
                         "candidate consumer device selection mismatch");
   if(dsv41::runtime_index_diagnostics_enabled()&&
      (consumer_batch[i].rows!=one.rows||consumer_batch[i].candidates!=one.candidates))
    throw std::runtime_error("candidate consumer chunk mismatch");
   auto diagnostic_rows=dsv41::runtime_index_diagnostics_enabled()?source_batch[i].rows:
    std::vector<std::int32_t>{};
   auto diagnostic_candidates=dsv41::runtime_index_diagnostics_enabled()?candidates[i]:
    std::vector<std::uint8_t>{};
   dsv41::SharedAttentionReference publication(prefixes[i],source_batch[i].device_rows,
    source_batch[i].device_candidates,i,i?128:1,20,1,std::move(diagnostic_rows),
    std::move(diagnostic_candidates));
   publication.republish(24,consumer_batch[i].device_rows,consumer_batch[i].device_candidates,
    dsv41::runtime_index_diagnostics_enabled()?consumer_batch[i].rows:std::vector<std::int32_t>{});
   same(publication.device_indices(24,i,i?128:1),consumer_batch[i].device_rows,
        "device republish rows mismatch");
   same(publication.device_candidates(),consumer_batch[i].device_candidates,
        "device republish candidates mismatch");
   if(dsv41::runtime_index_diagnostics_enabled()&&publication.candidates()!=candidates[i])
    throw std::runtime_error("diagnostic candidate publication was not preserved");
  }
 }
 {
  // Cross the 512-row boundary with a full chunk and force level-one pruning
  // (80 blocks -> 64), comparing every publication prefix.
  dsv41::GlobalKVProducerReference producer(catalog,20);dsv41::GlobalKVState state;
  for(int start=0;start<512;start+=128){
   auto values=mx::arange(start*5120,(start+128)*5120,mx::float32);
   auto chunk=mx::astype(mx::reshape(mx::sin(values),{128,5120}),mx::bfloat16);
   producer.append(chunk,state,start);
  }
  auto serial=state;
  auto tail_values=mx::arange(512*5120,640*5120,mx::float32);
  auto tail=mx::astype(mx::reshape(mx::sin(tail_values),{128,5120}),mx::bfloat16);
  auto prefixes=producer.append_chunk(tail,state,512);
  std::vector<dsv41::GlobalKVState> serial_prefixes;serial_prefixes.reserve(128);
  for(int i=0;i<128;++i){
   producer.append(mx::slice(tail,{i,0},{i+1,5120}),serial,512+i);
   serial_prefixes.push_back(serial);
  }
  same_global(state,serial);
  for(int i=0;i<128;++i)same_global(prefixes[i],serial_prefixes[i]);
  auto qr=mx::astype(mx::reshape(mx::cos(mx::arange(128*1280,mx::float32)),{128,1280}),mx::bfloat16);
  dsv41::IndexQueryReference source(catalog,20,true,false,64,8);
  auto source_batch=source.forward_chunk(tail,qr,prefixes,512);
  std::vector<std::vector<std::uint8_t>> candidates;candidates.reserve(128);
  std::vector<mx::array> device_candidates;device_candidates.reserve(128);
  for(int i=0;i<128;++i){
   auto one=source.forward(mx::slice(tail,{i,0},{i+1,5120}),mx::slice(qr,{i,0},{i+1,1280}),
                           prefixes[i],512+i,128);
   same_device_selection(source_batch[i],one,128,&one.candidates,
                         "512-boundary candidate source device mismatch");
   if(dsv41::runtime_index_diagnostics_enabled()&&
      (source_batch[i].rows!=one.rows||source_batch[i].candidates!=one.candidates))
    throw std::runtime_error("512-boundary candidate source mismatch token="+std::to_string(i));
   if(one.rows.size()!=512)throw std::runtime_error("512-boundary top-k width mismatch");
   candidates.push_back(one.candidates);device_candidates.push_back(source_batch[i].device_candidates);
  }
  dsv41::IndexQueryReference consumer(catalog,24,false,true,64,8);
  auto consumer_batch=dsv41::runtime_index_diagnostics_enabled()?
   consumer.forward_chunk(tail,qr,prefixes,512,&candidates):
   consumer.forward_chunk(tail,qr,prefixes,512,nullptr,&device_candidates);
  for(int i=0;i<128;++i){
   auto one=consumer.forward(mx::slice(tail,{i,0},{i+1,5120}),mx::slice(qr,{i,0},{i+1,1280}),
                             prefixes[i],512+i,128,&candidates[i]);
   same_device_selection(consumer_batch[i],one,128,&candidates[i],
                         "512-boundary candidate consumer device mismatch");
   if(dsv41::runtime_index_diagnostics_enabled()&&
      (consumer_batch[i].rows!=one.rows||consumer_batch[i].candidates!=one.candidates))
    throw std::runtime_error("512-boundary candidate consumer mismatch token="+std::to_string(i));
  }
  auto zero_x=mx::zeros({128,5120},mx::bfloat16),zero_qr=mx::zeros({128,1280},mx::bfloat16);
  dsv41::IndexQueryReference tied(catalog,20,false,false);
  auto tied_batch=tied.forward_chunk(zero_x,zero_qr,prefixes,512);
  for(int i=0;i<128;++i){
   auto one=tied.forward(mx::slice(zero_x,{i,0},{i+1,5120}),
                         mx::slice(zero_qr,{i,0},{i+1,1280}),prefixes[i],512+i,128);
   same_device_selection(tied_batch[i],one,128,nullptr,"512-boundary device tie mismatch");
   if(dsv41::runtime_index_diagnostics_enabled()&&
      (tied_batch[i].rows!=one.rows||tied_batch[i].rows.front()!=128||tied_batch[i].rows.back()!=639))
    throw std::runtime_error("512-boundary lowest-ID tie mismatch token="+std::to_string(i));
  }
 }
 std::cout<<"PASS: producer recurrent 128-token chunk, device candidate/top-k through 640 rows, all publication prefixes, lowest-ID boundary ties, packed cache, pending state and rejection match token-serial bits; index_diagnostics="<<dsv41::runtime_index_diagnostics_enabled()<<"\n";
 return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
