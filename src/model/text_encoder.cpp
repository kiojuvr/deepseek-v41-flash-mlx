#include "dsv41/text_encoder.hpp"
#include "dsv41/sweep_telemetry.hpp"
#include "dsv41/layer_owner.hpp"
#include "dsv41/execution_policy.hpp"
#include "dsv41/runtime_profile.hpp"
#include <algorithm>
#include <iterator>
#include <stdexcept>
namespace dsv41 {
namespace mx=mlx::core;
namespace {
std::shared_ptr<const EngramMetadata> validate(std::shared_ptr<const EngramMetadata> m){
 if(!m||m->layer_ids!=std::vector<std::uint64_t>{1,14})throw std::runtime_error("invalid encoder Engram metadata");
 return m;
}
int producer_slot(int layer){
 if(layer==2)return 0;
 if(layer==8)return 1;
 if(layer==14)return 2;
 throw std::runtime_error("invalid encoder producer layer");
}
int reuse_slot(int layer){
 if(layer>=3&&layer<=7)return layer-3;
 if(layer>=9&&layer<=13)return layer-9+5;
 if(layer>=15&&layer<=19)return layer-15+10;
 throw std::runtime_error("invalid encoder reuse layer");
}
}
void TextEncoderState::reset(){
 hash.reset();for(auto& s:swa)s.reset();for(auto& s:producer)s.reset();for(auto& s:reuse)s.reset();
}
BlockResult TextEncoderReference::forward_packed_chunk(std::span<const std::uint32_t> ids,
 TextEncoderState& state,std::uint64_t start) const{
 if(state.metadata!=metadata_||state.hash.position()!=start||ids.empty()||ids.size()>128||
    start>=1048576||ids.size()>1048576-start)throw std::runtime_error("invalid packed encoder request");
 for(const auto& s:state.swa)if(s.position()!=start)throw std::runtime_error("invalid packed encoder SWA position");
 for(const auto& s:state.producer)if(s.position()!=start)throw std::runtime_error("invalid packed encoder producer position");
 for(const auto& s:state.reuse)if(s.position()!=start)throw std::runtime_error("invalid packed encoder reuse position");
 for(auto id:ids)if(id>=129280||id==129264)throw std::runtime_error("packed encoder requires text IDs");
 auto next=state;auto hashes=next.hash.append(ids,{},start);
 auto entry=entry_.forward(ids);BlockResult out{entry.hidden,entry.pre_mix};
 auto engram=[&](const EngramLayerReference& layer,int slot){
  std::vector<std::uint64_t> rows;rows.reserve(ids.size()*24);
  for(std::size_t t=0;t<ids.size();++t)
   rows.insert(rows.end(),hashes.begin()+t*48+slot*24,hashes.begin()+t*48+slot*24+24);
  out.hidden=layer.forward(out.hidden,rows).output;
 };
 // Release on exceptions too: failed requests must not retain a full bank.
 auto run_layer=[&](int layer,const auto& block,auto&& forward){
  auto started=runtime_profile_start();
  try{out=forward();}catch(...){block.release_packed_bank();throw;}
  block.release_packed_bank();
  record_runtime_layer(layer,runtime_profile_elapsed(started));
 };
 for(int layer=0;layer<2;++layer){
  if(layer==1)engram(engram1_,0);
  run_layer(layer,*swa_[layer],[&]{return swa_[layer]->forward_packed_chunk(out.hidden,out.pre_mix,next.swa[layer],start);});
 }
 for(int source:{2,8,14}){
  if(source==14)engram(engram14_,1);
  const int slot=producer_slot(source);std::vector<SharedAttentionReference> publications;
  run_layer(source,*producer_[slot],[&]{return producer_[slot]->forward_packed_chunk(out.hidden,out.pre_mix,next.producer[slot],start,&publications);});
  for(int layer=source+1;layer<source+6;++layer){
   const int reuse_index=reuse_slot(layer);
   run_layer(layer,*reuse_[reuse_index],[&]{return reuse_[reuse_index]->forward_packed_chunk(out.hidden,out.pre_mix,next.reuse[reuse_index],publications,start);});
  }
  next.producer[slot].publication()=publications.back();
 }
 mx::eval(out.hidden,out.pre_mix);state=std::move(next);return out;
}
BlockResult TextEncoderReference::forward_packed_sweep(std::span<const std::uint32_t> ids,
 TextEncoderState& state,std::uint64_t start) const{
 if(state.metadata!=metadata_||state.hash.position()!=start||ids.empty()||
    start>=1048576||ids.size()>1048576-start)throw std::runtime_error("invalid packed encoder sweep request");
 for(const auto& s:state.swa)if(s.position()!=start)throw std::runtime_error("invalid packed encoder sweep SWA position");
 for(const auto& s:state.producer)if(s.position()!=start)throw std::runtime_error("invalid packed encoder sweep producer position");
 for(const auto& s:state.reuse)if(s.position()!=start)throw std::runtime_error("invalid packed encoder sweep reuse position");
 for(auto id:ids)if(id>=129280||id==129264)throw std::runtime_error("packed encoder sweep requires text IDs");
 auto next=state;std::vector<std::uint64_t> hashes;hashes.reserve(ids.size()*48);
 std::vector<mx::array> entry_hidden,entry_pre;
 for(std::size_t offset=0;offset<ids.size();offset+=128){
  const auto count=std::min<std::size_t>(128,ids.size()-offset);
  auto tile=ids.subspan(offset,count);auto tile_hashes=next.hash.append(tile,{},start+offset);
  hashes.insert(hashes.end(),tile_hashes.begin(),tile_hashes.end());
  auto entry=entry_.forward(tile);entry_hidden.push_back(entry.hidden);entry_pre.push_back(entry.pre_mix);
 }
 BlockResult out{mx::concatenate(entry_hidden,0),mx::concatenate(entry_pre,0)};
 const bool defer_layers=runtime_defer_decode_layer_eval(ids.size());
 auto materialize=[&](std::vector<mx::array>& hidden,std::vector<mx::array>& pre){
  out={mx::concatenate(hidden,0),mx::concatenate(pre,0)};
  if(!defer_layers)mx::eval(out.hidden,out.pre_mix);
  record_sweep_layer(defer_layers);
 };
 auto tile_input=[&](std::size_t offset,std::size_t count){
  return BlockResult{
   mx::slice(out.hidden,{int(offset),0,0},{int(offset+count),4,5120}),
   mx::slice(out.pre_mix,{int(offset),0},{int(offset+count),4})};
 };
 auto apply_engram=[&](const EngramLayerReference& layer,int slot){
  std::vector<mx::array> hidden;hidden.reserve((ids.size()+127)/128);
  for(std::size_t offset=0;offset<ids.size();offset+=128){
   const auto count=std::min<std::size_t>(128,ids.size()-offset);
   std::vector<std::uint64_t> rows;rows.reserve(count*24);
   for(std::size_t token=offset;token<offset+count;++token)
    rows.insert(rows.end(),hashes.begin()+token*48+slot*24,hashes.begin()+token*48+slot*24+24);
   auto input=mx::slice(out.hidden,{int(offset),0,0},{int(offset+count),4,5120});
   hidden.push_back(layer.forward(input,rows).output);
  }
  out.hidden=mx::concatenate(hidden,0);
  if(!defer_layers){mx::eval(out.hidden);record_sweep_engram();}
 };
 auto run_plain=[&](int layer,const auto& block,auto& layer_state){
  auto started=runtime_profile_start();std::vector<mx::array> hidden,pre;
  hidden.reserve((ids.size()+127)/128);pre.reserve(hidden.capacity());
  try{
   for(std::size_t offset=0;offset<ids.size();offset+=128){
    const auto count=std::min<std::size_t>(128,ids.size()-offset);auto input=tile_input(offset,count);
    auto result=block.forward_packed_chunk(input.hidden,input.pre_mix,layer_state,start+offset);
    hidden.push_back(result.hidden);pre.push_back(result.pre_mix);
   }
  }catch(...){block.release_packed_bank();throw;}
  block.release_packed_bank();materialize(hidden,pre);
  record_runtime_layer(layer,runtime_profile_elapsed(started));
 };
 auto run_producer=[&](int layer,const auto& block,auto& layer_state,
                       std::vector<SharedAttentionReference>& publications){
  auto started=runtime_profile_start();std::vector<mx::array> hidden,pre;publications.clear();
  hidden.reserve((ids.size()+127)/128);pre.reserve(hidden.capacity());publications.reserve(ids.size());
  try{
   for(std::size_t offset=0;offset<ids.size();offset+=128){
    const auto count=std::min<std::size_t>(128,ids.size()-offset);auto input=tile_input(offset,count);
    std::vector<SharedAttentionReference> tile_publications;
    auto result=block.forward_packed_chunk(input.hidden,input.pre_mix,layer_state,start+offset,&tile_publications);
    hidden.push_back(result.hidden);pre.push_back(result.pre_mix);
    publications.insert(publications.end(),std::make_move_iterator(tile_publications.begin()),
                         std::make_move_iterator(tile_publications.end()));
   }
  }catch(...){block.release_packed_bank();throw;}
  block.release_packed_bank();materialize(hidden,pre);
  record_runtime_layer(layer,runtime_profile_elapsed(started));
 };
 auto run_reuse=[&](int layer,const auto& block,auto& layer_state,
                    std::vector<SharedAttentionReference>& publications){
  auto started=runtime_profile_start();std::vector<mx::array> hidden,pre;
  hidden.reserve((ids.size()+127)/128);pre.reserve(hidden.capacity());
  try{
   for(std::size_t offset=0;offset<ids.size();offset+=128){
    const auto count=std::min<std::size_t>(128,ids.size()-offset);auto input=tile_input(offset,count);
    std::vector<SharedAttentionReference> tile_publications(
     publications.begin()+offset,publications.begin()+offset+count);
    auto result=block.forward_packed_chunk(input.hidden,input.pre_mix,layer_state,
                                           tile_publications,start+offset);
    hidden.push_back(result.hidden);pre.push_back(result.pre_mix);
    std::move(tile_publications.begin(),tile_publications.end(),publications.begin()+offset);
   }
  }catch(...){block.release_packed_bank();throw;}
  block.release_packed_bank();materialize(hidden,pre);
  record_runtime_layer(layer,runtime_profile_elapsed(started));
 };
 run_plain(0,*swa_[0],next.swa[0]);apply_engram(engram1_,0);run_plain(1,*swa_[1],next.swa[1]);
 for(int source:{2,8,14}){
  if(source==14)apply_engram(engram14_,1);
  const int slot=producer_slot(source);std::vector<SharedAttentionReference> publications;
  run_producer(source,*producer_[slot],next.producer[slot],publications);
  for(int layer=source+1;layer<source+6;++layer){
   const int index=reuse_slot(layer);run_reuse(layer,*reuse_[index],next.reuse[index],publications);
  }
  next.producer[slot].publication()=publications.back();
 }
 // Evaluate before publication even for callers using the encoder directly.
 if(defer_layers){mx::eval(out.hidden,out.pre_mix);record_sweep_stack();}
 state=std::move(next);return out;
}
TextEncoderReference::TextEncoderReference(WeightCatalog& c,std::shared_ptr<const EngramMetadata> m,
 std::shared_ptr<const ResidentExpertAtlas> atlas)
 :metadata_(validate(std::move(m))),entry_(c),
  engram1_(c,*metadata_,0,1e-20f,EngramReadMode::Mmap),engram14_(c,*metadata_,1,1e-20f,EngramReadMode::Mmap){
 swa_[0]=std::make_unique<BlockReference>(c,0,atlas);
 swa_[1]=std::make_unique<BlockReference>(c,1,atlas);
 producer_[0]=std::make_unique<CompressedBlockReference>(c,2,atlas);
 producer_[1]=std::make_unique<CompressedBlockReference>(c,8,atlas);
 producer_[2]=std::make_unique<CompressedBlockReference>(c,14,atlas);
 for(int layer=0;layer<20;++layer){
  if(layer<=1||layer==2||layer==8||layer==14)continue;
  reuse_[reuse_slot(layer)]=std::make_unique<ReusedBlockReference>(c,layer,atlas);
 }
}
BlockResult TextEncoderReference::forward(std::span<const std::uint32_t> ids,TextEncoderState& state,std::uint64_t start,TraceSink* trace) const{
 if(state.metadata!=metadata_||state.hash.position()!=start||
    state.swa[0].position()!=start||state.swa[1].position()!=start||
    ids.empty()||ids.size()>128||start>=1048576||ids.size()>1048576-start)throw std::runtime_error("invalid encoder request state");
 for(int i=0;i<3;++i)if(state.producer[i].position()!=start)throw std::runtime_error("invalid encoder producer position");
 for(auto& s:state.reuse)if(s.position()!=start)throw std::runtime_error("invalid encoder reuse position");
 for(auto id:ids)if(id>=129280||id==129264)throw std::runtime_error("encoder reference requires text token IDs");
 set_active_trace_sink(trace);
 auto next=state;std::vector<mx::array> hidden,pre;
 std::array<std::vector<mx::array>,20> layer_hidden,layer_pre;
 std::vector<mx::array> entry_hidden,entry_pre;
 auto capture=[&](int layer,const BlockResult& r){layer_hidden[layer].push_back(r.hidden);layer_pre[layer].push_back(r.pre_mix);};
 for(std::size_t t=0;t<ids.size();++t){
  auto token=ids.subspan(t,1);
  std::uint64_t pos=start+t;
  set_route_trace_token(pos);
  auto hashes=next.hash.append(token,{},pos);
  if(hashes.size()!=48)throw std::runtime_error("unexpected encoder Engram hash width");
  auto entry=entry_.forward(token);
  if(trace){entry_hidden.push_back(entry.hidden);entry_pre.push_back(entry.pre_mix);}
  // Layer 0: pure SWA.
  auto h=swa_[0]->forward(entry.hidden,entry.pre_mix,next.swa[0],pos);capture(0,h);
  // Layer 1: Engram then pure SWA.
  auto e1=engram1_.forward(h.hidden,std::span<const std::uint64_t>(hashes).first(24));
  h=swa_[1]->forward(e1.output,h.pre_mix,next.swa[1],pos);capture(1,h);
  // Layer 2: compressed producer, then layers 3..7 reuse its publication.
  h=producer_[0]->forward(h.hidden,h.pre_mix,next.producer[0],pos);capture(2,h);
  for(int layer=3;layer<=7;++layer){
   auto& pub=next.producer[producer_slot(2)].publication();
   if(!pub)throw std::runtime_error("missing encoder publication layer 2");
   h=reuse_[reuse_slot(layer)]->forward(h.hidden,h.pre_mix,next.reuse[reuse_slot(layer)],*pub,pos);capture(layer,h);
  }
  // Layer 8: compressed producer, then layers 9..13 reuse.
  h=producer_[1]->forward(h.hidden,h.pre_mix,next.producer[1],pos);capture(8,h);
  for(int layer=9;layer<=13;++layer){
   auto& pub=next.producer[producer_slot(8)].publication();
   if(!pub)throw std::runtime_error("missing encoder publication layer 8");
   h=reuse_[reuse_slot(layer)]->forward(h.hidden,h.pre_mix,next.reuse[reuse_slot(layer)],*pub,pos);capture(layer,h);
  }
  // Layer 14: Engram then compressed producer, then layers 15..19 reuse.
  auto e14=engram14_.forward(h.hidden,std::span<const std::uint64_t>(hashes).subspan(24,24));
  h=producer_[2]->forward(e14.output,h.pre_mix,next.producer[2],pos);capture(14,h);
  for(int layer=15;layer<=19;++layer){
   auto& pub=next.producer[producer_slot(14)].publication();
   if(!pub)throw std::runtime_error("missing encoder publication layer 14");
   h=reuse_[reuse_slot(layer)]->forward(h.hidden,h.pre_mix,next.reuse[reuse_slot(layer)],*pub,pos);capture(layer,h);
  }
  hidden.push_back(h.hidden);pre.push_back(h.pre_mix);
 }
 if(trace){
  trace->record("encoder.entry.hidden",mx::concatenate(entry_hidden,0));
  trace->record("encoder.entry.pre_mix",mx::concatenate(entry_pre,0));
  for(int layer=0;layer<20;++layer){
   auto name=[&](const char* suffix){return "encoder.layer"+std::to_string(layer)+"."+suffix;};
   trace->record(name("hidden"),mx::concatenate(layer_hidden[layer],0));
   trace->record(name("pre_mix"),mx::concatenate(layer_pre[layer],0));
  }
 }
 BlockResult result{mx::concatenate(hidden,0),mx::concatenate(pre,0)};
 mx::eval(result.hidden,result.pre_mix);set_active_trace_sink(nullptr);state=std::move(next);return result;
}
}
