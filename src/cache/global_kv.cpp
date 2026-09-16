#include "dsv41/global_kv.hpp"
#include "dsv41/kv_quant.hpp"
#include "dsv41/layer_owner.hpp"
#include <stdexcept>
namespace dsv41 {
namespace mx=mlx::core;
GlobalKVState::GlobalKVState():main_(mx::zeros({0,256},mx::uint8)),main_scale_(mx::zeros({0,32},mx::uint8)),
 index_(mx::zeros({0,64},mx::uint8)),index_scale_(mx::zeros({0,4},mx::uint8)){}
void GlobalKVState::reset(){*this=GlobalKVState();}
GlobalKVProducerReference::GlobalKVProducerReference(WeightCatalog& c,int layer)
 :layer_(checked_producer_layer(layer)),ratio_(layer_compress_ratio(layer)),compressor_(c,layer),index_(c,layer){}
void GlobalKVProducerReference::append(const mx::array& hidden,GlobalKVState& state,std::uint64_t start) const{
 (void)append_chunk(hidden,state,start);
}
std::vector<GlobalKVState> GlobalKVProducerReference::append_chunk(
 const mx::array& hidden,GlobalKVState& state,std::uint64_t start) const{
 if(start!=state.position()||state.rows()!=start/std::uint64_t(ratio_))throw std::runtime_error("invalid global KV publication position");
 auto next=state;std::vector<CompressorState> compressor_prefixes;
 auto latent=compressor_.forward(hidden,next.compressor_,start,&compressor_prefixes);
 if(!latent.positions.empty()){
  for(std::size_t i=0;i<latent.positions.size();++i)
   if(latent.positions[i]!=std::uint64_t(ratio_)*(state.rows()+i))throw std::runtime_error("non-contiguous global KV group");
  // Both consumers receive the pre-RoPE latent; neither mutates the other's input.
  auto index=kv_quant_reference(index_.before_quantization(latent.values,latent.positions),KVQuantFormat::IndexE8M0);
  auto main=kv_quant_reference(compressed_rope_reference(latent.values,latent.positions),KVQuantFormat::MainE4M3);
  next.main_=mx::concatenate({state.main_,main.packed},0);
  next.main_scale_=mx::concatenate({state.main_scale_,main.scales},0);
  next.index_=mx::concatenate({state.index_,index.packed},0);
  next.index_scale_=mx::concatenate({state.index_scale_,index.scales},0);
  mx::eval(next.main_,next.main_scale_,next.index_,next.index_scale_);
 }
 std::vector<GlobalKVState> prefixes;prefixes.reserve(hidden.shape(0));
 for(int i=0;i<hidden.shape(0);++i){
  const auto position=start+std::uint64_t(i)+1;
  const int rows=int(position/std::uint64_t(ratio_));
  GlobalKVState prefix;prefix.compressor_=std::move(compressor_prefixes[i]);
  if(rows){
   prefix.main_=rows==int(next.rows())?next.main_:mx::slice(next.main_,{0,0},{rows,256});
   prefix.main_scale_=rows==int(next.rows())?next.main_scale_:mx::slice(next.main_scale_,{0,0},{rows,32});
   prefix.index_=rows==int(next.rows())?next.index_:mx::slice(next.index_,{0,0},{rows,64});
   prefix.index_scale_=rows==int(next.rows())?next.index_scale_:mx::slice(next.index_scale_,{0,0},{rows,4});
  }
  prefixes.push_back(std::move(prefix));
 }
 // Atomic at the synchronous API boundary: even compressor progress waits for both caches.
 state=std::move(next);
 return prefixes;
}
}
