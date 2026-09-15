#include "dsv41/compressor.hpp"
#include "dsv41/model_entry.hpp"
#include "dsv41/layer_owner.hpp"
#include <stdexcept>
namespace dsv41 {
namespace mx=mlx::core;
namespace {
mx::array load(WeightCatalog& c,const std::string& name,mx::Shape shape){
 auto t=c.tensor(name);
 std::vector<std::uint64_t> expected(shape.begin(),shape.end());std::size_t count=1;for(auto n:shape)count*=n;
 if(t.dtype!="BF16"||t.shape!=expected||t.size_bytes!=2*count)throw std::runtime_error("unexpected compressor weight");
 std::vector<std::uint16_t> v(count);t.read(0,{reinterpret_cast<std::byte*>(v.data()),t.size_bytes});
 return mx::view(mx::array(v.begin(),shape,mx::uint16),mx::bfloat16);
}
}
CompressorState::CompressorState():kv_(mx::zeros({0,512})),scores_(mx::zeros({0,512})){}
void CompressorState::reset(){kv_=mx::zeros({0,512});scores_=mx::zeros({0,512});position_=0;}
CompressorReference::CompressorReference(WeightCatalog& c,int layer)
 :layer_(layer),ratio_(layer_compress_ratio(layer)),
  kv_weight_(load(c,("layers."+std::to_string(layer))+".attn.compressor.wkv.weight",{512,5120})),
  gate_weight_(ratio_>1?load(c,("layers."+std::to_string(layer))+".attn.compressor.wgate.weight",{512,5120}):mx::array(0)),
  norm_(load(c,("layers."+std::to_string(layer))+".attn.compressor.norm.weight",{512})){
 if(!is_kv_source_layer(layer))throw std::runtime_error("compressor requires a kv_source layer");
 if(ratio_>1){
  kv_weight_=mx::astype(kv_weight_,mx::float32);
  gate_weight_=mx::astype(gate_weight_,mx::float32);
 }
}
CompressedLatents CompressorReference::forward(const mx::array& h,CompressorState& state,std::uint64_t start) const{
 if(h.dtype()!=mx::bfloat16||h.ndim()!=2||h.shape(0)<1||h.shape(0)>128||h.shape(1)!=5120||
    start!=state.position_||start>=1048576||std::uint64_t(h.shape(0))>1048576-start)
  throw std::runtime_error("invalid compressor input/position");
 auto next=state;std::vector<mx::array> outputs;std::vector<std::uint64_t> positions;
 auto finite=mx::all(mx::isfinite(h));mx::eval(finite);if(!finite.item<bool>())throw std::runtime_error("nonfinite compressor input");
 for(int i=0;i<h.shape(0);++i){
  if(ratio_==1){
   // One token per group: plain projection and norm in the checkpoint's bf16.
   auto latent=rms_norm_reference(mx::matmul(mx::slice(h,{i,0},{i+1,5120}),mx::transpose(kv_weight_)),norm_,1e-20f);
   auto valid=mx::all(mx::isfinite(latent));mx::eval(latent,valid);if(!valid.item<bool>())throw std::runtime_error("nonfinite compressor latent");
   outputs.push_back(latent);positions.push_back(start+i);
  }else{
   auto x=mx::astype(mx::slice(h,{i,0},{i+1,5120}),mx::float32);
   auto kv=mx::matmul(x,mx::transpose(kv_weight_)),score=mx::matmul(x,mx::transpose(gate_weight_));
   auto keys=mx::concatenate({next.kv_,kv},0),scores=mx::concatenate({next.scores_,score},0);
   auto ok=mx::logical_and(mx::all(mx::isfinite(keys)),mx::all(mx::isfinite(scores)));mx::eval(keys,scores,ok);
   if(!ok.item<bool>())throw std::runtime_error("nonfinite compressor projection");
   if((start+i+1)%std::uint64_t(ratio_)==0){
    // Each channel has its own softmax over the ratio tokens, not over channels.
    auto exp=mx::exp(mx::subtract(scores,mx::max(scores,0,true)));
    auto pooled=mx::sum(mx::multiply(keys,mx::divide(exp,mx::sum(exp,0,true))),0,true);
    auto latent=rms_norm_reference(mx::astype(pooled,mx::bfloat16),norm_,1e-20f);
    auto valid=mx::all(mx::isfinite(latent));mx::eval(latent,valid);if(!valid.item<bool>())throw std::runtime_error("nonfinite compressor latent");
    outputs.push_back(latent);positions.push_back(start+i-(ratio_-1));
    next.kv_=mx::zeros({0,512});next.scores_=mx::zeros({0,512});
   }else{next.kv_=keys;next.scores_=scores;}
  }
  next.position_=start+i+1;
 }
 auto result=outputs.empty()?mx::zeros({0,512},mx::bfloat16):mx::concatenate(outputs,0);
 mx::eval(result);state=std::move(next);return {result,std::move(positions)};
}
}
