#include "dsv41/compressor.hpp"
#include "dsv41/model_entry.hpp"
#include "dsv41/execution_policy.hpp"
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
CompressedLatents CompressorReference::forward(const mx::array& h,CompressorState& state,std::uint64_t start,
 std::vector<CompressorState>* prefixes) const{
 if(h.dtype()!=mx::bfloat16||h.ndim()!=2||h.shape(0)<1||h.shape(0)>128||h.shape(1)!=5120||
    start!=state.position_||start>=1048576||std::uint64_t(h.shape(0))>1048576-start)
  throw std::runtime_error("invalid compressor input/position");
 auto next=state;std::vector<std::uint64_t> positions;
 if(runtime_layer_finite_checks_enabled()){
  auto finite=mx::all(mx::isfinite(h));mx::eval(finite);if(!finite.item<bool>())throw std::runtime_error("nonfinite compressor input");
 }
 mx::array result(0);
 if(ratio_==1){
  // Keep the one-row reduction geometry used by the reference, but materialize
  // every result in one lazy graph and synchronize only at the commit boundary.
  std::vector<mx::array> rows;rows.reserve(h.shape(0));
  for(int i=0;i<h.shape(0);++i)
   rows.push_back(rms_norm_reference(mx::matmul(mx::slice(h,{i,0},{i+1,5120}),mx::transpose(kv_weight_)),norm_,1e-20f));
  result=rows.size()==1?rows.front():mx::concatenate(rows,0);
  positions.reserve(h.shape(0));
  for(int i=0;i<h.shape(0);++i)positions.push_back(start+std::uint64_t(i));
  next.kv_=mx::zeros({0,512});next.scores_=mx::zeros({0,512});
  if(prefixes){
   prefixes->clear();prefixes->reserve(h.shape(0));
   for(int i=0;i<h.shape(0);++i){
    CompressorState prefix;prefix.position_=start+std::uint64_t(i)+1;
    prefixes->push_back(std::move(prefix));
   }
  }
 }else{
  // Encoder groups contain two tokens. The only recurrent input is the possible
  // first row of an incomplete group from the preceding chunk. Project the new
  // rows together, prepend that row, then pool every complete pair on device.
  std::vector<mx::array> kv_rows,score_rows;kv_rows.reserve(h.shape(0));score_rows.reserve(h.shape(0));
  for(int i=0;i<h.shape(0);++i){
   auto x=mx::astype(mx::slice(h,{i,0},{i+1,5120}),mx::float32);
   kv_rows.push_back(mx::matmul(x,mx::transpose(kv_weight_)));
   score_rows.push_back(mx::matmul(x,mx::transpose(gate_weight_)));
  }
  auto kv=kv_rows.size()==1?kv_rows.front():mx::concatenate(kv_rows,0);
  auto scores=score_rows.size()==1?score_rows.front():mx::concatenate(score_rows,0);
  const int pending=next.kv_.shape(0);
  if(pending<0||pending>=ratio_||next.scores_.shape()!=next.kv_.shape())
   throw std::runtime_error("invalid compressor pending state");
  auto keys=pending?mx::concatenate({next.kv_,kv},0):kv;
  auto all_scores=pending?mx::concatenate({next.scores_,scores},0):scores;
  if(prefixes){
   prefixes->clear();prefixes->reserve(h.shape(0));
   for(int i=0;i<h.shape(0);++i){
    CompressorState prefix;prefix.position_=start+std::uint64_t(i)+1;
    const int prefix_rows=pending+i+1;
    if(prefix_rows%ratio_){
     prefix.kv_=mx::slice(keys,{prefix_rows-1,0},{prefix_rows,512});
     prefix.scores_=mx::slice(all_scores,{prefix_rows-1,0},{prefix_rows,512});
    }
    prefixes->push_back(std::move(prefix));
   }
  }
  const int complete=keys.shape(0)/ratio_,used=complete*ratio_;
  if(complete){
   std::vector<mx::array> latent_rows;latent_rows.reserve(complete);
   for(int i=0;i<complete;++i){
    auto group_keys=mx::slice(keys,{i*ratio_,0},{(i+1)*ratio_,512});
    auto group_scores=mx::slice(all_scores,{i*ratio_,0},{(i+1)*ratio_,512});
    // Each channel has its own softmax over the ratio tokens, not over channels.
    auto exp=mx::exp(mx::subtract(group_scores,mx::max(group_scores,0,true)));
    auto pooled=mx::sum(mx::multiply(group_keys,mx::divide(exp,mx::sum(exp,0,true))),0,true);
    latent_rows.push_back(rms_norm_reference(mx::astype(pooled,mx::bfloat16),norm_,1e-20f));
   }
   result=latent_rows.size()==1?latent_rows.front():mx::concatenate(latent_rows,0);
   positions.reserve(complete);
   const auto first=start-std::uint64_t(pending);
   for(int i=0;i<complete;++i)positions.push_back(first+std::uint64_t(i*ratio_));
  }else result=mx::zeros({0,512},mx::bfloat16);
  const int leftover=keys.shape(0)-used;
  if(!used){next.kv_=keys;next.scores_=all_scores;}
  else if(leftover){
   next.kv_=mx::slice(keys,{used,0},{keys.shape(0),512});
   next.scores_=mx::slice(all_scores,{used,0},{all_scores.shape(0),512});
  }else{
   next.kv_=mx::zeros({0,512});next.scores_=mx::zeros({0,512});
  }
  if(runtime_layer_finite_checks_enabled()){
   auto projected_ok=mx::logical_and(mx::all(mx::isfinite(keys)),mx::all(mx::isfinite(all_scores)));
   auto ok=complete?mx::logical_and(projected_ok,mx::all(mx::isfinite(result))):projected_ok;
   if(complete)mx::eval(result,next.kv_,next.scores_,ok);
   else mx::eval(next.kv_,next.scores_,ok);
   if(!ok.item<bool>())throw std::runtime_error("nonfinite compressor projection/latent");
  }
 }
 next.position_=start+std::uint64_t(h.shape(0));
 if(runtime_layer_finite_checks_enabled()&&ratio_==1){
  auto valid=mx::all(mx::isfinite(result));mx::eval(result,valid);
  if(!valid.item<bool>())throw std::runtime_error("nonfinite compressor latent");
 }else if(ratio_==1)mx::eval(result);
 else if(!runtime_layer_finite_checks_enabled()){
  if(result.size())mx::eval(result,next.kv_,next.scores_);
  else mx::eval(next.kv_,next.scores_);
 }
 state=std::move(next);return {result,std::move(positions)};
}
}
