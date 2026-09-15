#include "dsv41/compressed_layer.hpp"
#include "dsv41/reused_layer.hpp"
#include "dsv41/swa_attention.hpp"
#include "dsv41/model_entry.hpp"
#include "dsv41/engram.hpp"
#include <stdexcept>
#include "dsv41/layer_owner.hpp"
namespace dsv41 {
namespace mx=mlx::core;
namespace {
mx::array grouped_weight(WeightCatalog& c,int layer){
 auto w=c.tensor(("layers."+std::to_string(layer))+".attn.wo_a.weight"),s=c.tensor(("layers."+std::to_string(layer))+".attn.wo_a.scale");
 if(w.dtype!="F8_E4M3"||w.shape!=std::vector<std::uint64_t>{8192,4096}||w.size_bytes!=8192ull*4096||
    s.dtype!="F8_E8M0"||s.shape!=std::vector<std::uint64_t>{256,128}||s.size_bytes!=256*128)
  throw std::runtime_error("unexpected SWA wo_a layout");
 std::vector<std::uint8_t> weights(w.size_bytes),scales(s.size_bytes);
 w.read(0,{reinterpret_cast<std::byte*>(weights.data()),weights.size()});
 s.read(0,{reinterpret_cast<std::byte*>(scales.data()),scales.size()});
 std::vector<std::uint16_t> decoded(weights.size());
 for(std::size_t i=0;i<weights.size();++i){
  auto code=weights[i],scale=scales[(i/4096/32)*128+(i%4096)/32];
  if((code&127)==127||scale==255)throw std::runtime_error("nonfinite SWA wo_a weight");
  decoded[i]=engram_bf16(code,scale);
 }
 return mx::view(mx::array(decoded.begin(),{8,1024,4096},mx::uint16),mx::bfloat16);
}
mx::array sink(WeightCatalog& c,int layer){
 auto t=c.tensor(("layers."+std::to_string(layer))+".attn.attn_sink");
 if(t.dtype!="F32"||t.shape!=std::vector<std::uint64_t>{64}||t.size_bytes!=256)throw std::runtime_error("unexpected SWA sink");
 std::vector<float> v(64);t.read(0,{reinterpret_cast<std::byte*>(v.data()),256});return mx::array(v.begin(),{64},mx::float32);
}
}
namespace {
mx::array norm(WeightCatalog& c,const std::string& name,int size,int layer){
 auto t=c.tensor("layers."+std::to_string(layer)+".attn."+name);
 if(t.dtype!="BF16"||t.shape!=std::vector<std::uint64_t>{std::uint64_t(size)}||t.size_bytes!=2ull*size)throw std::runtime_error("compressed norm layout mismatch");
 std::vector<std::uint16_t> v(size);t.read(0,{reinterpret_cast<std::byte*>(v.data()),t.size_bytes});return mx::view(mx::array(v.begin(),{size},mx::uint16),mx::bfloat16);
}
mx::array main_rows(const GlobalKVState& state,const std::vector<std::int32_t>& selected,int offset){
 std::vector<int> ids;for(auto index:selected){int row=index-offset;if(row<0||std::size_t(row)>=state.rows())throw std::runtime_error("invalid selected global row");ids.push_back(row);}
 auto indices=mx::array(ids.begin(),{int(ids.size())},mx::int32);
 auto p=mx::take(state.main_bytes(),indices,0),s=mx::take(state.main_scales(),indices,0);
 auto low=mx::bitwise_and(p,mx::array(15,mx::uint8)),high=mx::right_shift(p,mx::array(4,mx::uint8));
 auto codes=mx::reshape(mx::stack({low,high},-1),{int(ids.size()),512});
 auto levels=mx::array({0.0f,0.5f,1.0f,1.5f,2.0f,3.0f,4.0f,6.0f});
 auto v=mx::take(levels,mx::astype(mx::bitwise_and(codes,mx::array(7,mx::uint8)),mx::int32));
 v=mx::where(mx::greater_equal(codes,mx::array(8,mx::uint8)),mx::negative(v),v);
 auto exponent=mx::astype(mx::right_shift(s,mx::array(3,mx::uint8)),mx::float32);
 auto mantissa=mx::astype(mx::bitwise_and(s,mx::array(7,mx::uint8)),mx::float32);
 auto scale=mx::where(mx::equal(exponent,mx::array(0.0f)),mx::multiply(mantissa,mx::array(0x1p-9f)),mx::multiply(mx::add(mantissa,mx::array(8.0f)),mx::power(mx::array(2.0f),mx::subtract(exponent,mx::array(10.0f)))));
 return mx::astype(mx::reshape(mx::multiply(mx::reshape(v,{int(ids.size()),32,16}),mx::expand_dims(scale,-1)),{int(ids.size()),512}),mx::bfloat16);
}
}
CompressedLayerReference::CompressedLayerReference(WeightCatalog& c,int layer):layer_(checked_producer_layer(layer)),ratio_(layer_compress_ratio(layer)),
 qa_(c,("layers."+std::to_string(layer_))+".attn.wq_a"),qb_(c,("layers."+std::to_string(layer_))+".attn.wq_b"),kv_(c,("layers."+std::to_string(layer_))+".attn.wkv"),output_(c,("layers."+std::to_string(layer_))+".attn.wo_b"),
 qnorm_(norm(c,"q_norm.weight",1280,layer_)),kvnorm_(norm(c,"kv_norm.weight",512,layer_)),grouped_(grouped_weight(c,layer_)),sink_(sink(c,layer_)),producer_(c,layer_),index_(c,layer_,layer_==20,layer_>20){
 if(qa_.input_dims()!=5120||qa_.output_dims()!=1280||qb_.input_dims()!=1280||qb_.output_dims()!=32768||kv_.input_dims()!=5120||kv_.output_dims()!=512||output_.input_dims()!=8192||output_.output_dims()!=5120||qa_.bits()!=8||qb_.bits()!=8||kv_.bits()!=8||output_.bits()!=8)throw std::runtime_error("compressed projection layout mismatch");
 mx::eval(grouped_,sink_);
}
mx::array CompressedLayerReference::forward(const mx::array& h,CompressedLayerState& state,std::uint64_t start) const{
 if(h.dtype()!=mx::bfloat16||h.ndim()!=2||h.shape(0)<1||h.shape(0)>128||h.shape(1)!=5120||state.position()!=start||start>=1048576||std::uint64_t(h.shape(0))>1048576-start)throw std::runtime_error("invalid compressed attention input/state");
 auto next=state;std::vector<mx::array> outputs;
 for(int i=0;i<h.shape(0);++i){
  std::uint64_t pos=start+i;auto x=mx::slice(h,{i,0},{i+1,5120});
  auto qr=rms_norm_reference(qa_.forward(x),qnorm_,1e-20f);
  auto q=compressed_rope_reference(mx::reshape(qb_.forward(qr),{1,64,512}),std::span(&pos,1));
  auto kv=linear_activation_reference(compressed_rope_reference(rms_norm_reference(kv_.forward(x),kvnorm_,1e-20f),std::span(&pos,1))).decoded;
  auto window=mx::concatenate({next.window_,kv},0);if(window.shape(0)>128)window=mx::slice(window,{window.shape(0)-128,0},{window.shape(0),512});
  int padding=pos==0?0:128-window.shape(0);
  auto ordered=padding?mx::concatenate({mx::zeros({padding,512},mx::bfloat16),window},0):window;
  int offset=ordered.shape(0);producer_.append(x,next.global_,pos);
  auto selection=index_.forward(x,qr,next.global_,pos,offset);
  auto publication=SharedAttentionReference(next.global_,selection.rows,pos,offset,layer_,ratio_);
  if(!selection.candidates.empty())publication.republish(layer_,selection.rows,selection.candidates);
  if(!selection.rows.empty())ordered=mx::concatenate({ordered,main_rows(next.global_,selection.rows,offset)},0);
  auto valid=mx::greater_equal(mx::arange(ordered.shape(0),mx::int32),mx::array(padding));
  auto o=swa_attention_masked_reference(mx::reshape(q,{64,512}),ordered,sink_,valid);
  o=compressed_rope_reference(mx::reshape(o,{1,64,512}),std::span(&pos,1),true);
  auto projected=mx::matmul(mx::reshape(o,{8,1,4096}),mx::transpose(grouped_,{0,2,1}));
  auto y=output_.forward(mx::reshape(projected,{1,8192}));
  auto ok=mx::logical_and(mx::all(mx::isfinite(y)),mx::all(mx::isfinite(window)));mx::eval(y,window,ok);
  if(!ok.item<bool>())throw std::runtime_error("nonfinite compressed attention output/state");
  next.window_=window;next.publication_=std::move(publication);outputs.push_back(y);
 }
 auto result=mx::concatenate(outputs,0);mx::eval(result);state=std::move(next);return result;
}
ReusedLayerReference::ReusedLayerReference(WeightCatalog& c,int layer):layer_(checked_reused_layer(layer)),ratio_(layer_compress_ratio(layer_)),is_index_source_(is_index_source_layer(layer_)),uses_candidates_(layer_>20),qa_(c,("layers."+std::to_string(layer_))+".attn.wq_a"),qb_(c,("layers."+std::to_string(layer_))+".attn.wq_b"),kv_(c,("layers."+std::to_string(layer_))+".attn.wkv"),output_(c,("layers."+std::to_string(layer_))+".attn.wo_b"),qnorm_(norm(c,"q_norm.weight",1280,layer_)),kvnorm_(norm(c,"kv_norm.weight",512,layer_)),grouped_(grouped_weight(c,layer_)),sink_(sink(c,layer_)){
 if(qa_.input_dims()!=5120||qa_.output_dims()!=1280||qb_.input_dims()!=1280||qb_.output_dims()!=32768||kv_.input_dims()!=5120||kv_.output_dims()!=512||output_.input_dims()!=8192||output_.output_dims()!=5120||qa_.bits()!=8||qb_.bits()!=8||kv_.bits()!=8||output_.bits()!=8)throw std::runtime_error("compressed projection layout mismatch");
 if(is_index_source_)index_=std::make_unique<IndexQueryReference>(c,layer_,false,uses_candidates_);
 mx::eval(grouped_,sink_);
}
mx::array ReusedLayerReference::forward(const mx::array& x,ReusedLayerState& state,SharedAttentionReference& publication,std::uint64_t pos) const{
 if(x.dtype()!=mx::bfloat16||x.shape()!=mx::Shape({1,5120})||state.position()!=pos||pos>=1048576)throw std::runtime_error("invalid reuse layer input/state");
 int offset=pos==0?1:128;
 auto finite=mx::all(mx::isfinite(x));mx::eval(finite);if(!finite.item<bool>())throw std::runtime_error("nonfinite reuse layer input");
 auto qr=rms_norm_reference(qa_.forward(x),qnorm_,1e-20f);
 std::vector<std::int32_t> selected;
 if(is_index_source_){
  auto selection=index_->forward(x,qr,publication.cache(),pos,offset,uses_candidates_?&publication.candidates():nullptr);
  publication.republish(layer_,selection.rows,std::move(selection.candidates));
  selected=publication.indices(layer_,pos,offset);
 }else{
  selected=publication.indices(layer_,pos,offset);
 }
 auto q=compressed_rope_reference(mx::reshape(qb_.forward(qr),{1,64,512}),std::span(&pos,1));
 auto kv=linear_activation_reference(compressed_rope_reference(rms_norm_reference(kv_.forward(x),kvnorm_,1e-20f),std::span(&pos,1))).decoded;
 auto window=mx::concatenate({state.window_,kv},0);
 if(window.shape(0)>128)window=mx::slice(window,{window.shape(0)-128,0},{window.shape(0),512});
 int padding=offset-window.shape(0);
 auto ordered=padding?mx::concatenate({mx::zeros({padding,512},mx::bfloat16),window},0):window;
 if(!selected.empty())ordered=mx::concatenate({ordered,main_rows(publication.cache(),selected,offset)},0);
 auto valid=mx::greater_equal(mx::arange(ordered.shape(0),mx::int32),mx::array(padding));
 auto o=swa_attention_masked_reference(mx::reshape(q,{64,512}),ordered,sink_,valid);
 o=compressed_rope_reference(mx::reshape(o,{1,64,512}),std::span(&pos,1),true);
 auto projected=mx::matmul(mx::reshape(o,{8,1,4096}),mx::transpose(grouped_,{0,2,1}));
 auto y=output_.forward(mx::reshape(projected,{1,8192}));
 auto ok=mx::logical_and(mx::all(mx::isfinite(y)),mx::all(mx::isfinite(window)));mx::eval(y,window,ok);
 if(!ok.item<bool>())throw std::runtime_error("nonfinite reuse layer output/state");
 state.window_=window;state.position_=pos+1;return y;
}
}
