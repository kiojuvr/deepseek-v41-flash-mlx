#include "dsv41/swa_layer.hpp"
#include "dsv41/attention_telemetry.hpp"
#include "dsv41/execution_policy.hpp"
#include "dsv41/swa_attention.hpp"
#include "dsv41/engram.hpp"
#include <stdexcept>
#include "dsv41/layer_owner.hpp"
namespace dsv41 {
namespace mx=mlx::core;
namespace {
mx::array grouped_weight(WeightCatalog& c,int layer){
 auto w=c.tensor(pure_swa_prefix(layer)+".attn.wo_a.weight"),s=c.tensor(pure_swa_prefix(layer)+".attn.wo_a.scale");
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
 auto t=c.tensor(pure_swa_prefix(layer)+".attn.attn_sink");
 if(t.dtype!="F32"||t.shape!=std::vector<std::uint64_t>{64}||t.size_bytes!=256)throw std::runtime_error("unexpected SWA sink");
 std::vector<float> v(64);t.read(0,{reinterpret_cast<std::byte*>(v.data()),256});return mx::array(v.begin(),{64},mx::float32);
}
}
SwaLayerState::SwaLayerState():rows_(mx::zeros({0,512},mx::bfloat16)){}
void SwaLayerState::reset(){rows_=mx::zeros({0,512},mx::bfloat16);position_=0;}
SwaLayerReference::SwaLayerReference(WeightCatalog& c,int layer):layer_(layer),input_(c,layer),output_(c,pure_swa_prefix(layer)+".attn.wo_b"),grouped_weight_(grouped_weight(c,layer)),sink_(sink(c,layer)){
 if(output_.input_dims()!=8192||output_.output_dims()!=5120||output_.bits()!=8)throw std::runtime_error("unexpected SWA wo_b layout");
 mx::eval(grouped_weight_,sink_);
}
mx::array SwaLayerReference::forward(const mx::array& h,SwaLayerState& state,std::uint64_t start) const{
 if(h.dtype()!=mx::bfloat16||h.ndim()!=2||h.shape(1)!=5120||h.shape(0)<1||h.shape(0)>128||
    start!=state.position_||start>=1048576||std::uint64_t(h.shape(0))>1048576-start)
  throw std::runtime_error("invalid SWA layer input/position");
 if(runtime_layer_finite_checks_enabled()){
  auto finite=mx::all(mx::isfinite(h));mx::eval(finite);if(!finite.item<bool>())throw std::runtime_error("nonfinite SWA input");
 }
 const std::string prefix="encoder.layer"+std::to_string(layer_)+".";
 auto next=state;std::vector<mx::array> outputs;outputs.reserve(h.shape(0));
 auto qkv=input_.forward(h,start);
 auto all_rows=mx::concatenate({state.rows_,qkv.kv},0);
 { std::lock_guard l(attention_telemetry_mutex()); auto& t=attention_telemetry(); ++t.concat_calls; t.concat_input_bytes+=(state.rows_.size()+qkv.kv.size())*2; t.concat_output_bytes+=all_rows.size()*2; t.cumulative_bytes_copied+=all_rows.size()*2; }
 for(int i=0;i<h.shape(0);++i){
 { std::lock_guard l(attention_telemetry_mutex()); auto& t=attention_telemetry();t.logical_tokens++;t.attention_rows++;t.token_serial_attention_calls++; }
  auto pos=start+i;
  auto end=state.rows_.shape(0)+i+1;
  auto begin=std::max(0,end-128);
  auto rows=mx::slice(all_rows,{begin,0},{end,512});
  // Decode lists unseen ring slots first, then chronological live rows.
  int pad=pos==0?0:128-rows.shape(0);
  auto ordered=pad?mx::concatenate({mx::zeros({pad,512},mx::bfloat16),rows},0):rows;
  auto valid=mx::greater_equal(mx::arange(ordered.shape(0),mx::int32),mx::array(pad));
  auto query=mx::reshape(mx::slice(qkv.query,{i,0,0},{i+1,64,512}),{64,512});
  auto o=swa_attention_masked_reference(query,ordered,sink_,valid);
  trace_record(prefix+"attn_o_raw",mx::reshape(o,{1,64,512}));
  o=swa_rope_reference(mx::reshape(o,{1,64,512}),pos,true);
  trace_record(prefix+"attn_o",o);
  auto groups=mx::reshape(o,{8,1,4096});
  auto projected=mx::matmul(groups,mx::transpose(grouped_weight_,{0,2,1}));
  auto y=output_.forward(mx::reshape(projected,{1,8192}));
  if(runtime_layer_finite_checks_enabled()){
   auto ok=mx::logical_and(mx::all(mx::isfinite(y)),mx::all(mx::isfinite(rows)));
   mx::eval(y,rows,ok);if(!ok.item<bool>())throw std::runtime_error("nonfinite SWA output/state");
  }
  outputs.push_back(y);
 }
 next.rows_=mx::slice(all_rows,{std::max(0,all_rows.shape(0)-128),0},{all_rows.shape(0),512});
 next.position_=start+h.shape(0);
 auto result=mx::concatenate(outputs,0);
 if(runtime_layer_finite_checks_enabled()) mx::eval(result,next.rows_);
 state=std::move(next);return result;
}
}
