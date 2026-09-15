#include "dsv41/swa_projection.hpp"
#include "dsv41/model_entry.hpp"
#include <stdexcept>
#include "dsv41/layer_owner.hpp"
namespace dsv41 {
namespace mx=mlx::core;
namespace {
mx::array norm_weight(WeightCatalog& catalog,const std::string& name,int width){
 auto t=catalog.tensor(name);
 if(t.dtype!="BF16"||t.shape!=std::vector<std::uint64_t>{std::uint64_t(width)}||t.size_bytes!=std::uint64_t(width)*2)
  throw std::runtime_error("unexpected SWA norm weight");
 std::vector<std::uint16_t> data(width);t.read(0,{reinterpret_cast<std::byte*>(data.data()),t.size_bytes});
 return mx::view(mx::array(data.begin(),{width},mx::uint16),mx::bfloat16);
}
}
mx::array swa_rope_reference(const mx::array& input,std::uint64_t start,bool inverse){
 if(input.dtype()!=mx::bfloat16||(input.ndim()!=2&&input.ndim()!=3)||input.shape(-1)!=512||
    input.shape(0)<1||input.shape(0)>128||(input.ndim()==3&&input.shape(1)!=64)||
    start>=1048576||std::uint64_t(input.shape(0))>1048576-start)
  throw std::runtime_error("invalid pure SWA RoPE input/position");
 auto begin=mx::Shape(input.ndim(),0),end=input.shape();begin.back()=448;
 auto tail=mx::astype(mx::slice(input,begin,end),mx::float32);
 auto pair_shape=tail.shape();pair_shape.back()=32;pair_shape.push_back(2);
 auto pairs=mx::reshape(tail,pair_shape);
 auto real=mx::take(pairs,mx::array(0),-1),imag=mx::take(pairs,mx::array(1),-1);
 auto frequency=mx::divide(mx::array(1.0f),mx::power(mx::array(10000.0f),mx::divide(mx::arange(0,64,2,mx::float32),mx::array(64.0f))));
 auto angle=mx::multiply(mx::reshape(mx::arange(int(start),int(start)+input.shape(0),mx::float32),{input.shape(0),1}),frequency);
 if(input.ndim()==3)angle=mx::expand_dims(angle,1);
 auto c=mx::cos(angle),s=mx::sin(angle);if(inverse)s=mx::negative(s);
 auto rotated=mx::stack({mx::subtract(mx::multiply(real,c),mx::multiply(imag,s)),
                        mx::add(mx::multiply(real,s),mx::multiply(imag,c))},-1);
 begin.back()=0;end.back()=448;
 return mx::concatenate({mx::slice(input,begin,end),mx::astype(mx::reshape(rotated,tail.shape()),mx::bfloat16)},-1);
}
SwaProjectionReference::SwaProjectionReference(WeightCatalog& c,int layer):layer_(layer),
 q_a_(c,pure_swa_prefix(layer)+".attn.wq_a"),q_b_(c,pure_swa_prefix(layer)+".attn.wq_b"),kv_(c,pure_swa_prefix(layer)+".attn.wkv"),
 q_norm_(norm_weight(c,pure_swa_prefix(layer)+".attn.q_norm.weight",1280)),
 kv_norm_(norm_weight(c,pure_swa_prefix(layer)+".attn.kv_norm.weight",512)){
 if(q_a_.input_dims()!=5120||q_a_.output_dims()!=1280||q_b_.input_dims()!=1280||q_b_.output_dims()!=32768||
    kv_.input_dims()!=5120||kv_.output_dims()!=512||q_a_.bits()!=8||q_b_.bits()!=8||kv_.bits()!=8)
  throw std::runtime_error("unexpected layer 0 SWA projection geometry");
}
SwaProjectedQKV SwaProjectionReference::forward(const mx::array& h,std::uint64_t start) const{
 if(h.dtype()!=mx::bfloat16||h.ndim()!=2||h.shape(1)!=5120||h.shape(0)<1||h.shape(0)>128||
    start>=1048576||std::uint64_t(h.shape(0))>1048576-start)
  throw std::runtime_error("invalid SWA projection input/position");
 const std::string prefix="encoder.layer"+std::to_string(layer_)+".";
 auto qr=rms_norm_reference(q_a_.forward(h),q_norm_,1e-20f);
 trace_record(prefix+"attn_qr",qr);
 auto qb=q_b_.forward(qr);
 trace_record(prefix+"attn_qb",qb);
 auto q=swa_rope_reference(mx::reshape(qb,{h.shape(0),64,512}),start);
 trace_record(prefix+"attn_q",q);
 auto kv_normed=rms_norm_reference(kv_.forward(h),kv_norm_,1e-20f);
 trace_record(prefix+"attn_kv_norm",kv_normed);
 auto kv=swa_rope_reference(kv_normed,start);
 trace_record(prefix+"attn_kv",kv);
 auto decoded=linear_activation_reference(kv).decoded;
 trace_record(prefix+"attn_kv_quant",decoded);
 return {q,decoded};
}
}
