#include "dsv41/swa_attention.hpp"
#include "chunk_attention.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <limits>
namespace dsv41 {
namespace mx=mlx::core;
mx::array swa_attention_reference(const mx::array& q,const mx::array& kv,const mx::array& sink){
 if(kv.ndim()!=2||kv.shape(0)>128)throw std::runtime_error("invalid SWA KV rank/window");
 return swa_attention_masked_reference(q,kv,sink,mx::ones({kv.shape(0)},mx::bool_));
}
mx::array swa_attention_masked_reference(const mx::array& q,const mx::array& kv,const mx::array& sink,const mx::array& valid){
 if(q.dtype()!=mx::bfloat16||q.shape()!=mx::Shape({64,512})||kv.dtype()!=mx::bfloat16||
    kv.ndim()!=2||kv.shape(0)<1||kv.shape(0)>640||kv.shape(1)!=512||
    sink.dtype()!=mx::float32||sink.shape()!=mx::Shape({64})||valid.dtype()!=mx::bool_||valid.shape()!=mx::Shape({kv.shape(0)}))
  throw std::runtime_error("invalid SWA reference query/KV/sink geometry");
 auto qf=mx::astype(q,mx::float32);
 auto maximum=mx::full({64,1},-1e30f,mx::float32);
 auto denominator=mx::zeros({64,1},mx::float32), accumulated=mx::zeros({64,512},mx::float32);
 for(int first=0;first<kv.shape(0);first+=64){
  auto mask=mx::slice(valid,{first},{std::min(first+64,kv.shape(0))});
  auto keys=mx::where(mx::expand_dims(mask,-1),mx::astype(mx::slice(kv,{first,0},{std::min(first+64,kv.shape(0)),512}),mx::float32),mx::array(0.0f));
  auto scores=mx::multiply(mx::matmul(qf,mx::transpose(keys)),mx::array(float(std::pow(512.0,-0.5))));
  scores=mx::where(mask,scores,mx::array(-std::numeric_limits<float>::infinity()));
  auto next_max=mx::maximum(maximum,mx::max(scores,-1,true));
  auto rescale=mx::exp(mx::subtract(maximum,next_max));
  auto exponent=mx::exp(mx::subtract(scores,next_max));
  denominator=mx::add(mx::multiply(denominator,rescale),mx::sum(exponent,-1,true));
  // Official casts unnormalized exponentials to BF16 before value GEMM.
  auto rounded=mx::astype(mx::astype(exponent,mx::bfloat16),mx::float32);
  accumulated=mx::add(mx::multiply(accumulated,rescale),mx::matmul(rounded,keys));
  maximum=next_max;
 }
 denominator=mx::add(denominator,mx::exp(mx::subtract(mx::expand_dims(sink,-1),maximum)));
 return mx::astype(mx::divide(accumulated,denominator),mx::bfloat16);
}
mx::array swa_attention_masked_chunk(const mx::array& q,const mx::array& kv,
 const mx::array& sink,const mx::array& valid){
 if(q.dtype()!=mx::bfloat16||q.ndim()!=3||q.shape(0)<1||q.shape(0)>128||q.shape(1)!=64||q.shape(2)!=512||
    kv.dtype()!=mx::bfloat16||kv.ndim()!=3||kv.shape(0)!=q.shape(0)||kv.shape(1)<1||kv.shape(1)>640||kv.shape(2)!=512||
    sink.dtype()!=mx::float32||sink.shape()!=mx::Shape({64})||valid.dtype()!=mx::bool_||
    valid.shape()!=mx::Shape({q.shape(0),kv.shape(1)}))throw std::runtime_error("invalid chunk attention geometry");
 const int tokens=q.shape(0),rows=kv.shape(1);
 static auto kernel=mx::fast::metal_kernel("dsv41_chunk_attention",
  {"q","kv","sink","valid","params"},{"output"},dsv41_chunk_attention_source);
 return kernel({q,kv,sink,valid,mx::array({std::uint32_t(tokens),std::uint32_t(rows)})},
  {{tokens,64,512}},{mx::bfloat16},{tokens*8*256,1,1},{256,1,1},{},
  std::nullopt,false,mx::Device::gpu).front();
}
}
