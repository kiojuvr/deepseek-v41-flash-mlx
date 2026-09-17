#include "dsv41/swa_attention.hpp"
#include "dsv41/attention_telemetry.hpp"
#include "dsv41/execution_policy.hpp"
#include "fused_chunk_attention.hpp"
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
 const int tokens=q.shape(0),rows=kv.shape(1);auto qf=mx::astype(q,mx::float32);
 if(runtime_fused_chunk_attention_enabled()){
  static auto kernel=mx::fast::metal_kernel("dsv41_fused_chunk_attention",
   {"queries","keys","sinks","valid","meta","scale"},{"output"},
   dsv41_fused_chunk_attention_source);
  auto out=kernel({q,kv,sink,valid,mx::array({std::int32_t(rows)},mx::int32),
                   mx::array({float(std::pow(512.0,-0.5))},mx::float32)},
   {{tokens,64,512}},{mx::bfloat16},{64*32,tokens,1},{32,1,1},{{"T",mx::bfloat16}},std::nullopt,false,
   mx::Device::gpu);
  { std::lock_guard l(attention_telemetry_mutex());++attention_telemetry().chunk_fused_attention_calls; }
  return out[0];
 }
 auto maximum=mx::full({tokens,64,1},-1e30f,mx::float32);
 auto denominator=mx::zeros({tokens,64,1},mx::float32);
 auto accumulated=mx::zeros({tokens,64,512},mx::float32);
 for(int first=0;first<rows;first+=64){
  const int last=std::min(first+64,rows);
  auto mask=mx::slice(valid,{0,first},{tokens,last});
  auto keys=mx::where(mx::expand_dims(mask,-1),
   mx::astype(mx::slice(kv,{0,first,0},{tokens,last,512}),mx::float32),mx::array(0.0f));
  // MLX 0.32.2 sends each scalar-oracle QK shape through Steel split-K,
  // while rank-3 batch QK selects a different regular GEMM reduction. Keep
  // only this qualified reduction token-wise; softmax and AV remain batched.
  std::vector<mx::array> score_rows;score_rows.reserve(tokens);
  for(int token=0;token<tokens;++token){
   auto token_q=mx::reshape(mx::slice(qf,{token,0,0},{token+1,64,512}),{64,512});
   auto token_keys=mx::reshape(mx::slice(keys,{token,0,0},{token+1,last-first,512}),{last-first,512});
   score_rows.push_back(mx::expand_dims(mx::matmul(token_q,mx::transpose(token_keys)),0));
  }
  { std::lock_guard l(attention_telemetry_mutex());attention_telemetry().chunk_scalar_qk_calls+=tokens; }
  auto scores=mx::multiply(mx::concatenate(score_rows,0),mx::array(float(std::pow(512.0,-0.5))));
  scores=mx::where(mx::expand_dims(mask,1),scores,mx::array(-std::numeric_limits<float>::infinity()));
  auto next_max=mx::maximum(maximum,mx::max(scores,-1,true));auto rescale=mx::exp(mx::subtract(maximum,next_max));
  auto exponent=mx::exp(mx::subtract(scores,next_max));
  denominator=mx::add(mx::multiply(denominator,rescale),mx::sum(exponent,-1,true));
  auto rounded=mx::astype(mx::astype(exponent,mx::bfloat16),mx::float32);
  accumulated=mx::add(mx::multiply(accumulated,rescale),mx::matmul(rounded,keys));maximum=next_max;
  { std::lock_guard l(attention_telemetry_mutex());++attention_telemetry().chunk_av_batches; }
 }
 denominator=mx::add(denominator,mx::exp(mx::subtract(mx::reshape(sink,{1,64,1}),maximum)));
 return mx::astype(mx::divide(accumulated,denominator),mx::bfloat16);
}
}
