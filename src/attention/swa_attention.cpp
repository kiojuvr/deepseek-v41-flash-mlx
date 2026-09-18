#include "dsv41/swa_attention.hpp"
#include "dsv41/attention_telemetry.hpp"
#include "dsv41/execution_policy.hpp"
#include "batched_splitk_qk.hpp"
#include "packed_chunk_attention.hpp"
#include <cmath>
#include <algorithm>
#include <bit>
#include <stdexcept>
#include <limits>
namespace dsv41 {
namespace mx=mlx::core;
namespace {
mx::array batched_splitk_qk(const mx::array& queries,const mx::array& keys){
 const int tokens=queries.shape(0),columns=keys.shape(1);
 const int bn=columns<40?16:32;
 const int tiles_n=(columns+bn-1)/bn;
 const int partition_seed=32/(2*((columns+31)/32));
 const int partitions=std::min(std::max(2,int(std::bit_ceil(unsigned(partition_seed)))),32);
 static auto splitk=mx::fast::metal_kernel("dsv41_batched_splitk_qk",
  {"queries","keys","meta"},{"partial"},dsv41_batched_splitk_qk_source,
  dsv41_batched_splitk_header);
 auto partial=splitk({queries,keys,mx::array({keys.shape(1),columns,tiles_n,tokens},mx::int32)},
  {{tokens,partitions,64,columns}},{mx::float32},
  {tiles_n*32,2*2,tokens*partitions*2},{32,2,2},
  {{"BN",bn},{"PARTITIONS",partitions},{"MN_ALIGNED",columns%bn==0}},
  std::nullopt,false,mx::Device::gpu).front();
 static auto accumulate=mx::fast::metal_kernel("dsv41_batched_splitk_accum",
  {"partial","meta"},{"scores"},dsv41_batched_splitk_accum_source);
 return accumulate({partial,mx::array({64*columns,partitions,tokens},mx::int32)},
  {{tokens,64,columns}},{mx::float32},{64*columns,tokens,1},{256,1,1},{},
  std::nullopt,false,mx::Device::gpu).front();
}
}
mx::array swa_packed_attention_chunk(const mx::array& q,const mx::array& local,
 const mx::array& pooled_values,const mx::array& pooled_scales,const mx::array& topk,
 const mx::array& sink,std::uint64_t start,int ratio){
 if(q.dtype()!=mx::bfloat16||q.ndim()!=3||q.shape(0)<1||q.shape(0)>128||
    q.shape(1)!=64||q.shape(2)!=512||local.dtype()!=mx::bfloat16||local.ndim()!=2||
    local.shape(0)<q.shape(0)||local.shape(0)>256||local.shape(1)!=512||
    pooled_values.dtype()!=mx::uint8||pooled_values.ndim()!=2||pooled_values.shape(1)!=256||
    pooled_scales.dtype()!=mx::uint8||pooled_scales.shape()!=mx::Shape({pooled_values.shape(0),32})||
    topk.dtype()!=mx::int32||topk.ndim()!=2||topk.shape(0)!=q.shape(0)||
    topk.shape(1)<1||topk.shape(1)>512||
    sink.dtype()!=mx::float32||sink.shape()!=mx::Shape({64})||start>=1048576||
    std::uint64_t(q.shape(0))>1048576-start||ratio<1)
  throw std::runtime_error("invalid packed chunk attention geometry");
 static auto kernel=mx::fast::metal_kernel("dsv41_packed_chunk_attention",
  {"queries","local_kv","pooled_values","pooled_scales","topk","sinks","meta","scale"},
  {"output"},dsv41_packed_chunk_attention_source,dsv41_packed_chunk_attention_header);
 const int tokens=q.shape(0);
 auto cq=mx::contiguous(q,false,mx::Device::gpu);
 auto cl=mx::contiguous(local,false,mx::Device::gpu);
 auto cp=mx::contiguous(pooled_values,false,mx::Device::gpu);
 auto cs=mx::contiguous(pooled_scales,false,mx::Device::gpu);
 auto ct=mx::contiguous(topk,false,mx::Device::gpu);
 auto metadata=mx::array({tokens,local.shape(0),pooled_values.shape(0),int(start),ratio,topk.shape(1)},mx::int32);
 return kernel({cq,
                cl,cp,cs,ct,sink,metadata,
                mx::array(float(std::pow(512.0,-0.5)))},
               {{tokens,64,512}},{mx::bfloat16},{tokens*32,8,1},{32,8,1},{},
               std::nullopt,false,mx::Device::gpu).front();
}
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
 const bool batched_splitk=runtime_batched_splitk_qk_enabled();
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
  mx::array scores=mx::array(0.0f);
  // The standalone custom compilation differs by a few float32 ULPs from
  // native Steel on short-N tails. Batch only complete 64-key blocks; keep
  // every tail on the native scalar oracle.
  if(batched_splitk&&last-first==64){
   scores=batched_splitk_qk(qf,keys);
   if(runtime_batched_splitk_qk_diagnostics_enabled()){
    std::vector<mx::array> rows;rows.reserve(tokens);
    for(int token=0;token<tokens;++token){
     auto token_q=mx::reshape(mx::slice(qf,{token,0,0},{token+1,64,512}),{64,512});
     auto token_keys=mx::reshape(mx::slice(keys,{token,0,0},{token+1,last-first,512}),{last-first,512});
     rows.push_back(mx::expand_dims(mx::matmul(token_q,mx::transpose(token_keys)),0));
    }
    auto oracle=mx::concatenate(rows,0);
    if(scores.shape()!=oracle.shape())throw std::runtime_error("batched split-K QK diagnostic shape mismatch");
    auto equal=mx::all(mx::equal(scores,oracle));mx::eval(equal);
    if(!equal.item<bool>()){
     auto difference=mx::abs(mx::subtract(scores,oracle));
     auto maximum=mx::max(difference),mean=mx::mean(difference);
     auto mismatches=mx::sum(mx::astype(mx::not_equal(scores,oracle),mx::uint32));
     mx::eval(maximum,mean,mismatches);
     throw std::runtime_error("batched split-K QK float32 mismatch: tokens="+
      std::to_string(tokens)+" columns="+std::to_string(last-first)+" max_abs="+
      std::to_string(maximum.item<float>())+" mean_abs="+std::to_string(mean.item<float>())+
      " mismatches="+std::to_string(mismatches.item<std::uint32_t>()));
    }
   }
   { std::lock_guard l(attention_telemetry_mutex());++attention_telemetry().chunk_batched_splitk_qk_calls; }
  }else{
   std::vector<mx::array> score_rows;score_rows.reserve(tokens);
   for(int token=0;token<tokens;++token){
    auto token_q=mx::reshape(mx::slice(qf,{token,0,0},{token+1,64,512}),{64,512});
    auto token_keys=mx::reshape(mx::slice(keys,{token,0,0},{token+1,last-first,512}),{last-first,512});
    score_rows.push_back(mx::expand_dims(mx::matmul(token_q,mx::transpose(token_keys)),0));
   }
   { std::lock_guard l(attention_telemetry_mutex());attention_telemetry().chunk_scalar_qk_calls+=tokens; }
   scores=mx::concatenate(score_rows,0);
  }
  scores=mx::multiply(scores,mx::array(float(std::pow(512.0,-0.5))));
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
