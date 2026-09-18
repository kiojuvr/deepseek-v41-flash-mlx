#include "dsv41/swa_attention.hpp"
#include "dsv41/attention_telemetry.hpp"
#include "dsv41/execution_policy.hpp"
#include "batched_splitk_qk.hpp"
#include "ragged_tail_qk.hpp"
#include "ragged_tail_av.hpp"
#include "packed_chunk_attention.hpp"
#include "wide_chunk_attention.hpp"
#include "packed_attention_worklist.hpp"
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
mx::array ragged_tail_qk(const mx::array& queries,const mx::array& keys,
 const mx::array& widths){
 const int tokens=queries.shape(0),rows=keys.shape(1);
 static auto splitk=mx::fast::metal_kernel("dsv41_ragged_tail_qk",
  {"queries","keys","widths","meta"},{"partial"},dsv41_ragged_tail_qk_source,
  dsv41_batched_splitk_header);
 static auto accumulate=mx::fast::metal_kernel("dsv41_ragged_tail_accum",
  {"partial","widths","meta"},{"scores"},dsv41_ragged_tail_accum_source);
 auto run=[&](int bn,int partitions,int minimum,int maximum){
  const int tiles=(maximum+bn-1)/bn;
  auto partial=splitk({queries,keys,widths,mx::array({tokens,rows},mx::int32)},
   {{tokens,partitions,64,64}},{mx::float32},
   {tiles*32,2*2,tokens*partitions*2},{32,2,2},
   {{"BN",bn},{"PARTITIONS",partitions},{"MIN_WIDTH",minimum},{"MAX_WIDTH",maximum}},
   std::nullopt,false,mx::Device::gpu).front();
  return accumulate({partial,widths,mx::array({tokens},mx::int32)},
   {{tokens,64,64}},{mx::float32},{64*64,tokens,1},{256,1,1},
   {{"PARTITIONS",partitions},{"MIN_WIDTH",minimum},{"MAX_WIDTH",maximum}},
   std::nullopt,false,mx::Device::gpu).front();
 };
 auto small=run(16,16,1,32),middle=run(16,8,33,39),large=run(32,8,40,63);
 auto remainder=mx::remainder(widths,mx::array(64,mx::int32));
 auto small_mask=mx::reshape(mx::less_equal(remainder,mx::array(32,mx::int32)),{tokens,1,1});
 auto middle_mask=mx::reshape(mx::less_equal(remainder,mx::array(39,mx::int32)),{tokens,1,1});
 return mx::where(small_mask,small,mx::where(middle_mask,middle,large));
}
mx::array ragged_tail_av(const mx::array& probabilities,const mx::array& keys,
 const mx::array& widths,int block){
 const int tokens=probabilities.shape(0),rows=keys.shape(1);
 static auto kernel=mx::fast::metal_kernel("dsv41_ragged_tail_av",
  {"probabilities","keys","widths","meta"},{"output"},dsv41_ragged_tail_av_source,
  dsv41_batched_splitk_header);
 return kernel({probabilities,keys,widths,mx::array({tokens,rows,block},mx::int32)},
  {{tokens,64,512}},{mx::float32},{16*32,2*2,tokens*2},{32,2,2},{},
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
mx::array swa_wide_attention_chunk(const mx::array& q,const mx::array& local,
 const mx::array& pooled_values,const mx::array& pooled_scales,const mx::array& topk,
 const mx::array& sink,std::uint64_t start,int ratio){
 if(q.dtype()!=mx::bfloat16||q.ndim()!=3||q.shape(0)<1||q.shape(0)>128||
    q.shape(1)!=64||q.shape(2)!=512||local.dtype()!=mx::bfloat16||local.ndim()!=2||
    local.shape(0)<q.shape(0)||local.shape(0)>256||local.shape(1)!=512||
    pooled_values.dtype()!=mx::uint8||pooled_values.ndim()!=2||pooled_values.shape(1)!=256||
    pooled_scales.dtype()!=mx::uint8||pooled_scales.shape()!=mx::Shape({pooled_values.shape(0),32})||
    topk.dtype()!=mx::int32||topk.ndim()!=2||topk.shape(0)!=q.shape(0)||
    topk.shape(1)<1||topk.shape(1)>512||sink.dtype()!=mx::float32||
    sink.shape()!=mx::Shape({64})||start>=1048576||
    std::uint64_t(q.shape(0))>1048576-start||ratio<1)
  throw std::runtime_error("invalid wide chunk attention geometry");
 static auto kernel=mx::fast::metal_kernel("dsv41_wide_chunk_attention",
  {"queries","local_kv","pooled_values","pooled_scales","topk","sinks","meta","scale"},
  {"output"},dsv41_wide_chunk_attention_source,dsv41_packed_chunk_attention_header);
 const int tokens=q.shape(0);
 return kernel({mx::contiguous(q,false,mx::Device::gpu),
                mx::contiguous(local,false,mx::Device::gpu),
                mx::contiguous(pooled_values,false,mx::Device::gpu),
                mx::contiguous(pooled_scales,false,mx::Device::gpu),
                mx::contiguous(topk,false,mx::Device::gpu),sink,
                mx::array({tokens,local.shape(0),pooled_values.shape(0),int(start),ratio,topk.shape(1)},mx::int32),
                mx::array(float(std::pow(512.0,-0.5)))},
               {{tokens,64,512}},{mx::bfloat16},{tokens*32,64,1},{32,8,1},{},
               std::nullopt,false,mx::Device::gpu).front();
}
PackedAttentionWorkList swa_packed_attention_work_list(const mx::array& local,
 const mx::array& pooled_values,const mx::array& pooled_scales,const mx::array& topk,
 std::uint64_t start,int ratio,int selected_count,bool fixed_window){
 if(local.dtype()!=mx::bfloat16||local.ndim()!=2||local.shape(0)<1||local.shape(0)>256||
    local.shape(1)!=512||pooled_values.dtype()!=mx::uint8||pooled_values.ndim()!=2||
    pooled_values.shape(1)!=256||pooled_scales.dtype()!=mx::uint8||
    pooled_scales.shape()!=mx::Shape({pooled_values.shape(0),32})||topk.dtype()!=mx::int32||
    topk.ndim()!=2||topk.shape(0)<1||topk.shape(0)>128||topk.shape(1)<1||topk.shape(1)>512||
    local.shape(0)<topk.shape(0)||start>=1048576||std::uint64_t(topk.shape(0))>1048576-start||ratio<1||
    selected_count<-1||selected_count>topk.shape(1)||(selected_count==0&&topk.shape(1)!=1))
  throw std::runtime_error("invalid packed attention work-list geometry");
 if(selected_count<0)selected_count=topk.shape(1);
 const int tokens=topk.shape(0),window=fixed_window?128:(start==0?std::min(128,tokens):128);
 const int rows=window+selected_count;
 static auto kernel=mx::fast::metal_kernel("dsv41_packed_attention_worklist",
  {"local_kv","pooled_values","pooled_scales","topk","meta"},{"ordered","valid","widths"},
  dsv41_packed_attention_worklist_source);
 auto result=kernel({mx::contiguous(local,false,mx::Device::gpu),
                     mx::contiguous(pooled_values,false,mx::Device::gpu),
                     mx::contiguous(pooled_scales,false,mx::Device::gpu),
                     mx::contiguous(topk,false,mx::Device::gpu),
                     mx::array({tokens,local.shape(0),pooled_values.shape(0),int(start),ratio,selected_count,
                                fixed_window?1:0},mx::int32)},
                    {{tokens,rows,512},{tokens,rows},{tokens}},{mx::bfloat16,mx::bool_,mx::int32},
                    {512,rows,tokens},{32,1,1},{},std::nullopt,false,mx::Device::gpu);
 return {result[0],result[1],result[2]};
}
mx::array swa_fixed_tile_attention_chunk(const mx::array& q,const mx::array& local,
 const mx::array& pooled_values,const mx::array& pooled_scales,const mx::array& topk,
 const mx::array& sink,std::uint64_t start,int ratio){
 if(topk.dtype()!=mx::int32||topk.ndim()!=2||topk.shape(0)!=q.shape(0)||topk.shape(1)!=512)
  throw std::runtime_error("fixed-tile attention requires a dense [tokens,512] plan");
 auto work=swa_packed_attention_work_list(local,pooled_values,pooled_scales,topk,start,ratio,512,true);
 if(work.ordered.shape(1)!=640)
  throw std::runtime_error("fixed-tile attention must materialize ten 64-row tiles");
 return runtime_ragged_tail_qk_enabled()?swa_attention_fixed_tile_core(
  q,work,sink,runtime_ragged_tail_av_enabled()):
  swa_attention_masked_chunk(q,work.ordered,sink,work.valid);
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
namespace {
mx::array swa_attention_masked_chunk_impl(const mx::array& q,const mx::array& kv,
 const mx::array& sink,const mx::array& valid,const mx::array* tail_widths,bool ragged_av){
 if(q.dtype()!=mx::bfloat16||q.ndim()!=3||q.shape(0)<1||q.shape(0)>128||q.shape(1)!=64||q.shape(2)!=512||
    kv.dtype()!=mx::bfloat16||kv.ndim()!=3||kv.shape(0)!=q.shape(0)||kv.shape(1)<1||kv.shape(1)>640||kv.shape(2)!=512||
    sink.dtype()!=mx::float32||sink.shape()!=mx::Shape({64})||valid.dtype()!=mx::bool_||
    valid.shape()!=mx::Shape({q.shape(0),kv.shape(1)}))throw std::runtime_error("invalid chunk attention geometry");
 const int tokens=q.shape(0),rows=kv.shape(1);auto qf=mx::astype(q,mx::float32);
 const bool batched_splitk=runtime_batched_splitk_qk_enabled();
 mx::array tail_scores=mx::array(0.0f),tail_blocks=mx::array(0,mx::int32),tail_remainder=mx::array(0,mx::int32);
 if(tail_widths){
  if(tail_widths->dtype()!=mx::int32||tail_widths->shape()!=mx::Shape({tokens})||rows!=640)
   throw std::runtime_error("invalid ragged tail QK metadata");
  tail_scores=ragged_tail_qk(qf,mx::astype(kv,mx::float32),*tail_widths);
  tail_blocks=mx::floor_divide(*tail_widths,mx::array(64,mx::int32));
  tail_remainder=mx::remainder(*tail_widths,mx::array(64,mx::int32));
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
  if(tail_widths&&first>=128){
   const int pooled_block=(first-128)/64;
   auto use_tail=mx::logical_and(mx::equal(tail_blocks,mx::array(pooled_block,mx::int32)),
                                 mx::greater(tail_remainder,mx::array(0,mx::int32)));
   scores=mx::where(mx::reshape(use_tail,{tokens,1,1}),tail_scores,scores);
  }
  scores=mx::multiply(scores,mx::array(float(std::pow(512.0,-0.5))));
  scores=mx::where(mx::expand_dims(mask,1),scores,mx::array(-std::numeric_limits<float>::infinity()));
  auto next_max=mx::maximum(maximum,mx::max(scores,-1,true));auto rescale=mx::exp(mx::subtract(maximum,next_max));
  auto exponent=mx::exp(mx::subtract(scores,next_max));
  denominator=mx::add(mx::multiply(denominator,rescale),mx::sum(exponent,-1,true));
  auto rounded=mx::astype(mx::astype(exponent,mx::bfloat16),mx::float32);
  auto av=mx::matmul(rounded,keys);
  if(tail_widths&&ragged_av&&first>=128){
   const int pooled_block=(first-128)/64;
   auto use_tail=mx::logical_and(mx::equal(tail_blocks,mx::array(pooled_block,mx::int32)),
                                 mx::greater(tail_remainder,mx::array(0,mx::int32)));
   auto exact_tail=ragged_tail_av(rounded,mx::astype(kv,mx::float32),*tail_widths,pooled_block);
   av=mx::where(mx::reshape(use_tail,{tokens,1,1}),exact_tail,av);
  }
  accumulated=mx::add(mx::multiply(accumulated,rescale),av);maximum=next_max;
  { std::lock_guard l(attention_telemetry_mutex());++attention_telemetry().chunk_av_batches; }
 }
 denominator=mx::add(denominator,mx::exp(mx::subtract(mx::reshape(sink,{1,64,1}),maximum)));
 return mx::astype(mx::divide(accumulated,denominator),mx::bfloat16);
}
}
mx::array swa_attention_masked_chunk(const mx::array& q,const mx::array& kv,
 const mx::array& sink,const mx::array& valid){
 return swa_attention_masked_chunk_impl(q,kv,sink,valid,nullptr,false);
}
mx::array swa_attention_fixed_tile_core(const mx::array& q,
 const PackedAttentionWorkList& work,const mx::array& sink,bool ragged_av){
 return swa_attention_masked_chunk_impl(q,work.ordered,sink,work.valid,&work.widths,ragged_av);
}
AttentionTailDiagnostics swa_attention_tail_diagnostics(const mx::array& q,
 const mx::array& kv,const mx::array& sink,const mx::array& valid){
 if(q.dtype()!=mx::bfloat16||q.ndim()!=3||q.shape(0)<1||q.shape(0)>128||
    q.shape(1)!=64||q.shape(2)!=512||kv.dtype()!=mx::bfloat16||kv.ndim()!=3||
    kv.shape(0)!=q.shape(0)||kv.shape(1)<1||kv.shape(1)>640||kv.shape(2)!=512||
    sink.dtype()!=mx::float32||sink.shape()!=mx::Shape({64})||valid.dtype()!=mx::bool_||
    valid.shape()!=mx::Shape({q.shape(0),kv.shape(1)}))
  throw std::runtime_error("invalid attention tail diagnostic geometry");
 const int tokens=q.shape(0),rows=kv.shape(1);auto qf=mx::astype(q,mx::float32);
 auto run=[&](bool pad_qk,bool pad_av){
  auto maximum=mx::full({tokens,64,1},-1e30f,mx::float32);
  auto denominator=mx::zeros({tokens,64,1},mx::float32);
  auto accumulated=mx::zeros({tokens,64,512},mx::float32);
  for(int first=0;first<rows;first+=64){
   const int last=std::min(first+64,rows),width=last-first;
   auto mask=mx::slice(valid,{0,first},{tokens,last});
   auto keys=mx::where(mx::expand_dims(mask,-1),
    mx::astype(mx::slice(kv,{0,first,0},{tokens,last,512}),mx::float32),mx::array(0.0f));
   mx::array scores=mx::array(0.0f);
   if(width==64||(pad_qk&&width<64)){
    auto qk_keys=width==64?keys:mx::concatenate(
     {keys,mx::zeros({tokens,64-width,512},mx::float32)},1);
    scores=batched_splitk_qk(qf,qk_keys);
    if(width<64)scores=mx::slice(scores,{0,0,0},{tokens,64,width});
   }else{
    std::vector<mx::array> score_rows;score_rows.reserve(tokens);
    for(int token=0;token<tokens;++token){
     auto token_q=mx::reshape(mx::slice(qf,{token,0,0},{token+1,64,512}),{64,512});
     auto token_keys=mx::reshape(mx::slice(keys,{token,0,0},{token+1,width,512}),{width,512});
     score_rows.push_back(mx::expand_dims(mx::matmul(token_q,mx::transpose(token_keys)),0));
    }
    scores=mx::concatenate(score_rows,0);
   }
   scores=mx::multiply(scores,mx::array(float(std::pow(512.0,-0.5))));
   scores=mx::where(mx::expand_dims(mask,1),scores,mx::array(-std::numeric_limits<float>::infinity()));
   auto next_max=mx::maximum(maximum,mx::max(scores,-1,true));
   auto rescale=mx::exp(mx::subtract(maximum,next_max));
   auto exponent=mx::exp(mx::subtract(scores,next_max));
   denominator=mx::add(mx::multiply(denominator,rescale),mx::sum(exponent,-1,true));
   auto rounded=mx::astype(mx::astype(exponent,mx::bfloat16),mx::float32);
   if(pad_av&&width<64){
    rounded=mx::concatenate({rounded,mx::zeros({tokens,64,64-width},mx::float32)},2);
    keys=mx::concatenate({keys,mx::zeros({tokens,64-width,512},mx::float32)},1);
   }
   accumulated=mx::add(mx::multiply(accumulated,rescale),mx::matmul(rounded,keys));
   maximum=next_max;
  }
  denominator=mx::add(denominator,mx::exp(mx::subtract(mx::reshape(sink,{1,64,1}),maximum)));
  return mx::astype(mx::divide(accumulated,denominator),mx::bfloat16);
 };
 return {run(true,false),run(false,true)};
}
}
