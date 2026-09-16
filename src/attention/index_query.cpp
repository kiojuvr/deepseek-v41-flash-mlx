#include "dsv41/index_query.hpp"
#include "dsv41/attention_telemetry.hpp"
#include "dsv41/kv_quant.hpp"
#include "dsv41/layer_owner.hpp"
#include "dsv41/execution_policy.hpp"
#include <algorithm>
#include <bit>
#include <numeric>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace dsv41 {
namespace mx=mlx::core;
namespace {
std::size_t& tie_counter(){static std::size_t count=0;return count;}
std::vector<IndexTieRecord>& tie_records(){static std::vector<IndexTieRecord> records;return records;}
constexpr std::size_t kRecordedTieLimit=256;
std::vector<std::int32_t> topk_cpu(const std::vector<float>& raw,int offset,
 const std::vector<std::uint8_t>* mask,bool strict,IndexTieRecord* record){
 int n=int(raw.size());
 if(n==0)return {};
 if(offset<0||offset>128)throw std::runtime_error("invalid index offset");
 for(int i=0;i<n;++i)if(!std::isfinite(raw[i]))throw std::runtime_error("nonfinite index score");
 std::vector<float> p=raw;
 if(mask){
  if(mask->size()!=std::size_t(n))throw std::runtime_error("candidate mask width mismatch");
  for(int i=0;i<n;++i)if(!(*mask)[i])p[i]=-std::numeric_limits<float>::infinity();
 }
 int count=std::min(n,512);
 std::vector<int> order(n);std::iota(order.begin(),order.end(),0);
 auto better=[&](int a,int b){return p[a]!=p[b]?p[a]>p[b]:a<b;};
 const int boundary=std::min(n,count+1);
 std::partial_sort(order.begin(),order.begin()+boundary,order.end(),better);
 const bool tied=n>count&&p[order[count-1]]==p[order[count]];
 IndexTieRecord observed=record?*record:IndexTieRecord{};
 if(tied){
  observed.selected_id=order[count-1];observed.excluded_id=order[count];
  observed.selected_score=p[order[count-1]];observed.excluded_score=p[order[count]];
  observed.candidates=mask!=nullptr;observed.tied=true;
  if(record)*record=observed;
  if(strict)throw std::runtime_error(
   "ambiguous index top-k boundary requires oracle: selected_id="+std::to_string(observed.selected_id)+
   " excluded_id="+std::to_string(observed.excluded_id)+
   " score_bits="+std::to_string(std::bit_cast<std::uint32_t>(observed.selected_score)));
  ++tie_counter();if(tie_records().size()<kRecordedTieLimit)tie_records().push_back(observed);
 }
 order.resize(count);std::sort(order.begin(),order.end());
 std::vector<std::int32_t> out;for(auto i:order)out.push_back(i+offset);return out;
}
}
std::size_t index_tie_count(){return tie_counter();}
void reset_index_tie_count(){tie_counter()=0;}
std::vector<IndexTieRecord> index_tie_records(){return tie_records();}
void reset_index_tie_records(){tie_records().clear();}
mlx::core::array restore_index_reference(const GlobalKVState& state){
 int rows=int(state.rows());
 auto packed=state.index_bytes();
 auto low=mx::bitwise_and(packed,mx::array(15,mx::uint8));
 auto high=mx::right_shift(packed,mx::array(4,mx::uint8));
 auto codes=mx::reshape(mx::stack({low,high},-1),{rows,128});
 auto levels=mx::array({0.0f,0.5f,1.0f,1.5f,2.0f,3.0f,4.0f,6.0f});
 auto magnitude=mx::take(levels,mx::astype(mx::bitwise_and(codes,mx::array(7,mx::uint8)),mx::int32));
 auto value=mx::where(mx::greater_equal(codes,mx::array(8,mx::uint8)),mx::negative(magnitude),magnitude);
 auto bits=mx::left_shift(mx::astype(state.index_scales(),mx::uint32),mx::array(23,mx::uint32));
 auto scales=mx::view(bits,mx::float32);
 return mx::astype(mx::reshape(mx::multiply(mx::reshape(value,{rows,4,32}),mx::expand_dims(scales,-1)),{rows,128}),mx::bfloat16);
}
std::vector<std::int32_t> index_topk_reference(const mx::array& scores,int offset,bool strict,IndexTieRecord* tie){
 if(scores.ndim()!=1||scores.shape(0)>524288)throw std::runtime_error("invalid index scores");
 auto cpu=mx::astype(scores,mx::float32,mx::Device::cpu);mx::eval(cpu);
 const auto* p=cpu.data<float>();
 std::vector<float> values(p,p+cpu.shape(0));
 return topk_cpu(values,offset,nullptr,strict,tie);
}
std::vector<std::uint8_t> select_candidate_blocks_reference(const std::vector<float>& logits,
 int compress_len,int topk_blocks,int block_size){
 int width=int(logits.size());
 if(width==0)return {};
 if(compress_len<1||compress_len>width||topk_blocks<1||block_size<1)throw std::runtime_error("invalid candidate block selection");
 int num_blocks=(width+block_size-1)/block_size;
 const float inf=std::numeric_limits<float>::infinity();
 std::vector<float> block_scores(num_blocks,-inf);
 for(int b=0;b<num_blocks;++b){
  float best=-inf;
  for(int i=0;i<block_size;++i){int idx=b*block_size+i;if(idx<width)best=std::max(best,logits[idx]);}
  block_scores[b]=best;
 }
 // Pin the block holding this query's newest position, which may be only partly filled.
 int last=(compress_len-1)/block_size;
 if(last>=0&&last<num_blocks)block_scores[last]=inf;
 std::vector<int> order(num_blocks);std::iota(order.begin(),order.end(),0);
 int keep_n=std::min(topk_blocks,num_blocks);
 std::partial_sort(order.begin(),order.begin()+keep_n,order.end(),
  [&](int a,int b){return block_scores[a]!=block_scores[b]?block_scores[a]>block_scores[b]:a<b;});
 std::vector<std::uint8_t> block_keep(num_blocks,0);
 for(int i=0;i<keep_n;++i){int b=order[i];if(block_scores[b]>-inf)block_keep[b]=1;}
 std::vector<std::uint8_t> mask(width,0);
 for(int b=0;b<num_blocks;++b)if(block_keep[b])for(int i=0;i<block_size;++i){int idx=b*block_size+i;if(idx<width)mask[idx]=1;}
 return mask;
}
IndexQueryReference::IndexQueryReference(WeightCatalog& c,int layer,
 bool is_candidate_source,bool uses_candidates,int candidate_topk_blocks,int candidate_block_size)
 :layer_(layer),ratio_(layer_compress_ratio(layer)),candidate_topk_blocks_(candidate_topk_blocks),candidate_block_size_(candidate_block_size),
  is_candidate_source_(is_candidate_source),uses_candidates_(uses_candidates),
  query_(c,("layers."+std::to_string(layer))+".attn.indexer.wq_b"),weights_(mx::array(0)){
 if(!is_index_source_layer(layer))throw std::runtime_error("index query requires an index source layer");
 if(is_candidate_source_&&uses_candidates_)throw std::runtime_error("candidate source cannot also use candidates");
 if((is_candidate_source_||uses_candidates_)&&(candidate_topk_blocks_<1||candidate_block_size_<1))
  throw std::runtime_error("invalid candidate config");
 auto t=c.tensor(("layers."+std::to_string(layer))+".attn.indexer.weights_proj.weight");
 if(t.dtype!="BF16"||t.shape!=std::vector<std::uint64_t>{32,5120}||t.size_bytes!=32*5120*2||query_.input_dims()!=1280||query_.output_dims()!=4096||query_.bits()!=8)throw std::runtime_error("unexpected index query weight");
 std::vector<std::uint16_t> data(t.size_bytes/2);t.read(0,{reinterpret_cast<std::byte*>(data.data()),t.size_bytes});
 weights_=mx::view(mx::array(data.begin(),{32,5120},mx::uint16),mx::bfloat16);
}
IndexSelection IndexQueryReference::forward(const mx::array& x,const mx::array& qr,const GlobalKVState& state,std::uint64_t pos,int offset,const std::vector<std::uint8_t>* incoming_candidates) const{
 if(x.dtype()!=mx::bfloat16||x.shape()!=mx::Shape({1,5120})||qr.dtype()!=mx::bfloat16||qr.shape()!=mx::Shape({1,1280})||
    pos>=1048576||state.position()!=pos+1||state.rows()!=(pos+1)/std::uint64_t(ratio_)||offset<0||offset>128)throw std::runtime_error("index query requires cache through current token");
 IndexSelection result;
 if(state.rows()==0)return result;
 { std::lock_guard l(attention_telemetry_mutex()); attention_telemetry().indexer_rows+=state.rows(); }
 auto q=compressed_rope_reference(mx::reshape(query_.forward(qr),{1,32,128}),std::span(&pos,1));
 q=kv_quant_reference(mx::reshape(q,{32,128}),KVQuantFormat::IndexE8M0).decoded;
 auto k=restore_index_reference(state);
 auto weights=mx::astype(mx::multiply(mx::matmul(x,mx::transpose(weights_)),mx::array(float(std::pow(128.0,-0.5)*std::pow(32.0,-0.5)))),mx::bfloat16);
 auto scores=mx::matmul(q,mx::transpose(k));
 auto weighted=mx::astype(mx::multiply(mx::maximum(scores,mx::array(0,mx::bfloat16)),mx::reshape(weights,{32,1})),mx::bfloat16);
 auto combined=mx::astype(mx::sum(weighted,0),mx::float32,mx::Device::cpu);mx::eval(combined);
 const auto* p=combined.data<float>();
 std::vector<float> raw(p,p+combined.shape(0));
 if(is_candidate_source_){
  result.candidates=select_candidate_blocks_reference(raw,int(state.rows()),candidate_topk_blocks_,candidate_block_size_);
  IndexTieRecord tie{layer_,pos};
  result.rows=topk_cpu(raw,offset,&result.candidates,false,&tie);
 }else if(uses_candidates_){
  if(!incoming_candidates||incoming_candidates->size()!=raw.size())throw std::runtime_error("missing candidate mask for candidate consumer");
  IndexTieRecord tie{layer_,pos};
  result.rows=topk_cpu(raw,offset,incoming_candidates,false,&tie);
 }else{
  IndexTieRecord tie{layer_,pos};
  result.rows=topk_cpu(raw,offset,nullptr,false,&tie);
 }
 return result;
}
std::vector<IndexSelection> IndexQueryReference::forward_chunk(const mx::array& x,const mx::array& qr,
 const std::vector<GlobalKVState>& caches,std::uint64_t start,
 const std::vector<std::vector<std::uint8_t>>* incoming_candidates,
 const std::vector<mx::array>* incoming_device_candidates,bool force_diagnostics) const{
 if(x.dtype()!=mx::bfloat16||x.ndim()!=2||x.shape(0)<1||x.shape(0)>128||x.shape(1)!=5120||
    qr.dtype()!=mx::bfloat16||qr.shape()!=mx::Shape({x.shape(0),1280})||caches.size()!=std::size_t(x.shape(0))||
    start>=1048576||std::uint64_t(x.shape(0))>1048576-start||
    (incoming_candidates&&incoming_candidates->size()!=caches.size())||
    (incoming_device_candidates&&incoming_device_candidates->size()!=caches.size())||
    (uses_candidates_&&!incoming_device_candidates&&!incoming_candidates))
  throw std::runtime_error("invalid index query chunk");
 const int tokens=x.shape(0),rows=int(caches.back().rows());
 std::vector<int> visible;visible.reserve(tokens);
 for(int i=0;i<tokens;++i){
  const auto pos=start+std::uint64_t(i);const auto expected=(pos+1)/std::uint64_t(ratio_);
  if(caches[i].position()!=pos+1||caches[i].rows()!=expected||int(caches[i].rows())>rows)
   throw std::runtime_error("invalid index cache prefix");
  visible.push_back(int(caches[i].rows()));
 }
 std::vector<IndexSelection> results(tokens);
 if(!rows){for(auto& result:results){result.device_rows=mx::zeros({0},mx::int32);
  result.device_candidates=mx::zeros({0},mx::uint8);}return results;}
 {
  std::lock_guard l(attention_telemetry_mutex());auto& telemetry=attention_telemetry();
  for(auto count:visible)telemetry.indexer_rows+=std::size_t(count);
 }
 std::vector<std::uint64_t> positions;positions.reserve(tokens);
 for(int i=0;i<tokens;++i)positions.push_back(start+std::uint64_t(i));
 auto projected=compressed_rope_reference(mx::reshape(query_.forward(qr),{tokens,32,128}),positions);
 std::vector<mx::array> quantized_q;quantized_q.reserve(tokens);
 for(int i=0;i<tokens;++i)
  quantized_q.push_back(mx::reshape(kv_quant_reference(
   mx::reshape(mx::slice(projected,{i,0,0},{i+1,32,128}),{32,128}),KVQuantFormat::IndexE8M0).decoded,
   {1,32,128}));
 auto q=tokens==1?quantized_q.front():mx::concatenate(quantized_q,0);
 auto k=restore_index_reference(caches.back());
 std::vector<mx::array> weight_rows;weight_rows.reserve(tokens);
 const auto scale=mx::array(float(std::pow(128.0,-0.5)*std::pow(32.0,-0.5)));
 for(int i=0;i<tokens;++i)
  weight_rows.push_back(mx::matmul(mx::slice(x,{i,0},{i+1,5120}),mx::transpose(weights_)));
 auto weight_projection=tokens==1?weight_rows.front():mx::concatenate(weight_rows,0);
 auto weights=mx::astype(mx::multiply(weight_projection,scale),mx::bfloat16);
 auto scores=mx::reshape(mx::matmul(mx::reshape(q,{tokens*32,128}),mx::transpose(k)),{tokens,32,rows});
 auto weighted=mx::astype(mx::multiply(mx::maximum(scores,mx::array(0,mx::bfloat16)),
                                       mx::reshape(weights,{tokens,32,1})),mx::bfloat16);
 auto raw_combined=mx::astype(mx::sum(weighted,1),mx::float32);
 auto causal=mx::less(mx::reshape(mx::arange(rows,mx::int32),{1,rows}),
                      mx::reshape(mx::array(visible.begin(),{tokens},mx::int32),{tokens,1}));
 auto combined=mx::where(causal,raw_combined,mx::array(-std::numeric_limits<float>::infinity()));
 auto errors=mx::any(mx::logical_and(causal,mx::logical_not(mx::isfinite(raw_combined))),1);
 mx::array candidate_mask=causal;
 if(is_candidate_source_){
  const int blocks=(rows+candidate_block_size_-1)/candidate_block_size_;
  const int padded=blocks*candidate_block_size_;
  auto padded_scores=rows==padded?combined:mx::concatenate(
   {combined,mx::full({tokens,padded-rows},-std::numeric_limits<float>::infinity(),mx::float32)},1);
  auto block_scores=mx::max(mx::reshape(padded_scores,{tokens,blocks,candidate_block_size_}),2);
  std::vector<std::uint32_t> last_blocks;last_blocks.reserve(tokens);
  for(auto count:visible)last_blocks.push_back(std::uint32_t((count-1)/candidate_block_size_));
  auto last=mx::array(last_blocks.begin(),{tokens,1},mx::uint32);
  block_scores=mx::put_along_axis(block_scores,last,
                                  mx::full({tokens,1},std::numeric_limits<float>::infinity()),1);
  const int keep=std::min(candidate_topk_blocks_,blocks);
  auto block_order=mx::slice(mx::argsort(mx::negative(block_scores),1),{0,0},{tokens,keep});
  auto selected_block_scores=mx::take_along_axis(block_scores,block_order,1);
  auto keep_values=mx::astype(mx::greater(selected_block_scores,
                              mx::array(-std::numeric_limits<float>::infinity())),mx::uint8);
  auto block_mask=mx::put_along_axis(mx::zeros({tokens,blocks},mx::uint8),block_order,keep_values,1);
  candidate_mask=mx::slice(mx::reshape(mx::broadcast_to(mx::expand_dims(block_mask,2),
                                  {tokens,blocks,candidate_block_size_}),{tokens,padded}),
                           {0,0},{tokens,rows});
  combined=mx::where(mx::astype(candidate_mask,mx::bool_),combined,
                     mx::array(-std::numeric_limits<float>::infinity()));
 }else if(uses_candidates_){
  if(incoming_device_candidates){
   std::vector<mx::array> masks;masks.reserve(tokens);
   for(int token=0;token<tokens;++token){
    const auto& incoming=(*incoming_device_candidates)[token];
    if(incoming.dtype()!=mx::uint8||incoming.shape()!=mx::Shape({visible[token]}))
     throw std::runtime_error("missing device candidate mask for candidate consumer");
    auto padded=visible[token]==rows?incoming:mx::concatenate(
     {incoming,mx::zeros({rows-visible[token]},mx::uint8)},0);
    masks.push_back(mx::expand_dims(padded,0));
   }
   candidate_mask=tokens==1?masks.front():mx::concatenate(masks,0);
  }else{
   std::vector<std::uint8_t> host_mask(std::size_t(tokens)*rows,0);
   for(int token=0;token<tokens;++token){
    const auto& incoming=(*incoming_candidates)[token];
    if(incoming.size()!=std::size_t(visible[token]))throw std::runtime_error("missing candidate mask for candidate consumer");
    std::copy(incoming.begin(),incoming.end(),host_mask.begin()+std::size_t(token)*rows);
   }
   candidate_mask=mx::array(host_mask.begin(),{tokens,rows},mx::uint8);
  }
  combined=mx::where(mx::astype(candidate_mask,mx::bool_),combined,
                     mx::array(-std::numeric_limits<float>::infinity()));
 }
 // Stable ascending sort of negative scores is score-descending with the
 // lowest row ID first at exact ties, matching the runtime tie contract.
 const int selected_width=std::min(rows,512),boundary_width=std::min(rows,513);
 auto score_order=mx::slice(mx::argsort(mx::negative(combined),1),{0,0},{tokens,boundary_width});
 auto ordered_scores=mx::take_along_axis(combined,score_order,1);
 auto selected=mx::sort(mx::slice(score_order,{0,0},{tokens,selected_width}),1);
 for(int token=0;token<tokens;++token){
  const int count=std::min(visible[token],512);
  results[token].device_rows=mx::astype(mx::reshape(
   mx::slice(selected,{token,0},{token+1,count}),{count}),mx::int32);
  results[token].device_candidates=(is_candidate_source_||uses_candidates_)?
   mx::astype(mx::reshape(mx::slice(candidate_mask,{token,0},{token+1,visible[token]}),
                          {visible[token]}),mx::uint8):mx::zeros({0},mx::uint8);
 }
 if(!force_diagnostics&&!runtime_index_diagnostics_enabled())return results;
 { std::lock_guard l(attention_telemetry_mutex());++attention_telemetry().index_host_readbacks; }
 auto selected_cpu=mx::contiguous(mx::astype(selected,mx::uint32,mx::Device::cpu),false,mx::Device::cpu);
 auto order_cpu=mx::contiguous(mx::astype(score_order,mx::uint32,mx::Device::cpu),false,mx::Device::cpu);
 auto score_cpu=mx::contiguous(mx::astype(ordered_scores,mx::float32,mx::Device::cpu),false,mx::Device::cpu);
 auto error_cpu=mx::contiguous(mx::astype(errors,mx::uint8,mx::Device::cpu),false,mx::Device::cpu);
 mx::array candidate_cpu(0);
 if(is_candidate_source_)candidate_cpu=mx::contiguous(
  mx::astype(candidate_mask,mx::uint8,mx::Device::cpu),false,mx::Device::cpu);
 if(is_candidate_source_)mx::eval(selected_cpu,order_cpu,score_cpu,error_cpu,candidate_cpu);
 else mx::eval(selected_cpu,order_cpu,score_cpu,error_cpu);
 const auto* selected_data=selected_cpu.data<std::uint32_t>();
 const auto* order_data=order_cpu.data<std::uint32_t>();
 const auto* score_data=score_cpu.data<float>();
 const auto* error_data=error_cpu.data<std::uint8_t>();
 const auto* candidate_data=is_candidate_source_?candidate_cpu.data<std::uint8_t>():nullptr;
 for(int token=0;token<tokens;++token){
  if(error_data[token])throw std::runtime_error("nonfinite index score");
  const auto pos=start+std::uint64_t(token);const int offset=pos==0?1:128;
  const int count=std::min(visible[token],512);
  results[token].rows.reserve(count);
  for(int i=0;i<count;++i)results[token].rows.push_back(
   std::int32_t(selected_data[std::size_t(token)*selected_width+i])+offset);
  if(is_candidate_source_)
   results[token].candidates.assign(candidate_data+std::size_t(token)*rows,
                                    candidate_data+std::size_t(token)*rows+visible[token]);
  if(visible[token]>count){
   const auto selected_score=score_data[std::size_t(token)*boundary_width+count-1];
   const auto excluded_score=score_data[std::size_t(token)*boundary_width+count];
   if(selected_score==excluded_score){
    IndexTieRecord tie{layer_,pos,
     int(order_data[std::size_t(token)*boundary_width+count-1]),
     int(order_data[std::size_t(token)*boundary_width+count]),
     selected_score,excluded_score,is_candidate_source_||uses_candidates_,true};
    ++tie_counter();if(tie_records().size()<kRecordedTieLimit)tie_records().push_back(tie);
   }
  }
 }
 return results;
}
}
