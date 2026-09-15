#include "dsv41/index_query.hpp"
#include "dsv41/kv_quant.hpp"
#include "dsv41/layer_owner.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace dsv41 {
namespace mx=mlx::core;
namespace {
std::vector<std::int32_t> topk_cpu(const std::vector<float>& raw,int offset,const std::vector<std::uint8_t>* mask){
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
 std::sort(order.begin(),order.end(),[&](int a,int b){return p[a]!=p[b]?p[a]>p[b]:a<b;});
 if(n>count&&p[order[count-1]]==p[order[count]])throw std::runtime_error("ambiguous index top-k boundary requires oracle");
 order.resize(count);std::sort(order.begin(),order.end());
 std::vector<std::int32_t> out;for(auto i:order)out.push_back(i+offset);return out;
}
}
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
std::vector<std::int32_t> index_topk_reference(const mx::array& scores,int offset){
 if(scores.ndim()!=1||scores.shape(0)>524288)throw std::runtime_error("invalid index scores");
 auto cpu=mx::astype(scores,mx::float32,mx::Device::cpu);mx::eval(cpu);
 const auto* p=cpu.data<float>();
 std::vector<float> values(p,p+cpu.shape(0));
 return topk_cpu(values,offset,nullptr);
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
 std::sort(order.begin(),order.end(),[&](int a,int b){return block_scores[a]!=block_scores[b]?block_scores[a]>block_scores[b]:a<b;});
 int keep_n=std::min(topk_blocks,num_blocks);
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
 if(uses_candidates_&&(candidate_topk_blocks_<1||candidate_block_size_<1))throw std::runtime_error("invalid candidate config");
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
  result.rows=topk_cpu(raw,offset,&result.candidates);
 }else if(uses_candidates_){
  if(!incoming_candidates||incoming_candidates->size()!=raw.size())throw std::runtime_error("missing candidate mask for candidate consumer");
  result.rows=topk_cpu(raw,offset,incoming_candidates);
 }else{
  result.rows=topk_cpu(raw,offset,nullptr);
 }
 return result;
}
}
