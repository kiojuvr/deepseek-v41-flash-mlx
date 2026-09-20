#include "dsv41/shared_attention.hpp"
#include "dsv41/layer_owner.hpp"
#include <algorithm>
#include <stdexcept>
namespace dsv41 {
namespace mx=mlx::core;
namespace {
// selected is +offset and position-sorted; store the relative compressed positions.
std::vector<std::int32_t> to_relative(const std::vector<std::int32_t>& selected,std::size_t rows,int offset){
 std::vector<std::int32_t> relative;relative.reserve(selected.size());
 int previous=-1;
 for(auto index:selected){
  int row=index-offset;
  if(row<0||std::size_t(row)>=rows||row<=previous)throw std::runtime_error("invalid shared attention candidates");
  relative.push_back(row);previous=row;
 }
 return relative;
}
}
SharedAttentionReference::SharedAttentionReference(const GlobalKVState& cache,
 std::vector<std::int32_t> selected,std::uint64_t pos,int offset,int source_layer,int ratio)
 :cache_(cache),position_(pos),source_layer_(source_layer),ratio_(ratio),index_source_layer_(source_layer){
 if(!is_kv_source_layer(source_layer)||ratio!=layer_compress_ratio(source_layer))
  throw std::runtime_error("invalid shared attention source layer/ratio");
 if(pos>=1048576||cache.position()!=pos+1||cache.rows()!=(pos+1)/std::uint64_t(ratio)||offset!=(pos==0?1:128)||
    selected.size()!=std::min<std::size_t>(512,cache.rows()))throw std::runtime_error("invalid shared attention publication");
 rows_=to_relative(selected,cache.rows(),offset);
 device_rows_=rows_.empty()?mx::zeros({0},mx::int32):
  mx::array(rows_.begin(),{int(rows_.size())},mx::int32);
 device_candidates_=mx::zeros({0},mx::uint8);
}
SharedAttentionReference::SharedAttentionReference(const GlobalKVState& cache,
 mx::array relative_selected,mx::array device_candidates,std::uint64_t pos,int offset,
 int source_layer,int ratio,std::vector<std::int32_t> diagnostic_selected,
 std::vector<std::uint8_t> diagnostic_candidates)
 :cache_(cache),device_rows_(std::move(relative_selected)),
  device_candidates_(std::move(device_candidates)),position_(pos),source_layer_(source_layer),
  ratio_(ratio),index_source_layer_(source_layer){
 if(!is_kv_source_layer(source_layer)||ratio!=layer_compress_ratio(source_layer)||pos>=1048576||
    cache.position()!=pos+1||cache.rows()!=(pos+1)/std::uint64_t(ratio)||offset!=(pos==0?1:128)||
    device_rows_.dtype()!=mx::int32||device_rows_.shape()!=mx::Shape({int(std::min<std::size_t>(512,cache.rows()))})||
    device_candidates_.dtype()!=mx::uint8||
    (device_candidates_.size()!=0&&device_candidates_.shape()!=mx::Shape({int(cache.rows())})))
  throw std::runtime_error("invalid device shared attention publication");
 if(!diagnostic_selected.empty())rows_=to_relative(diagnostic_selected,cache.rows(),offset);
 if(!diagnostic_candidates.empty()){
  if(diagnostic_candidates.size()!=cache.rows())throw std::runtime_error("invalid diagnostic candidate width");
  candidates_=std::move(diagnostic_candidates);
 }
}
std::vector<std::int32_t> SharedAttentionReference::indices(int layer,std::uint64_t pos,int offset) const{
 if(layer<3||layer>=kBackboneLayers||kv_source_for_layer(layer)!=source_layer_||pos!=position_||offset!=(pos==0?1:128))
  throw std::runtime_error("shared attention source/position mismatch");
 std::vector<std::int32_t> out;for(auto row:rows_)out.push_back(row+offset);return out;
}
const mx::array& SharedAttentionReference::device_indices(int layer,std::uint64_t pos,int offset) const{
 if(layer<3||layer>=kBackboneLayers||kv_source_for_layer(layer)!=source_layer_||pos!=position_||
    offset!=(pos==0?1:128))throw std::runtime_error("shared attention device source/position mismatch");
 return device_rows_;
}
void SharedAttentionReference::publish_chunk_plan(int layer,mx::array rows,
 std::uint64_t start,int tokens){
 if(layer!=index_source_layer_||tokens<1||tokens>128||start>=1048576||
    std::uint64_t(tokens)>1048576-start||position_<start||position_>=start+std::uint64_t(tokens)||
    rows.dtype()!=mx::int32||rows.shape()!=mx::Shape({tokens,512}))
  throw std::runtime_error("invalid shared attention chunk plan");
 device_chunk_rows_=std::move(rows);chunk_start_=start;chunk_tokens_=tokens;
 chunk_index_source_layer_=layer;
}
const mx::array& SharedAttentionReference::device_chunk_indices(int layer,
 std::uint64_t start,int tokens) const{
 if(layer<3||layer>=kBackboneLayers||kv_source_for_layer(layer)!=source_layer_||
    index_source_layer_!=chunk_index_source_layer_||start!=chunk_start_||tokens!=chunk_tokens_||
    device_chunk_rows_.shape()!=mx::Shape({tokens,512}))
  throw std::runtime_error("shared attention chunk plan mismatch");
 return device_chunk_rows_;
}
bool SharedAttentionReference::chunk_plan_matches(int layer,std::uint64_t start,int tokens) const{
 return layer>=3&&layer<kBackboneLayers&&kv_source_for_layer(layer)==source_layer_&&
  index_source_layer_==chunk_index_source_layer_&&start==chunk_start_&&tokens==chunk_tokens_&&
  device_chunk_rows_.shape()==mx::Shape({tokens,512});
}
void SharedAttentionReference::republish(int index_source_layer,std::vector<std::int32_t> selected,
 std::vector<std::uint8_t> candidates){
 if(!is_index_source_layer(index_source_layer)||kv_source_for_layer(index_source_layer)!=source_layer_)
  throw std::runtime_error("republish requires an index source in this kv group");
 if(selected.size()!=std::min<std::size_t>(512,cache_.rows()))throw std::runtime_error("invalid republished row count");
 rows_=to_relative(selected,cache_.rows(),position_==0?1:128);
 device_rows_=rows_.empty()?mx::zeros({0},mx::int32):
  mx::array(rows_.begin(),{int(rows_.size())},mx::int32);
 // The candidate mask belongs to the candidate source and persists across later index sources.
 if(!candidates.empty()){
  if(candidates.size()!=cache_.rows())throw std::runtime_error("invalid republished candidate width");
  candidates_=std::move(candidates);
  device_candidates_=mx::array(candidates_.begin(),{int(candidates_.size())},mx::uint8);
 }
 index_source_layer_=index_source_layer;
 device_chunk_rows_=mx::array(0);chunk_tokens_=0;chunk_index_source_layer_=-1;
}
void SharedAttentionReference::republish(int index_source_layer,mx::array relative_selected,
 mx::array device_candidates,std::vector<std::int32_t> diagnostic_selected,
 std::vector<std::uint8_t> diagnostic_candidates){
 if(!is_index_source_layer(index_source_layer)||kv_source_for_layer(index_source_layer)!=source_layer_||
    relative_selected.dtype()!=mx::int32||
    relative_selected.shape()!=mx::Shape({int(std::min<std::size_t>(512,cache_.rows()))})||
    device_candidates.dtype()!=mx::uint8||
    (device_candidates.size()!=0&&device_candidates.shape()!=mx::Shape({int(cache_.rows())})))
  throw std::runtime_error("invalid device republished rows");
 device_rows_=std::move(relative_selected);device_candidates_=std::move(device_candidates);
 rows_.clear();
 if(!diagnostic_selected.empty())rows_=to_relative(diagnostic_selected,cache_.rows(),position_==0?1:128);
 if(!diagnostic_candidates.empty()){
  if(diagnostic_candidates.size()!=cache_.rows())throw std::runtime_error("invalid diagnostic republished candidate width");
  candidates_=std::move(diagnostic_candidates);
 }
 index_source_layer_=index_source_layer;
 device_chunk_rows_=mx::array(0);chunk_tokens_=0;chunk_index_source_layer_=-1;
}
}
