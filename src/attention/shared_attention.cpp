#include "dsv41/shared_attention.hpp"
#include "dsv41/layer_owner.hpp"
#include <algorithm>
#include <stdexcept>
namespace dsv41 {
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
}
std::vector<std::int32_t> SharedAttentionReference::indices(int layer,std::uint64_t pos,int offset) const{
 if(layer<3||layer>=kBackboneLayers||kv_source_for_layer(layer)!=source_layer_||pos!=position_||offset!=(pos==0?1:128))
  throw std::runtime_error("shared attention source/position mismatch");
 std::vector<std::int32_t> out;for(auto row:rows_)out.push_back(row+offset);return out;
}
void SharedAttentionReference::republish(int index_source_layer,std::vector<std::int32_t> selected,
 std::vector<std::uint8_t> candidates){
 if(!is_index_source_layer(index_source_layer)||kv_source_for_layer(index_source_layer)!=source_layer_)
  throw std::runtime_error("republish requires an index source in this kv group");
 if(selected.size()!=std::min<std::size_t>(512,cache_.rows()))throw std::runtime_error("invalid republished row count");
 rows_=to_relative(selected,cache_.rows(),position_==0?1:128);
 // The candidate mask belongs to the candidate source and persists across later index sources.
 if(!candidates.empty()){
  if(candidates.size()!=cache_.rows())throw std::runtime_error("invalid republished candidate width");
  candidates_=std::move(candidates);
 }
 index_source_layer_=index_source_layer;
}
}
