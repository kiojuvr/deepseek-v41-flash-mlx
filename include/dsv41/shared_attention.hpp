#pragma once
#include "dsv41/global_kv.hpp"
#include <cstdint>
#include <vector>
namespace dsv41 {
// Immutable per-query publication from one kv_source layer, consumed by the layers that reuse it.
// Ownership stays with one request; cross-request routing is the caller's responsibility.
// The selected rows may be republished by a later index source (decoder layers 24/28/32/36).
class SharedAttentionReference {
public:
 SharedAttentionReference(const GlobalKVState& cache,std::vector<std::int32_t> selected,
                          std::uint64_t query_position,int window_offset,int source_layer,int ratio);
 const GlobalKVState& cache() const{return cache_;}
 int source_layer() const{return source_layer_;}
 int index_source_layer() const{return index_source_layer_;}
 const std::vector<std::uint8_t>& candidates() const{return candidates_;}
 // Validates that `consumer_layer` belongs to this source's group and returns the row ids.
 std::vector<std::int32_t> indices(int consumer_layer,std::uint64_t query_position,
                                   int window_offset) const;
 // Index source republish: rows are +offset and sorted; candidates is the level-one mask.
 void republish(int index_source_layer,std::vector<std::int32_t> selected,
                std::vector<std::uint8_t> candidates);
private:
 GlobalKVState cache_;
 std::vector<std::int32_t> rows_;
 std::vector<std::uint8_t> candidates_;
 std::uint64_t position_;
 int source_layer_,ratio_,index_source_layer_;
};
}
