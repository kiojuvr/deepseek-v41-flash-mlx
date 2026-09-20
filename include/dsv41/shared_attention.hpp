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
 SharedAttentionReference(const GlobalKVState& cache,mlx::core::array relative_selected,
                          mlx::core::array device_candidates,std::uint64_t query_position,
                          int window_offset,int source_layer,int ratio,
                          std::vector<std::int32_t> diagnostic_selected={},
                          std::vector<std::uint8_t> diagnostic_candidates={});
 const GlobalKVState& cache() const{return cache_;}
 int source_layer() const{return source_layer_;}
 int index_source_layer() const{return index_source_layer_;}
 const std::vector<std::uint8_t>& candidates() const{return candidates_;}
 const mlx::core::array& device_candidates() const{return device_candidates_;}
 // Validates that `consumer_layer` belongs to this source's group and returns the row ids.
 std::vector<std::int32_t> indices(int consumer_layer,std::uint64_t query_position,
                                   int window_offset) const;
 const mlx::core::array& device_indices(int consumer_layer,std::uint64_t query_position,
                                        int window_offset) const;
 // Optimized chunk publication: every token publication shares the same
 // fixed-width device row plan until the next index-source republish.
 void publish_chunk_plan(int index_source_layer,mlx::core::array relative_rows,
                         std::uint64_t start,int tokens);
 const mlx::core::array& device_chunk_indices(int consumer_layer,
                         std::uint64_t start,int tokens) const;
 bool chunk_plan_matches(int consumer_layer,std::uint64_t start,int tokens) const;
 // Index source republish: rows are +offset and sorted; candidates is the level-one mask.
 void republish(int index_source_layer,std::vector<std::int32_t> selected,
                std::vector<std::uint8_t> candidates);
 void republish(int index_source_layer,mlx::core::array relative_selected,
                mlx::core::array device_candidates,
                std::vector<std::int32_t> diagnostic_selected={},
                std::vector<std::uint8_t> diagnostic_candidates={});
private:
 GlobalKVState cache_;
 std::vector<std::int32_t> rows_;
 std::vector<std::uint8_t> candidates_;
 mlx::core::array device_rows_{0},device_candidates_{0};
 mlx::core::array device_chunk_rows_{0};
 std::uint64_t chunk_start_=0;
 int chunk_tokens_=0,chunk_index_source_layer_=-1;
 std::uint64_t position_;
 int source_layer_,ratio_,index_source_layer_;
};
}
