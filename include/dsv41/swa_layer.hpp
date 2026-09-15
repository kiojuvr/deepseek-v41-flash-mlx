#pragma once
#include "dsv41/swa_projection.hpp"
namespace dsv41 {
class SwaLayerReference;
// Immutable MLX arrays on copy; no host KV transfer. Synchronous reference commit.
class SwaLayerState {
public:
 SwaLayerState();
 void reset();
 std::uint64_t position() const{return position_;}
 const mlx::core::array& rows() const{return rows_;}
private:
 mlx::core::array rows_;
 std::uint64_t position_=0;
 friend class SwaLayerReference;
};
class SwaLayerReference {
public:
 explicit SwaLayerReference(WeightCatalog& catalog,int layer=0);
 // Pure SWA layer 0/1 attention sublayer only, token-serial schedule for all chunks.
 mlx::core::array forward(const mlx::core::array& hidden,SwaLayerState& state,
                          std::uint64_t start_position) const;
private:
 int layer_;
 SwaProjectionReference input_;
 PackedLinearReference output_;
 mlx::core::array grouped_weight_,sink_;
};
}
