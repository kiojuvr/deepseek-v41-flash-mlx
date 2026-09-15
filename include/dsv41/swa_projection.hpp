#pragma once
#include "dsv41/linear.hpp"
#include "dsv41/trace.hpp"
namespace dsv41 {
// Pure SWA only: adjacent pairs in the last 64 channels, theta=10000, no YaRN.
mlx::core::array swa_rope_reference(const mlx::core::array& input,
    std::uint64_t start_position, bool inverse=false);
struct SwaProjectedQKV { mlx::core::array query, kv; };
class SwaProjectionReference {
public:
    explicit SwaProjectionReference(WeightCatalog& catalog,int layer=0);
    SwaProjectedQKV forward(const mlx::core::array& hidden, std::uint64_t start_position) const;
private:
    int layer_;
    PackedLinearReference q_a_,q_b_,kv_;
    mlx::core::array q_norm_,kv_norm_;
};
}
