#pragma once
#include <mlx/mlx.h>
#include <cstdint>
#include <span>
namespace dsv41 {
// Reference sampling: greedy when temperature <= 0, otherwise Gumbel-max over softmax.
// The RNG is a documented splitmix64 reference stream; it is not the official torch RNG,
// so sampling exactness against the official kernel is a separate, unverified gate.
struct SamplingConfig {
 float temperature=0.0f;
 std::uint64_t seed=0;
};
std::uint32_t greedy_reference(const mlx::core::array& logits);
// Official-formula boundary with caller-supplied positive Exp(1) values.
// This separates sampling arithmetic from backend-specific RNG streams.
std::uint32_t sample_from_exponentials_reference(const mlx::core::array& logits,
 float temperature,std::span<const float> exponentials);
std::uint32_t sample_reference(const mlx::core::array& logits,float temperature,std::uint64_t& rng_state);
}
