#pragma once
#include <mlx/mlx.h>
#include <span>
#include <cstdint>
namespace dsv41 {
// Explicit positions support compressor group starts, not just consecutive queries.
// Last 64 channels, theta=160000, YaRN factor=16 / original=65536 / beta=32,1.
mlx::core::array compressed_rope_reference(const mlx::core::array& input,
 std::span<const std::uint64_t> positions,bool inverse=false);
}
