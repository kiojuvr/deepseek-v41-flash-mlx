#pragma once
#include "dsv41/engram.hpp"
#include <mlx/array.h>

namespace dsv41 {
// Owned input arrays retain copied staging bytes until Metal completes.
mlx::core::array engram_lookup_mlx(const PackedEngramRows& rows);
}
