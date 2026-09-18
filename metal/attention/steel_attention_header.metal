#include <metal_stdlib>
#include "mlx/backend/metal/kernels/steel/attn/attn.h"
#include "mlx/backend/metal/kernels/steel/attn/params.h"

// Adapted from oMLX's DeepSeek V4.1 packed attention kernel.
// Copyright © 2026 OpenAI. Licensed under Apache-2.0.
// This version reads the runtime's already-round-tripped BF16 local window
// directly and keeps the official packed E4M3-scale pooled cache split across
// value and scale buffers.
using namespace mlx::steel;

struct Dsv41SparseMaxOp {
  template <typename T> METAL_FUNC static constexpr T apply(T x, T y) {
    return metal::max(x, y);
  }
};
struct Dsv41SparseSumOp {
  template <typename T> METAL_FUNC static constexpr T apply(T x, T y) {
    return x + y;
  }
};
struct Dsv41SparseMulOp {
  template <typename T> METAL_FUNC static constexpr T apply(T x, T y) {
    return x * y;
  }
};
struct Dsv41SparseExpSubOp {
  template <typename T> METAL_FUNC static constexpr T apply(T x, T y) {
    return metal::exp(x - y);
  }
};
struct Dsv41SparseDivOp {
  template <typename T> METAL_FUNC static constexpr T apply(T x, T y) {
    return x / y;
  }
};

METAL_FUNC float dsv41_attention_fp8(uchar code) {
  const uint a = code & 127, exponent = a >> 3, mantissa = a & 7;
  const float value = exponent == 0 ? float(mantissa) * 0x1p-9f
      : as_type<float>(((exponent + 120u) << 23) | (mantissa << 20));
  return code & 128 ? -value : value;
}

METAL_FUNC float dsv41_pooled_value(
    const device uchar* values, const device uchar* scales, int d) {
  constexpr float levels[8] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6};
  const uint code = (values[d / 2] >> ((d % 2) * 4)) & 15;
  const float value = levels[code & 7] * ((code & 8) ? -1.0f : 1.0f);
  return value * dsv41_attention_fp8(scales[d / 16]);
}
