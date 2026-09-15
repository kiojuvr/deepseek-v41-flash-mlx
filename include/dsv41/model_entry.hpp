#pragma once
#include "dsv41/weights.hpp"
#include <mlx/mlx.h>
#include <span>

namespace dsv41 {
// Plain BF16 boundary, no fused/reordered norm or activation quantization.
mlx::core::array rms_norm_reference(const mlx::core::array& input,
    const mlx::core::array& weight, float eps);
struct TextEntryResult {
    mlx::core::array hidden;  // [tokens,4,5120], batch one
    mlx::core::array pre_mix; // [tokens,4], FP32 [1,0,0,0]
};
// Fixed V4.1 text entry only; no encoder, generation or image processing.
class TextEntryReference {
public:
    explicit TextEntryReference(WeightCatalog& catalog);
    TextEntryResult forward(std::span<const std::uint32_t> token_ids) const;
    static constexpr int vocab_size = 129280, dim = 5120, hc_mult = 4;
private:
    mlx::core::array embedding_;
};
}
