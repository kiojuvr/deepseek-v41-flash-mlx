#pragma once
#include "dsv41/compressed_rope.hpp"
#include "dsv41/weights.hpp"
namespace dsv41 {
// Index key projection/norm/RoPE boundary for a kv_source layer; output is PRE FP4 quantization.
// Never publish this BF16 intermediate as the canonical index cache.
class IndexKeyReference {
public:
 explicit IndexKeyReference(WeightCatalog& catalog,int layer);
 mlx::core::array before_quantization(const mlx::core::array& latent,
                                     std::span<const std::uint64_t> positions) const;
private:
 mlx::core::array weight_,norm_;
};
}
