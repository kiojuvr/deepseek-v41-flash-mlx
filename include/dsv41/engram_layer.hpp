#pragma once
#include "dsv41/engram_mlx.hpp"
#include <optional>

namespace dsv41 {
struct EngramGateResult {
    mlx::core::array dot, gate, output;
};
struct EngramActivation {
    mlx::core::array quantized, scales;
};
EngramActivation engram_activation_reference(const mlx::core::array& values);
// Explicit FP32 operations; BF16 cast only at the residual output boundary.
EngramGateResult engram_gate_reference(const mlx::core::array& hidden,
    const mlx::core::array& kv, const mlx::core::array& q_weight,
    const mlx::core::array& k_weight, const std::optional<mlx::core::array>& mask,
    float eps);
struct EngramForwardResult {
    mlx::core::array quantized, scales, kv, dot, gate, output;
};
// Experimental batch-one local reference. Packed projection weights stay resident.
// Hash state is owned by the caller; forward has no mutable convolution/cache state.
class EngramLayerReference {
public:
    EngramLayerReference(WeightCatalog& catalog, const EngramMetadata& metadata,
                         std::size_t layer_index, float eps, EngramReadMode mode);
    ~EngramLayerReference();
    EngramForwardResult forward(const mlx::core::array& hidden,
        std::span<const std::uint64_t> rows,
        const std::optional<mlx::core::array>& mask = std::nullopt) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
