#pragma once
#include "dsv41/weights.hpp"
#include <mlx/mlx.h>

namespace dsv41 {
struct LinearActivation {
    mlx::core::array values, scales, decoded;
};
// Official 32-element E4M3 activation quantization, including the 1e-4 absmax floor.
LinearActivation linear_activation_reference(const mlx::core::array& input);
class PackedLinearReference {
public:
    PackedLinearReference(WeightCatalog& catalog, const std::string& prefix);
    mlx::core::array forward(const mlx::core::array& input) const;
    mlx::core::array project_quantized(const LinearActivation& activation) const;
    // Diagnostic only: shape-dependent QMM reduction is not reference schedule exact.
    mlx::core::array project_batch_diagnostic(const LinearActivation& activation) const;
    const mlx::core::array& packed_weight() const;
    const mlx::core::array& scales() const;
    int input_dims() const;
    int output_dims() const;
    int bits() const;
    ~PackedLinearReference();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
