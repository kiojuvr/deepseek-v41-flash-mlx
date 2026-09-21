#pragma once
#include "dsv41/weights.hpp"
#include <cstddef>
#include <mlx/mlx.h>
namespace dsv41 {
struct HCMixes { mlx::core::array pre, post, comb; };
// Process-wide count of fused mHC kernel invocations. This is candidate
// invocation proof for the paired performance runner, not a Metal count.
std::size_t fused_mhc_invocation_count();
void reset_fused_mhc_invocation_count();
mlx::core::array hc_pre_reference(const mlx::core::array& hidden,const mlx::core::array& pre);
mlx::core::array hc_post_reference(const mlx::core::array& output,const mlx::core::array& residual,
                                  const HCMixes& mixes);
class HCReference {
public:
    HCReference(WeightCatalog& catalog,int layer,const std::string& kind);
    HCMixes mixes(const mlx::core::array& hidden) const;
private:
    mlx::core::array fn_,scale_,base_;
};
}
