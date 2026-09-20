#pragma once
#include <mlx/mlx.h>

namespace dsv41 {
// MIT-derived oMLX short-token pipeline adapted to the checkpoint's existing
// FP8 roundtrip semantics. The activation must remain an unevaluated custom
// kernel node when passed to grouped_expert_pipeline().
mlx::core::array fused_swiglu_fp8_roundtrip(const mlx::core::array& gate,
                                             const mlx::core::array& up);
mlx::core::array grouped_expert_pipeline(const mlx::core::array& gate,
                                          const mlx::core::array& up,
                                          const mlx::core::array& activation,
                                          const mlx::core::array& down);
}
