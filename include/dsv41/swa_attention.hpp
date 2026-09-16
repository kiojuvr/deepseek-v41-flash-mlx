#pragma once
#include <mlx/mlx.h>
namespace dsv41 {
// Up to 640 ordered slots (128 window + 512 global); false entries receive -inf.
mlx::core::array swa_attention_masked_reference(const mlx::core::array& query,
 const mlx::core::array& ordered_kv,const mlx::core::array& sink,const mlx::core::array& valid);
mlx::core::array swa_attention_masked_chunk(const mlx::core::array& queries,
 const mlx::core::array& ordered_kv,const mlx::core::array& sink,const mlx::core::array& valid);
// One already projected/rotated query [64,512] and chronological visible KV [1..128,512].
// KV is already official FP8 round-tripped BF16. No RoPE/projection/state mutation here.
mlx::core::array swa_attention_reference(const mlx::core::array& query,
 const mlx::core::array& visible_kv,const mlx::core::array& sink);
}
