#pragma once
#include <mlx/mlx.h>
#include <cstdint>
namespace dsv41 {
struct PackedAttentionWorkList {
 mlx::core::array ordered,valid;
};
// Up to 640 ordered slots (128 window + 512 global); false entries receive -inf.
mlx::core::array swa_attention_masked_reference(const mlx::core::array& query,
 const mlx::core::array& ordered_kv,const mlx::core::array& sink,const mlx::core::array& valid);
mlx::core::array swa_attention_masked_chunk(const mlx::core::array& queries,
 const mlx::core::array& ordered_kv,const mlx::core::array& sink,const mlx::core::array& valid);
// oMLX-style one-threadgroup-per-token fused prefill attention. Local KV is
// the existing official quantization round-trip in BF16; pooled KV stays in
// its persistent packed 4-bit/E4M3-scale representation. Top-k is one
// device-resident [tokens,1..512] chunk work list padded with -1.
mlx::core::array swa_packed_attention_chunk(const mlx::core::array& queries,
 const mlx::core::array& local_kv,const mlx::core::array& pooled_values,
 const mlx::core::array& pooled_scales,const mlx::core::array& topk,
 const mlx::core::array& sink,std::uint64_t start,int compress_ratio);
PackedAttentionWorkList swa_packed_attention_work_list(
 const mlx::core::array& local_kv,const mlx::core::array& pooled_values,
 const mlx::core::array& pooled_scales,const mlx::core::array& topk,
 std::uint64_t start,int compress_ratio);
// One already projected/rotated query [64,512] and chronological visible KV [1..128,512].
// KV is already official FP8 round-tripped BF16. No RoPE/projection/state mutation here.
mlx::core::array swa_attention_reference(const mlx::core::array& query,
 const mlx::core::array& visible_kv,const mlx::core::array& sink);
}
