#pragma once
#include <mlx/mlx.h>
#include <cstdint>
namespace dsv41 {
struct PackedAttentionWorkList {
 mlx::core::array ordered,valid,widths;
 bool request_boundary=false;
};
struct AttentionTailDiagnostics {
 mlx::core::array padded_qk,padded_av;
};
struct FixedTileWidthOneDiagnostics {
 mlx::core::array native_qk,native_av;
};
// Up to 640 ordered slots (128 window + 512 global); false entries receive -inf.
mlx::core::array swa_attention_masked_reference(const mlx::core::array& query,
 const mlx::core::array& ordered_kv,const mlx::core::array& sink,const mlx::core::array& valid);
mlx::core::array swa_attention_masked_chunk(const mlx::core::array& queries,
 const mlx::core::array& ordered_kv,const mlx::core::array& sink,const mlx::core::array& valid);
mlx::core::array swa_attention_fixed_tile_core(const mlx::core::array& queries,
 const PackedAttentionWorkList& work,const mlx::core::array& sink,bool ragged_av=false);
// Qualification-only attribution for selected-width one on the real fixed
// work list. Each result substitutes only the native token-scalar QK or AV
// shape while leaving metadata, softmax, and the other operation unchanged.
FixedTileWidthOneDiagnostics swa_attention_fixed_tile_width_one_diagnostics(
 const mlx::core::array& queries,const PackedAttentionWorkList& work,
 const mlx::core::array& sink,bool ragged_av);
// Qualification-only attribution for the final ragged block. Each result
// changes only one GEMM shape to 64 columns/rows while retaining the exact
// work-list content and online-softmax schedule.
AttentionTailDiagnostics swa_attention_tail_diagnostics(
 const mlx::core::array& queries,const mlx::core::array& ordered_kv,
 const mlx::core::array& sink,const mlx::core::array& valid);
// oMLX-style one-threadgroup-per-token fused prefill attention. Local KV is
// the existing official quantization round-trip in BF16; pooled KV stays in
// its persistent packed 4-bit/E4M3-scale representation. Top-k is one
// device-resident [tokens,1..512] chunk work list padded with -1.
mlx::core::array swa_packed_attention_chunk(const mlx::core::array& queries,
 const mlx::core::array& local_kv,const mlx::core::array& pooled_values,
 const mlx::core::array& pooled_scales,const mlx::core::array& topk,
 const mlx::core::array& sink,std::uint64_t start,int compress_ratio);
// DwarfStar-style one layer-chunk dispatch. The fixed-width top-k matrix stays
// device resident; negative entries are inactive metadata, not padded KV rows.
mlx::core::array swa_wide_attention_chunk(const mlx::core::array& queries,
 const mlx::core::array& local_kv,const mlx::core::array& pooled_values,
 const mlx::core::array& pooled_scales,const mlx::core::array& topk,
 const mlx::core::array& sink,std::uint64_t start,int compress_ratio);
// One logical fixed-tile layer-chunk operation.  The dense plan always owns
// 512 pooled slots; negative rows remain invalid, so the reduction topology is
// exactly ten 64-row tiles including the 128-row local window.
mlx::core::array swa_fixed_tile_attention_chunk(const mlx::core::array& queries,
 const mlx::core::array& local_kv,const mlx::core::array& pooled_values,
 const mlx::core::array& pooled_scales,const mlx::core::array& topk,
 const mlx::core::array& sink,std::uint64_t start,int compress_ratio);
// Materialize an exact-shape work list. selected_count separates the logical
// pooled width from the one-column dummy storage used for empty lists.
PackedAttentionWorkList swa_packed_attention_work_list(
 const mlx::core::array& local_kv,const mlx::core::array& pooled_values,
 const mlx::core::array& pooled_scales,const mlx::core::array& topk,
 std::uint64_t start,int compress_ratio,int selected_count=-1,bool fixed_window=false);
// One already projected/rotated query [64,512] and chronological visible KV [1..128,512].
// KV is already official FP8 round-tripped BF16. No RoPE/projection/state mutation here.
mlx::core::array swa_attention_reference(const mlx::core::array& query,
 const mlx::core::array& visible_kv,const mlx::core::array& sink);
}
