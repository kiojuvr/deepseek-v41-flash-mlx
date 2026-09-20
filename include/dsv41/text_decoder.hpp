#pragma once
#include "dsv41/compressed_block.hpp"
#include "dsv41/reused_block.hpp"
#include "dsv41/trace.hpp"
#include <array>
#include <memory>
namespace dsv41 {
// Decoder layers 20..39: ratio-1 producer 20 (candidate + index source), reuse 21..39 with
// index sources 24/28/32/36. Final hc_pre collapse, RMSNorm and head logits.
struct TextDecoderState {
 TextDecoderState()=default;
 void reset(){producer.reset();for(auto& s:reuse)s.reset();}
 CompressedLayerState producer;
 std::array<ReusedLayerState,19> reuse;
};
class TextDecoderReference {
public:
 explicit TextDecoderReference(WeightCatalog& catalog,
                               std::shared_ptr<const ResidentExpertAtlas> atlas={});
 BlockResult forward_packed_chunk(const mlx::core::array& hidden,const mlx::core::array& pre_mix,
                                  TextDecoderState& state,std::uint64_t start) const;
 BlockResult forward_packed_sweep(const mlx::core::array& hidden,const mlx::core::array& pre_mix,
                                  TextDecoderState& state,std::uint64_t start) const;
 // Exact CED completion. Returns only the final row while advancing every
 // decoder state to start + input rows.
 BlockResult forward_deferred_suffix(const mlx::core::array& hidden,
                                     const mlx::core::array& pre_mix,
                                     TextDecoderState& state,std::uint64_t start) const;
 // DwarfStar-style cross-sweep CED: publish only producer cache/window state
 // for an encoder-only prefix, then complete the exact suffix from a later
 // encoder sweep. Reuse states remain private and invalid between these calls.
 void prepare_deferred_prefix(const mlx::core::array& hidden,
                              const mlx::core::array& pre_mix,
                              TextDecoderState& state,std::uint64_t start) const;
 BlockResult resume_deferred_suffix(const mlx::core::array& hidden,
                                    const mlx::core::array& pre_mix,
                                    TextDecoderState& state,std::uint64_t start) const;
 // Returns the final collapsed hidden [1,5120] after layer 39 (pre-norm).
 BlockResult forward(const mlx::core::array& hidden,const mlx::core::array& pre_mix,
                     TextDecoderState& state,std::uint64_t start,TraceSink* trace=nullptr) const;
 // Final collapse + RMSNorm + head logits [tokens,129280] FP32.
 mlx::core::array logits(const BlockResult& final_hidden,TraceSink* trace=nullptr) const;
private:
 BlockResult forward_deferred_suffix_impl(const mlx::core::array& hidden,
                                          const mlx::core::array& pre_mix,
                                          TextDecoderState& state,std::uint64_t start,
                                          bool resume_prefix) const;
 int reuse_slot(int layer) const;
 std::unique_ptr<CompressedBlockReference> producer_;
 std::array<std::unique_ptr<ReusedBlockReference>,19> reuse_;
 mlx::core::array norm_,head_;
};
}
