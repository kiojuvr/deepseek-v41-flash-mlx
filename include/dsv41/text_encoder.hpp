#pragma once
#include "dsv41/block.hpp"
#include "dsv41/compressed_block.hpp"
#include "dsv41/reused_block.hpp"
#include "dsv41/engram_layer.hpp"
#include "dsv41/model_entry.hpp"
#include "dsv41/trace.hpp"
#include <array>
#include <memory>
namespace dsv41 {
// Encoder layers 0..19: pure SWA 0/1, producers 2/8/14, reuse 3-7/9-13/15-19, Engram 1/14.
struct TextEncoderState {
 explicit TextEncoderState(std::shared_ptr<const EngramMetadata> m):metadata(std::move(m)),hash(metadata){}
 void reset();
 std::shared_ptr<const EngramMetadata> metadata;
 EngramHashState hash;
 std::array<SwaLayerState,2> swa;
 std::array<CompressedLayerState,3> producer;
 std::array<ReusedLayerState,15> reuse;
};
class TextEncoderReference {
public:
 TextEncoderReference(WeightCatalog& catalog,std::shared_ptr<const EngramMetadata> metadata,
                      std::shared_ptr<const ResidentExpertAtlas> atlas={});
 BlockResult forward(std::span<const std::uint32_t> ids,TextEncoderState& state,std::uint64_t start,
                     TraceSink* trace=nullptr) const;
 // Explicit layer-major candidate; releases each packed bank after evaluation.
 BlockResult forward_packed_chunk(std::span<const std::uint32_t> ids,
                                  TextEncoderState& state,std::uint64_t start) const;
 // Transactional request-wide layer sweep. Primitive kernels retain their
 // 128-row contract; the sweep owns microtiling and publishes state once.
 BlockResult forward_packed_sweep(std::span<const std::uint32_t> ids,
                                  TextEncoderState& state,std::uint64_t start) const;
private:
 std::shared_ptr<const EngramMetadata> metadata_;
 TextEntryReference entry_;
 std::array<std::unique_ptr<BlockReference>,2> swa_;
 std::array<std::unique_ptr<CompressedBlockReference>,3> producer_;
 std::array<std::unique_ptr<ReusedBlockReference>,15> reuse_;
 EngramLayerReference engram1_,engram14_;
};
}
