#pragma once
#include "dsv41/block.hpp"
#include "dsv41/model_entry.hpp"
namespace dsv41 {
// Tokenized text -> embedding -> layer 0. No tokenizer, logits or sampling yet.
class TextFrontReference {
public:
 explicit TextFrontReference(WeightCatalog& catalog):entry_(catalog),block_(catalog){}
 BlockResult forward(std::span<const std::uint32_t> ids,SwaLayerState& state,
                     std::uint64_t start_position) const;
private:
 TextEntryReference entry_;
 BlockReference block_;
};
}
