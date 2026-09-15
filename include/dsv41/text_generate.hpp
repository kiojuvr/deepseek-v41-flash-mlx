#pragma once
#include "dsv41/text_backbone.hpp"
#include "dsv41/sampling.hpp"
#include "dsv41/generation_loop.hpp"
#include <span>
#include <vector>
namespace dsv41 {
// Plain autoregressive reference: prefill the prompt, then decode one token per step.
class TextGenerationReference {
public:
 TextGenerationReference(WeightCatalog& catalog,std::shared_ptr<const EngramMetadata> metadata)
  :metadata_(metadata),model_(catalog,std::move(metadata)){}
 GenerationResult generate(std::span<const std::uint32_t> prompt,std::size_t max_new_tokens,
                           SamplingConfig config,std::span<const std::uint32_t> stop_ids={},
                           const GenerationControl& control={}) const;
private:
 std::shared_ptr<const EngramMetadata> metadata_;
 TextBackboneReference model_;
};
}
