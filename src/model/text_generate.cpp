#include "dsv41/text_generate.hpp"
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <optional>
namespace dsv41 {
namespace mx=mlx::core;
GenerationResult TextGenerationReference::generate(std::span<const std::uint32_t> prompt,
 std::size_t max_new_tokens,SamplingConfig config,std::span<const std::uint32_t> stop_ids,
 const GenerationControl& control) const{
 if(!std::isfinite(config.temperature) || config.temperature<0)
  throw std::runtime_error("generation requires a finite nonnegative temperature");
 std::optional<TextBackboneState> state;
 std::optional<mx::array> last;
 std::uint64_t rng=config.seed;
 return run_generation_loop(prompt.size(),max_new_tokens,stop_ids,control,[&]{
  state.emplace(metadata_);
  // Encoder/decoder kernels accept at most 128 tokens per call. Keep one
  // stateful prefill while feeding long protocol-rendered prompts in bounded
  // chunks; the final chunk supplies the logits for the first decode step.
  constexpr std::size_t kPrefillChunk = 128;
  std::optional<BlockResult> prefill;
  run_prefill_chunks(prompt.size(),kPrefillChunk,[&](std::size_t offset,std::size_t count){
   prefill.emplace(model_.forward(prompt.subspan(offset,count),*state,offset));
  });
  auto logits=model_.logits(*prefill);
  const int n=int(prefill->hidden.shape(0));
  last=mx::slice(logits,{n-1,0},{n,129280});
 },[&]{return sample_reference(*last,config.temperature,rng);},
 [&](std::uint32_t token,std::uint64_t position){
  const std::array<std::uint32_t,1> one{token};
  auto out=model_.forward(std::span<const std::uint32_t>(one),*state,position);
  auto step_logits=model_.logits(out);
  last=mx::slice(step_logits,{0,0},{1,129280});
 });
}
}
