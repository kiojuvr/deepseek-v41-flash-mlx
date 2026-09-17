#include "dsv41/text_generate.hpp"
#include "dsv41/execution_policy.hpp"
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
  const bool sweep=runtime_layer_sweep_enabled();
  if(sweep&&(!runtime_packed_expert_bank_enabled()||runtime_group_selected_experts_enabled()))
   throw std::runtime_error(
    "layer sweep requires packed expert bank enabled and selected grouping disabled");
  // The oracle keeps its 128-token request schedule.  The optimized path owns
  // up to 4096 tokens transactionally so one 2K prompt crosses all 40 layers
  // once.  Longer prompts retain bounded state commits until Phase 5.
  const std::size_t prefill_chunk=sweep?4096:128;
  std::optional<BlockResult> prefill;
  run_prefill_chunks(prompt.size(),prefill_chunk,[&](std::size_t offset,std::size_t count){
   prefill.emplace(sweep?model_.forward_packed_sweep(prompt.subspan(offset,count),*state,offset):
                         model_.forward(prompt.subspan(offset,count),*state,offset));
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
