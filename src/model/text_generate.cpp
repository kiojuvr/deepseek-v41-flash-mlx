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
 if(const auto limit=runtime_effective_mlx_cache_limit_bytes(prompt.size()+max_new_tokens);limit)
  mx::set_cache_limit(limit);
 const bool sweep=runtime_layer_sweep_enabled();
 if(sweep&&(!runtime_packed_expert_bank_enabled()||runtime_group_selected_experts_enabled()))
  throw std::runtime_error(
   "layer sweep requires packed expert bank enabled and selected grouping disabled");
 std::optional<TextBackboneState> state;
 std::optional<mx::array> last;
 std::uint64_t rng=config.seed;
 return run_generation_loop(prompt.size(),max_new_tokens,stop_ids,control,[&]{
  state.emplace(metadata_);
  std::optional<BlockResult> prefill;
  // The oracle keeps its 128-token request schedule. The opt-in sweep can hold
  // one 16K encoder-only frontier and finish it with a later >=8K sweep, then returns to
  // the unchanged 4096-token packed sweep for a short tail.
  const bool deferred=sweep&&runtime_deferred_decoder_enabled();
  std::optional<DeferredDecoderTransaction> pending_decoder;
  for(std::size_t offset=0;offset<prompt.size();){
   const auto step=runtime_prefill_step(
    prompt.size()-offset,sweep,deferred,pending_decoder.has_value());
   auto input=prompt.subspan(offset,step.tokens);
   if(step.action==RuntimePrefillStep::Action::BeginDeferredDecoder){
    pending_decoder.emplace(model_.begin_deferred_decoder(input,*state,offset));
   }else if(step.action==RuntimePrefillStep::Action::FinishDeferredDecoder){
    prefill.emplace(model_.finish_deferred_decoder(std::move(*pending_decoder),input,*state));
    pending_decoder.reset();
   }else{
    prefill.emplace(sweep?model_.forward_packed_sweep(input,*state,offset):
                           model_.forward(input,*state,offset));
   }
   offset+=step.tokens;
  }
  if(pending_decoder)throw std::runtime_error("generation ended with an unpublished decoder frontier");
  auto logits=model_.logits(*prefill);
  const int n=int(prefill->hidden.shape(0));
  last=mx::slice(logits,{n-1,0},{n,129280});
 },[&]{return sample_reference(*last,config.temperature,rng);},
 [&](std::uint32_t token,std::uint64_t position){
  const std::array<std::uint32_t,1> one{token};
  auto out=sweep?model_.forward_packed_sweep(one,*state,position):
                 model_.forward(std::span<const std::uint32_t>(one),*state,position);
  auto step_logits=model_.logits(out);
  last=mx::slice(step_logits,{0,0},{1,129280});
 });
}
}
