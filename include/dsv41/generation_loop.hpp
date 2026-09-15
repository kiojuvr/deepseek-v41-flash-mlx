#pragma once
#include <algorithm>
#include <cstdint>
#include <functional>
#include <span>
#include <stdexcept>
#include <vector>

namespace dsv41 {
struct GenerationResult {
 std::vector<std::uint32_t> tokens;
 // Logical position after committed output; not a resumable KV cursor.
 std::uint64_t next_position=0;
 bool stopped=false;
 bool cancelled=false;
};
struct GenerationControl {
 // Synchronous notification after commitment. False requests cancellation.
 // index is zero-based; callbacks must not reenter the same model.
 std::function<bool(std::uint32_t token,std::uint64_t index)> on_token;
 // May read an atomic flag set by another thread. Polled between model steps.
 std::function<bool()> is_cancelled;
};

// Deterministic bounded prefill schedule shared by the model path and its
// checkpoint-free boundary tests. feed(offset, count) receives nonempty,
// contiguous ranges whose union is exactly [0, prompt_size).
template<class Feed>
void run_prefill_chunks(std::size_t prompt_size,std::size_t max_chunk,Feed&& feed) {
 if(prompt_size==0 || max_chunk==0)
  throw std::runtime_error("prefill schedule requires positive sizes");
 for(std::size_t offset=0;offset<prompt_size;){
  const auto count=std::min(max_chunk,prompt_size-offset);
  feed(offset,count);
  offset+=count;
 }
}

// Shared control flow for batch and streaming reference generation. Model work
// remains in native callables; no checkpoint or MLX dependency is needed here.
template<class Prefill,class Sample,class Advance>
GenerationResult run_generation_loop(std::size_t prompt_size,std::size_t max_new_tokens,
 std::span<const std::uint32_t> stop_ids,const GenerationControl& control,
 Prefill&& prefill,Sample&& sample,Advance&& advance) {
 if(prompt_size==0)throw std::runtime_error("generation requires a nonempty prompt");
 constexpr std::size_t limit=262144;
 if(prompt_size>limit || max_new_tokens>limit-prompt_size)
  throw std::runtime_error("generation exceeds total context limit");
 GenerationResult result;
 result.next_position=prompt_size;
 auto cancelled=[&]{return control.is_cancelled && control.is_cancelled();};
 if(cancelled()){result.cancelled=true;return result;}
 if(max_new_tokens==0)return result;
 prefill();
 for(std::size_t step=0;step<max_new_tokens;++step){
  if(cancelled()){result.cancelled=true;break;}
  auto token=sample();
  // Cancellation during evaluation does not commit an unreported token.
  if(cancelled()){result.cancelled=true;break;}
  result.tokens.push_back(token);
  ++result.next_position;
  result.stopped=std::find(stop_ids.begin(),stop_ids.end(),token)!=stop_ids.end();
  if(control.on_token && !control.on_token(token,step))result.cancelled=true;
  if(cancelled())result.cancelled=true;
  if(result.cancelled || result.stopped || step+1==max_new_tokens)break;
  advance(token,result.next_position-1);
 }
 return result;
}
}
