#include "dsv41/attention_telemetry.hpp"
#include "dsv41/checkpoint_atlas.hpp"
#include "dsv41/execution_policy.hpp"
#include "dsv41/sampling.hpp"
#include "dsv41/text_backbone.hpp"
#include "dsv41/sweep_telemetry.hpp"
#include <mlx/mlx.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace mx=mlx::core;
using Clock=std::chrono::steady_clock;
using J=nlohmann::json;

struct Turn { std::size_t append_tokens=0,decode_tokens=0; };

double seconds(Clock::time_point start) {
 return std::chrono::duration<double>(Clock::now()-start).count();
}

double wall_budget_seconds() {
 const char* text=std::getenv("DSV41_CUMULATIVE_WALL_BUDGET_SECONDS");
 if(text==nullptr||*text=='\0')return 0.0;
 std::size_t used=0;const double value=std::stod(text,&used);
 if(text[used]!='\0'||!std::isfinite(value)||value<0.0)
  throw std::runtime_error("invalid DSV41_CUMULATIVE_WALL_BUDGET_SECONDS");
 return value;
}

double percentile(std::vector<double> values,double q) {
 if(values.empty())throw std::runtime_error("percentile requires samples");
 std::sort(values.begin(),values.end());
 const auto index=std::min(values.size()-1,
  static_cast<std::size_t>(std::ceil(q*values.size())-1));
 return values[index];
}

std::vector<std::uint32_t> read_tokens(const std::filesystem::path& path) {
 std::ifstream in(path);if(!in)throw std::runtime_error("cannot open token file");
 std::vector<std::uint32_t> result;std::uint64_t value;
 while(in>>value){
  if(value>=129280||value==129264)throw std::runtime_error("invalid text token");
  result.push_back(static_cast<std::uint32_t>(value));
 }
 if(!in.eof()||result.empty())throw std::runtime_error("invalid token file");
 return result;
}

std::vector<Turn> read_schedule(const std::filesystem::path& path) {
 std::ifstream in(path);if(!in)throw std::runtime_error("cannot open turn schedule");
 std::vector<Turn> result;Turn turn;
 while(in>>turn.append_tokens>>turn.decode_tokens){
  if(!turn.append_tokens||!turn.decode_tokens||turn.decode_tokens>128)
   throw std::runtime_error("turn schedule requires positive append and 1..128 decode tokens");
  result.push_back(turn);
 }
 if(!in.eof()||result.empty())throw std::runtime_error("invalid turn schedule");
 return result;
}

void require_state_position(const dsv41::TextBackboneState& state,std::uint64_t expected) {
 if(state.encoder.hash.position()!=expected)throw std::runtime_error("Engram state position mismatch");
 for(const auto& value:state.encoder.swa)if(value.position()!=expected)
  throw std::runtime_error("SWA state position mismatch");
 for(const auto& value:state.encoder.producer)if(value.position()!=expected)
  throw std::runtime_error("encoder producer position mismatch");
 for(const auto& value:state.encoder.reuse)if(value.position()!=expected)
  throw std::runtime_error("encoder reuse position mismatch");
 if(state.decoder.producer.position()!=expected)
  throw std::runtime_error("decoder producer position mismatch");
 for(const auto& value:state.decoder.reuse)if(value.position()!=expected)
  throw std::runtime_error("decoder reuse position mismatch");
}

J memory() {
 return {{"active_bytes",mx::get_active_memory()},{"cache_bytes",mx::get_cache_memory()},
         {"peak_bytes",mx::get_peak_memory()},{"limit_bytes",mx::get_memory_limit()}};
}

void evaluate(const dsv41::BlockResult& value) {
 mx::eval(value.hidden,value.pre_mix);mx::synchronize();
 auto hidden=mx::all(mx::isfinite(value.hidden));
 auto pre=mx::all(mx::isfinite(value.pre_mix));mx::eval(hidden,pre);
 if(!hidden.item<bool>()||!pre.item<bool>())
  throw std::runtime_error("nonfinite backbone output");
}

J attention_delta(const dsv41::AttentionTelemetry& before,
                  const dsv41::AttentionTelemetry& after) {
#define DSV41_DELTA(name) {#name,after.name-before.name}
 return {DSV41_DELTA(concat_calls),DSV41_DELTA(concat_input_bytes),
  DSV41_DELTA(concat_output_bytes),DSV41_DELTA(cumulative_bytes_copied),
  DSV41_DELTA(logical_tokens),DSV41_DELTA(attention_rows),
  DSV41_DELTA(indexer_rows),DSV41_DELTA(index_host_readbacks),
  DSV41_DELTA(token_serial_attention_calls),DSV41_DELTA(chunk_attention_calls),
  DSV41_DELTA(packed_chunk_attention_calls),DSV41_DELTA(wide_attention_calls),
  DSV41_DELTA(fixed_tile_attention_calls),
  DSV41_DELTA(chunk_batched_splitk_qk_calls),DSV41_DELTA(chunk_scalar_qk_calls),
  DSV41_DELTA(chunk_scalar_av_calls),DSV41_DELTA(chunk_av_batches)};
#undef DSV41_DELTA
}
}

int main(int argc,char** argv) { try {
 if(argc!=8)throw std::runtime_error(
  "usage: dsv41-cumulative-session checkpoint m1-summary metadata tokens-file schedule-file output-json max-context");
 const auto checkpoint=std::filesystem::path(argv[1]);
 const auto output=std::filesystem::path(argv[6]);
 if(output.lexically_normal().string().starts_with(checkpoint.lexically_normal().string()))
  throw std::runtime_error("checkpoint is read-only");
 const std::size_t max_context=std::stoull(argv[7]);
 if(max_context<1||max_context>262144)throw std::runtime_error("invalid max context");
 const auto schedule=read_schedule(argv[5]);
 auto corpus=read_tokens(argv[4]);
 std::size_t corpus_required=0,total_context=0;
 for(const auto& turn:schedule){
  corpus_required+=turn.append_tokens;total_context+=turn.append_tokens+turn.decode_tokens;
  if(total_context>max_context)throw std::runtime_error("turn schedule exceeds max context");
 }
 if(corpus.size()!=corpus_required)throw std::runtime_error("token file length does not match turn schedule");

 const bool sweep=dsv41::runtime_layer_sweep_enabled();
 const bool deferred=sweep&&dsv41::runtime_deferred_decoder_enabled();
 const double wall_budget=wall_budget_seconds();
 if(!sweep)throw std::runtime_error("cumulative production measurement requires layer sweep");
 if(!dsv41::runtime_packed_expert_bank_enabled()||dsv41::runtime_group_selected_experts_enabled())
  throw std::runtime_error("layer sweep requires packed bank and device expert grouping");
 if(!dsv41::runtime_fixed_tile_attention_enabled()||
    !dsv41::runtime_ragged_tail_qk_enabled()||!dsv41::runtime_ragged_tail_av_enabled())
  throw std::runtime_error("cumulative production measurement requires fixed-tile/ragged attention");

 mx::set_default_device(mx::Device::gpu);
 const auto cache_limit=dsv41::runtime_effective_mlx_cache_limit_bytes(max_context);
 if(cache_limit)mx::set_cache_limit(cache_limit);
 auto proof=dsv41::read_json_file("artifacts/engram/fixture-provenance.json");
 std::ifstream meta_file(argv[3],std::ios::binary);
 std::string meta_text{std::istreambuf_iterator<char>(meta_file),{}};
 if(dsv41::sha256_text(meta_text)!=proof.at("fixture_sha256").at("metadata.json").get<std::string>())
  throw std::runtime_error("Engram metadata identity mismatch");

 const auto process_started=Clock::now();
 J report={{"schema_version",1},{"status","measurement_completed_requires_review"},
  {"scope","One cumulative multi-turn native session. No external oracle or qualification claim."},
  {"max_context_tokens",max_context},{"scheduled_context_tokens",total_context},
  {"corpus_tokens",corpus_required},{"turn_count",schedule.size()},
  {"wall_budget_seconds",wall_budget},
  {"resident_expert_atlas",dsv41::runtime_resident_expert_atlas_enabled()},
  {"grouped_expert_pipeline",dsv41::runtime_grouped_expert_pipeline_enabled()},
  {"fixed_tile_attention",dsv41::runtime_fixed_tile_attention_enabled()},
  {"ragged_tail_qk",dsv41::runtime_ragged_tail_qk_enabled()},
  {"ragged_tail_av",dsv41::runtime_ragged_tail_av_enabled()},
  {"batched_splitk_qk",dsv41::runtime_batched_splitk_qk_enabled()},
  {"deferred_decoder",deferred},
  {"deferred_decoder_clear_cache",dsv41::runtime_deferred_decoder_clear_cache_enabled()},
  {"effective_mlx_cache_limit_bytes",cache_limit},{"turns",J::array()}};
 std::ofstream progress(output.string()+".progress.jsonl");
 if(!progress)throw std::runtime_error("cannot create progress JSONL");

 dsv41::reset_packed_expert_bank_construction_count();
 dsv41::reset_expert_bank_io_stats();dsv41::reset_attention_telemetry();
 dsv41::reset_route_execution_stats();dsv41::reset_route_union_stats();
 auto construction_started=Clock::now();
 dsv41::WeightCatalog catalog(argv[1],argv[2]);
 auto metadata=dsv41::EngramMetadata::load(argv[3]);
 dsv41::TextBackboneReference model(catalog,metadata);mx::synchronize();
 report["model_construction"]={{"seconds",seconds(construction_started)},{"memory",memory()}};
 dsv41::TextBackboneState state(metadata);
 const auto session_started=Clock::now();
 std::optional<dsv41::BlockResult> last;
 std::optional<std::uint32_t> pending_token;
 std::size_t corpus_offset=0,context_position=0,total_deferred=0;
 std::vector<double> all_decode_latencies;

 for(std::size_t turn_index=0;turn_index<schedule.size();++turn_index){
  const auto turn=schedule[turn_index];
  const auto turn_started=Clock::now();
  const auto attention_before=dsv41::read_attention_telemetry();
  const auto routes_before=dsv41::route_execution_stats();
  const auto memory_before=memory();
  const auto revision_before=state.revision();
  std::vector<std::uint32_t> input;
  input.reserve(turn.append_tokens+(pending_token?1:0));
  if(pending_token)input.push_back(*pending_token);
  input.insert(input.end(),corpus.begin()+static_cast<std::ptrdiff_t>(corpus_offset),
               corpus.begin()+static_cast<std::ptrdiff_t>(corpus_offset+turn.append_tokens));
  corpus_offset+=turn.append_tokens;pending_token.reset();
  const auto append_started=Clock::now();
  std::size_t deferred_chunks=0;
  std::optional<dsv41::DeferredDecoderTransaction> pending_decoder;
  for(std::size_t offset=0;offset<input.size();){
   const auto step=dsv41::runtime_prefill_step(
    input.size()-offset,true,deferred,pending_decoder.has_value());
   const auto piece=std::span<const std::uint32_t>(input).subspan(offset,step.tokens);
   bool has_output=true;
   if(step.action==dsv41::RuntimePrefillStep::Action::BeginDeferredDecoder){
    pending_decoder.emplace(model.begin_deferred_decoder(piece,state,context_position+offset));
    has_output=false;
   }else if(step.action==dsv41::RuntimePrefillStep::Action::FinishDeferredDecoder){
    last.emplace(model.finish_deferred_decoder(std::move(*pending_decoder),piece,state));
    pending_decoder.reset();++deferred_chunks;
   }else last.emplace(model.forward_packed_sweep(piece,state,context_position+offset));
   if(has_output)evaluate(*last);
   offset+=step.tokens;
  }
  if(pending_decoder)throw std::runtime_error("turn ended with unpublished deferred decoder state");
  context_position+=input.size();
  require_state_position(state,context_position);
  const double append_seconds=seconds(append_started);

  std::vector<std::uint32_t> generated;
  std::vector<double> latencies;
  generated.reserve(turn.decode_tokens);latencies.reserve(turn.decode_tokens);
  auto first_started=Clock::now();
  auto logits=model.logits(*last);mx::eval(logits);mx::synchronize();
  auto finite=mx::all(mx::isfinite(logits));mx::eval(finite);
  if(!finite.item<bool>())throw std::runtime_error("nonfinite logits");
  generated.push_back(dsv41::greedy_reference(
   mx::slice(logits,{int(logits.shape(0))-1,0},{int(logits.shape(0)),129280})));
  latencies.push_back(seconds(first_started));
  for(std::size_t i=1;i<turn.decode_tokens;++i){
   auto decode_started=Clock::now();
   const std::array<std::uint32_t,1> token{generated.back()};
   last.emplace(model.forward_packed_sweep(token,state,context_position));evaluate(*last);++context_position;
   auto step_logits=model.logits(*last);mx::eval(step_logits);mx::synchronize();
   generated.push_back(dsv41::greedy_reference(step_logits));
   latencies.push_back(seconds(decode_started));
  }
  pending_token=generated.back();
  require_state_position(state,context_position);
  const std::size_t next_position=context_position+1;
  double decode_sum=0.0;
  for(double value:latencies){decode_sum+=value;all_decode_latencies.push_back(value);}
  total_deferred+=deferred_chunks;
  J turn_report={{"turn",turn_index+1},{"context_start",next_position-turn.append_tokens-turn.decode_tokens},
   {"new_prompt_tokens",turn.append_tokens},{"prefill_input_tokens",input.size()},
   {"decode_tokens",turn.decode_tokens},{"append_seconds",append_seconds},
   {"append_tokens_per_second",input.size()/append_seconds},
   {"deferred_decoder_chunks",deferred_chunks},{"first_decode_seconds",latencies.front()},
   {"decode_mean_seconds",decode_sum/latencies.size()},
   {"decode_p50_seconds",percentile(latencies,0.50)},
   {"decode_p95_seconds",percentile(latencies,0.95)},
   {"decode_p99_seconds",percentile(latencies,0.99)},
   {"decode_max_seconds",*std::max_element(latencies.begin(),latencies.end())},
   {"decode_latency_seconds",latencies},{"generated_token_ids",generated},
   {"state_position",context_position},{"next_position",next_position},
   {"state_revision_before",revision_before},{"state_revision_after",state.revision()},
   {"memory_before",memory_before},{"memory_after",memory()},
   {"attention_telemetry",attention_delta(attention_before,dsv41::read_attention_telemetry())},
   {"route_execution_stats",{{"device_batches",dsv41::route_execution_stats().device_batches-routes_before.device_batches},
    {"diagnostic_readbacks",dsv41::route_execution_stats().diagnostic_readbacks-routes_before.diagnostic_readbacks}}},
   {"turn_wall_seconds",seconds(turn_started)}};
  report["turns"].push_back(turn_report);progress<<turn_report.dump()<<'\n'<<std::flush;
  std::cout<<"turn "<<turn_index+1<<'/'<<schedule.size()<<" next_position="<<next_position
           <<" append_s="<<append_seconds<<" decode_mean_s="<<decode_sum/latencies.size()<<std::endl;
  if(wall_budget>0.0&&seconds(session_started)>wall_budget)
   throw std::runtime_error("cumulative session wall budget exceeded; completed turn progress is preserved");
 }
 if(corpus_offset!=corpus.size()||context_position+1!=total_context)
  throw std::runtime_error("final cumulative context geometry mismatch");

 const auto at=dsv41::read_attention_telemetry();
 const auto routes=dsv41::route_execution_stats();
 if(dsv41::runtime_resident_expert_atlas_enabled()){
  if(dsv41::packed_expert_bank_construction_count()!=40)
   throw std::runtime_error("resident atlas did not construct exactly 40 banks");
  if(routes.diagnostic_readbacks!=0)
   throw std::runtime_error("production session performed a route diagnostic host readback");
  if(at.index_host_readbacks!=0)
   throw std::runtime_error("production session performed an index diagnostic host readback");
 }
 if(at.fixed_tile_attention_calls==0||at.chunk_scalar_qk_calls!=0||at.chunk_scalar_av_calls!=0)
  throw std::runtime_error("production session left fixed-tile attention topology");
 double decode_sum=0.0;for(double value:all_decode_latencies)decode_sum+=value;
 report["aggregates"]={{"session_wall_seconds",seconds(session_started)},
  {"deferred_decoder_chunks",total_deferred},
  {"decode_tokens",all_decode_latencies.size()},
  {"decode_mean_seconds",decode_sum/all_decode_latencies.size()},
  {"decode_p50_seconds",percentile(all_decode_latencies,0.50)},
  {"decode_p95_seconds",percentile(all_decode_latencies,0.95)},
  {"decode_p99_seconds",percentile(all_decode_latencies,0.99)},
  {"decode_max_seconds",*std::max_element(all_decode_latencies.begin(),all_decode_latencies.end())}};
 report["final_state"]={{"state_position",context_position},{"next_position",context_position+1},
  {"revision",state.revision()},{"memory",memory()}};
 report["attention_telemetry"]={{"fixed_tile_attention_calls",at.fixed_tile_attention_calls},
  {"chunk_batched_splitk_qk_calls",at.chunk_batched_splitk_qk_calls},
  {"chunk_scalar_qk_calls",at.chunk_scalar_qk_calls},{"chunk_scalar_av_calls",at.chunk_scalar_av_calls},
  {"chunk_av_batches",at.chunk_av_batches},{"index_host_readbacks",at.index_host_readbacks},
  {"attention_rows",at.attention_rows},{"indexer_rows",at.indexer_rows},
  {"cumulative_bytes_copied",at.cumulative_bytes_copied}};
 report["route_execution_stats"]={{"device_batches",routes.device_batches},
  {"diagnostic_readbacks",routes.diagnostic_readbacks}};
 report["packed_expert_bank_constructions"]=dsv41::packed_expert_bank_construction_count();
 const auto sweep_stats=dsv41::read_sweep_telemetry();
 report["decode_stack_graph"]=dsv41::runtime_decode_stack_graph_enabled();
 report["sweep_telemetry"]={{"layer_evaluations",sweep_stats.layer_evaluations},
  {"deferred_layer_evaluations",sweep_stats.deferred_layer_evaluations},
  {"decode_stack_evaluations",sweep_stats.decode_stack_evaluations},
  {"engram_evaluations",sweep_stats.engram_evaluations}};
 last.reset();pending_token.reset();state.reset();mx::synchronize();
 report["recovery"]={{"after_state_reset",memory()}};
 mx::clear_cache();mx::synchronize();
 report["recovery"]["after_idle_cache_clear"]=memory();
 report["total_process_seconds"]=seconds(process_started);
 std::ofstream out(output);if(!out)throw std::runtime_error("cannot create output JSON");
 out<<report.dump(2)<<'\n';
 std::cout<<"PASS: completed cumulative session measurement; review required, not qualification"<<std::endl;
 }catch(const std::exception& error){std::cerr<<error.what()<<std::endl;return 1;}
}
