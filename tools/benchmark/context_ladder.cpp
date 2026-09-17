#include "dsv41/checkpoint_atlas.hpp"
#include "dsv41/execution_policy.hpp"
#include "dsv41/sampling.hpp"
#include "dsv41/text_backbone.hpp"
#include "dsv41/attention_telemetry.hpp"
#include "dsv41/runtime_profile.hpp"
#include <mlx/mlx.h>
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <vector>

namespace {
namespace mx=mlx::core;
using Clock=std::chrono::steady_clock;
using J=nlohmann::json;

double seconds(Clock::time_point start) {
 return std::chrono::duration<double>(Clock::now()-start).count();
}

double percentile(std::vector<double> values,double q) {
 std::sort(values.begin(),values.end());
 const auto index=std::min(values.size()-1,static_cast<std::size_t>(std::ceil(q*values.size())-1));
 return values[index];
}

std::vector<std::uint32_t> read_tokens(const std::filesystem::path& path) {
 std::ifstream in(path); if(!in) throw std::runtime_error("cannot open token file");
 std::vector<std::uint32_t> result; std::uint64_t value;
 while(in>>value) {
  if(value>=129280||value==129264) throw std::runtime_error("invalid text token");
  result.push_back(static_cast<std::uint32_t>(value));
 }
 if(!in.eof()||result.empty()) throw std::runtime_error("invalid token file");
 return result;
}

void require_state_position(const dsv41::TextBackboneState& state,std::uint64_t expected) {
 if(state.encoder.hash.position()!=expected) throw std::runtime_error("Engram state position mismatch");
 for(const auto& value:state.encoder.swa) if(value.position()!=expected) throw std::runtime_error("SWA state position mismatch");
 for(const auto& value:state.encoder.producer) if(value.position()!=expected) throw std::runtime_error("encoder producer position mismatch");
 for(const auto& value:state.encoder.reuse) if(value.position()!=expected) throw std::runtime_error("encoder reuse position mismatch");
 if(state.decoder.producer.position()!=expected) throw std::runtime_error("decoder producer position mismatch");
 for(const auto& value:state.decoder.reuse) if(value.position()!=expected) throw std::runtime_error("decoder reuse position mismatch");
}

J memory() {
 return {{"active_bytes",mx::get_active_memory()},{"cache_bytes",mx::get_cache_memory()},
         {"peak_bytes",mx::get_peak_memory()},{"limit_bytes",mx::get_memory_limit()}};
}

double nonnegative_environment_seconds(const char* name) {
 const char* text=std::getenv(name);
 if(text==nullptr||*text=='\0') return 0.0;
 std::size_t used=0;
 const double value=std::stod(text,&used);
 if(text[used]!='\0'||!std::isfinite(value)||value<0.0)
  throw std::runtime_error(std::string("invalid ")+name);
 return value;
}

void evaluate(const dsv41::BlockResult& value) {
 mx::eval(value.hidden,value.pre_mix); mx::synchronize();
 auto finite_hidden=mx::all(mx::isfinite(value.hidden));
 auto finite_pre=mx::all(mx::isfinite(value.pre_mix));
 mx::eval(finite_hidden,finite_pre);
 if(!finite_hidden.item<bool>()||!finite_pre.item<bool>()) throw std::runtime_error("nonfinite backbone output");
}
}

int main(int argc,char** argv) { try {
 const auto program_started=Clock::now();
 if(argc!=10) throw std::runtime_error(
  "usage: dsv41-context-ladder checkpoint m1-summary metadata tokens-file output-json context teacher tail decode");
 const auto checkpoint=std::filesystem::path(argv[1]);
 const auto output=std::filesystem::path(argv[5]);
 if(output.lexically_normal().string().starts_with(checkpoint.lexically_normal().string()))
  throw std::runtime_error("checkpoint is read-only");
 const std::size_t context=std::stoull(argv[6]),teacher=std::stoull(argv[7]);
 const std::size_t tail=std::stoull(argv[8]),decode=std::stoull(argv[9]);
 if(context<1||context>262144||decode<1||teacher<tail||context<=teacher+decode)
  throw std::runtime_error("invalid context phase geometry");
 const std::size_t prefill=context-decode,base=prefill-teacher,teacher_head=teacher-tail;
 const char* mode_env=std::getenv("DSV41_CONTEXT_EXECUTION");
 const std::string mode=mode_env?mode_env:"individual";
 if(mode!="individual"&&mode!="layer_major")throw std::runtime_error("invalid DSV41_CONTEXT_EXECUTION");
 const bool layer_major=mode=="layer_major";
 if(layer_major&&(!dsv41::runtime_packed_expert_bank_enabled()||dsv41::runtime_group_selected_experts_enabled()))
  throw std::runtime_error("layer_major requires packed bank enabled and selected grouping disabled");
 const double wall_budget=nonnegative_environment_seconds("DSV41_CONTEXT_WALL_BUDGET_SECONDS");
 const double projected_wall_limit=
  nonnegative_environment_seconds("DSV41_CONTEXT_PROJECTED_WALL_LIMIT_SECONDS");
 auto ids=read_tokens(argv[4]);
 if(ids.size()!=prefill) throw std::runtime_error("token file length does not match prefill geometry");

 mx::set_default_device(mx::Device::gpu);
 if(const auto limit=dsv41::runtime_mlx_cache_limit_bytes();limit)mx::set_cache_limit(limit);
 auto proof=dsv41::read_json_file("artifacts/engram/fixture-provenance.json");
 std::ifstream meta_file(argv[3],std::ios::binary);
 std::string meta_text{std::istreambuf_iterator<char>(meta_file),{}};
 if(dsv41::sha256_text(meta_text)!=proof.at("fixture_sha256").at("metadata.json").get<std::string>())
  throw std::runtime_error("Engram metadata identity mismatch");

 J report={{"schema_version",1},{"status","measurement_completed_requires_review"},
  {"scope","One native context-ladder measurement. No external oracle, cold-cache proof, acceptance threshold, or 256K qualification."},
  {"context_tokens",context},{"base_prefill_tokens",base},{"teacher_continuation_tokens",teacher},
  {"tail_teacher_tokens",tail},{"decode_tokens",decode},{"prefill_chunk_tokens",128},
  {"wall_budget_seconds",wall_budget},{"projected_wall_limit_seconds",projected_wall_limit},
  {"layer_finite_checks",dsv41::runtime_layer_finite_checks_enabled()},
  {"execution",mode},{"bank_construction_included_in_prefill",
   !dsv41::runtime_resident_expert_atlas_enabled()},
  {"packed_expert_bank",dsv41::runtime_packed_expert_bank_enabled()},
  {"resident_expert_atlas",dsv41::runtime_resident_expert_atlas_enabled()},
  {"compact_expert_bank",dsv41::runtime_compact_expert_bank_enabled()},
  {"route_diagnostics",dsv41::runtime_route_diagnostics_enabled()},
  {"index_diagnostics",dsv41::runtime_index_diagnostics_enabled()},
  {"chunk_attention",dsv41::runtime_chunk_attention_enabled()},
  {"mlx_cache_limit_bytes",dsv41::runtime_mlx_cache_limit_bytes()},
  {"expert_assignment_chunk",dsv41::runtime_expert_assignment_chunk()},
  {"component_profile",dsv41::runtime_component_profile_enabled()},
  {"token_file",argv[4]},{"phases",J::object()}};

 std::ofstream progress(output.string()+".progress.jsonl");
 if(!progress) throw std::runtime_error("cannot create progress JSONL");

 dsv41::reset_packed_expert_bank_construction_count();
 dsv41::reset_expert_bank_io_stats();
 auto started=Clock::now();
 dsv41::WeightCatalog catalog(argv[1],argv[2]);
 auto metadata=dsv41::EngramMetadata::load(argv[3]);
 dsv41::TextBackboneReference model(catalog,metadata);
 mx::synchronize();
 const auto construction_io=dsv41::expert_bank_io_stats();
 report["phases"]["model_construction"]={{"seconds",seconds(started)},{"memory",memory()},
  {"bank_constructions",dsv41::packed_expert_bank_construction_count()},
  {"loaded_experts",dsv41::packed_expert_bank_loaded_expert_count()},
  {"bank_read_calls",construction_io.read_calls},{"bank_read_seconds",construction_io.read_seconds},
  {"bank_total_seconds",construction_io.total_seconds}};
 dsv41::TextBackboneState state(metadata);
 dsv41::reset_attention_telemetry();
 dsv41::reset_runtime_profile();
 dsv41::reset_route_tie_count(); dsv41::reset_route_tie_records();
 dsv41::reset_route_union_stats();
 dsv41::reset_route_execution_stats();
 dsv41::reset_index_tie_count(); dsv41::reset_index_tie_records();
 std::optional<dsv41::BlockResult> last;

 auto feed=[&](std::size_t begin,std::size_t count,const char* label) {
  auto phase=Clock::now(),interval=phase; std::size_t chunks=0,interval_begin=begin;
 for(std::size_t offset=begin;offset<begin+count;) {
   const auto constructions_before=dsv41::packed_expert_bank_construction_count();
   const auto experts_before=dsv41::packed_expert_bank_loaded_expert_count();
   const auto union_before=dsv41::route_union_stats();
   const auto size=std::min<std::size_t>(128,begin+count-offset);
   auto chunk_started=Clock::now();
   auto input=std::span(ids).subspan(offset,size);
   last.emplace(layer_major?model.forward_packed_chunk(input,state,offset):model.forward(input,state,offset));
   evaluate(*last); offset+=size; ++chunks;
   if(dsv41::runtime_resident_expert_atlas_enabled()&&
      dsv41::packed_expert_bank_construction_count()!=constructions_before)
    throw std::runtime_error("resident expert atlas constructed a bank on the request path");
   if(const auto limit=dsv41::runtime_mlx_cache_limit_bytes();limit) mx::set_cache_limit(limit);
   if(layer_major) {
    J chunk_point={{"event","chunk"},{"phase",label},{"position",offset},{"tokens",size},
     {"seconds",seconds(chunk_started)},{"memory",memory()},
     {"bank_constructions",dsv41::packed_expert_bank_construction_count()-constructions_before},
     {"loaded_experts",dsv41::packed_expert_bank_loaded_expert_count()-experts_before},
     {"route_union_batches",dsv41::route_union_stats().batches-union_before.batches},
     {"route_union_selected_experts",dsv41::route_union_stats().selected_experts-union_before.selected_experts},
     {"route_union_unique_experts",dsv41::route_union_stats().unique_experts-union_before.unique_experts},
     {"route_union_overlap_experts",dsv41::route_union_stats().overlap_experts-union_before.overlap_experts},
     {"route_union_exact_reuses",dsv41::route_union_stats().exact_reuses-union_before.exact_reuses}};
    progress<<chunk_point.dump()<<'\n'<<std::flush;
   }
   if(layer_major||chunks%8==0||offset==begin+count) {
    const auto now=Clock::now();
    const double phase_wall=std::chrono::duration<double>(now-phase).count();
    const double interval_wall=std::chrono::duration<double>(now-interval).count();
    const std::size_t phase_tokens=offset-begin,interval_tokens=offset-interval_begin;
    const double projected_prefill=phase_tokens?phase_wall*double(prefill)/double(phase_tokens):0.0;
    const auto mem=memory();
    J point={{"phase",label},{"position",offset},{"prefill_tokens",prefill},
     {"phase_elapsed_seconds",phase_wall},{"interval_seconds",interval_wall},
     {"interval_tokens",interval_tokens},{"interval_tokens_per_second",interval_tokens/interval_wall},
     {"phase_tokens_per_second",phase_tokens/phase_wall},
     {"projected_prefill_seconds_at_current_average",projected_prefill},{"memory",mem},
     {"route_tie_count",dsv41::route_tie_count()},{"index_tie_count",dsv41::index_tie_count()}};
    progress<<point.dump()<<'\n'<<std::flush;
    std::cout<<"progress: "<<label<<" position="<<offset<<"/"<<prefill
             <<" interval_tps="<<point["interval_tokens_per_second"]
             <<" active_gib="<<double(mem["active_bytes"].get<std::uint64_t>())/(1ull<<30)
             <<" cache_gib="<<double(mem["cache_bytes"].get<std::uint64_t>())/(1ull<<30)
             <<" projected_prefill_s="<<projected_prefill<<std::endl;
    interval=now; interval_begin=offset;
    if(wall_budget>0.0&&seconds(program_started)>wall_budget)
     throw std::runtime_error("context measurement wall budget exceeded; partial progress JSONL is preserved");
    if(projected_wall_limit>0.0&&offset>=2048&&projected_prefill>projected_wall_limit)
     throw std::runtime_error("projected prefill wall exceeds limit; partial progress JSONL is preserved");
   }
  }
  require_state_position(state,begin+count);
  const double wall=seconds(phase);
  report["phases"][label]={{"tokens",count},{"chunks",chunks},{"seconds",wall},
                            {"tokens_per_second",count/wall},{"end_position",begin+count},
                            {"memory",memory()}};
 };

 feed(0,base,"base_prefill");
 feed(base,teacher_head,"teacher_head");
 feed(base+teacher_head,tail,"teacher_tail");
 if(dsv41::runtime_resident_expert_atlas_enabled()&&
    dsv41::packed_expert_bank_construction_count()!=40)
  throw std::runtime_error("resident expert atlas did not construct exactly one bank per layer");

 const double base_seconds=report["phases"]["base_prefill"]["seconds"].get<double>();
 const double teacher_seconds=report["phases"]["teacher_head"]["seconds"].get<double>()+
  report["phases"]["teacher_tail"]["seconds"].get<double>();
 const double prefill_seconds=base_seconds+teacher_seconds;
 report["aggregates"]={{"prefill_seconds",prefill_seconds},
  {"prefill_tokens_per_second",prefill/prefill_seconds},
  {"teacher_continuation_seconds",teacher_seconds},
  {"teacher_continuation_tokens_per_second",teacher/teacher_seconds}};

 auto first_started=Clock::now();
 auto logits=model.logits(*last); mx::eval(logits); mx::synchronize();
 auto finite_logits=mx::all(mx::isfinite(logits)); mx::eval(finite_logits);
 if(!finite_logits.item<bool>()) throw std::runtime_error("nonfinite logits");
 std::vector<std::uint32_t> generated{dsv41::greedy_reference(
  mx::slice(logits,{int(logits.shape(0))-1,0},{int(logits.shape(0)),129280}))};
 std::vector<double> latencies{seconds(first_started)};
 for(std::size_t i=1;i<decode;++i) {
  auto decode_started=Clock::now();
  const std::array<std::uint32_t,1> one{generated.back()};
  auto value=layer_major?model.forward_packed_chunk(one,state,prefill+i-1):model.forward(one,state,prefill+i-1); evaluate(value);
  auto step=model.logits(value); mx::eval(step); mx::synchronize();
  generated.push_back(dsv41::greedy_reference(step));
  latencies.push_back(seconds(decode_started));
  std::cout<<"decode: "<<i+1<<"/"<<decode<<" token="<<generated.back()<<std::endl;
 }
 require_state_position(state,prefill+decode-1);
 double sum=0,maximum=0; for(double value:latencies){sum+=value;maximum=std::max(maximum,value);}
 const double request_to_first=
  report["phases"]["base_prefill"]["seconds"].get<double>()+
  report["phases"]["teacher_head"]["seconds"].get<double>()+
  report["phases"]["teacher_tail"]["seconds"].get<double>()+latencies.front();
 report["phases"]["decode"]={{"tokens",decode},{"latency_seconds",latencies},
  {"first_token_seconds",latencies.front()},{"request_to_first_token_seconds",request_to_first},
  {"mean_seconds",sum/decode},{"p50_seconds",percentile(latencies,0.50)},
  {"p95_seconds",percentile(latencies,0.95)},{"p99_seconds",percentile(latencies,0.99)},
  {"max_seconds",maximum},
  {"generated_token_ids",generated},{"state_position",prefill+decode-1},{"next_position",context},
  {"memory",memory()}};
 if(dsv41::runtime_resident_expert_atlas_enabled()&&!dsv41::runtime_route_diagnostics_enabled()){
  const auto route_stats=dsv41::route_execution_stats();
  const auto expert_stats=dsv41::expert_bank_io_stats();
  if(route_stats.diagnostic_readbacks!=0)
   throw std::runtime_error("resident production path performed a route diagnostic readback");
  if(expert_stats.expert_major_batches!=route_stats.device_batches)
   throw std::runtime_error("resident route batch did not use expert-major execution");
 }
 report["route_tie_count"]=dsv41::route_tie_count();
 report["packed_expert_bank_constructions"]=dsv41::packed_expert_bank_construction_count();
 report["packed_expert_bank_loaded_experts"]=dsv41::packed_expert_bank_loaded_expert_count();
 auto bank_io=dsv41::expert_bank_io_stats();
 report["expert_bank_io_stats"]={{"constructions",bank_io.constructions},
  {"read_calls",bank_io.read_calls},{"qmm_dispatches",bank_io.qmm_dispatches},
  {"qmm_rows_total",bank_io.qmm_rows_total},{"qmm_rows_max",bank_io.qmm_rows_max},
  {"expert_major_batches",bank_io.expert_major_batches},
  {"expert_major_assignments",bank_io.expert_major_assignments},
  {"read_seconds",bank_io.read_seconds},
  {"total_seconds",bank_io.total_seconds}};
 auto route_execution=dsv41::route_execution_stats();
 report["route_execution_stats"]={{"device_batches",route_execution.device_batches},
  {"diagnostic_readbacks",route_execution.diagnostic_readbacks}};
 auto at=dsv41::read_attention_telemetry();
 if(dsv41::runtime_resident_expert_atlas_enabled()&&!dsv41::runtime_index_diagnostics_enabled()&&
    at.index_host_readbacks!=0)throw std::runtime_error("resident production path performed an index result readback");
 if(dsv41::runtime_chunk_attention_enabled()){
  if(at.chunk_attention_calls==0)throw std::runtime_error("chunk attention was enabled but never invoked");
  if(at.chunk_av_batches==0)throw std::runtime_error("chunk attention produced no AV batches");
  if(at.chunk_scalar_av_calls!=0)throw std::runtime_error("chunk attention regressed to scalar AV calls");
 }
 report["attention_telemetry"]={{"concat_calls",at.concat_calls},{"concat_input_bytes",at.concat_input_bytes},
  {"concat_output_bytes",at.concat_output_bytes},{"cumulative_bytes_copied",at.cumulative_bytes_copied},
  {"logical_tokens",at.logical_tokens},{"attention_rows",at.attention_rows},{"indexer_rows",at.indexer_rows},
  {"index_host_readbacks",at.index_host_readbacks},{"token_serial_attention_calls",at.token_serial_attention_calls},
  {"chunk_attention_calls",at.chunk_attention_calls},{"chunk_scalar_qk_calls",at.chunk_scalar_qk_calls},
  {"chunk_scalar_av_calls",at.chunk_scalar_av_calls},{"chunk_av_batches",at.chunk_av_batches}};
 if(dsv41::runtime_component_profile_enabled()){
  const auto profile=dsv41::read_runtime_profile();J layers=J::array();
  double layer_total=0.0,attention_total=0.0,moe_total=0.0,post_total=0.0;
  for(int layer=0;layer<40;++layer){
   layer_total+=profile.layer_seconds[layer];attention_total+=profile.attention_path_seconds[layer];
   moe_total+=profile.moe_path_seconds[layer];post_total+=profile.post_moe_seconds[layer];
   layers.push_back({{"layer",layer},{"calls",profile.layer_calls[layer]},
    {"component_calls",profile.component_calls[layer]},{"layer_seconds",profile.layer_seconds[layer]},
    {"attention_path_seconds",profile.attention_path_seconds[layer]},
    {"moe_path_seconds",profile.moe_path_seconds[layer]},
    {"post_moe_seconds",profile.post_moe_seconds[layer]}});
  }
  report["runtime_component_profile"]={{"measurement_semantics",
   "GPU completion wall with synchronization after attention, MoE, and post-MoE; perturbs the normal lazy schedule"},
   {"layer_seconds",layer_total},{"attention_path_seconds",attention_total},
   {"moe_path_seconds",moe_total},{"post_moe_seconds",post_total},{"layers",std::move(layers)}};
 }
 auto union_stats=dsv41::route_union_stats();
 report["route_union_stats"]={{"batches",union_stats.batches},
  {"selected_experts",union_stats.selected_experts},
  {"unique_experts",union_stats.unique_experts},
  {"overlap_experts",union_stats.overlap_experts},
  {"exact_reuses",union_stats.exact_reuses},
  {"mean_unique_per_batch",union_stats.batches?double(union_stats.unique_experts)/union_stats.batches:0.0},
  {"overlap_ratio",union_stats.unique_experts?double(union_stats.overlap_experts)/union_stats.unique_experts:0.0}};
 report["route_ties"]=J::array();
 for(const auto& tie:dsv41::route_tie_records()) report["route_ties"].push_back({
  {"layer",tie.layer},{"token",tie.token},{"sixth_id",tie.sixth_id},{"seventh_id",tie.seventh_id},
  {"sixth_score",tie.sixth_score},{"seventh_score",tie.seventh_score}});
 report["index_tie_count"]=dsv41::index_tie_count();
 report["index_tie_records_truncated"]=dsv41::index_tie_count()>dsv41::index_tie_records().size();
 report["index_ties"]=J::array();
 for(const auto& tie:dsv41::index_tie_records()) report["index_ties"].push_back({
  {"layer",tie.layer},{"token",tie.token},{"selected_id",tie.selected_id},
  {"excluded_id",tie.excluded_id},{"selected_score",tie.selected_score},
  {"excluded_score",tie.excluded_score},{"score_bits",std::bit_cast<std::uint32_t>(tie.selected_score)},
  {"candidates",tie.candidates}});
 report["final_memory"]=memory();
 report["total_process_seconds"]=seconds(program_started);
 std::ofstream out(output); if(!out) throw std::runtime_error("cannot create output JSON");
 out<<report.dump(2)<<'\n';
 std::cout<<"PASS: completed repo canonical "<<context<<"-token measurement; review required, not qualification"<<std::endl;
 } catch(const std::exception& e) { std::cerr<<e.what()<<std::endl; return 1; }
}
