#include "dsv41/text_backbone.hpp"
#include "dsv41/execution_policy.hpp"
#include "dsv41/attention_telemetry.hpp"
#include "dsv41/generation_loop.hpp"
#include "dsv41/deferred_decoder_plan.hpp"
#include <bit>
#include <cstdlib>
#include <iostream>
#include "dsv41/checkpoint_atlas.hpp"
#include <fstream>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <tuple>
#include <map>
namespace mx=mlx::core;
class MemoryTraceSink final:public dsv41::TraceSink {
public:
 void record(const std::string& name,const mx::array& value) override {
  auto [it,inserted]=values.emplace(name,value);
  if(!inserted)it->second=mx::concatenate({it->second,value},0);
 }
 const mx::array& at(const std::string& name) const {
  auto it=values.find(name);if(it==values.end())throw std::runtime_error("missing layer trace: "+name);
  return it->second;
 }
private:
 std::map<std::string,mx::array> values;
};
void same(const mx::array& a,const mx::array& b,const char* what){
 if(a.shape()!=b.shape()||a.dtype()!=b.dtype())throw std::runtime_error(std::string(what)+": shape/dtype mismatch");
 auto dtype=a.dtype()==mx::bfloat16?mx::uint16:mx::uint32;
 auto ok=mx::all(mx::equal(mx::view(a,dtype),mx::view(b,dtype)));mx::eval(ok);
 if(!ok.item<bool>())throw std::runtime_error(std::string(what)+": bit mismatch");
}
void semantic_same(const mx::array& candidate,const mx::array& reference,const char* what,bool logits=false){
 if(candidate.shape()!=reference.shape()||candidate.dtype()!=reference.dtype())
  throw std::runtime_error(std::string(what)+": shape/dtype mismatch");
 auto c=mx::astype(candidate,mx::float32),r=mx::astype(reference,mx::float32),d=mx::subtract(c,r);
 auto relative=mx::sqrt(mx::divide(mx::sum(mx::multiply(d,d)),
  mx::maximum(mx::sum(mx::multiply(r,r)),mx::array(1e-30f))));
 auto maximum=mx::max(mx::abs(d)),mean=mx::mean(mx::abs(d));
 auto mismatches=mx::sum(mx::astype(mx::not_equal(candidate,reference),mx::uint32));
 auto finite=mx::logical_and(mx::all(mx::isfinite(c)),mx::all(mx::isfinite(r)));
 mx::array decisions(true);
 if(logits)decisions=mx::all(mx::equal(mx::argmax(c,-1),mx::argmax(r,-1)));
 mx::eval(relative,maximum,mean,mismatches,finite,decisions);
 std::cout<<what<<" relative_rms="<<relative.item<float>()<<" max_abs="<<maximum.item<float>()
          <<" mean_abs="<<mean.item<float>()<<" bit_mismatches="<<mismatches.item<std::uint32_t>()<<std::endl;
 if(!finite.item<bool>()||relative.item<float>()>=0.002f||!decisions.item<bool>())
  throw std::runtime_error(std::string(what)+": semantic gate failed");
}
void bytes(const mx::array& a,const mx::array& b,const char* what){
 if(a.shape()!=b.shape()||a.dtype()!=b.dtype())throw std::runtime_error(std::string(what)+": shape/dtype mismatch");
 auto ok=mx::all(mx::equal(a,b));mx::eval(ok);if(!ok.item<bool>())throw std::runtime_error(std::string(what)+": byte mismatch");
}
void producer_same(const dsv41::CompressedLayerState& a,const dsv41::CompressedLayerState& b,const char* what){
 if(a.position()!=b.position()||a.global().rows()!=b.global().rows())throw std::runtime_error(std::string(what)+": position mismatch");
 same(a.window(),b.window(),"producer window");
 bytes(a.global().main_bytes(),b.global().main_bytes(),"producer main bytes");
 bytes(a.global().main_scales(),b.global().main_scales(),"producer main scales");
 bytes(a.global().index_bytes(),b.global().index_bytes(),"producer index bytes");
 bytes(a.global().index_scales(),b.global().index_scales(),"producer index scales");
 same(a.global().compressor().pending_kv(),b.global().compressor().pending_kv(),"producer pending kv");
 same(a.global().compressor().pending_scores(),b.global().compressor().pending_scores(),"producer pending scores");
}
void state_same(const dsv41::TextBackboneState& a,const dsv41::TextBackboneState& b){
 if(a.encoder.hash.position()!=b.encoder.hash.position())throw std::runtime_error("hash position mismatch");
 for(int i=0;i<2;++i){if(a.encoder.swa[i].position()!=b.encoder.swa[i].position())throw std::runtime_error("swa position mismatch");same(a.encoder.swa[i].rows(),b.encoder.swa[i].rows(),"swa rows");}
 for(int i=0;i<3;++i)producer_same(a.encoder.producer[i],b.encoder.producer[i],"encoder producer");
 for(int i=0;i<15;++i){if(a.encoder.reuse[i].position()!=b.encoder.reuse[i].position())throw std::runtime_error("encoder reuse position mismatch");same(a.encoder.reuse[i].window(),b.encoder.reuse[i].window(),"encoder reuse window");}
 producer_same(a.decoder.producer,b.decoder.producer,"decoder producer");
 for(int i=0;i<19;++i){if(a.decoder.reuse[i].position()!=b.decoder.reuse[i].position())throw std::runtime_error("decoder reuse position mismatch");same(a.decoder.reuse[i].window(),b.decoder.reuse[i].window(),"decoder reuse window");}
}
void ties_same(const std::vector<dsv41::RouteTieRecord>& a,const std::vector<dsv41::RouteTieRecord>& b,const char* what){
 if(a.size()!=b.size())throw std::runtime_error(std::string(what)+": count mismatch");
 for(std::size_t i=0;i<a.size();++i){
  if(a[i].layer!=b[i].layer||a[i].token!=b[i].token||a[i].sixth_id!=b[i].sixth_id||
     a[i].seventh_id!=b[i].seventh_id||a[i].sixth_score!=b[i].sixth_score||
     a[i].seventh_score!=b[i].seventh_score||a[i].tied!=b[i].tied)
   throw std::runtime_error(std::string(what)+": record mismatch");
 }
}
std::vector<std::uint32_t> tokens(const char* path){
 std::ifstream file(path);if(!file)throw std::runtime_error("cannot open token file");
 std::vector<std::uint32_t> result;std::uint64_t token;
 while(file>>token){if(token>=129280||token==129264)throw std::runtime_error("invalid text token");result.push_back(std::uint32_t(token));}
 if(!file.eof()||result.empty()||result.size()>262144)throw std::runtime_error("invalid token file");
 return result;
}
int main(int argc,char** argv){try{
 if(argc!=4&&argc!=5)throw std::runtime_error("usage: dsv41-text-backbone-test checkpoint m1-summary metadata [tokens]");
 mx::set_default_device(mx::Device::gpu);dsv41::WeightCatalog c(argv[1],argv[2]);
 auto metadata=dsv41::EngramMetadata::load(argv[3]);
 auto proof=dsv41::read_json_file("artifacts/engram/fixture-provenance.json");
 std::ifstream file(argv[3],std::ios::binary);std::string raw{std::istreambuf_iterator<char>(file),{}};
 if(dsv41::sha256_text(raw)!=proof.at("fixture_sha256").at("metadata.json").get<std::string>())throw std::runtime_error("Engram metadata identity mismatch");
 if(std::getenv("DSV41_CHECK_FILE_BACKED_EXACTNESS")){
  if(!std::getenv("DSV41_ALLOW_UNSAFE_IN_PROCESS_FILE_EXACTNESS"))
   throw std::runtime_error("in-process file-backed exactness is rejected for host-memory safety; use process-separated digest runner");
  const char* configured=std::getenv("DSV41_RUNTIME_EXPERT_BACKING_DIR");
  if(!configured||!*configured)throw std::runtime_error("file-backed exactness requires backing directory");
  const std::string backing=configured;setenv("DSV41_RUNTIME_WIRED_LIMIT_BYTES","0",1);unsetenv("DSV41_RUNTIME_EXPERT_BACKING_DIR");
  dsv41::TextBackboneState expected_state(metadata);std::vector<dsv41::BlockResult> outputs;std::vector<mx::array> logits;std::vector<dsv41::TextBackboneState> states;
  std::vector<std::uint32_t> prompt(129);for(std::size_t i=0;i<prompt.size();++i)prompt[i]=std::uint32_t((i*7919)%129263);
  dsv41::reset_route_tie_records();
  {
   auto baseline=std::make_unique<dsv41::TextBackboneReference>(c,metadata);outputs.push_back(baseline->forward_packed_sweep(prompt,expected_state,0));logits.push_back(baseline->logits(outputs.back()));states.push_back(expected_state);
   for(std::uint64_t pos=129;pos<133;++pos){const std::array<std::uint32_t,1> token{std::uint32_t(pos*17)};outputs.push_back(baseline->forward_packed_sweep(token,expected_state,pos));logits.push_back(baseline->logits(outputs.back()));states.push_back(expected_state);}
   for(auto& x:outputs)mx::eval(x.hidden,x.pre_mix);for(auto& x:logits)mx::eval(x);mx::synchronize();
  }
  auto expected_ties=dsv41::route_tie_records();mx::clear_cache();setenv("DSV41_RUNTIME_EXPERT_BACKING_DIR",backing.c_str(),1);dsv41::reset_route_tie_records();
  dsv41::TextBackboneReference candidate(c,metadata);if(!candidate.expert_backing_file_backed())throw std::runtime_error("candidate did not map file-backed experts");dsv41::TextBackboneState actual_state(metadata);
  auto check=[&](std::size_t i,const dsv41::BlockResult& actual){same(actual.hidden,outputs[i].hidden,"file-backed hidden");same(actual.pre_mix,outputs[i].pre_mix,"file-backed pre-mix");same(candidate.logits(actual),logits[i],"file-backed logits");state_same(actual_state,states[i]);};
  auto actual=candidate.forward_packed_sweep(prompt,actual_state,0);check(0,actual);
  for(std::uint64_t pos=129;pos<133;++pos){const std::array<std::uint32_t,1> token{std::uint32_t(pos*17)};actual=candidate.forward_packed_sweep(token,actual_state,pos);check(std::size_t(pos-128),actual);}
  ties_same(dsv41::route_tie_records(),expected_ties,"file-backed route ties");
  auto saved=actual_state;const std::array<std::uint32_t,1> invalid{129264};bool rejected=false;try{candidate.forward_packed_sweep(invalid,actual_state,133);}catch(const std::exception&){rejected=true;}
  if(!rejected||actual_state.revision()!=saved.revision())throw std::runtime_error("file-backed invalid request atomicity");state_same(actual_state,saved);
  auto ah=actual_state.encoder.hash,bh=expected_state.encoder.hash;const std::array<std::uint32_t,1> suffix{42};if(ah.append(suffix,{},133)!=bh.append(suffix,{},133))throw std::runtime_error("file-backed hash mismatch");
  std::cout<<"PASS: file-backed prefill/decode hidden, pre-mix, logits, state, route ties, hash, revision and invalid-request atomicity are bit-exact; no performance qualification"<<std::endl;return 0;
 }
 std::cout<<"Loading full backbone layers 0..39 on-demand experts and Engram 1/14 mmap backing"<<std::endl;
 dsv41::TextBackboneReference model(c,metadata);
 if(std::getenv("DSV41_CHECK_FUSED_MHC_BACKBONE_PARITY")){
  // The fused mHC kernel is a faithful fusion of the reference arithmetic, so
  // the primary requirement is bit identity through all 40 layers. A mismatch
  // is investigated (reduction order / precision / contraction), never
  // silently weakened to a semantic gate.
  if(!dsv41::runtime_resident_expert_atlas_enabled())
   throw std::runtime_error("fused mHC parity requires the resident expert atlas");
  std::vector<std::uint32_t> input(256);
  for(std::size_t i=0;i<input.size();++i)input[i]=std::uint32_t((i*7919)%129263);
  auto run=[&](bool fused,dsv41::TextBackboneState& state){
   if(setenv("DSV41_RUNTIME_FUSED_MHC",fused?"1":"0",1)!=0)
    throw std::runtime_error("cannot select mHC policy");
   dsv41::reset_route_tie_count();dsv41::reset_route_tie_records();
   std::vector<mx::array> hidden,pre;
   dsv41::run_prefill_chunks(input.size(),128,[&](std::size_t offset,std::size_t count){
    auto part=model.forward_packed_sweep(std::span(input).subspan(offset,count),state,offset);
    hidden.push_back(part.hidden);pre.push_back(part.pre_mix);
   });
   dsv41::BlockResult result{mx::concatenate(hidden,0),mx::concatenate(pre,0)};
   return std::pair{std::move(result),dsv41::route_tie_records()};
  };
  dsv41::TextBackboneState reference_state(metadata),fused_state(metadata);
  auto [reference,reference_ties]=run(false,reference_state);
  auto [fused,fused_ties]=run(true,fused_state);
  if(setenv("DSV41_RUNTIME_FUSED_MHC","0",1)!=0)throw std::runtime_error("cannot restore mHC policy");
  same(fused.hidden,reference.hidden,"fused mHC hidden bits");
  same(fused.pre_mix,reference.pre_mix,"fused mHC pre-mix bits");
  const int row=int(input.size()-1);
  dsv41::BlockResult fused_last{mx::slice(fused.hidden,{row,0,0},{row+1,4,5120}),
                                mx::slice(fused.pre_mix,{row,0},{row+1,4})};
  dsv41::BlockResult reference_last{mx::slice(reference.hidden,{row,0,0},{row+1,4,5120}),
                                    mx::slice(reference.pre_mix,{row,0},{row+1,4})};
  same(model.logits(fused_last),model.logits(reference_last),"fused mHC logits bits");
  state_same(fused_state,reference_state);
  ties_same(fused_ties,reference_ties,"fused mHC route ties");
  std::cout<<"PASS: fused mHC 2x128-token 40-layer hidden, pre-mix, state, logits and "
   <<"route ties are bit-exact; no performance qualification"<<std::endl;
  return 0;
 }
 if(std::getenv("DSV41_CHECK_FUSED_MHC_DECODE_PARITY")){
  if(!dsv41::runtime_resident_expert_atlas_enabled())
   throw std::runtime_error("fused mHC decode parity requires the resident expert atlas");
  dsv41::TextBackboneState baseline(metadata),candidate(metadata);
  std::vector<std::uint32_t> prompt(129);
  for(std::size_t i=0;i<prompt.size();++i)prompt[i]=std::uint32_t((i*7919)%129263);
  if(setenv("DSV41_RUNTIME_FUSED_MHC","0",1)!=0)throw std::runtime_error("cannot select reference mHC");
  model.forward_packed_sweep(prompt,baseline,0);candidate=baseline;
  for(std::uint64_t pos=129;pos<133;++pos){
   const std::array<std::uint32_t,1> token{std::uint32_t(pos*17)};
   if(setenv("DSV41_RUNTIME_FUSED_MHC","0",1)!=0)throw std::runtime_error("cannot select reference mHC");
   auto expected=model.forward_packed_sweep(token,baseline,pos);
   if(setenv("DSV41_RUNTIME_FUSED_MHC","1",1)!=0)throw std::runtime_error("cannot select fused mHC");
   auto actual=model.forward_packed_sweep(token,candidate,pos);
   same(actual.hidden,expected.hidden,"fused mHC decode hidden");
   same(actual.pre_mix,expected.pre_mix,"fused mHC decode pre-mix");
   same(model.logits(actual),model.logits(expected),"fused mHC decode logits");
   state_same(candidate,baseline);
   if(candidate.revision()!=baseline.revision())throw std::runtime_error("fused mHC decode revision mismatch");
   auto publication_same=[&](const auto& a,const auto& b,int consumer){
    const auto& p=*a.publication();const auto& q=*b.publication();
    if(p.source_layer()!=q.source_layer()||p.index_source_layer()!=q.index_source_layer())
     throw std::runtime_error("fused mHC decode publication source mismatch");
    bytes(p.device_indices(consumer,pos,128),q.device_indices(consumer,pos,128),"fused mHC decode indices");
    bytes(p.device_candidates(),q.device_candidates(),"fused mHC decode candidates");
   };
   for(int i=0;i<3;++i)publication_same(candidate.encoder.producer[i],baseline.encoder.producer[i],3+6*i);
   publication_same(candidate.decoder.producer,baseline.decoder.producer,39);
  }
  if(setenv("DSV41_RUNTIME_FUSED_MHC","0",1)!=0)throw std::runtime_error("cannot restore mHC policy");
  auto saved=candidate;const std::array<std::uint32_t,1> invalid{129264};bool rejected=false;
  try{model.forward_packed_sweep(invalid,candidate,133);}catch(const std::exception&){rejected=true;}
  if(!rejected||candidate.revision()!=saved.revision())throw std::runtime_error("fused mHC decode invalid request atomicity");
  state_same(candidate,saved);
  auto ah=candidate.encoder.hash,bh=baseline.encoder.hash;
  const std::array<std::uint32_t,1> suffix{42};
  if(ah.append(suffix,{},133)!=bh.append(suffix,{},133))throw std::runtime_error("fused mHC decode hash mismatch");
  std::cout<<"PASS: fused mHC decode advancing-token hidden/pre-mix/logits/state/publication/"
   <<"hash/revision and invalid request atomicity are bit-exact; no performance qualification"<<std::endl;
  return 0;
 }
 if(std::getenv("DSV41_CHECK_DECODE_STACK_GRAPH")||std::getenv("DSV41_CHECK_RESIDENCY")){
  const bool residency_check=std::getenv("DSV41_CHECK_RESIDENCY")!=nullptr;
  if(residency_check&&(!model.wired_limit_bytes()||dsv41::runtime_decode_stack_graph_enabled()))
   throw std::runtime_error("residency parity requires a wired model and decode stack graph OFF");
  auto select=[&](bool candidate){
   if(residency_check){mx::synchronize();mx::set_wired_limit(candidate?model.wired_limit_bytes():0);}
   else setenv("DSV41_RUNTIME_DECODE_STACK_GRAPH",candidate?"1":"0",1);
  };
  if(!dsv41::runtime_resident_expert_atlas_enabled())throw std::runtime_error("decode graph gate requires resident atlas");
  dsv41::TextBackboneState baseline(metadata),candidate(metadata);
  std::vector<std::uint32_t> prompt(129);
  for(std::size_t i=0;i<prompt.size();++i)prompt[i]=std::uint32_t((i*7919)%129263);
  select(false);
  auto prefix_expected=model.forward_packed_sweep(prompt,baseline,0);
  if(residency_check){
   select(true);auto prefix_actual=model.forward_packed_sweep(prompt,candidate,0);
   same(prefix_actual.hidden,prefix_expected.hidden,"residency prefill hidden");
   same(prefix_actual.pre_mix,prefix_expected.pre_mix,"residency prefill pre-mix");
   same(model.logits(prefix_actual),model.logits(prefix_expected),"residency prefill logits");
   state_same(candidate,baseline);
  }else candidate=baseline;
  for(std::uint64_t pos=129;pos<133;++pos){
   const std::array<std::uint32_t,1> token{std::uint32_t(pos*17)};
   select(false);
   auto expected=model.forward_packed_sweep(token,baseline,pos);
   select(true);
   auto actual=model.forward_packed_sweep(token,candidate,pos);
   same(actual.hidden,expected.hidden,"decode graph hidden");
   same(actual.pre_mix,expected.pre_mix,"decode graph pre-mix");
   same(model.logits(actual),model.logits(expected),"decode graph logits");
   state_same(candidate,baseline);
   if(candidate.revision()!=baseline.revision())throw std::runtime_error("decode graph revision mismatch");
   auto publication_same=[&](const auto& a,const auto& b,int consumer){
    const auto& p=*a.publication();const auto& q=*b.publication();
    if(p.source_layer()!=q.source_layer()||p.index_source_layer()!=q.index_source_layer())
     throw std::runtime_error("decode graph publication source mismatch");
    bytes(p.device_indices(consumer,pos,128),q.device_indices(consumer,pos,128),"decode graph indices");
    bytes(p.device_candidates(),q.device_candidates(),"decode graph candidates");
   };
   for(int i=0;i<3;++i)publication_same(candidate.encoder.producer[i],baseline.encoder.producer[i],3+6*i);
   publication_same(candidate.decoder.producer,baseline.decoder.producer,39);
  }
  auto saved=candidate;const std::array<std::uint32_t,1> invalid{129264};bool rejected=false;
  try{model.forward_packed_sweep(invalid,candidate,133);}catch(const std::exception&){rejected=true;}
  if(!rejected||candidate.revision()!=saved.revision())throw std::runtime_error("decode graph invalid request atomicity");
  state_same(candidate,saved);
  auto ah=candidate.encoder.hash,bh=baseline.encoder.hash;
  const std::array<std::uint32_t,1> suffix{42};
  if(ah.append(suffix,{},133)!=bh.append(suffix,{},133))throw std::runtime_error("decode graph hash mismatch");
  std::cout<<"PASS: "<<(residency_check?"residency policy":"decode stack graph")
   <<" hidden/pre-mix/logits/state/publication/hash/revision and invalid request atomicity; no performance qualification"<<std::endl;
  return 0;
 }
 if(std::getenv("DSV41_CHECK_LAYER_MAJOR_BACKBONE")){
  const bool compact_check=std::getenv("DSV41_CHECK_COMPACT_LAYER_MAJOR_BACKBONE")!=nullptr;
  const bool chunk_attention_check=std::getenv("DSV41_CHECK_CHUNK_ATTENTION_BACKBONE")!=nullptr;
  if(dsv41::runtime_packed_expert_bank_enabled())throw std::runtime_error("comparison must start in individual mode");
  if(setenv("DSV41_RUNTIME_PACKED_EXPERT_BANK","1",1)!=0||
     setenv("DSV41_RUNTIME_COMPACT_EXPERT_BANK",compact_check?"1":"0",1)!=0)
   throw std::runtime_error("cannot select packed model");
  dsv41::TextBackboneReference packed(c,metadata);
  dsv41::reset_packed_expert_bank_construction_count();
  dsv41::TextBackboneState expected_state(metadata),actual_state(metadata);
  auto full_state_same=[&](const auto& a,const auto& b){
   state_same(a,b);
   auto ah=a.encoder.hash,bh=b.encoder.hash;const std::array<std::uint32_t,4> suffix{42,17,1000,7};
   if(ah.append(suffix,{},ah.position())!=bh.append(suffix,{},bh.position()))throw std::runtime_error("hash history mismatch");
   auto publication_same=[&](const auto& x,const auto& y,int consumer){
    if(bool(x.publication())!=bool(y.publication()))throw std::runtime_error("publication presence mismatch");
    if(!x.publication())return;
    const auto& p=*x.publication();const auto& q=*y.publication();const auto pos=x.position()-1;
    if(p.source_layer()!=q.source_layer()||p.index_source_layer()!=q.index_source_layer()||
       p.indices(consumer,pos,pos?128:1)!=q.indices(consumer,pos,pos?128:1)||p.candidates()!=q.candidates())
     throw std::runtime_error("publication routing mismatch");
    bytes(p.cache().main_bytes(),q.cache().main_bytes(),"publication main");
    bytes(p.cache().main_scales(),q.cache().main_scales(),"publication main scales");
    bytes(p.cache().index_bytes(),q.cache().index_bytes(),"publication index");
    bytes(p.cache().index_scales(),q.cache().index_scales(),"publication index scales");
   };
   for(int i=0;i<3;++i)publication_same(a.encoder.producer[i],b.encoder.producer[i],3+6*i);
   publication_same(a.decoder.producer,b.decoder.producer,39);
  };
  if(std::getenv("DSV41_CHECK_DEFERRED_PENDING")){
   constexpr std::size_t first_count=16384,second_count=8192,total=first_count+second_count;
   std::vector<std::uint32_t> input(total);for(std::size_t i=0;i<total;++i)
    input[i]=std::uint32_t((i*7919)%129263);
   dsv41::reset_route_tie_records();
   std::optional<dsv41::BlockResult> expected;
   dsv41::run_prefill_chunks(total,4096,[&](std::size_t offset,std::size_t count){
    expected.emplace(packed.forward_packed_sweep(std::span(input).subspan(offset,count),
                                                  expected_state,offset));
   });
   auto expected_ties=dsv41::route_tie_records();
   dsv41::reset_route_tie_records();
   auto initial=actual_state;const auto initial_revision=actual_state.revision();
   auto transaction=packed.begin_deferred_decoder(std::span(input).first(first_count),actual_state,0);
   if(!transaction.pending()||transaction.start()!=0||transaction.tokens()!=first_count)
    throw std::runtime_error("invalid pending decoder transaction metadata");
   full_state_same(actual_state,initial);
   auto actual=packed.finish_deferred_decoder(std::move(transaction),
    std::span(input).subspan(first_count,second_count),actual_state);
   if(transaction.pending())throw std::runtime_error("finished pending decoder remained publishable");
   if(actual_state.revision()!=initial_revision+1)
    throw std::runtime_error("pending decoder finish revision mismatch");
   dsv41::BlockResult expected_output{
    mlx::core::slice(expected->hidden,{int(expected->hidden.shape(0))-1,0,0},
                                      {int(expected->hidden.shape(0)),4,5120}),
    mlx::core::slice(expected->pre_mix,{int(expected->pre_mix.shape(0))-1,0},
                                       {int(expected->pre_mix.shape(0)),4})};
   same(actual.hidden,expected_output.hidden,"pending decoder hidden");
   same(actual.pre_mix,expected_output.pre_mix,"pending decoder pre-mix");
   same(packed.logits(actual),packed.logits(expected_output),"pending decoder logits");
   full_state_same(actual_state,expected_state);
   expected_ties.erase(std::remove_if(expected_ties.begin(),expected_ties.end(),[&](const auto& tie){
    return tie.layer>=20&&tie.token<first_count+
     dsv41::deferred_decoder_span(second_count,std::size_t(tie.layer)).first;}),expected_ties.end());
   auto actual_ties=dsv41::route_tie_records();
   auto order=[](const auto& x,const auto& y){return std::tie(x.token,x.layer)<std::tie(y.token,y.layer);};
   std::sort(actual_ties.begin(),actual_ties.end(),order);std::sort(expected_ties.begin(),expected_ties.end(),order);
   ties_same(actual_ties,expected_ties,"pending decoder route ties");
   bool reused=false;try{packed.finish_deferred_decoder(std::move(transaction),
    std::span(input).subspan(first_count,second_count),actual_state);}catch(const std::exception&){reused=true;}
   if(!reused)throw std::runtime_error("finished pending decoder transaction was reused");
   std::cout<<"PASS: DwarfStar-style 16K encoder-only plus 8K exact decoder resume; "
    <<"hidden/pre-mix/logits/state/publication/hash/route ties exact; source atomicity exact; "
    <<"performance unqualified"<<std::endl;
   return 0;
  }
  if(std::getenv("DSV41_CHECK_DEFERRED_TRANSACTION")||std::getenv("DSV41_CHECK_DEFERRED_SUFFIX")){
   const bool suffix_check=std::getenv("DSV41_CHECK_DEFERRED_SUFFIX")!=nullptr;
   const std::size_t count=suffix_check?2541:256;
   std::vector<std::uint32_t> input(count);for(std::size_t i=0;i<count;++i)input[i]=std::uint32_t((i*7919)%129263);
   dsv41::reset_route_tie_records();
   auto expected=packed.forward_packed_sweep(input,expected_state,0);
   auto expected_ties=dsv41::route_tie_records();
   dsv41::reset_route_tie_records();
   auto initial=actual_state;
   const auto initial_revision=actual_state.revision();
   auto transaction=packed.begin_deferred_prefill(input,actual_state,0);
   if(!transaction.pending()||transaction.start()!=0||transaction.tokens()!=count)
    throw std::runtime_error("invalid deferred transaction metadata");
   full_state_same(actual_state,initial);
   if(actual_state.revision()!=initial_revision)throw std::runtime_error("deferred encoder published a revision");
   auto moved=std::move(transaction);
   if(transaction.pending())throw std::runtime_error("moved deferred transaction retained ownership");
   auto actual=packed.finish_deferred_prefill(std::move(moved),actual_state);
   if(moved.pending())throw std::runtime_error("finished deferred transaction remained publishable");
   if(actual_state.revision()!=initial_revision+1)throw std::runtime_error("deferred finish revision mismatch");
   dsv41::BlockResult expected_output=expected;
   if(suffix_check)expected_output={
    mlx::core::slice(expected.hidden,{int(count-1),0,0},{int(count),4,5120}),
    mlx::core::slice(expected.pre_mix,{int(count-1),0},{int(count),4})};
   same(actual.hidden,expected_output.hidden,"deferred transaction hidden");
   same(actual.pre_mix,expected_output.pre_mix,"deferred transaction pre-mix");
   same(packed.logits(actual),packed.logits(expected_output),"deferred transaction logits");
   full_state_same(actual_state,expected_state);
   auto actual_ties=dsv41::route_tie_records();
   if(suffix_check)expected_ties.erase(std::remove_if(expected_ties.begin(),expected_ties.end(),
    [&](const auto& tie){return tie.layer>=20&&tie.token<
     dsv41::deferred_decoder_span(count,std::size_t(tie.layer)).first;}),expected_ties.end());
   auto order=[](const auto& x,const auto& y){return std::tie(x.token,x.layer)<std::tie(y.token,y.layer);};
   std::sort(actual_ties.begin(),actual_ties.end(),order);std::sort(expected_ties.begin(),expected_ties.end(),order);
   ties_same(actual_ties,expected_ties,"deferred transaction route ties");
   bool reused=false;try{packed.finish_deferred_prefill(std::move(moved),actual_state);}
   catch(const std::exception&){reused=true;}
   if(!reused)throw std::runtime_error("finished deferred transaction was reused");
   auto invalid=input;invalid[129]=129264;auto saved=actual_state;bool rejected=false;
   try{(void)packed.begin_deferred_prefill(invalid,actual_state,256);}
   catch(const std::exception&){rejected=true;}
   if(!rejected)throw std::runtime_error("invalid deferred encoder token accepted");
   full_state_same(saved,actual_state);
   std::cout<<"PASS: deferred encoder transaction is unpublished until exact "
    <<(suffix_check?"shrinking decoder suffix":"full decoder finish")<<"; "
    <<"hidden/pre-mix/logits/state/publication/hash/route ties exact; discard/reuse/invalid atomicity exact; "
    <<(suffix_check?"suffix performance unqualified":"decoder suffix and performance unqualified")<<std::endl;
   return 0;
  }
  if(std::getenv("DSV41_CHECK_FIXED_TILE_LAYER_LOCALIZATION")){
   std::vector<std::uint32_t> input(128);for(int i=0;i<128;++i)input[i]=std::uint32_t((i*7919)%129263);
   MemoryTraceSink expected_trace,actual_trace;
   model.forward(input,expected_state,0,&expected_trace);
   dsv41::set_active_trace_sink(&actual_trace);
   packed.forward_packed_chunk(input,actual_state,0);
   dsv41::set_active_trace_sink(nullptr);
   int first_layer=-1;std::string first_stage;float first_rms=0.0f;
   auto report_stage=[&](int layer,const std::string& stage,const std::string& reference_stage=""){
    const std::string prefix=(layer<20?"encoder.layer":"decoder.layer")+
     std::to_string(layer)+".";
    const auto& candidate=actual_trace.at(prefix+stage);
    const auto& reference=expected_trace.at(prefix+(reference_stage.empty()?stage:reference_stage));
    if(candidate.shape()!=reference.shape()||candidate.dtype()!=reference.dtype())
     throw std::runtime_error("layer trace shape mismatch: "+prefix+stage);
    auto c=mx::astype(candidate,mx::float32),r=mx::astype(reference,mx::float32),d=mx::subtract(c,r);
    auto relative=mx::sqrt(mx::divide(mx::sum(mx::multiply(d,d)),
     mx::maximum(mx::sum(mx::multiply(r,r)),mx::array(1e-30f))));
    auto maximum=mx::max(mx::abs(d));
    auto mismatches=mx::sum(mx::astype(mx::not_equal(candidate,reference),mx::uint32));
    mx::eval(relative,maximum,mismatches);const float rms=relative.item<float>();
    std::cout<<"fixed-tile layer="<<layer<<" stage="<<stage<<" relative_rms="<<rms
             <<" max_abs="<<maximum.item<float>()
             <<" bit_mismatches="<<mismatches.item<std::uint32_t>()<<std::endl;
    if(first_layer<0&&rms>=0.002f){first_layer=layer;first_stage=stage;first_rms=rms;}
   };
   for(int layer=0;layer<40;++layer){
    const std::string prefix=(layer<20?"encoder.layer":"decoder.layer")+std::to_string(layer)+".";
    for(const char* stage:{"attn_in","attn_out","post_attn","ffn_in","moe_out","hidden","pre_mix"}){
     const auto& candidate=actual_trace.at(prefix+stage);const auto& reference=expected_trace.at(prefix+stage);
     if(candidate.shape()!=reference.shape()||candidate.dtype()!=reference.dtype())
      throw std::runtime_error("layer trace shape mismatch: "+prefix+stage);
     auto c=mx::astype(candidate,mx::float32),r=mx::astype(reference,mx::float32),d=mx::subtract(c,r);
     auto relative=mx::sqrt(mx::divide(mx::sum(mx::multiply(d,d)),
      mx::maximum(mx::sum(mx::multiply(r,r)),mx::array(1e-30f))));
     auto maximum=mx::max(mx::abs(d));
     auto mismatches=mx::sum(mx::astype(mx::not_equal(candidate,reference),mx::uint32));
     mx::eval(relative,maximum,mismatches);const float rms=relative.item<float>();
     std::cout<<"fixed-tile layer="<<layer<<" stage="<<stage<<" relative_rms="<<rms
              <<" max_abs="<<maximum.item<float>()
              <<" bit_mismatches="<<mismatches.item<std::uint32_t>()<<std::endl;
     if(first_layer<0&&rms>=0.002f){first_layer=layer;first_stage=stage;first_rms=rms;}
    }
    if(layer==3||layer==4||layer==20||layer==21)
     for(const char* stage:{"attn_qr","attn_q","attn_kv","attn_core",
                            "attn_inverse_rope","attn_grouped","attn_linear"})
     report_stage(layer,stage);
    if(layer==3||layer==20){
     for(const char* stage:{"attn_core_native_width1_qk","attn_core_native_width1_av"})
      report_stage(layer,stage,"attn_core");
    }
    if(layer==20)report_stage(layer,"attn_core_compact_token0","attn_core");
    if(layer==3||layer==20){
     const std::string prefix=(layer<20?"encoder.layer":"decoder.layer")+std::to_string(layer)+".";
     const auto& candidate=actual_trace.at(prefix+"attn_core");
     const auto& reference=expected_trace.at(prefix+"attn_core");
     auto token_mask=mx::any(mx::not_equal(candidate,reference),std::vector<int>{1,2});
     auto ids=mx::arange(candidate.shape(0),mx::int32);
     auto count=mx::sum(mx::astype(token_mask,mx::uint32));
     auto first=mx::min(mx::where(token_mask,ids,mx::array(candidate.shape(0),mx::int32)));
     auto last=mx::max(mx::where(token_mask,ids,mx::array(-1,mx::int32)));
     mx::eval(count,first,last);
     auto widths=actual_trace.at(prefix+"attn_widths");
     const int first_id=first.item<std::int32_t>(),last_id=last.item<std::int32_t>();
     auto first_width=first_id<candidate.shape(0)?mx::take(widths,mx::array(first_id)):mx::array(-1);
     auto last_width=last_id>=0?mx::take(widths,mx::array(last_id)):mx::array(-1);
     mx::eval(first_width,last_width);
     std::cout<<"fixed-tile layer="<<layer<<" stage=attn_core_tokens mismatch_tokens="
              <<count.item<std::uint32_t>()<<" first="<<first_id
              <<" first_width="<<first_width.item<std::int32_t>()<<" last="<<last_id
              <<" last_width="<<last_width.item<std::int32_t>()<<std::endl;
    }
   }
   if(first_layer<0)full_state_same(actual_state,expected_state);
   std::cout<<"PASS: fixed-tile layer localization completed; first_gate_failure_layer="
            <<first_layer<<" first_gate_failure_stage="<<(first_layer<0?"none":first_stage)
            <<" first_gate_failure_rms="<<first_rms
            <<(first_layer<0?"; final state equality asserted":"; final state equality not asserted after semantic divergence")
            <<"; diagnostic only"<<std::endl;
   return 0;
  }
  if(std::getenv("DSV41_CHECK_LAYER_SWEEP_BACKBONE")){
   std::vector<std::uint32_t> input(256);for(int i=0;i<256;++i)input[i]=std::uint32_t((i*7919)%129263);
   if(setenv("DSV41_RUNTIME_CHUNK_ATTENTION","0",1)!=0)throw std::runtime_error("cannot select sweep oracle attention");
   dsv41::reset_route_tie_records();std::vector<mx::array> expected_hidden,expected_pre;
   for(int offset:{0,128}){auto part=model.forward(std::span(input).subspan(offset,128),expected_state,offset);
    expected_hidden.push_back(part.hidden);expected_pre.push_back(part.pre_mix);}
   dsv41::BlockResult expected{mx::concatenate(expected_hidden,0),mx::concatenate(expected_pre,0)};
   auto expected_ties=dsv41::route_tie_records();dsv41::reset_route_tie_records();
   if(setenv("DSV41_RUNTIME_CHUNK_ATTENTION","1",1)!=0)throw std::runtime_error("cannot select sweep attention");
   auto actual=packed.forward_packed_sweep(input,actual_state,0);auto actual_ties=dsv41::route_tie_records();
   semantic_same(actual.hidden,expected.hidden,"layer-sweep hidden");
   semantic_same(actual.pre_mix,expected.pre_mix,"layer-sweep pre-mix");
   semantic_same(packed.logits(actual),model.logits(expected),"layer-sweep logits",true);
   full_state_same(actual_state,expected_state);
   auto order=[](const auto& x,const auto& y){return std::tie(x.token,x.layer)<std::tie(y.token,y.layer);};
   std::sort(actual_ties.begin(),actual_ties.end(),order);std::sort(expected_ties.begin(),expected_ties.end(),order);
   ties_same(actual_ties,expected_ties,"layer-sweep route ties");
   auto saved=actual_state;auto invalid=input;invalid[129]=129264;bool rejected=false;
   try{packed.forward_packed_sweep(invalid,actual_state,256);}catch(const std::exception&){rejected=true;}
   if(!rejected)throw std::runtime_error("invalid sweep token accepted");full_state_same(saved,actual_state);
   std::cout<<"PASS: transactional 256-token/40-layer sweep hidden/pre-mix/logits bounded; "
    <<"state/publication/hash/route ties and invalid-request atomicity exact; active_bytes="<<mx::get_active_memory()
    <<" cache_bytes="<<mx::get_cache_memory()<<" peak_bytes="<<mx::get_peak_memory()
    <<" bank_constructions="<<dsv41::packed_expert_bank_construction_count()
    <<" loaded_experts="<<dsv41::packed_expert_bank_loaded_expert_count()
    <<"; performance/2K/32K unqualified"<<std::endl;
   return 0;
  }
  std::vector<std::uint32_t> input(128);for(int i=0;i<128;++i)input[i]=std::uint32_t((i*7919)%129263);
  dsv41::reset_attention_telemetry();
  for(int chunk_index=0;chunk_index<2;++chunk_index){
   const auto start=std::uint64_t(chunk_index*128);
   if(chunk_attention_check&&setenv("DSV41_RUNTIME_CHUNK_ATTENTION","0",1)!=0)
    throw std::runtime_error("cannot select token-serial attention oracle");
   dsv41::reset_route_tie_records();auto expected=model.forward(input,expected_state,start);
   auto expected_ties=dsv41::route_tie_records();dsv41::reset_route_tie_records();
   if(chunk_attention_check&&setenv("DSV41_RUNTIME_CHUNK_ATTENTION","1",1)!=0)
    throw std::runtime_error("cannot select chunk attention candidate");
   auto actual=packed.forward_packed_chunk(input,actual_state,start);auto actual_ties=dsv41::route_tie_records();
   auto actual_logits=packed.logits(actual),expected_logits=model.logits(expected);
   if(chunk_attention_check){semantic_same(actual.hidden,expected.hidden,"layer-major hidden");
    semantic_same(actual.pre_mix,expected.pre_mix,"layer-major pre-mix");
    semantic_same(actual_logits,expected_logits,"layer-major logits",true);
   }else{same(actual.hidden,expected.hidden,"layer-major hidden");same(actual.pre_mix,expected.pre_mix,"layer-major pre-mix");
    same(actual_logits,expected_logits,"layer-major logits");}
   full_state_same(actual_state,expected_state);
   auto order=[](const auto& x,const auto& y){return std::tie(x.token,x.layer)<std::tie(y.token,y.layer);};
   std::sort(actual_ties.begin(),actual_ties.end(),order);std::sort(expected_ties.begin(),expected_ties.end(),order);
   ties_same(actual_ties,expected_ties,"layer-major route ties");
   if(actual_state.decoder.producer.publication()->index_source_layer()!=36)throw std::runtime_error("decoder republishing did not reach layer 36");
   std::cout<<"PASS: 40-layer chunk "<<chunk_index<<" hidden/pre-mix/logits "
            <<(chunk_attention_check?"bounded; state/publication/hash/route ties exact":"and state/publication/hash/route ties exact")<<std::endl;
  }
  auto saved=actual_state;
  for(std::uint32_t invalid:{129264u,129280u}){bool rejected=false;
   try{packed.forward_packed_chunk(std::span(&invalid,1),actual_state,256);}catch(const std::exception&){rejected=true;}
   if(!rejected)throw std::runtime_error("invalid token accepted");full_state_same(saved,actual_state);
  }
  const auto attention=dsv41::attention_telemetry();
  if(chunk_attention_check&&dsv41::runtime_batched_splitk_qk_enabled()&&
     attention.chunk_batched_splitk_qk_calls==0)
   throw std::runtime_error("batched split-K QK candidate was never invoked");
  if(chunk_attention_check&&dsv41::runtime_packed_chunk_attention_enabled()&&
     attention.packed_chunk_attention_calls==0)
   throw std::runtime_error("packed chunk attention candidate was never invoked");
  if(chunk_attention_check&&dsv41::runtime_wide_attention_enabled()&&
     attention.wide_attention_calls==0)
   throw std::runtime_error("wide attention candidate was never invoked");
  if(chunk_attention_check&&dsv41::runtime_fixed_tile_attention_enabled()&&
     (attention.fixed_tile_attention_calls==0||attention.chunk_scalar_qk_calls!=0||
      attention.chunk_scalar_av_calls!=0))
   throw std::runtime_error("fixed-tile attention candidate topology mismatch");
  std::cout<<"PASS: layer-major backbone 2x128 tokens and invalid-token atomicity; active_bytes="<<mx::get_active_memory()
   <<" cache_bytes="<<mx::get_cache_memory()<<" peak_bytes="<<mx::get_peak_memory()
   <<" bank_constructions="<<dsv41::packed_expert_bank_construction_count()
   <<" loaded_experts="<<dsv41::packed_expert_bank_loaded_expert_count()
   <<" packed_attention_calls="<<attention.packed_chunk_attention_calls
   <<" wide_attention_calls="<<attention.wide_attention_calls
   <<" fixed_tile_attention_calls="<<attention.fixed_tile_attention_calls
   <<" batched_splitk_qk_calls="<<attention.chunk_batched_splitk_qk_calls
   <<" scalar_qk_calls="<<attention.chunk_scalar_qk_calls
   <<"; performance/32K unqualified"<<std::endl;
  return 0;
 }
 const auto ids=argc==5?tokens(argv[4]):std::vector<std::uint32_t>{0,42,1000};
 dsv41::TextBackboneState chunk(metadata),serial(metadata);
 dsv41::reset_route_tie_count();dsv41::reset_route_tie_records();
 std::vector<mx::array> batch_hidden,batch_pre;
 dsv41::run_prefill_chunks(ids.size(),128,[&](std::size_t offset,std::size_t count){
  auto part=model.forward(std::span(ids).subspan(offset,count),chunk,offset);
  batch_hidden.push_back(part.hidden);batch_pre.push_back(part.pre_mix);
 });
 dsv41::BlockResult batch{mx::concatenate(batch_hidden,0),mx::concatenate(batch_pre,0)};
 const auto chunk_ties=dsv41::route_tie_records();
 dsv41::reset_route_tie_count();dsv41::reset_route_tie_records();
 dsv41::BlockResult serial_last{mx::zeros({1,4,5120},mx::bfloat16),mx::zeros({1,4},mx::float32)};
 for(std::size_t i=0;i<ids.size();++i){auto one=model.forward(std::span(ids).subspan(i,1),serial,i);
  const int row=int(i);
  same(one.hidden,mx::slice(batch.hidden,{row,0,0},{row+1,4,5120}),"backbone hidden chunk bits");
  same(one.pre_mix,mx::slice(batch.pre_mix,{row,0},{row+1,4}),"backbone pre-mix chunk bits");
  serial_last=one;}
 const auto serial_ties=dsv41::route_tie_records();ties_same(chunk_ties,serial_ties,"backbone route ties chunk/token");
 state_same(chunk,serial);
 const int last=int(ids.size()-1);
 dsv41::BlockResult last_slice{mx::slice(batch.hidden,{last,0,0},{last+1,4,5120}),mx::slice(batch.pre_mix,{last,0},{last+1,4})};
 auto logits_chunk=model.logits(last_slice);auto logits_serial=model.logits(serial_last);
 same(logits_chunk,logits_serial,"backbone logits chunk bits");
 if(std::getenv("DSV41_CHECK_PACKED_EXPERT_BACKBONE_PARITY")||
    std::getenv("DSV41_CHECK_COMPACT_EXPERT_BACKBONE_PARITY")){
  const bool compact_check=std::getenv("DSV41_CHECK_COMPACT_EXPERT_BACKBONE_PARITY")!=nullptr;
  if(dsv41::runtime_packed_expert_bank_enabled())
   throw std::runtime_error("packed expert backbone parity must start from the individual path");
  dsv41::TextBackboneState individual_state(metadata);
  dsv41::reset_route_tie_count();dsv41::reset_route_tie_records();
  auto individual_result=model.forward(std::span(ids).subspan(0,1),individual_state,0);
  auto individual_logits=model.logits(individual_result);mx::eval(individual_logits);
  const auto individual_ties=dsv41::route_tie_records();
  if(setenv("DSV41_RUNTIME_PACKED_EXPERT_BANK","1",1)!=0||
     setenv("DSV41_RUNTIME_COMPACT_EXPERT_BANK",compact_check?"1":"0",1)!=0)
   throw std::runtime_error("cannot enable packed expert bank");
  {
   dsv41::TextBackboneReference packed_model(c,metadata);
   dsv41::TextBackboneState packed_state(metadata);
   dsv41::reset_route_tie_count();dsv41::reset_route_tie_records();
   auto packed_result=packed_model.forward(std::span(ids).subspan(0,1),packed_state,0);
   auto packed_logits=packed_model.logits(packed_result);
   same(packed_result.hidden,individual_result.hidden,"packed expert backbone hidden bits");
   same(packed_result.pre_mix,individual_result.pre_mix,"packed expert backbone pre-mix bits");
   same(packed_logits,individual_logits,"packed expert backbone logits bits");
   state_same(packed_state,individual_state);
   ties_same(dsv41::route_tie_records(),individual_ties,"packed expert backbone route ties");
  }
  if(setenv("DSV41_RUNTIME_PACKED_EXPERT_BANK","0",1)!=0||
     setenv("DSV41_RUNTIME_COMPACT_EXPERT_BANK","0",1)!=0)
   throw std::runtime_error("cannot restore packed expert bank policy");
  std::cout<<"PASS: one-token full-backbone individual/"<<(compact_check?"compact":"packed")
           <<" expert hidden, pre-mix, state, logits and route ties are bit-exact"<<std::endl;
 }
 if(std::getenv("DSV41_CHECK_LAYER_FINITE_POLICY_PARITY")){
  const bool original=dsv41::runtime_layer_finite_checks_enabled();
  if(setenv("DSV41_RUNTIME_LAYER_FINITE_CHECKS",original?"0":"1",1)!=0)
   throw std::runtime_error("cannot switch layer finite policy");
  dsv41::TextBackboneState alternate(metadata);
  dsv41::reset_route_tie_count();dsv41::reset_route_tie_records();
  std::vector<mx::array> alternate_hidden,alternate_pre;
  dsv41::run_prefill_chunks(ids.size(),128,[&](std::size_t offset,std::size_t count){
   auto part=model.forward(std::span(ids).subspan(offset,count),alternate,offset);
   alternate_hidden.push_back(part.hidden);alternate_pre.push_back(part.pre_mix);
  });
  dsv41::BlockResult alternate_result{mx::concatenate(alternate_hidden,0),mx::concatenate(alternate_pre,0)};
  same(alternate_result.hidden,batch.hidden,"layer finite policy hidden bits");
  same(alternate_result.pre_mix,batch.pre_mix,"layer finite policy pre-mix bits");
  state_same(alternate,chunk);
  dsv41::BlockResult alternate_last{
   mx::slice(alternate_result.hidden,{last,0,0},{last+1,4,5120}),
   mx::slice(alternate_result.pre_mix,{last,0},{last+1,4})};
  same(model.logits(alternate_last),logits_chunk,"layer finite policy logits bits");
  ties_same(dsv41::route_tie_records(),chunk_ties,"layer finite policy route ties");
  if(setenv("DSV41_RUNTIME_LAYER_FINITE_CHECKS",original?"1":"0",1)!=0)
   throw std::runtime_error("cannot restore layer finite policy");
  std::cout<<"PASS: layer finite check on/off hidden, pre-mix, state, logits and route ties are bit-exact"<<std::endl;
 }
 auto finite=mx::all(mx::isfinite(logits_chunk));mx::eval(finite);if(!finite.item<bool>())throw std::runtime_error("nonfinite backbone logits");
 auto saved=chunk;
 for(std::uint32_t invalid:{129264u,129280u}){
  bool failed=false;try{model.forward(std::span(&invalid,1),chunk,ids.size());}catch(const std::exception&){failed=true;}
  if(!failed||chunk.encoder.hash.position()!=ids.size())throw std::runtime_error("invalid token changed backbone state");state_same(saved,chunk);
 }
 auto fork=chunk;const std::array<std::uint32_t,1> next{42};
 auto resumed=model.forward(next,fork,ids.size());auto repeated=model.forward(next,serial,ids.size());
 same(resumed.hidden,repeated.hidden,"backbone continuation hidden");same(resumed.pre_mix,repeated.pre_mix,"backbone continuation pre-mix");
 state_same(fork,serial);
 if(chunk.encoder.hash.position()!=ids.size()||fork.encoder.hash.position()!=ids.size()+1)throw std::runtime_error("backbone fork position mismatch");state_same(saved,chunk);
 chunk.reset();dsv41::reset_route_tie_count();dsv41::reset_route_tie_records();std::vector<mx::array> reset_hidden,reset_pre;
 dsv41::run_prefill_chunks(ids.size(),128,[&](std::size_t offset,std::size_t count){
  auto part=model.forward(std::span(ids).subspan(offset,count),chunk,offset);
  reset_hidden.push_back(part.hidden);reset_pre.push_back(part.pre_mix);
 });
 dsv41::BlockResult reset{mx::concatenate(reset_hidden,0),mx::concatenate(reset_pre,0)};
 const auto reset_ties=dsv41::route_tie_records();ties_same(chunk_ties,reset_ties,"backbone route ties reset");
 same(reset.hidden,batch.hidden,"backbone reset hidden");same(reset.pre_mix,batch.pre_mix,"backbone reset pre-mix");state_same(saved,chunk);
 for(const auto& tie:chunk_ties)std::cout<<"route_tie: layer="<<tie.layer<<" token="<<tie.token
  <<" sixth_id="<<tie.sixth_id<<" seventh_id="<<tie.seventh_id
  <<" sixth_score_bits="<<std::bit_cast<std::uint32_t>(tie.sixth_score)
  <<" seventh_score_bits="<<std::bit_cast<std::uint32_t>(tie.seventh_score)<<std::endl;
 std::cout<<"PASS: "<<ids.size()<<" token IDs -> encoder 0..19 -> decoder 20..39 -> logits; <=128 chunk/token bits, state, continuation, fork, reset, invalid token rejection; route tie records exact across schedules = "<<chunk_ties.size()<<" (lowest-ID policy remains an external-oracle gap); external oracle/256K NOT qualified."<<std::endl;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
