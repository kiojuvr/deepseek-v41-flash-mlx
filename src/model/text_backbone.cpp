#include "dsv41/text_backbone.hpp"
#include "dsv41/deferred_decoder_plan.hpp"
#include "dsv41/execution_policy.hpp"
#include <stdexcept>
namespace dsv41 {
namespace mx=mlx::core;
namespace {
bool state_at(const TextBackboneState& state,std::uint64_t position){
 if(state.encoder.hash.position()!=position||state.decoder.producer.position()!=position)return false;
 for(const auto& value:state.encoder.swa)if(value.position()!=position)return false;
 for(const auto& value:state.encoder.producer)if(value.position()!=position)return false;
 for(const auto& value:state.encoder.reuse)if(value.position()!=position)return false;
 for(const auto& value:state.decoder.reuse)if(value.position()!=position)return false;
 return true;
}
}
TextBackboneReference::TextBackboneReference(WeightCatalog& catalog,
 std::shared_ptr<const EngramMetadata> metadata)
 :expert_atlas_(make_resident_expert_atlas(catalog)),residency_activation_(residency_),
  encoder_(catalog,std::move(metadata),expert_atlas_),decoder_(catalog,expert_atlas_){}

BlockResult TextBackboneReference::forward_packed_chunk(std::span<const std::uint32_t> ids,
 TextBackboneState& state,std::uint64_t start) const{
 auto next=state;
 auto encoded=encoder_.forward_packed_chunk(ids,next.encoder,start);
 auto result=decoder_.forward_packed_chunk(encoded.hidden,encoded.pre_mix,next.decoder,start);
 state=std::move(next);state.commit_revision();return result;
}
BlockResult TextBackboneReference::forward_packed_sweep(std::span<const std::uint32_t> ids,
 TextBackboneState& state,std::uint64_t start) const{
 if(ids.empty()||ids.size()>4096)throw std::runtime_error("packed sweep requires 1..4096 tokens");
 auto next=state;
 auto encoded=encoder_.forward_packed_sweep(ids,next.encoder,start);
 auto result=decoder_.forward_packed_sweep(encoded.hidden,encoded.pre_mix,next.decoder,start);
 state=std::move(next);state.commit_revision();return result;
}
DeferredPrefillTransaction TextBackboneReference::begin_deferred_prefill(
 std::span<const std::uint32_t> ids,const TextBackboneState& state,std::uint64_t start) const{
 if(ids.empty()||ids.size()>16384||!state_at(state,start))
  throw std::runtime_error("deferred encoder transaction requires 1..16384 tokens and an exact frontier");
 auto pending=state;
 auto encoded=encoder_.forward_packed_sweep(ids,pending.encoder,start);
 mx::eval(encoded.hidden,encoded.pre_mix);
 return DeferredPrefillTransaction(std::move(pending),std::move(encoded),&state,start,ids.size());
}
BlockResult TextBackboneReference::finish_deferred_prefill(
 DeferredPrefillTransaction&& transaction,TextBackboneState& state) const{
 if(!transaction.state_||!transaction.encoded_||!transaction.tokens_||
    transaction.source_!=&state||transaction.source_revision_!=state.revision()||
    !state_at(state,transaction.start_))
  throw std::runtime_error("invalid or stale deferred prefill transaction");
 // Consume before decoder execution: a partial failure is neither publishable
 // nor retryable. The caller retains the unchanged source state for replay.
 auto pending=std::move(*transaction.state_);
 auto encoded=std::move(*transaction.encoded_);
 const auto start=transaction.start_;
 const auto tokens=transaction.tokens_;
 transaction.clear();
 const auto suffix_min=deferred_decoder_suffix_rows(20)+kDecoderRawHistory;
 auto result=tokens>=suffix_min?
  decoder_.forward_deferred_suffix(encoded.hidden,encoded.pre_mix,pending.decoder,start):
  decoder_.forward_packed_sweep(encoded.hidden,encoded.pre_mix,pending.decoder,start);
 mx::eval(result.hidden,result.pre_mix);
 state=std::move(pending);state.commit_revision();
 return result;
}
DeferredDecoderTransaction TextBackboneReference::begin_deferred_decoder(
 std::span<const std::uint32_t> ids,const TextBackboneState& state,std::uint64_t start) const{
 if(ids.size()!=kDeferredDecoderMaxTokens||!state_at(state,start))
  throw std::runtime_error("deferred decoder begin requires one exact 16384-token encoder sweep");
 auto pending=state;
 auto encoded=encoder_.forward_packed_sweep(ids,pending.encoder,start);
 decoder_.prepare_deferred_prefix(encoded.hidden,encoded.pre_mix,pending.decoder,start);
 mx::eval(encoded.hidden,encoded.pre_mix);
 return DeferredDecoderTransaction(std::move(pending),&state,start,ids.size());
}
BlockResult TextBackboneReference::finish_deferred_decoder(
 DeferredDecoderTransaction&& transaction,std::span<const std::uint32_t> ids,
 TextBackboneState& state) const{
 if(!transaction.state_||transaction.tokens_!=kDeferredDecoderMaxTokens||
    ids.size()<kDeferredDecoderMinTokens||ids.size()>kDeferredDecoderMaxTokens||
    transaction.source_!=&state||transaction.source_revision_!=state.revision()||
    !state_at(state,transaction.start_))
  throw std::runtime_error("invalid or stale deferred decoder transaction");
 auto pending=std::move(*transaction.state_);
 const auto start=transaction.start_+transaction.tokens_;
 transaction.clear();
 if(pending.encoder.hash.position()!=start||pending.decoder.producer.position()!=start)
  throw std::runtime_error("invalid deferred decoder private frontier");
 auto encoded=encoder_.forward_packed_sweep(ids,pending.encoder,start);
 auto result=decoder_.resume_deferred_suffix(encoded.hidden,encoded.pre_mix,pending.decoder,start);
 mx::eval(result.hidden,result.pre_mix);
 state=std::move(pending);state.commit_revision();
 if(runtime_deferred_decoder_clear_cache_enabled())mx::clear_cache();
 return result;
}
BlockResult TextBackboneReference::forward(std::span<const std::uint32_t> ids,TextBackboneState& state,std::uint64_t start,TraceSink* trace) const{
 auto next=state;
 auto encoder_out=encoder_.forward(ids,next.encoder,start,trace);
 if(trace){trace->record("encoder.out.hidden",encoder_out.hidden);trace->record("encoder.out.pre_mix",encoder_out.pre_mix);}
 auto result=decoder_.forward(encoder_out.hidden,encoder_out.pre_mix,next.decoder,start,trace);
 if(trace){trace->record("decoder.out.hidden",result.hidden);trace->record("decoder.out.pre_mix",result.pre_mix);}
 state=std::move(next);state.commit_revision();return result;
}
}
