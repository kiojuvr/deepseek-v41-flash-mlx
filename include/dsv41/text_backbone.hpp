#pragma once
#include "dsv41/text_encoder.hpp"
#include "dsv41/text_decoder.hpp"
#include "dsv41/trace.hpp"
#include "dsv41/runtime_residency.hpp"
#include <cstdint>
#include <optional>
namespace dsv41 {
// Full text backbone: token IDs -> encoder 0..19 -> decoder 20..39 -> final collapse/norm/head.
struct TextBackboneState {
 explicit TextBackboneState(std::shared_ptr<const EngramMetadata> m):encoder(std::move(m)){}
 void reset(){encoder.reset();decoder.reset();++revision_;}
 std::uint64_t revision() const{return revision_;}
 TextEncoderState encoder;
 TextDecoderState decoder;
private:
 void commit_revision(){++revision_;}
 std::uint64_t revision_=0;
 friend class TextBackboneReference;
};
// Owns an unpublished encoder frontier. Destruction discards it; only
// finish_deferred_prefill can atomically publish the completed decoder state.
class DeferredPrefillTransaction {
public:
 DeferredPrefillTransaction(DeferredPrefillTransaction&& other)
  :state_(std::move(other.state_)),encoded_(std::move(other.encoded_)),source_(other.source_),
   source_revision_(other.source_revision_),start_(other.start_),tokens_(other.tokens_){other.clear();}
 DeferredPrefillTransaction& operator=(DeferredPrefillTransaction&& other){
  if(this!=&other){
   state_=std::move(other.state_);encoded_=std::move(other.encoded_);source_=other.source_;
   source_revision_=other.source_revision_;start_=other.start_;tokens_=other.tokens_;other.clear();
  }
  return *this;
 }
 DeferredPrefillTransaction(const DeferredPrefillTransaction&)=delete;
 DeferredPrefillTransaction& operator=(const DeferredPrefillTransaction&)=delete;
 bool pending() const{return state_.has_value();}
 std::uint64_t start() const{return start_;}
 std::size_t tokens() const{return tokens_;}
private:
 DeferredPrefillTransaction(TextBackboneState state,BlockResult encoded,
                            const TextBackboneState* source,std::uint64_t start,std::size_t tokens)
  :state_(std::move(state)),encoded_(std::move(encoded)),source_(source),
   source_revision_(source->revision()),start_(start),tokens_(tokens){}
 void clear(){
  state_.reset();encoded_.reset();source_=nullptr;source_revision_=0;start_=0;tokens_=0;
 }
 std::optional<TextBackboneState> state_;
 std::optional<BlockResult> encoded_;
 const TextBackboneState* source_=nullptr;
 std::uint64_t source_revision_=0;
 std::uint64_t start_=0;
 std::size_t tokens_=0;
 friend class TextBackboneReference;
};
// Cross-sweep CED ownership. The first 16K encoder sweep and its decoder
// producer cache are private; a later >=8K encoder sweep completes all decoder
// states and publishes once. Destruction discards the partial frontier.
class DeferredDecoderTransaction {
public:
 DeferredDecoderTransaction(DeferredDecoderTransaction&& other)
  :state_(std::move(other.state_)),source_(other.source_),
   source_revision_(other.source_revision_),start_(other.start_),tokens_(other.tokens_){other.clear();}
 DeferredDecoderTransaction& operator=(DeferredDecoderTransaction&& other){
  if(this!=&other){state_=std::move(other.state_);source_=other.source_;
   source_revision_=other.source_revision_;start_=other.start_;tokens_=other.tokens_;other.clear();}
  return *this;
 }
 DeferredDecoderTransaction(const DeferredDecoderTransaction&)=delete;
 DeferredDecoderTransaction& operator=(const DeferredDecoderTransaction&)=delete;
 bool pending() const{return state_.has_value();}
 std::uint64_t start() const{return start_;}
 std::size_t tokens() const{return tokens_;}
private:
 DeferredDecoderTransaction(TextBackboneState state,const TextBackboneState* source,
                            std::uint64_t start,std::size_t tokens)
  :state_(std::move(state)),source_(source),source_revision_(source->revision()),
   start_(start),tokens_(tokens){}
 void clear(){state_.reset();source_=nullptr;source_revision_=0;start_=0;tokens_=0;}
 std::optional<TextBackboneState> state_;
 const TextBackboneState* source_=nullptr;
 std::uint64_t source_revision_=0;
 std::uint64_t start_=0;
 std::size_t tokens_=0;
 friend class TextBackboneReference;
};
class TextBackboneReference {
public:
 TextBackboneReference(WeightCatalog& catalog,std::shared_ptr<const EngramMetadata> metadata);
 ~TextBackboneReference(){residency_.synchronize();}
 std::size_t wired_limit_bytes() const{return residency_.requested_bytes();}
 bool expert_backing_file_backed() const{return expert_atlas_&&expert_atlas_->file_backed();}
 BlockResult forward(std::span<const std::uint32_t> ids,TextBackboneState& state,std::uint64_t start,TraceSink* trace=nullptr) const;
 BlockResult forward_packed_chunk(std::span<const std::uint32_t> ids,
                                  TextBackboneState& state,std::uint64_t start) const;
 BlockResult forward_packed_sweep(std::span<const std::uint32_t> ids,
                                  TextBackboneState& state,std::uint64_t start) const;
 // CED ownership boundary: run the encoder into private state, then finish
 // either the short full decoder or the exact long-context suffix and publish once.
 DeferredPrefillTransaction begin_deferred_prefill(std::span<const std::uint32_t> ids,
                                  const TextBackboneState& state,std::uint64_t start) const;
 BlockResult finish_deferred_prefill(DeferredPrefillTransaction&& transaction,
                                  TextBackboneState& state) const;
 DeferredDecoderTransaction begin_deferred_decoder(std::span<const std::uint32_t> ids,
                                  const TextBackboneState& state,std::uint64_t start) const;
 BlockResult finish_deferred_decoder(DeferredDecoderTransaction&& transaction,
                                  std::span<const std::uint32_t> ids,
                                  TextBackboneState& state) const;
 mlx::core::array logits(const BlockResult& final_hidden,TraceSink* trace=nullptr) const{return decoder_.logits(final_hidden,trace);}
private:
 // First constructed, last destroyed: budget covers all model-owned buffers.
 RuntimeResidencyLease residency_;
 std::shared_ptr<const ResidentExpertAtlas> expert_atlas_;
 RuntimeResidencyActivation residency_activation_;
 TextEncoderReference encoder_;
 TextDecoderReference decoder_;
};
}
