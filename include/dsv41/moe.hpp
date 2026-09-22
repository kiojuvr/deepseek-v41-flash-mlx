#pragma once
#include "dsv41/linear.hpp"
#include <array>
#include <memory>
#include <unordered_map>
#include <vector>
namespace dsv41 {
struct RouteReference { std::array<int,6> ids; mlx::core::array weights; };
struct RouteTieRecord { int layer; std::uint64_t token; int sixth_id,seventh_id; float sixth_score,seventh_score; bool tied; };
struct RouteBatchReference {
 std::vector<std::array<int,6>> ids;
 mlx::core::array weights;
 std::vector<RouteTieRecord> ties;
 mlx::core::array device_ids;
 mlx::core::array device_lhs;
 mlx::core::array reduction_slots;
};
struct RouteUnionStats {
 std::size_t batches=0, selected_experts=0, unique_experts=0, overlap_experts=0, exact_reuses=0;
};
struct ExpertBankIoStats {
 std::size_t constructions=0, read_calls=0, qmm_dispatches=0, qmm_rows_total=0, qmm_rows_max=0;
 std::size_t expert_major_batches=0,expert_major_assignments=0;
 std::size_t grouped_pipeline_batches=0,grouped_pipeline_assignments=0;
 double read_seconds=0.0, total_seconds=0.0;
};
struct RouteExecutionStats { std::size_t device_batches=0,diagnostic_readbacks=0; };
ExpertBankIoStats expert_bank_io_stats();
void reset_expert_bank_io_stats();
RouteExecutionStats route_execution_stats();
void reset_route_execution_stats();
RouteUnionStats route_union_stats();
void reset_route_union_stats();
struct GateDiagnostic { mlx::core::array raw_scores, corrected_scores; RouteReference route; };
struct MoEComponents { mlx::core::array shared, routed, total; };
struct ExpertComponents { mlx::core::array gate, up, activation, output; };
struct GroupedExpertComponents {
 mlx::core::array gate,up,activation,down,weighted,accumulated,routed;
};
struct GroupedExpertBatchResult { mlx::core::array accumulated,routed; };
std::size_t packed_expert_bank_construction_count();
std::size_t packed_expert_bank_loaded_expert_count();
void reset_packed_expert_bank_construction_count();
// Text layer 0, one token. Raw and corrected scores cross one explicit CPU
// synchronization, then a fixed top-7 selection supplies top-6 and its boundary.
// strict throws on an exact top-6 boundary tie; otherwise ties break to the lowest expert ID.
// Official torch 2.13 CPU selected the other boundary candidate in one reviewed fixture;
// torch does not specify tie ordering, so callers that use this canonical policy must record it.
RouteReference select_routes_reference(const mlx::core::array& scores,const mlx::core::array& bias,
 bool strict=true,RouteTieRecord* tie=nullptr);
// Select routes for a complete prefill tile. Gate scores cross one GPU-to-CPU
// synchronization for the whole tile, rather than one synchronization per token.
// The selection and normalization policy is identical to select_routes_reference.
RouteBatchReference select_routes_batch_reference(const mlx::core::array& scores,
 const mlx::core::array& bias,bool strict=true,int layer=0,std::uint64_t start_position=0);
// Parallel Metal selector used by the prefill path. Qualification diagnostics
// transfer seven IDs, two boundary scores, and one status byte per token;
// diagnostics=false leaves selection, weights, and assignments on device.
RouteBatchReference select_routes_batch_device(const mlx::core::array& scores,
 const mlx::core::array& bias,bool strict=true,int layer=0,std::uint64_t start_position=0,
 bool diagnostics=true);
class GateReference {
public:
 explicit GateReference(WeightCatalog& catalog,int layer=0);
 RouteReference forward(const mlx::core::array& input,RouteTieRecord* tie=nullptr) const;
 RouteBatchReference forward_batch(const mlx::core::array& input,
                                   std::uint64_t start_position,
                                   bool diagnostics=true) const;
 GateDiagnostic diagnose(const mlx::core::array& input,RouteTieRecord* tie=nullptr) const;
private:
 int layer_;
 mlx::core::array weight_,bias_;
};
mlx::core::array expert_activation_reference(const mlx::core::array& gate,
 const mlx::core::array& up,const mlx::core::array& route_weight);
// Process-wide count of top-6 boundary ties broken by lowest expert ID (unqualified oracle gap).
std::size_t route_tie_count();
void reset_route_tie_count();
std::vector<RouteTieRecord> route_tie_records();
void reset_route_tie_records();
// Current token position used to tag routing tie records; set by the reference loops.
void set_route_trace_token(std::uint64_t token);
std::uint64_t route_trace_token();
class ExpertReference {
public:
 // expert -1 selects the shared FP8 expert; 0..383 select canonical FP4 experts.
 ExpertReference(WeightCatalog& catalog,int expert,int layer=0);
 mlx::core::array forward(const mlx::core::array& input,const mlx::core::array& route_weight) const;
 ExpertComponents components(const mlx::core::array& input,const mlx::core::array& route_weight) const;
 const PackedLinearReference& gate_projection() const{return w1_;}
 const PackedLinearReference& down_projection() const{return w2_;}
 const PackedLinearReference& up_projection() const{return w3_;}
private:
 PackedLinearReference w1_,w2_,w3_;
};
// One layer's 384 routed experts in the [E,N,packed] layout required by
// gather_qmm. Construction reads checkpoint tensors without modifying them and
// avoids retaining a second set of per-expert MLX arrays.
class ExpertBackingStore;
class PackedExpertBank {
public:
 PackedExpertBank(WeightCatalog& catalog,int layer);
 // Compact bank for a sorted unique subset of global expert IDs. This is the
 // storage primitive for route-first, expert-major prefill scheduling.
 PackedExpertBank(WeightCatalog& catalog,int layer,const std::vector<int>& expert_ids);
 PackedExpertBank(WeightCatalog& catalog,int layer,std::shared_ptr<const ExpertBackingStore> backing);
 GroupedExpertComponents forward_selected(const mlx::core::array& input,
  const std::array<int,6>& expert_ids,const mlx::core::array& route_weights) const;
 GroupedExpertBatchResult forward_batch_selected(const mlx::core::array& input,
  const std::vector<std::array<int,6>>& expert_ids,
  const mlx::core::array& route_weights) const;
 GroupedExpertBatchResult forward_batch_selected(const mlx::core::array& input,
  // IDs are bank-local. They equal global IDs for the full identity bank;
  // compact-bank callers should normally use the host-ID overload above.
  const mlx::core::array& expert_ids,const mlx::core::array& lhs_ids,
  const mlx::core::array& reduction_slots,const mlx::core::array& route_weights) const;
 // Device-only resident path: stable-sort assignments by expert, execute the
 // three gathered QMMs in expert-major row order, then reduce in canonical
 // per-token expert-ID order.
 // diagnostic=true suppresses io counters and routed-stage timers; used only by
 // the profiling warm-repeat, which must not change observable topology counts.
 GroupedExpertBatchResult forward_batch_expert_major(const mlx::core::array& input,
  const mlx::core::array& expert_ids,const mlx::core::array& lhs_ids,
  const mlx::core::array& reduction_slots,const mlx::core::array& route_weights,
  bool diagnostic=false) const;
 std::size_t packed_bytes() const{return packed_bytes_;}
 std::size_t expert_count() const{return expert_ids_.size();}
private:
 std::uint32_t local_expert_id(int global_id) const;
 int layer_=0;
 mlx::core::array w1_,s1_,w2_,s2_,w3_,s3_;
 std::vector<int> expert_ids_;
 std::array<int,384> global_to_local_{};
 std::size_t packed_bytes_=0;
};
// Immutable hardware-native routed weights for the complete backbone.  The
// constructor builds into an unpublished object; callers receive the shared
// atlas only after all 40 layer banks have completed successfully.
class ResidentExpertAtlas {
public:
 explicit ResidentExpertAtlas(WeightCatalog& catalog);
 const PackedExpertBank& bank(int layer) const;
 std::size_t packed_bytes() const{return packed_bytes_;}
 bool file_backed() const{return bool(backing_);}
private:
 std::shared_ptr<const ExpertBackingStore> backing_;
 std::array<std::shared_ptr<const PackedExpertBank>,40> banks_;
 std::size_t packed_bytes_=0;
};
std::shared_ptr<const ResidentExpertAtlas> make_resident_expert_atlas(WeightCatalog& catalog);
GroupedExpertComponents forward_grouped_selected(
 const mlx::core::array& input,const std::array<int,6>& expert_ids,
 const std::array<const ExpertReference*,6>& experts,
 const mlx::core::array& route_weights);
class MoEReference {
public:
 // Reference/compact modes retain their existing on-demand ownership.  The
 // optimized backbone supplies one immutable model-owned resident atlas.
 explicit MoEReference(WeightCatalog& catalog,int layer=0,
                       std::shared_ptr<const ResidentExpertAtlas> atlas={});
 mlx::core::array forward(const mlx::core::array& input) const;
 MoEComponents forward_components(const mlx::core::array& input) const;
 MoEComponents forward_batch_components(const mlx::core::array& input,
                                         std::uint64_t start_position) const;
 void release_packed_bank() const;
 bool packed_bank_loaded() const;
 std::size_t packed_bank_expert_count() const;
 mlx::core::array expert_contribution(const mlx::core::array& input, int expert_id,
                                      const mlx::core::array& route_weight) const;
 ExpertComponents expert_components(const mlx::core::array& input, int expert_id,
                                    const mlx::core::array& route_weight) const;
 // Number of top-6 boundary ties broken by lowest expert ID (unqualified oracle gap).
 std::size_t tie_count() const{return tie_count_;}
private:
 ExpertReference& expert(int id) const;
 WeightCatalog* catalog_;
 int layer_;
 GateReference gate_;
 ExpertReference shared_;
 mutable std::unordered_map<int,std::unique_ptr<ExpertReference>> experts_;
 bool packed_experts_;
 bool group_selected_experts_;
 bool resident_expert_atlas_;
 bool compact_expert_bank_;
 std::shared_ptr<const ResidentExpertAtlas> resident_atlas_;
 mutable std::unique_ptr<PackedExpertBank> expert_bank_;
 mutable std::vector<int> previous_compact_ids_;
 mutable std::size_t tie_count_=0;
};
}
