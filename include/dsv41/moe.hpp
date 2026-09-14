#pragma once
#include "dsv41/linear.hpp"
#include <array>
#include <memory>
#include <unordered_map>
#include <vector>
namespace dsv41 {
struct RouteReference { std::array<int,6> ids; mlx::core::array weights; };
struct GateDiagnostic { mlx::core::array raw_scores, corrected_scores; RouteReference route; };
struct RouteTieRecord { int layer; std::uint64_t token; int sixth_id,seventh_id; float sixth_score,seventh_score; bool tied; };
// Text layer 0, one token. CPU selection is an explicit reference synchronization.
// strict throws on an exact top-6 boundary tie; otherwise ties break to the lowest expert ID.
// The official topk tie-break is not oracle-verified, so callers that proceed must record it.
RouteReference select_routes_reference(const mlx::core::array& scores,const mlx::core::array& bias,
 bool strict=true,RouteTieRecord* tie=nullptr);
class GateReference {
public:
 explicit GateReference(WeightCatalog& catalog,int layer=0);
 RouteReference forward(const mlx::core::array& input,RouteTieRecord* tie=nullptr) const;
 GateDiagnostic diagnose(const mlx::core::array& input,RouteTieRecord* tie=nullptr) const;
private:
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
private:
 PackedLinearReference w1_,w2_,w3_;
};
class MoEReference {
public:
 // Routed experts load on first use and stay cached; the shared expert loads eagerly.
 // Numerically identical to a fully resident load, but bounds memory for the full backbone.
 explicit MoEReference(WeightCatalog& catalog,int layer=0);
 mlx::core::array forward(const mlx::core::array& input) const;
 // Number of top-6 boundary ties broken by lowest expert ID (unqualified oracle gap).
 std::size_t tie_count() const{return tie_count_;}
private:
 ExpertReference& expert(int id) const;
 WeightCatalog* catalog_;
 int layer_;
 GateReference gate_;
 ExpertReference shared_;
 mutable std::unordered_map<int,std::unique_ptr<ExpertReference>> experts_;
 mutable std::size_t tie_count_=0;
};
}
