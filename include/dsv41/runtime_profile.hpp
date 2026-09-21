#pragma once
#include <mlx/mlx.h>
#include <array>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string_view>

namespace dsv41 {
enum class ProfileComponent { AttentionPath, MoEPath, PostMoE };
// Sub-phases of MoEPath. All are nested inside moe_path_seconds and must never be
// added to layer wall; they exist only to split the ~90% decode MoE bucket.
enum class ProfileSubcomponent { MoeInput, MoeRoute, MoeRouted, MoeShared, MoeCombine, MoeRoutedWarm, MoeSyncNoop };
struct RuntimeProfileTelemetry {
 std::array<double,40> layer_seconds{},attention_path_seconds{},moe_path_seconds{},post_moe_seconds{};
 std::array<std::size_t,40> layer_calls{},component_calls{};
 std::array<double,40> moe_input_seconds{},moe_route_seconds{},moe_routed_seconds{},
  moe_shared_seconds{},moe_combine_seconds{},moe_routed_warm_seconds{},moe_sync_noop_seconds{};
 std::array<std::size_t,40> moe_subcomponent_calls{};
 // Sub-phases of MoeRouted (the expert-major routed service), nested inside
 // moe_routed_seconds. stage 0 prep (sort/take + input FP8 roundtrip),
 // 1 gate/up gather-QMM, 2 swiglu + down-input FP8 roundtrip,
 // 3 down gather-QMM + route weighting, 4 inverse-sort + device reduce.
 std::array<std::array<double,40>,5> routed_stage_seconds{};
 std::array<std::size_t,40> routed_stage_calls{};
};
inline bool runtime_component_profile_enabled(){
 const char* value=std::getenv("DSV41_RUNTIME_COMPONENT_PROFILE");
 if(value==nullptr||std::string_view(value)=="0")return false;
 if(std::string_view(value)=="1")return true;
 throw std::runtime_error("DSV41_RUNTIME_COMPONENT_PROFILE must be 0 or 1");
}
inline RuntimeProfileTelemetry& runtime_profile_telemetry(){static RuntimeProfileTelemetry value;return value;}
inline std::mutex& runtime_profile_mutex(){static std::mutex value;return value;}
inline void reset_runtime_profile(){std::lock_guard lock(runtime_profile_mutex());runtime_profile_telemetry()={};}
inline RuntimeProfileTelemetry read_runtime_profile(){std::lock_guard lock(runtime_profile_mutex());return runtime_profile_telemetry();}
using RuntimeProfileClock=std::chrono::steady_clock;
inline RuntimeProfileClock::time_point runtime_profile_start(){return RuntimeProfileClock::now();}
inline double runtime_profile_elapsed(RuntimeProfileClock::time_point start){
 return std::chrono::duration<double>(RuntimeProfileClock::now()-start).count();
}
inline void record_runtime_layer(int layer,double seconds){
 if(!runtime_component_profile_enabled())return;
 std::lock_guard lock(runtime_profile_mutex());auto& p=runtime_profile_telemetry();
 p.layer_seconds.at(layer)+=seconds;++p.layer_calls.at(layer);
}
inline void finish_runtime_component(int layer,ProfileComponent component,
 RuntimeProfileClock::time_point start,const mlx::core::array& value){
 if(!runtime_component_profile_enabled())return;
 mlx::core::eval(value);mlx::core::synchronize();const double elapsed=runtime_profile_elapsed(start);
 std::lock_guard lock(runtime_profile_mutex());auto& p=runtime_profile_telemetry();
 auto* target=component==ProfileComponent::AttentionPath?&p.attention_path_seconds:
              component==ProfileComponent::MoEPath?&p.moe_path_seconds:&p.post_moe_seconds;
 target->at(layer)+=elapsed;if(component==ProfileComponent::PostMoE)++p.component_calls.at(layer);
}
inline void finish_runtime_subcomponent(int layer,ProfileSubcomponent component,
 RuntimeProfileClock::time_point start,const mlx::core::array& value){
 if(!runtime_component_profile_enabled())return;
 mlx::core::eval(value);mlx::core::synchronize();const double elapsed=runtime_profile_elapsed(start);
 std::lock_guard lock(runtime_profile_mutex());auto& p=runtime_profile_telemetry();
 auto* target=component==ProfileSubcomponent::MoeInput?&p.moe_input_seconds:
              component==ProfileSubcomponent::MoeRoute?&p.moe_route_seconds:
              component==ProfileSubcomponent::MoeRouted?&p.moe_routed_seconds:
              component==ProfileSubcomponent::MoeRoutedWarm?&p.moe_routed_warm_seconds:
              component==ProfileSubcomponent::MoeSyncNoop?&p.moe_sync_noop_seconds:
              component==ProfileSubcomponent::MoeShared?&p.moe_shared_seconds:&p.moe_combine_seconds;
 target->at(layer)+=elapsed;if(component==ProfileSubcomponent::MoeCombine)++p.moe_subcomponent_calls.at(layer);
}
inline void finish_runtime_routed_stage(int layer,int stage,
 RuntimeProfileClock::time_point start,const mlx::core::array& value){
 if(!runtime_component_profile_enabled())return;
 if(stage<0||stage>=5)return;
 mlx::core::eval(value);mlx::core::synchronize();const double elapsed=runtime_profile_elapsed(start);
 std::lock_guard lock(runtime_profile_mutex());auto& p=runtime_profile_telemetry();
 p.routed_stage_seconds.at(stage).at(layer)+=elapsed;if(stage==4)++p.routed_stage_calls.at(layer);
}
}
