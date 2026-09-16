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
struct RuntimeProfileTelemetry {
 std::array<double,40> layer_seconds{},attention_path_seconds{},moe_path_seconds{},post_moe_seconds{};
 std::array<std::size_t,40> layer_calls{},component_calls{};
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
}
