#pragma once
#include <cstddef>
#include <mutex>

namespace dsv41 {
// Explicit host evaluation boundaries, NOT actual Metal dispatches.
// Component diagnostics and lower-level state evaluation are separate.
struct SweepTelemetry {
 std::size_t layer_evaluations=0, deferred_layer_evaluations=0;
 std::size_t decode_stack_evaluations=0, engram_evaluations=0;
};
inline SweepTelemetry& sweep_counters(){static SweepTelemetry value;return value;}
inline std::mutex& sweep_mutex(){static std::mutex value;return value;}
inline SweepTelemetry read_sweep_telemetry(){std::lock_guard lock(sweep_mutex());return sweep_counters();}
inline void record_sweep_layer(bool deferred){
 std::lock_guard lock(sweep_mutex());auto& s=sweep_counters();
 if(deferred)++s.deferred_layer_evaluations;else ++s.layer_evaluations;
}
inline void record_sweep_stack(){std::lock_guard lock(sweep_mutex());++sweep_counters().decode_stack_evaluations;}
inline void record_sweep_engram(){std::lock_guard lock(sweep_mutex());++sweep_counters().engram_evaluations;}
}
