#pragma once
#include <mlx/mlx.h>
#include <string>
namespace dsv41 {
// Optional observation hook for the reference forward. Recording must not change numerics.
class TraceSink {
public:
 virtual ~TraceSink()=default;
 virtual void record(const std::string& name,const mlx::core::array& value)=0;
};
// Active sink for sub-boundary recording inside blocks; set by the encoder/decoder loops.
void set_active_trace_sink(TraceSink* sink);
TraceSink* active_trace_sink();
// No-op when no sink is active.
void trace_record(const std::string& name,const mlx::core::array& value);
}
