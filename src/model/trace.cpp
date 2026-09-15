#include "dsv41/trace.hpp"
namespace dsv41 {
namespace {
TraceSink* g_sink=nullptr;
}
void set_active_trace_sink(TraceSink* sink){g_sink=sink;}
TraceSink* active_trace_sink(){return g_sink;}
void trace_record(const std::string& name,const mlx::core::array& value){
 if(g_sink)g_sink->record(name,value);
}
}
