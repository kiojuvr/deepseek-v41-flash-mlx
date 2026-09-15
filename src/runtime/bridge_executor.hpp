#pragma once
#include "dsv41/runtime_bridge.h"
#include "dsv41/generation_loop.hpp"
#include <memory>
namespace dsv41 {
// Internal injection boundary for checkpoint-free ABI lifecycle tests.
class BridgeExecutor {
public:
 virtual ~BridgeExecutor()=default;
 virtual GenerationResult generate(const dsv41_request_t&,const GenerationControl&)=0;
};
std::unique_ptr<BridgeExecutor> make_bridge_executor(const dsv41_model_config_t&);
dsv41_bridge_t* bridge_from_executor(std::unique_ptr<BridgeExecutor>);
}
