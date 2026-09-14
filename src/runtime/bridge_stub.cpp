#include "dsv41/runtime_bridge.h"

#include <atomic>
#include <cmath>
#include <cstring>

struct dsv41_bridge {
    std::atomic<uint64_t> next_request{1};
};

extern "C" dsv41_bridge_t *dsv41_bridge_create(uint32_t abi_version) {
    if (abi_version != DSV41_BRIDGE_ABI_VERSION) return nullptr;
    return new dsv41_bridge{};
}

extern "C" void dsv41_bridge_destroy(dsv41_bridge_t *bridge) { delete bridge; }

extern "C" int dsv41_bridge_submit(dsv41_bridge_t *bridge, const dsv41_request_t *request,
                                     dsv41_event_callback callback, void *context,
                                     uint64_t *request_id_out) {
    if (!bridge || !request || request->abi_version != DSV41_BRIDGE_ABI_VERSION ||
        (!request->input_tokens && request->input_token_count) || !callback ||
        request->input_token_count == 0 || request->max_new_tokens == 0 ||
        !std::isfinite(request->temperature) || request->temperature < 0.0f) {
        return DSV41_BRIDGE_INVALID_ARGUMENT;
    }
    const auto id = bridge->next_request.fetch_add(1, std::memory_order_relaxed);
    if (request_id_out) *request_id_out = id;
    const dsv41_event_t event{DSV41_EVENT_ERROR, id, 0, 0, nullptr,
                              "runtime_unavailable", "native runtime bridge is not connected"};
    (void)callback(&event, context);
    return DSV41_BRIDGE_UNAVAILABLE;
}

extern "C" int dsv41_bridge_cancel(dsv41_bridge_t *bridge, uint64_t request_id) {
    if (!bridge || request_id == 0) return DSV41_BRIDGE_INVALID_ARGUMENT;
    return DSV41_BRIDGE_UNAVAILABLE;
}
