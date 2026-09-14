#include "dsv41/runtime_bridge.h"
#include <cassert>
#include <cstdint>

struct Seen { int calls = 0; dsv41_event_kind_t kind{}; uint64_t id = 0; };
static int on_event(const dsv41_event_t *event, void *ctx) {
    auto *seen = static_cast<Seen *>(ctx);
    ++seen->calls; seen->kind = event->kind; seen->id = event->request_id; return 0;
}

int main() {
    assert(dsv41_bridge_create(999) == nullptr);
    auto *bridge = dsv41_bridge_create(DSV41_BRIDGE_ABI_VERSION);
    assert(bridge != nullptr);
    assert(dsv41_bridge_submit(nullptr, nullptr, on_event, nullptr, nullptr) == DSV41_BRIDGE_INVALID_ARGUMENT);
    const uint32_t tokens[] = {0, 42, 1000, 42};
    dsv41_request_t invalid{DSV41_BRIDGE_ABI_VERSION, tokens, 4, 0, 0.0f, 0};
    assert(dsv41_bridge_submit(bridge, &invalid, on_event, nullptr, nullptr) == DSV41_BRIDGE_INVALID_ARGUMENT);
    dsv41_request_t request{DSV41_BRIDGE_ABI_VERSION, tokens, 4, 16, 0.0f, 0};
    Seen seen;
    uint64_t request_id = 0;
    const int status = dsv41_bridge_submit(bridge, &request, on_event, &seen, &request_id);
    assert(status == DSV41_BRIDGE_UNAVAILABLE);
    assert(seen.calls == 1 && seen.kind == DSV41_EVENT_ERROR && seen.id == request_id && request_id != 0);
    assert(dsv41_bridge_cancel(bridge, request_id) == DSV41_BRIDGE_UNAVAILABLE);
    dsv41_bridge_destroy(bridge);
    return 0;
}
