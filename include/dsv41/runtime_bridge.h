#pragma once

// Stable C ABI boundary between the recipe/API process and the C++ runtime.
// Model execution is opt-in through create_model in the MLX bridge library.

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DSV41_BRIDGE_ABI_VERSION 1u
#define DSV41_BRIDGE_OK 0
#define DSV41_BRIDGE_INVALID_ARGUMENT -1
#define DSV41_BRIDGE_UNAVAILABLE -2
#define DSV41_BRIDGE_BUSY -3
#define DSV41_BRIDGE_RUNTIME_ERROR -4
#define DSV41_BRIDGE_NOT_FOUND -5
#define DSV41_BRIDGE_MAX_NEW_TOKENS 262144u

typedef struct dsv41_bridge dsv41_bridge_t;

dsv41_bridge_t *dsv41_bridge_create(uint32_t abi_version);
typedef struct {
    uint32_t abi_version;
    const char *checkpoint_path;
    const char *summary_path;
    const char *metadata_path;
    const char *provenance_path;
    const uint32_t *stop_tokens;
    size_t stop_token_count;
} dsv41_model_config_t;
// Copies configuration during this call; checkpoint remains read-only.
// Returns NULL on failure, optionally writing a NUL-terminated diagnostic.
// Model construction can be expensive. The shell library returns unavailable.
// MLX model create/submit/destroy must run on the same OS thread. cancel may
// run on another thread; mutual exclusion alone does not transfer MLX streams.
dsv41_bridge_t *dsv41_bridge_create_model(const dsv41_model_config_t *config,
                                         char *error_out, size_t error_capacity);
// Caller must join all submit/cancel calls before destroy. Never call in callback.
void dsv41_bridge_destroy(dsv41_bridge_t *bridge);

typedef struct {
    uint32_t abi_version;
    const uint32_t *input_tokens; // borrowed for the duration of submit
    size_t input_token_count;
    uint32_t max_new_tokens;
    float temperature;
    uint64_t seed;
} dsv41_request_t;

typedef enum {
    DSV41_EVENT_TOKEN = 1,
    DSV41_EVENT_FINISHED = 2,
    DSV41_EVENT_ERROR = 3,
} dsv41_event_kind_t;

typedef struct {
    dsv41_event_kind_t kind;
    uint64_t request_id;
    uint64_t committed_index; // TOKEN: zero-based index; terminal: committed count
    uint32_t token_id;       // valid for TOKEN
    const char *finish_reason; // borrowed until callback returns
    const char *error_code;    // borrowed until callback returns
    const char *error_message; // borrowed until callback returns
} dsv41_event_t;

// Return 0 to continue. A nonzero return requests cancellation after the
// current committed event. The event and pointed-to strings are borrowed only
// for the callback duration; copy them before returning.
typedef int (*dsv41_event_callback)(const dsv41_event_t *event, void *context);

// The bridge retains no request token memory after this call returns.
// Events are delivered synchronously in committed-token order.
// Accepted submissions emit exactly one terminal event. Invalid/busy requests
// emit no events and set request_id_out to zero. Cancellation returns OK with
// FINISHED("cancelled"); ordinary completion uses "stop" or "length".
// One model operation at a time per process; concurrent/reentrant calls return
// BUSY instead of queueing heavyweight model work. Callbacks must not throw.
int dsv41_bridge_submit(dsv41_bridge_t *bridge, const dsv41_request_t *request,
                        dsv41_event_callback callback, void *context,
                        uint64_t *request_id_out);
// Optional cancellation poll runs synchronously on the model owner thread at
// generation boundaries, including before prefill/first TOKEN. Nonzero cancels.
// It must not throw or block. context has the same lifetime as submit.
typedef int (*dsv41_cancel_poll)(void *context);
int dsv41_bridge_submit_cancellable(dsv41_bridge_t *bridge, const dsv41_request_t *request,
    dsv41_event_callback callback, void *context, dsv41_cancel_poll poll,
    uint64_t *request_id_out);

// Safe from the callback or another thread. Cancellation is cooperative and
// must produce either FINISHED(reason="cancelled") or ERROR before submit
// returns.
int dsv41_bridge_cancel(dsv41_bridge_t *bridge, uint64_t request_id);

#ifdef __cplusplus
}
#endif
