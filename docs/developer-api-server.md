# Developer API server contract

The server is deliberately available during M3, before model calculation is
qualified. Its purpose is to expose protocol failures early; it is not a model
backend and is never a RELEASE artifact.

Current endpoints:

| Endpoint | Behavior |
| --- | --- |
| `GET /health` | 200 JSON, includes `runtime: unconnected` and `release: false` |
| `GET /v1/models` | 200 OpenAI-style list with configured model ID |
| `POST /v1/chat/completions` | validates model and nonempty messages; returns explicit 501 `runtime_unavailable` |

The stub accepts JSON only to test request routing and validation. It does not
claim prompt encoding, tool calls, reasoning, image input, streaming, usage,
stop sequences or generation. Unsupported behavior must remain an explicit
error rather than a fabricated completion.

When built with `native-model` and `DSV41_NATIVE_MODEL=1`, the same endpoint is
connected to the native bridge and pinned `deepseek-recipe`. In that mode text
prompt encoding, reasoning settings, bounded SSE/cancellation, usage snapshots,
client function tools, historical tool results, and nonstream/SSE tool-call
parity are implemented. Request-local decoded stop sequences are connected to
the pinned recipe, including SSE cooperative cancellation. The developer server
still does not advertise release readiness or 256K qualification; vision input,
strict tool schemas, broader stop validation, and long-context qualification remain
separate gates. A short single-string stop passed native-model nonstream/SSE parity,
usage, cooperative cancellation and post-cancellation recovery checks.
Nonstream local-stop handling was subsequently changed from post-hoc parsing to a
bounded incremental parser/cancellation path. Its short full-model recheck passed at
the same seven-token usage/cancellation boundary for nonstream and SSE.

The handler depends on a `RuntimeBackend` trait in
`server/src/backend.rs`. `UnconnectedBackend` is the current implementation;
the future C++ bridge can replace it without changing protocol validation or
OpenAI-compatible response mapping.

The backend boundary now carries optional `max_tokens`, `temperature`, and
`seed` values. Non-finite or negative temperatures are rejected with HTTP 400;
valid options are preserved for the native bridge even while the current
backend returns `runtime_unavailable`.

The planned integration uses the pinned official `deepseek-recipe` revision and
a small C ABI event bridge. Recipe owns prompt/message encoding and stream
parsing; C++ owns tensor execution and committed token events. The bridge must
define request cancellation, backpressure, client disconnect, error propagation,
buffer lifetime and usage/finish accounting before the dependency is enabled.

The first ABI draft is [`include/dsv41/runtime_bridge.h`](../include/dsv41/runtime_bridge.h).
Requests borrow token IDs only for the synchronous submit call. Events are
delivered in committed-token order; token and error strings are borrowed until
the callback returns. A callback can request cooperative cancellation, which
must end in an explicit `cancelled` finish event or an error before submit
returns. The header is an interface contract only and is not a runtime
qualification result.

Protocol smoke checks can run against the stub before M3 runtime integration:

The reproducible form is:

```sh
bash tools/benchmark/run_api_smoke.sh
```

It stores per-response JSON, HTTP statuses, server log and exit code under
`artifacts/api-smoke/run-.../`. Set `DSV41_SMOKE_PORT` when port 18080 is in use;
the process is always cleaned up by the script trap.

```sh
cargo run --manifest-path server/Cargo.toml
curl -sS http://127.0.0.1:8080/health
curl -sS http://127.0.0.1:8080/v1/models
curl -sS -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"DeepSeek-V4.1-Flash","messages":[{"role":"user","content":"ping"}]}'
```

The last request must return HTTP 501 and `runtime_unavailable`. A 200 response
before native integration is a protocol test failure. OpenCode compatibility,
recipe fixtures, SSE, image/text boundaries and full cancellation tests remain
M6 requirements and cannot be inferred from this stub.

## Reviewed smoke test (2026-09-14)

The developer server was exercised manually. Health and model discovery returned
200; a known-model request returned explicit 501 `runtime_unavailable`; unknown
model and empty messages returned 404 and 400 respectively. Literal
`<｜deepseek_image｜>` text and structured content both reached the same explicit
backend-unavailable response rather than being rejected by a protocol-side
special-token check. A missing `model` field produced the expected 422 JSON
deserialization error. The checklist is recorded in
`artifacts/api-smoke-20260914.json`. These results validate only the stub's
HTTP/error contract and do not qualify recipe encoding or native inference.

The reproducible runner was then executed successfully at
`artifacts/api-smoke/run-20260914-221558-29823/` (exit code 0). Its readiness
loop suppresses transient connection errors and fails explicitly when the
server does not become ready within 20 seconds.

After generation option coverage was added, the runner passed again at
`artifacts/api-smoke/run-20260915-002706-32404/` (exit code 0). Valid
`max_tokens/temperature/seed` reached the explicit 501 backend response, while
negative temperature returned 400.

The server also rejects `max_tokens=0` and values above the current 262,144-token
admission limit with HTTP 400. This limit is an admission guard, not a 256K
qualification result; prompt tokens plus output reserve still require M4 testing.

The zero-limit validation was then exercised at
`artifacts/api-smoke/run-20260915-002902-32537/` (exit code 0). The complete
status set was 501 for valid backend requests, 404 for an unknown model, and
400 for empty messages, negative temperature, and `max_tokens=0`.

The 256K admission guard was verified at
`artifacts/api-smoke/run-20260915-003216-32696/` (exit code 0). A request with
`max_tokens=262145` returned 400, while the valid option request continued to
the explicit 501 backend-unavailable response.
