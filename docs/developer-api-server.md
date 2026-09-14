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

The planned integration uses the pinned official `deepseek-recipe` revision and
a small C ABI event bridge. Recipe owns prompt/message encoding and stream
parsing; C++ owns tensor execution and committed token events. The bridge must
define request cancellation, backpressure, client disconnect, error propagation,
buffer lifetime and usage/finish accounting before the dependency is enabled.

Protocol smoke checks can run against the stub before M3 runtime integration:

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
