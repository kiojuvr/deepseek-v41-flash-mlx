# Developer API server

This is a non-RELEASE protocol server. It intentionally runs before the native
runtime bridge is connected so OpenCode and other clients can exercise endpoint
shape, model discovery, validation errors and explicit backend-unavailable
responses while M3 work continues.

The target integration is Axum plus the pinned `deepseek-recipe` revision
`8cadfede7063c896b944e7bae05daa3549ae97ea`. That dependency remains commented
until its Cargo lock and C++ ABI event contract are fixed. The stub does not
parse prompts or generate tokens and must not be used as an inference server.

Run locally with:

```sh
cargo run --manifest-path server/Cargo.toml
```

Set `DSV41_BIND` (default `127.0.0.1:8080`) and `DSV41_MODEL` (default
`DeepSeek-V4.1-Flash`). `/health` and `/v1/models` return 200. Chat completions
validate model and nonempty messages, then return 501 with
`runtime_unavailable`; this explicit response prevents a protocol client from
mistaking a stub for successful inference. Streaming currently returns the same
501 and is not advertised as supported.

The developer server is excluded from RELEASE and qualification until recipe
prompt/token identity, SSE chunking, usage/finish fields, cancellation and the
native bridge lifecycle have real integration tests. See
`docs/developer-api-server.md`.
