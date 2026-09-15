# Developer API server

Non-RELEASE Axum server with an opt-in C++/MLX model backend and pinned
`deepseek-recipe` (`8cadfede7063c896b944e7bae05daa3549ae97ea`) encoding/output parser.

The default build is an unconnected protocol shell:

```sh
cargo run --manifest-path server/Cargo.toml
```

`DSV41_BIND` defaults to `127.0.0.1:8080`; `DSV41_MODEL` defaults to
`DeepSeek-V4.1-Flash`. Health/model discovery work; generation returns
`501 runtime_unavailable`. The `native-bridge` feature retains C ABI shell checks.

## Native model and SSE

SSE accepts `stream_options: {"include_usage": true}`. When requested, a final
`choices: []` usage chunk precedes `[DONE]`; ordinary chunks carry `usage: null`.
Omitting the option or setting it false omits SSE usage. Nonstream usage is unchanged.
The updated wire format passed a short full-model revalidation for both usage modes,
EOS completion, disconnect cancellation and the following request.

Requests support `thinking: {"type":"enabled"}` or `{"type":"disabled"}`;
omission preserves the enabled default. Prompt admission/encoding and recipe
output parsing share this request-local setting. A short disabled-thinking prompt
passed HTTP/SSE answer/EOS parity and recovery after an enabled-thinking request
was cancelled; see the evidence and command at the top of the bridge plan.
`reasoning_effort` accepts `none`, `minimal`, `low`, `medium`, `high`, `xhigh`,
and `max`, with the pinned recipe's normalization and explicit-thinking precedence.
A short full-model `low` request passed nonstream/SSE parity and disconnect recovery;
the other effort levels remain covered only by prompt/precedence tests.
Client function tools are rendered and parsed by the pinned recipe. Required
tool choice, `none`/named choice normalization, nonstream/SSE tool-call parity,
and a follow-up request containing the tool result passed checks; strict schemas
remain outside the current API contract. Request-local `stop` accepts one string
or up to 16 strings. The recipe suppresses the decoded match; SSE then requests
cooperative native cancellation. One short greedy, disabled-thinking, single-stop
request passed nonstream/SSE parity, usage, cancellation and recovery checks; wider
stop arrays, Unicode/overlap cases and reasoning/tool interactions remain unqualified.
Nonstream stop parsing now runs concurrently with native token production and joins
the cooperative terminal before returning. A short full-model recheck confirmed that
nonstream, SSE and post-disconnect recovery all cancel at the same seven-token usage
boundary. This remains narrower than general stop-sequence qualification.

`native-model` enables the MLX shared bridge. The model is constructed, executed
and destroyed on one dedicated OS thread. Nonstream HTTP passed the short
full-model check. Short SSE/nonstream parity and cancellation after the first
content delta also passed; the next request reproduced the same output and usage.
Long-context, sustained backpressure and broader disconnect validation remain pending.

From the repository root:

```sh
bash tools/benchmark/run_native_model_api_check.sh
```

This builds the server, loads the model, compares nonstream/SSE responses and
checks cancellation followed by a new request. It may take several minutes and
use hundreds of GB of unified memory. Checkpoint files remain read-only.
Logs/results go to `artifacts/native-model-api/run-*/`. See the
[bridge plan](../docs/m3-native-bridge-plan.md) for configuration, scope, timeouts
and failure recovery. M3 exit, performance/memory and 256K qualification remain
incomplete; this server is not RELEASE.
