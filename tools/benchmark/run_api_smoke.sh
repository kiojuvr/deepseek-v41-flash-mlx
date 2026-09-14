#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

port=${DSV41_SMOKE_PORT:-18080}
run_dir="artifacts/api-smoke/run-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$run_dir"
DSV41_BIND="127.0.0.1:${port}" cargo run --quiet --manifest-path server/Cargo.toml >"$run_dir/server.log" 2>&1 &
pid=$!
cleanup() { kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; }
trap 'status=$?; cleanup; echo "$status" > "$run_dir/exit-code.txt"' EXIT
ready=0
for _ in $(seq 1 100); do
  if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>/dev/null; then ready=1; break; fi
  sleep 0.2
done
if [[ "$ready" -ne 1 ]]; then
  echo "server did not become ready; inspect $run_dir/server.log" >&2
  exit 1
fi

curl -fsS "http://127.0.0.1:${port}/health" >"$run_dir/health.json"
curl -fsS "http://127.0.0.1:${port}/v1/models" >"$run_dir/models.json"
curl -sS -o "$run_dir/chat-501.json" -w '%{http_code}\n' \
  -X POST "http://127.0.0.1:${port}/v1/chat/completions" -H 'content-type: application/json' \
  -d '{"model":"DeepSeek-V4.1-Flash","messages":[{"role":"user","content":"literal <｜deepseek_image｜>"}]}' >"$run_dir/chat-501.status"
curl -sS -o "$run_dir/model-404.json" -w '%{http_code}\n' \
  -X POST "http://127.0.0.1:${port}/v1/chat/completions" -H 'content-type: application/json' \
  -d '{"model":"wrong-model","messages":[{"role":"user","content":"Hello"}]}' >"$run_dir/model-404.status"
curl -sS -o "$run_dir/empty-400.json" -w '%{http_code}\n' \
  -X POST "http://127.0.0.1:${port}/v1/chat/completions" -H 'content-type: application/json' \
  -d '{"model":"DeepSeek-V4.1-Flash","messages":[]}' >"$run_dir/empty-400.status"
curl -sS -o "$run_dir/options-501.json" -w '%{http_code}\n' \
  -X POST "http://127.0.0.1:${port}/v1/chat/completions" -H 'content-type: application/json' \
  -d '{"model":"DeepSeek-V4.1-Flash","messages":[{"role":"user","content":"options"}],"max_tokens":4,"temperature":0.7,"seed":7}' >"$run_dir/options-501.status"
curl -sS -o "$run_dir/temperature-400.json" -w '%{http_code}\n' \
  -X POST "http://127.0.0.1:${port}/v1/chat/completions" -H 'content-type: application/json' \
  -d '{"model":"DeepSeek-V4.1-Flash","messages":[{"role":"user","content":"bad"}],"temperature":-1}' >"$run_dir/temperature-400.status"
curl -sS -o "$run_dir/max-tokens-400.json" -w '%{http_code}\n' \
  -X POST "http://127.0.0.1:${port}/v1/chat/completions" -H 'content-type: application/json' \
  -d '{"model":"DeepSeek-V4.1-Flash","messages":[{"role":"user","content":"bad"}],"max_tokens":0}' >"$run_dir/max-tokens-400.status"
curl -sS -o "$run_dir/max-tokens-limit-400.json" -w '%{http_code}\n' \
  -X POST "http://127.0.0.1:${port}/v1/chat/completions" -H 'content-type: application/json' \
  -d '{"model":"DeepSeek-V4.1-Flash","messages":[{"role":"user","content":"bad"}],"max_tokens":262145}' >"$run_dir/max-tokens-limit-400.status"

python3 - "$run_dir" <<'PY'
import json, pathlib, sys
r=pathlib.Path(sys.argv[1])
assert json.loads((r/'health.json').read_text()) == {'status':'ok','runtime':'unconnected','release':False}
assert (r/'chat-501.status').read_text().strip() == '501'
assert (r/'model-404.status').read_text().strip() == '404'
assert (r/'empty-400.status').read_text().strip() == '400'
assert (r/'options-501.status').read_text().strip() == '501'
assert (r/'temperature-400.status').read_text().strip() == '400'
assert (r/'max-tokens-400.status').read_text().strip() == '400'
assert (r/'max-tokens-limit-400.status').read_text().strip() == '400'
print('PASS: developer API smoke contract')
PY
echo "API smoke completed: $run_dir"
