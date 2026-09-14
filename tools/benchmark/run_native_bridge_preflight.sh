#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
checkpoint=${DSV41_CHECKPOINT:-/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash}
tokenizer=${DSV41_TOKENIZER:-$checkpoint/tokenizer.json}
port=${DSV41_SMOKE_PORT:-18081}
run_dir="artifacts/api-smoke/preflight-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$run_dir"
trap 'status=$?; echo "$status" > "$run_dir/exit-code.txt"' EXIT
for path in "$checkpoint" "$tokenizer" build-mlx/libdsv41_runtime_bridge.a; do
  if [[ ! -e "$path" ]]; then
    echo "missing prerequisite: $path" | tee "$run_dir/preflight-error.txt" >&2
    exit 2
  fi
done
DSV41_CARGO_FEATURES='native-bridge recipe-adapter' DSV41_NATIVE_BRIDGE=1 \
DSV41_EXPECTED_RUNTIME=native-bridge DSV41_TOKENIZER="$tokenizer" \
DSV41_BRIDGE_LIB_DIR="$PWD/build-mlx" DSV41_SMOKE_PORT="$port" \
bash tools/benchmark/run_api_smoke.sh | tee "$run_dir/smoke.log"
echo "Native bridge preflight completed: $run_dir"
