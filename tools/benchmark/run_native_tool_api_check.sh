#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
run_dir="artifacts/native-tool-api/run-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$run_dir"
trap 'status=$?; echo "$status" > "$run_dir/exit-code.txt"; if ((status)); then echo "FAILED: inspect $run_dir; rerun with DSV41_SKIP_BUILD=1 after fixing a validation-only failure, or rerun normally for a fresh build."; fi' EXIT
echo "Logs: $run_dir"
export DSV41_CHECKPOINT=${DSV41_CHECKPOINT:-/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash}
export DSV41_TOKENIZER=${DSV41_TOKENIZER:-$DSV41_CHECKPOINT/tokenizer.json}
export DSV41_SUMMARY=${DSV41_SUMMARY:-artifacts/checkpoint/summary.json}
export DSV41_METADATA=${DSV41_METADATA:-artifacts/engram/metadata.json}
export DSV41_PROVENANCE=${DSV41_PROVENANCE:-artifacts/engram/fixture-provenance.json}
export DSV41_BRIDGE_LIB_DIR="$PWD/build-mlx"
git rev-parse HEAD > "$run_dir/revision.txt"
git diff --binary > "$run_dir/tracked.patch"
printf '%s\n' \
  "checkpoint=$DSV41_CHECKPOINT" \
  "tokenizer=$DSV41_TOKENIZER" \
  "summary=$DSV41_SUMMARY" \
  "metadata=$DSV41_METADATA" \
  "provenance=$DSV41_PROVENANCE" \
  "port=${DSV41_SMOKE_PORT:-18083}" \
  "tool_max_tokens=${DSV41_TOOL_MAX_TOKENS:-128}" \
  "skip_build=${DSV41_SKIP_BUILD:-0}" > "$run_dir/config.txt"
if [[ ${DSV41_SKIP_BUILD:-0} != 1 ]]; then
  cmake -S . -B build-mlx > "$run_dir/configure.log" 2>&1
  cmake --build build-mlx --target dsv41_runtime_bridge_mlx -j 4 > "$run_dir/native-build.log" 2>&1
  cargo test --manifest-path server/Cargo.toml --features native-model --locked --offline > "$run_dir/unit-tests.log" 2>&1
  cargo build --manifest-path server/Cargo.toml --features native-model --locked --offline > "$run_dir/server-build.log" 2>&1
fi
shasum -a 256 \
  server/target/debug/dsv41-developer-server \
  build-mlx/libdsv41_runtime_bridge_mlx.dylib \
  server/src/main.rs server/src/backend.rs server/src/native.rs server/src/owner_worker.rs \
  server/src/recipe_adapter.rs server/src/streaming.rs server/src/request_cancel.rs \
  src/model/text_generate.cpp include/dsv41/text_generate.hpp \
  server/Cargo.lock "$DSV41_TOKENIZER" "$DSV41_CHECKPOINT/config.json" \
  "$DSV41_SUMMARY" "$DSV41_METADATA" "$DSV41_PROVENANCE" \
  tools/reference/check_native_tool_api.py tools/benchmark/run_native_tool_api_check.sh \
  > "$run_dir/identity.txt"
"${PYTHON:-python3}" tools/reference/check_native_tool_api.py "$run_dir" 2>&1 | tee "$run_dir/test.log"
echo "Completed; results need review. Tool-call results require review; 256K remains unqualified."
