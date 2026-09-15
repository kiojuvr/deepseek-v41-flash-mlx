#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
run_dir="artifacts/bridge-model/run-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$run_dir"
trap 'status=$?; echo "$status" > "$run_dir/exit-code.txt"; if ((status)); then echo "FAILED: inspect $run_dir; rerun the same command in a fresh run directory."; fi' EXIT
echo "Logs: $run_dir"
checkpoint=${CHECKPOINT:-/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash}
max_new=${MAX_NEW:-4}
temperature=${TEMPERATURE:-0}
seed=${SEED:-0}
if [[ ! "$max_new" =~ ^[0-9]+$ ]] || ((max_new<2 || max_new>262140)); then
 echo "MAX_NEW must be 2..262140" >&2; exit 2
fi
git rev-parse HEAD > "$run_dir/revision.txt"
git diff --binary > "$run_dir/tracked.patch"
cmake -S . -B build-mlx > "$run_dir/configure.log" 2>&1
cmake --build build-mlx --target dsv41-text-generate dsv41-bridge-model-check -j 4 > "$run_dir/build.log" 2>&1
if [[ -n "${TOKENS_FILE:-}" ]]; then cp "$TOKENS_FILE" "$run_dir/prompt.txt"; else echo '0 42 1000 42' > "$run_dir/prompt.txt"; fi
shasum -a 256 build-mlx/dsv41-text-generate build-mlx/dsv41-bridge-model-check build-mlx/libdsv41_runtime_bridge_mlx.dylib artifacts/checkpoint/summary.json artifacts/engram/metadata.json artifacts/engram/fixture-provenance.json src/runtime/bridge_stub.cpp src/runtime/bridge_mlx.cpp src/runtime/bridge_executor.hpp include/dsv41/runtime_bridge.h include/dsv41/generation_loop.hpp src/model/text_generate.cpp tools/benchmark/bridge_model_check.cpp "$run_dir/prompt.txt" > "$run_dir/identity.txt"
reference=(build-mlx/dsv41-text-generate "$checkpoint" artifacts/checkpoint/summary.json artifacts/engram/metadata.json "$max_new" "$temperature" "$seed" "$run_dir/prompt.txt")
bridge=(build-mlx/dsv41-bridge-model-check "$checkpoint" artifacts/checkpoint/summary.json artifacts/engram/metadata.json artifacts/engram/fixture-provenance.json "$run_dir/prompt.txt" "$run_dir/expected.txt" "$temperature" "$seed")
printf '%q ' "${reference[@]}" > "$run_dir/command.txt"
printf '\n' >> "$run_dir/command.txt"
printf '%q ' "${bridge[@]}" >> "$run_dir/command.txt"
printf '\n' >> "$run_dir/command.txt"
env -u DSV41_CHECK_GENERATION_LIFECYCLE "${reference[@]}" 2>&1 | tee "$run_dir/reference.log"
awk '/^generated:/{sub(/^generated: */, ""); print; count++} END {if(count!=1)exit 1}' "$run_dir/reference.log" > "$run_dir/expected.txt"
shasum -a 256 "$run_dir/expected.txt" "$run_dir/reference.log" >> "$run_dir/identity.txt"
"${bridge[@]}" 2>&1 | tee "$run_dir/test.log"
echo "Completed; results need review. C ABI validation does not qualify HTTP/SSE or 256K."
