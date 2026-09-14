#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
checkpoint=/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash
run_dir="artifacts/text-generate/run-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$run_dir"
echo "Logs: $run_dir"
cmake --build build-mlx --target dsv41-text-generate -j 4 2>&1 | tee "$run_dir/build.log"
shasum -a 256 build-mlx/dsv41-text-generate artifacts/checkpoint/summary.json artifacts/engram/metadata.json artifacts/engram/fixture-provenance.json > "$run_dir/identity.txt"
set +e
build-mlx/dsv41-text-generate "$checkpoint" artifacts/checkpoint/summary.json artifacts/engram/metadata.json "${MAX_NEW:-16}" "${TEMPERATURE:-0}" "${SEED:-0}" 2>&1 | tee "$run_dir/test.log"
pipeline_status=("${PIPESTATUS[@]}")
status=${pipeline_status[0]}
if [ "$status" -eq 0 ] && [ "${pipeline_status[1]}" -ne 0 ]; then status=${pipeline_status[1]}; fi
set -e
echo "$status" > "$run_dir/exit-code.txt"
if [ "$status" -ne 0 ]; then
  echo "FAILED: inspect $run_dir/test.log; rerun this command to retry in a fresh run directory."
  exit "$status"
fi
echo "Completed. Generated tokens require review; RNG is not oracle-verified and this is not M2 qualification."
