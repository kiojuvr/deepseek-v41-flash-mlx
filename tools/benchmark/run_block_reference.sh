#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
checkpoint=/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash
run_dir="artifacts/block/run-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$run_dir"
echo "Logs: $run_dir"
shasum -a 256 build-mlx/dsv41-block-test artifacts/checkpoint/summary.json > "$run_dir/identity.txt"
set +e
build-mlx/dsv41-block-test "$checkpoint" artifacts/checkpoint/summary.json 2>&1 | tee "$run_dir/test.log"
status=${PIPESTATUS[0]}
set -e
echo "$status" > "$run_dir/exit-code.txt"
if [ "$status" -ne 0 ]; then
  echo "FAILED: inspect $run_dir/test.log; rerun this command to retry in a fresh run directory."
  exit "$status"
fi
echo "Completed. Result still requires review; this is not full-model qualification."
