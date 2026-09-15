#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
checkpoint=/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash
run_dir="artifacts/text-octet/run-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$run_dir"
echo "Logs: $run_dir"
shasum -a 256 build-mlx/dsv41-text-octet-test artifacts/checkpoint/summary.json artifacts/engram/metadata.json artifacts/engram/fixture-provenance.json > "$run_dir/identity.txt"
set +e
build-mlx/dsv41-text-octet-test "$checkpoint" artifacts/checkpoint/summary.json artifacts/engram/metadata.json 2>&1 | tee "$run_dir/test.log"
pipeline_status=("${PIPESTATUS[@]}")
status=${pipeline_status[0]}
if [ "$status" -eq 0 ] && [ "${pipeline_status[1]}" -ne 0 ]; then status=${pipeline_status[1]}; fi
set -e
echo "$status" > "$run_dir/exit-code.txt"
if [ "$status" -ne 0 ]; then
  echo "FAILED: inspect $run_dir/test.log; rerun this command to retry in a fresh run directory."
  exit "$status"
fi
echo "Completed. Result still requires review; this is not full-model qualification."
