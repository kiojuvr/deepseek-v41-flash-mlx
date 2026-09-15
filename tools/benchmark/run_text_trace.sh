#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
checkpoint=/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash
run_dir="artifacts/logits-trace/native-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$run_dir"
echo "Trace: $run_dir"
shasum -a 256 build-mlx/dsv41-text-trace artifacts/checkpoint/summary.json artifacts/engram/metadata.json artifacts/engram/fixture-provenance.json > "$run_dir/identity.txt"
extra=()
if [ -n "${TOKENS_FILE:-}" ]; then
  extra=("$TOKENS_FILE")
  shasum -a 256 "$TOKENS_FILE" >> "$run_dir/identity.txt"
fi
set +e
build-mlx/dsv41-text-trace "$checkpoint" artifacts/checkpoint/summary.json artifacts/engram/metadata.json "$run_dir" "${extra[@]}" 2>&1 | tee "$run_dir/trace.log"
pipeline_status=("${PIPESTATUS[@]}")
status=${pipeline_status[0]}
if [ "$status" -eq 0 ] && [ "${pipeline_status[1]}" -ne 0 ]; then status=${pipeline_status[1]}; fi
set -e
echo "$status" > "$run_dir/exit-code.txt"
if [ "$status" -ne 0 ]; then
  echo "FAILED: inspect $run_dir/trace.log; rerun this command to retry in a fresh run directory."
  exit "$status"
fi
echo "Completed. Native trace saved; oracle comparison still required."
