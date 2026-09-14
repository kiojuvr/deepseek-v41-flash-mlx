#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 <native-backbone-trace> <gate-trace> <expert-id> <output-trace> <output-report> <checkpoint>" >&2
  exit 2
fi

native_trace=$1
gate_trace=$2
expert_id=$3
output_trace=$4
output_report=$5
checkpoint=$6
summary=artifacts/checkpoint/summary.json

if [[ -e "$output_trace" || -e "$output_report" ]]; then
  echo "output paths must be fresh: $output_trace / $output_report" >&2
  exit 2
fi

cmake --build build-mlx --target dsv41-expert-trace -j 4
build-mlx/dsv41-expert-trace \
  "$checkpoint" "$summary" "$native_trace" "$gate_trace" "$output_trace" "$expert_id"

python_bin=${DSV41_PYTHON:-/Volumes/SDXC-512/deltafin/.venv/bin/python}
"$python_bin" tools/reference/compare_expert_stages.py \
  --checkpoint "$checkpoint" \
  --native "$output_trace" \
  --input "$native_trace" \
  --gate "$gate_trace" \
  --expert "$expert_id" \
  --output "$output_report"

echo "Expert stage comparison completed: $output_report"
