#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
checkpoint=/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash
python=/System/Volumes/Data/Users/kioju/.venvs/omlx-0.7.0.dev2/bin/python3
run_dir="artifacts/logits-trace/omlx-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$run_dir"
echo "Trace: $run_dir"
{
  "$python" -c "import mlx.core as mx, numpy; print('mlx', mx.__version__)"
  shasum -a 256 "$python" tools/reference/trace_omlx.py artifacts/engram/metadata.json artifacts/references/omlx-v0.7.0.dev2.json
} > "$run_dir/identity.txt"
extra=()
if [ -n "${TOKENS_FILE:-}" ]; then
  extra=(--tokens-file "$TOKENS_FILE")
  shasum -a 256 "$TOKENS_FILE" >> "$run_dir/identity.txt"
fi
set +e
"$python" tools/reference/trace_omlx.py --checkpoint "$checkpoint" \
  --metadata artifacts/engram/metadata.json --output "$run_dir" "${extra[@]}" 2>&1 | tee "$run_dir/trace.log"
pipeline_status=("${PIPESTATUS[@]}")
status=${pipeline_status[0]}
if [ "$status" -eq 0 ] && [ "${pipeline_status[1]}" -ne 0 ]; then status=${pipeline_status[1]}; fi
set -e
echo "$status" > "$run_dir/exit-code.txt"
if [ "$status" -ne 0 ]; then
  echo "FAILED: inspect $run_dir/trace.log; rerun this command to retry in a fresh run directory."
  exit "$status"
fi
echo "Completed. oMLX trace saved; native comparison still required."
