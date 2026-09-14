#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
reference_python=/Volumes/SDXC-512/deltafin/.venv/bin/python
native=${1:-artifacts/logits-trace/native-20260914-145608-21879}
run_dir="artifacts/cpu-attention/run-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$run_dir"
trap 'status=$?; echo "$status" > "$run_dir/exit-code.txt"' EXIT
echo "Logs: $run_dir"
shasum -a 256 tools/reference/cpu_reference.py tools/reference/trace_cpu_attention.py tools/reference/compare_traces.py artifacts/checkpoint/summary.json artifacts/checkpoint/verification.json "$native/manifest.json" "$native/encoder.layer0.attn_in.npy" > "$run_dir/identity.txt"
set +e
"$reference_python" tools/reference/trace_cpu_attention.py --checkpoint /Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash --native "$native" --output "$run_dir/trace" 2>&1 | tee "$run_dir/test.log"
codes=("${PIPESTATUS[@]}")
status=${codes[0]}
if [ "$status" -eq 0 ] && [ "${codes[1]}" -ne 0 ]; then status=${codes[1]}; fi
set -e
echo "$status" > "$run_dir/trace-exit-code.txt"
if [ "$status" -ne 0 ]; then exit "$status"; fi
"$reference_python" tools/reference/compare_traces.py --native "$native" --oracle "$run_dir/trace" --output "$run_dir/report.json" > "$run_dir/comparison.log" 2>&1
echo "Trace completed: $run_dir/report.json. Exit 0 does not mean numerical agreement."
