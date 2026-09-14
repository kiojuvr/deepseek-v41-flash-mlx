#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
native=${1:-artifacts/logits-trace/native-20260914-145608-21879}; gate=${2:-artifacts/cpu-attention/gate-native-weights-20260914}
run_dir="artifacts/cpu-attention/cpu-moe-$(date +%Y%m%d-%H%M%S)-$$"; mkdir -p "$run_dir"
trap 's=$?; echo "$s" > "$run_dir/exit-code.txt"' EXIT; echo "Logs: $run_dir"
shasum -a 256 tools/reference/cpu_moe_compare.py tools/reference/cpu_reference.py artifacts/checkpoint/summary.json artifacts/checkpoint/verification.json "$native/manifest.json" "$native/encoder.layer0.ffn_in.npy" "$native/encoder.layer0.moe_out.npy" "$gate/manifest.json" "$gate/encoder.layer0.route_weights.npy" > "$run_dir/identity.txt"
set +e
/Volumes/SDXC-512/deltafin/.venv/bin/python tools/reference/cpu_moe_compare.py --checkpoint /Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash --native "$native" --gate "$gate" --output "$run_dir/report.json" 2>&1 | tee "$run_dir/test.log"
codes=("${PIPESTATUS[@]}"); status=${codes[0]}; if [ "$status" -eq 0 ] && [ "${codes[1]}" -ne 0 ]; then status=${codes[1]}; fi; set -e; echo "$status" > "$run_dir/compute-exit-code.txt"; if [ "$status" -ne 0 ]; then exit "$status"; fi
echo "CPU MoE comparison completed: $run_dir/report.json. Exit 0 does not imply qualification."
