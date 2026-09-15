#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
run_dir="artifacts/m2-route-tie/official-frozen-gate-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$run_dir"
trap 'status=$?; echo "$status" > "$run_dir/exit-code.txt"; if ((status)); then echo "FAILED: inspect $run_dir; rerun this command for a fresh diagnostic."; fi' EXIT
echo "Logs: $run_dir"
checkpoint=${CHECKPOINT:-/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash}
torch_python=${TORCH_PYTHON:-/Volumes/SDXC-512/deltafin/.venv/bin/python}
native_trace=${NATIVE_TRACE:-artifacts/m2-route-tie/run-20260915-162023-52563/native}
layer=${TIE_LAYER:-8}
token=${TIE_TOKEN:-14}
if [[ ! -f "$native_trace/manifest.json" ]]; then
  echo "native trace manifest does not exist: $native_trace" >&2
  exit 2
fi
cmd=("$torch_python" tools/reference/check_official_frozen_gate.py
  --checkpoint "$checkpoint" --native-trace "$native_trace"
  --layer "$layer" --token "$token" --output "$run_dir/result.json")
printf '%q ' "${cmd[@]}" > "$run_dir/command.txt"
printf '\n' >> "$run_dir/command.txt"
printf '%s\n' "checkpoint=$checkpoint" "native_trace=$native_trace" "layer=$layer" "token=$token" "torch_python=$torch_python" > "$run_dir/config.txt"
git rev-parse HEAD > "$run_dir/revision.txt"
git diff --binary > "$run_dir/tracked.patch"
shasum -a 256 tools/reference/check_official_frozen_gate.py "$native_trace/manifest.json" \
  "$native_trace/encoder.layer${layer}.ffn_in.npy" "$checkpoint/inference/model.py" \
  "$checkpoint/config.json" "$checkpoint/model.safetensors.index.json" > "$run_dir/identity.txt"
"${cmd[@]}" 2>&1 | tee "$run_dir/test.log"
echo "Completed; official torch CPU observation requires review. CUDA topk, M2, performance, and 256K remain unqualified."
