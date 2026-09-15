#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
run_dir="artifacts/m2-route-tie/run-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$run_dir/native" "$run_dir/omlx"
trap 'status=$?; echo "$status" > "$run_dir/exit-code.txt"; if ((status)); then echo "FAILED: inspect $run_dir; completed trace directories can be reused for comparison."; fi' EXIT
echo "Logs: $run_dir"
checkpoint=${CHECKPOINT:-/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash}
reference_python=${OMLX_PYTHON:-/System/Volumes/Data/Users/kioju/.venvs/omlx-0.7.0.dev2/bin/python3}
tokens_file=${TOKENS_FILE:-artifacts/text-backbone/run-20260915-161212-52162/prompt.txt}
if [[ ! -f "$tokens_file" ]]; then
 echo "TOKENS_FILE does not exist: $tokens_file" >&2
 exit 2
fi
token_count=$(/usr/bin/wc -w < "$tokens_file" | tr -d ' ')
if ((token_count<1 || token_count>256)); then
 echo "trace comparison requires 1..256 tokens" >&2
 exit 2
fi
git rev-parse HEAD > "$run_dir/revision.txt"
git diff --binary > "$run_dir/tracked.patch"
printf '%s\n' "checkpoint=$checkpoint" "tokens_file=$tokens_file" "token_count=$token_count" "omlx_python=$reference_python" > "$run_dir/config.txt"
cmake --build build-mlx --target dsv41-text-trace -j 4 > "$run_dir/build.log" 2>&1
shasum -a 256 build-mlx/dsv41-text-trace include/dsv41/generation_loop.hpp tools/trace/text_trace.cpp \
  tools/reference/trace_omlx.py tools/reference/compare_traces.py tools/reference/compare_route_ties.py \
  "$tokens_file" artifacts/checkpoint/summary.json artifacts/engram/metadata.json \
  artifacts/engram/fixture-provenance.json artifacts/checkpoint/verification.json \
  artifacts/references/omlx-v0.7.0.dev2.json \
  "$checkpoint/inference/model.py" > "$run_dir/identity.txt"
native_cmd=(build-mlx/dsv41-text-trace "$checkpoint" artifacts/checkpoint/summary.json artifacts/engram/metadata.json "$run_dir/native" "$tokens_file")
omlx_cmd=("$reference_python" tools/reference/trace_omlx.py --checkpoint "$checkpoint" --metadata artifacts/engram/metadata.json --output "$run_dir/omlx" --tokens-file "$tokens_file")
trace_compare_cmd=("$reference_python" tools/reference/compare_traces.py --native "$run_dir/native" --oracle "$run_dir/omlx" --output "$run_dir/trace-comparison.json")
tie_compare_cmd=("$reference_python" tools/reference/compare_route_ties.py --native "$run_dir/native" --external "$run_dir/omlx" --output "$run_dir/tie-comparison.json")
write_command(){
 local label=$1
 shift
 printf '%s: ' "$label" >> "$run_dir/commands.txt"
 printf '%q ' "$@" >> "$run_dir/commands.txt"
 printf '\n' >> "$run_dir/commands.txt"
}
write_command native "${native_cmd[@]}"
write_command omlx "${omlx_cmd[@]}"
write_command trace_compare "${trace_compare_cmd[@]}"
write_command tie_compare "${tie_compare_cmd[@]}"
"${native_cmd[@]}" > "$run_dir/native.log" 2>&1
"${omlx_cmd[@]}" > "$run_dir/omlx.log" 2>&1
"${trace_compare_cmd[@]}" > "$run_dir/trace-comparison.log" 2>&1
"${tie_compare_cmd[@]}" \
  2>&1 | tee "$run_dir/test.log"
echo "Completed; trace/tie differences require review. Agreement does not qualify the official torch.topk tie policy, M2 exit, performance, or 256K."
