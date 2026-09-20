#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mode=${1:-bounded}
[[ "$mode" == bounded || "$mode" == 64k ]] || { echo 'usage: bash tools/benchmark/run_decode_stack_graph.sh [bounded|64k]' >&2; exit 2; }
root=${DSV41_DECODE_GRAPH_DIR:-artifacts/context-ladder/decode-stack-$(date +%Y%m%d-%H%M%S)-$$}
mkdir -p "$root"
printf '%s\n' \
 'scope=bounded: 129-token prefix + four teacher-forced decode state/logit comparisons, then fresh baseline/candidate 2063+8 full-path pair; 64k: eight 8128+64 turns per variant' \
 'resources=bounded approximately 10-20 minutes; 64k approximately 100-140 minutes; sequential processes, <=340 GB Unified Memory each, checkpoint read-only' \
 'logs=<root>/parity/{test.log,resource.log}; <root>/{bounded,64k}/{baseline,candidate}/ canonical result/config/identity/resource/system logs; comparison.json' \
 'failure=stop and retain artifacts; retry incomplete stage in a fresh process; never resume partial model state' \
 "resume=DSV41_DECODE_GRAPH_DIR=$root bash tools/benchmark/run_decode_stack_graph.sh $mode" \
 'gate=requires raw-log review; bounded pass cannot promote; a failed candidate returns to the next 64K bottleneck, never 128K' | tee "$root/README.txt"
if [[ ${DSV41_DRY_RUN:-0} == 1 ]]; then echo "DRY RUN: parity -> $mode baseline -> $mode candidate -> comparison; no model/build started"; exit 0; fi
if [[ "$mode" == 64k && ! -s "$root/bounded/comparison.json" ]]; then
 echo 'Run and review bounded mode in this root first; 64k cannot bypass its correctness gate.' >&2; exit 2
fi
if [[ "$mode" == 64k ]]; then
 jq -e '.status=="gates_passed_requires_raw_log_review_not_promotion"' "$root/bounded/comparison.json" >/dev/null
fi
# Include untracked implementation files; a changed source cannot reuse a stage.
identity=$( { git rev-parse HEAD; git diff --binary; rg --files src include tests tools/benchmark | LC_ALL=C sort | while IFS= read -r file; do shasum -a 256 "$file"; done; } | shasum -a 256 | awk '{print $1}')
if [[ -f "$root/source-identity.txt" ]]; then
 [[ $(<"$root/source-identity.txt") == "$identity" ]] || { echo 'source identity changed; use a new root' >&2; exit 2; }
else printf '%s\n' "$identity" > "$root/source-identity.txt"; fi
export DSV41_RUNTIME_GROUPED_EXPERT_PIPELINE=0 DSV41_RUNTIME_COMPONENT_PROFILE=0
export DSV41_RUNTIME_LAYER_FINITE_CHECKS=0 DSV41_RUNTIME_ROUTE_DIAGNOSTICS=0 DSV41_RUNTIME_INDEX_DIAGNOSTICS=0
export DSV41_RUNTIME_RESIDENT_EXPERT_ATLAS=1 DSV41_RUNTIME_PACKED_EXPERT_BANK=1
export DSV41_RUNTIME_COMPACT_EXPERT_BANK=0 DSV41_RUNTIME_GROUP_SELECTED_EXPERTS=0
export DSV41_RUNTIME_LAYER_SWEEP=1 DSV41_RUNTIME_BATCHED_DENSE_QMM=0
export DSV41_RUNTIME_FIXED_TILE_ATTENTION=1 DSV41_RUNTIME_RAGGED_TAIL_QK=1 DSV41_RUNTIME_RAGGED_TAIL_AV=1
export DSV41_RUNTIME_CHUNK_ATTENTION=0 DSV41_RUNTIME_PACKED_CHUNK_ATTENTION=0 DSV41_RUNTIME_WIDE_ATTENTION=0
export DSV41_RUNTIME_BATCHED_SPLITK_QK=1 DSV41_RUNTIME_BATCHED_SPLITK_QK_DIAGNOSTICS=0 DSV41_RUNTIME_FIXED_TILE_ATTENTION_DIAGNOSTICS=0
export DSV41_RUNTIME_DEFERRED_DECODER=1 DSV41_RUNTIME_DEFERRED_DECODER_CLEAR_CACHE=1
export DSV41_RUNTIME_MLX_CACHE_LIMIT_BYTES=0 DSV41_RUNTIME_LONG_CONTEXT_CACHE_LIMIT_BYTES=17179869184
export DSV41_RUNTIME_EXPERT_ASSIGNMENT_CHUNK=128 DSV41_RUNTIME_EXPERT_IO_THREADS=4
cmake --build build-mlx --target dsv41-text-backbone-test dsv41-context-ladder dsv41-cumulative-session -j 4 > "$root/build.log" 2>&1
binary_identity=$(shasum -a 256 build-mlx/dsv41-text-backbone-test build-mlx/dsv41-context-ladder build-mlx/dsv41-cumulative-session | shasum -a 256 | awk '{print $1}')
if [[ -f "$root/binary-identity.txt" ]]; then
 [[ $(<"$root/binary-identity.txt") == "$binary_identity" ]] || { echo 'binary identity changed; use a new root' >&2; exit 2; }
else printf '%s\n' "$binary_identity" > "$root/binary-identity.txt"; fi
checkpoint=${CHECKPOINT:-/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash}
input_identity=$( { printf '%s\n' "$checkpoint"; shasum -a 256 artifacts/checkpoint/summary.json artifacts/checkpoint/verification.json artifacts/engram/metadata.json "${PROMPT_PATTERN_FILE:-artifacts/logits-trace/prompt-tokens.txt}"; } | shasum -a 256 | awk '{print $1}')
if [[ -f "$root/input-identity.txt" ]]; then
 [[ $(<"$root/input-identity.txt") == "$input_identity" ]] || { echo 'input identity changed; use a new root' >&2; exit 2; }
else printf '%s\n' "$input_identity" > "$root/input-identity.txt"; fi
if [[ ! -f "$root/parity/completed" ]]; then
 if [[ -d "$root/parity" ]]; then mv "$root/parity" "$root/parity.failed-$(date +%Y%m%d-%H%M%S)-$$"; fi
 mkdir -p "$root/parity"
 DSV41_CHECK_DECODE_STACK_GRAPH=1 /usr/bin/time -l build-mlx/dsv41-text-backbone-test \
  "$checkpoint" artifacts/checkpoint/summary.json artifacts/engram/metadata.json \
  > "$root/parity/test.log" 2> "$root/parity/resource.log"
 grep -q 'PASS: decode stack graph' "$root/parity/test.log"
 touch "$root/parity/completed"
fi
mkdir -p "$root/$mode"
for variant in baseline candidate; do
 enabled=0; [[ "$variant" == candidate ]] && enabled=1
 dir="$root/$mode/$variant"
 if [[ -f "$dir/completed" && -s "$dir/result.json" && $(<"$dir/exit-code.txt") == 0 ]]; then echo "SKIP $dir"; continue; fi
 if [[ -d "$dir" ]]; then mv "$dir" "$dir.failed-$(date +%Y%m%d-%H%M%S)-$$"; fi
 export DSV41_RUNTIME_DECODE_STACK_GRAPH=$enabled
 if [[ "$mode" == bounded ]]; then
  DSV41_CONTEXT_RUN_DIR="$dir" CONTEXT_TOKENS=2071 TEACHER_TOKENS=0 TAIL_TEACHER_TOKENS=0 DECODE_TOKENS=8 \
   DSV41_CONTEXT_EXECUTION=sweep DSV41_CONTEXT_WALL_BUDGET_SECONDS=900 DSV41_CONTEXT_PROJECTED_WALL_LIMIT_SECONDS=900 \
   bash tools/benchmark/run_context_32k.sh
 else
  printf '8128 64\n%.0s' {1..8} > "$root/$mode/schedule.txt"
  DSV41_CUMULATIVE_RUN_DIR="$dir" TURN_SCHEDULE_FILE="$root/$mode/schedule.txt" CONTEXT_TOKENS=65536 \
   bash tools/benchmark/run_cumulative_session.sh
 fi
 [[ $(<"$dir/exit-code.txt") == 0 ]]
 touch "$dir/completed"
done
python3 tools/benchmark/compare_decode_stack_graph.py "$root/$mode" | tee "$root/$mode/comparison.json"
echo 'Review parity and all raw memory/system/config/result logs. No automatic promotion.'
