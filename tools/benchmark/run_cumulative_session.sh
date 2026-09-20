#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

run_dir=${DSV41_CUMULATIVE_RUN_DIR:-artifacts/context-ladder/cumulative-session-$(date +%Y%m%d-%H%M%S)-$$}
schedule=${TURN_SCHEDULE_FILE:?TURN_SCHEDULE_FILE is required}
context=${CONTEXT_TOKENS:?CONTEXT_TOKENS is required}
pattern=${PROMPT_PATTERN_FILE:-artifacts/logits-trace/prompt-tokens.txt}
checkpoint=${CHECKPOINT:-/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash}
mkdir -p "$run_dir"
finish(){
 status=$?
 [[ -f "$run_dir/system-after.txt" ]] || /usr/bin/vm_stat > "$run_dir/system-after.txt" 2>&1 || true
 echo "$status" > "$run_dir/exit-code.txt"
 if ((status)); then echo "FAILED: inspect $run_dir; request state is not resumable; retry this stage fresh."; fi
}
trap finish EXIT
echo "Logs: $run_dir"

if [[ ! "$context" =~ ^[0-9]+$ ]] || ((context<1 || context>262144)); then
 echo 'CONTEXT_TOKENS must be in 1..262144' >&2;exit 2
fi
[[ -f "$schedule" ]] || { echo "missing schedule: $schedule" >&2;exit 2; }
[[ -f "$pattern" ]] || { echo "missing prompt pattern: $pattern" >&2;exit 2; }
read -r prompt_tokens scheduled_context turn_count < <(awk '
 NF!=2 || $1!~/^[0-9]+$/ || $2!~/^[0-9]+$/ || $1<1 || $2<1 || $2>128 {exit 2}
 {prompt+=$1; total+=$1+$2; turns++}
 END {if(!turns)exit 2; print prompt,total,turns}' "$schedule") || {
 echo 'invalid turn schedule; expected: APPEND_TOKENS DECODE_TOKENS per line' >&2;exit 2
}
if ((scheduled_context!=context)); then
 echo "schedule context $scheduled_context does not equal CONTEXT_TOKENS $context" >&2;exit 2
fi

export DSV41_RUNTIME_LAYER_FINITE_CHECKS=${DSV41_RUNTIME_LAYER_FINITE_CHECKS:-0}
export DSV41_RUNTIME_PACKED_EXPERT_BANK=${DSV41_RUNTIME_PACKED_EXPERT_BANK:-1}
export DSV41_RUNTIME_GROUP_SELECTED_EXPERTS=${DSV41_RUNTIME_GROUP_SELECTED_EXPERTS:-0}
export DSV41_RUNTIME_RESIDENT_EXPERT_ATLAS=${DSV41_RUNTIME_RESIDENT_EXPERT_ATLAS:-1}
export DSV41_RUNTIME_COMPACT_EXPERT_BANK=${DSV41_RUNTIME_COMPACT_EXPERT_BANK:-0}
export DSV41_RUNTIME_GROUPED_EXPERT_PIPELINE=${DSV41_RUNTIME_GROUPED_EXPERT_PIPELINE:-0}
export DSV41_RUNTIME_ROUTE_DIAGNOSTICS=${DSV41_RUNTIME_ROUTE_DIAGNOSTICS:-0}
export DSV41_RUNTIME_INDEX_DIAGNOSTICS=${DSV41_RUNTIME_INDEX_DIAGNOSTICS:-0}
export DSV41_RUNTIME_CHUNK_ATTENTION=${DSV41_RUNTIME_CHUNK_ATTENTION:-0}
export DSV41_RUNTIME_BATCHED_SPLITK_QK=${DSV41_RUNTIME_BATCHED_SPLITK_QK:-1}
export DSV41_RUNTIME_PACKED_CHUNK_ATTENTION=${DSV41_RUNTIME_PACKED_CHUNK_ATTENTION:-0}
export DSV41_RUNTIME_WIDE_ATTENTION=${DSV41_RUNTIME_WIDE_ATTENTION:-0}
export DSV41_RUNTIME_FIXED_TILE_ATTENTION=${DSV41_RUNTIME_FIXED_TILE_ATTENTION:-1}
export DSV41_RUNTIME_RAGGED_TAIL_QK=${DSV41_RUNTIME_RAGGED_TAIL_QK:-1}
export DSV41_RUNTIME_RAGGED_TAIL_AV=${DSV41_RUNTIME_RAGGED_TAIL_AV:-1}
export DSV41_RUNTIME_LAYER_SWEEP=${DSV41_RUNTIME_LAYER_SWEEP:-1}
export DSV41_RUNTIME_DEFERRED_DECODER=${DSV41_RUNTIME_DEFERRED_DECODER:-1}
export DSV41_RUNTIME_DEFERRED_DECODER_CLEAR_CACHE=${DSV41_RUNTIME_DEFERRED_DECODER_CLEAR_CACHE:-1}
export DSV41_RUNTIME_BATCHED_DENSE_QMM=${DSV41_RUNTIME_BATCHED_DENSE_QMM:-0}
export DSV41_RUNTIME_MLX_CACHE_LIMIT_BYTES=${DSV41_RUNTIME_MLX_CACHE_LIMIT_BYTES:-0}
export DSV41_RUNTIME_LONG_CONTEXT_CACHE_LIMIT_BYTES=${DSV41_RUNTIME_LONG_CONTEXT_CACHE_LIMIT_BYTES:-17179869184}
export DSV41_RUNTIME_EXPERT_IO_THREADS=${DSV41_RUNTIME_EXPERT_IO_THREADS:-4}
export DSV41_CUMULATIVE_WALL_BUDGET_SECONDS=${DSV41_CUMULATIVE_WALL_BUDGET_SECONDS:-14400}

python=${PYTHON:-python3}
"$python" tools/reference/expand_token_pattern.py --input "$pattern" \
 --output "$run_dir/prompt.txt" --length "$prompt_tokens" > "$run_dir/prompt-build.log"
cp "$schedule" "$run_dir/turn-schedule.txt"
cmake --build build-mlx --target dsv41-cumulative-session -j 4 > "$run_dir/build.log" 2>&1
git rev-parse HEAD > "$run_dir/revision.txt"
git diff --binary > "$run_dir/tracked.patch"
printf '%s\n' "checkpoint=$checkpoint" "context=$context" "turn_count=$turn_count" \
 "prompt_tokens=$prompt_tokens" "schedule=$schedule" \
 "cache_condition=${CACHE_CONDITION:-unknown}" "run_conditions=${RUN_CONDITIONS:-unknown}" \
 "deferred_decoder=$DSV41_RUNTIME_DEFERRED_DECODER" \
 "deferred_decoder_clear_cache=$DSV41_RUNTIME_DEFERRED_DECODER_CLEAR_CACHE" \
 "grouped_expert_pipeline=$DSV41_RUNTIME_GROUPED_EXPERT_PIPELINE" \
 "long_context_cache_limit_bytes=$DSV41_RUNTIME_LONG_CONTEXT_CACHE_LIMIT_BYTES" \
 "wall_budget_seconds=$DSV41_CUMULATIVE_WALL_BUDGET_SECONDS" > "$run_dir/config.txt"
shasum -a 256 build-mlx/dsv41-cumulative-session tools/benchmark/cumulative_session.cpp \
 tools/benchmark/run_cumulative_session.sh tools/benchmark/run_cumulative_long_context_qualification.sh \
 tools/benchmark/summarize_cumulative_sessions.py \
 include/dsv41/text_backbone.hpp include/dsv41/execution_policy.hpp include/dsv41/deferred_decoder_plan.hpp \
 include/dsv41/moe.hpp include/dsv41/moe_pipeline.hpp src/moe/reference.cpp \
 src/moe/expert_bank.cpp src/moe/grouped_expert_pipeline.cpp \
 src/model/text_backbone.cpp src/model/text_encoder.cpp src/model/text_decoder.cpp src/model/text_generate.cpp \
 "$run_dir/prompt.txt" "$run_dir/turn-schedule.txt" "$pattern" \
 artifacts/checkpoint/summary.json artifacts/checkpoint/verification.json \
 artifacts/engram/metadata.json artifacts/engram/fixture-provenance.json > "$run_dir/identity.txt"
cmd=(build-mlx/dsv41-cumulative-session "$checkpoint" artifacts/checkpoint/summary.json \
 artifacts/engram/metadata.json "$run_dir/prompt.txt" "$run_dir/turn-schedule.txt" \
 "$run_dir/result.json" "$context")
printf '%q ' "${cmd[@]}" > "$run_dir/command.txt";printf '\n' >> "$run_dir/command.txt"
{
 /usr/bin/uname -a
 /usr/bin/sw_vers
 xcrun clang --version
 otool -L build-mlx/dsv41-cumulative-session
} > "$run_dir/environment.txt"
if [[ "${DSV41_DRY_RUN:-0}" == 1 ]]; then
 echo 'DRY RUN: inputs, build, identities, and command prepared; model was not started.' | tee "$run_dir/test.log"
 exit 0
fi
/usr/sbin/sysctl hw.memsize > "$run_dir/system-before.txt"
/usr/bin/vm_stat >> "$run_dir/system-before.txt"
(/usr/bin/time -l "${cmd[@]}") > >(tee "$run_dir/test.log") 2> >(tee "$run_dir/resource.log" >&2)
/usr/bin/vm_stat > "$run_dir/system-after.txt"
echo 'Completed; result/resource/config/identity logs require review. This single run is not qualification.'
