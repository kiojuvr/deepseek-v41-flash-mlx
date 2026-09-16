#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
run_dir="artifacts/context-ladder/32k-run-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$run_dir"
finish(){
 status=$?
 if [[ ! -f "$run_dir/system-after.txt" ]]; then /usr/bin/vm_stat > "$run_dir/system-after.txt" 2>&1 || true; fi
 echo "$status" > "$run_dir/exit-code.txt"
 if ((status)); then echo "FAILED: inspect $run_dir; model state cannot resume; rerun this command for a fresh run."; fi
}
trap finish EXIT
echo "Logs: $run_dir"

checkpoint=${CHECKPOINT:-/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash}
pattern=${PROMPT_PATTERN_FILE:-artifacts/logits-trace/prompt-tokens.txt}
context=${CONTEXT_TOKENS:-32768}
teacher=${TEACHER_TOKENS:-4096}
tail=${TAIL_TEACHER_TOKENS:-2048}
decode=${DECODE_TOKENS:-16}
export DSV41_CONTEXT_WALL_BUDGET_SECONDS=${DSV41_CONTEXT_WALL_BUDGET_SECONDS:-1800}
export DSV41_CONTEXT_PROJECTED_WALL_LIMIT_SECONDS=${DSV41_CONTEXT_PROJECTED_WALL_LIMIT_SECONDS:-3600}
export DSV41_RUNTIME_LAYER_FINITE_CHECKS=${DSV41_RUNTIME_LAYER_FINITE_CHECKS:-0}
export DSV41_RUNTIME_PACKED_EXPERT_BANK=${DSV41_RUNTIME_PACKED_EXPERT_BANK:-0}
export DSV41_RUNTIME_GROUP_SELECTED_EXPERTS=${DSV41_RUNTIME_GROUP_SELECTED_EXPERTS:-0}
export DSV41_RUNTIME_RESIDENT_EXPERT_ATLAS=${DSV41_RUNTIME_RESIDENT_EXPERT_ATLAS:-0}
export DSV41_RUNTIME_COMPACT_EXPERT_BANK=${DSV41_RUNTIME_COMPACT_EXPERT_BANK:-0}
export DSV41_RUNTIME_ROUTE_DIAGNOSTICS=${DSV41_RUNTIME_ROUTE_DIAGNOSTICS:-0}
export DSV41_RUNTIME_INDEX_DIAGNOSTICS=${DSV41_RUNTIME_INDEX_DIAGNOSTICS:-1}
export DSV41_RUNTIME_MLX_CACHE_LIMIT_BYTES=${DSV41_RUNTIME_MLX_CACHE_LIMIT_BYTES:-0}
export DSV41_RUNTIME_EXPERT_IO_THREADS=${DSV41_RUNTIME_EXPERT_IO_THREADS:-1}
export DSV41_RUNTIME_EXPERT_ASSIGNMENT_CHUNK=${DSV41_RUNTIME_EXPERT_ASSIGNMENT_CHUNK:-0}
export DSV41_RUNTIME_COMPONENT_PROFILE=${DSV41_RUNTIME_COMPONENT_PROFILE:-0}
if [[ ! "$decode" =~ ^[0-9]+$ ]] || ((decode<1 || decode>128)); then
 echo "DECODE_TOKENS must be an integer in 1..128" >&2
 exit 2
fi
if [[ ! "$context" =~ ^[0-9]+$ || ! "$teacher" =~ ^[0-9]+$ || ! "$tail" =~ ^[0-9]+$ ]] ||
   ((context<decode+1 || context>262144 || teacher<tail || teacher>=context-decode)); then
 echo "invalid CONTEXT_TOKENS/TEACHER_TOKENS/TAIL_TEACHER_TOKENS geometry" >&2
 exit 2
fi
prefill=$((context-decode))
if [[ ! -f "$pattern" ]]; then echo "prompt token pattern does not exist: $pattern" >&2; exit 2; fi

"${PYTHON:-python3}" tools/reference/expand_token_pattern.py \
 --input "$pattern" --output "$run_dir/prompt.txt" --length "$prefill" > "$run_dir/prompt-build.log"
cmake --build build-mlx --target dsv41-context-ladder -j 4 > "$run_dir/build.log" 2>&1
git rev-parse HEAD > "$run_dir/revision.txt"
git diff --binary > "$run_dir/tracked.patch"
printf '%s\n' "checkpoint=$checkpoint" "context=$context" "base_prefill=$((prefill-teacher))" \
 "teacher_continuation=$teacher" "teacher_tail=$tail" "decode=$decode" \
 "wall_budget_seconds=$DSV41_CONTEXT_WALL_BUDGET_SECONDS" \
 "projected_wall_limit_seconds=$DSV41_CONTEXT_PROJECTED_WALL_LIMIT_SECONDS" \
 "layer_finite_checks=$DSV41_RUNTIME_LAYER_FINITE_CHECKS" \
 "packed_expert_bank=$DSV41_RUNTIME_PACKED_EXPERT_BANK" \
 "group_selected_experts=$DSV41_RUNTIME_GROUP_SELECTED_EXPERTS" \
 "resident_expert_atlas=$DSV41_RUNTIME_RESIDENT_EXPERT_ATLAS" \
 "compact_expert_bank=$DSV41_RUNTIME_COMPACT_EXPERT_BANK" \
 "route_diagnostics=$DSV41_RUNTIME_ROUTE_DIAGNOSTICS" \
 "index_diagnostics=$DSV41_RUNTIME_INDEX_DIAGNOSTICS" \
 "mlx_cache_limit_bytes=$DSV41_RUNTIME_MLX_CACHE_LIMIT_BYTES" \
 "expert_io_threads=$DSV41_RUNTIME_EXPERT_IO_THREADS" \
 "expert_assignment_chunk=$DSV41_RUNTIME_EXPERT_ASSIGNMENT_CHUNK" \
 "component_profile=$DSV41_RUNTIME_COMPONENT_PROFILE" \
 "execution=${DSV41_CONTEXT_EXECUTION:-individual}" \
 "cache_condition=${CACHE_CONDITION:-unknown}" "run_conditions=${RUN_CONDITIONS:-unknown}" \
 "pattern=$pattern" > "$run_dir/config.txt"
shasum -a 256 build-mlx/dsv41-context-ladder tools/benchmark/context_ladder.cpp \
 tools/benchmark/run_context_32k.sh tools/benchmark/run_resident_atlas_prefill_check.sh \
 tools/benchmark/run_resident_atlas_prefill_measurement.sh \
 tools/benchmark/run_layer_component_profile.sh \
 tools/benchmark/run_resident_layer_component_profile.sh \
 include/dsv41/text_backbone.hpp include/dsv41/generation_loop.hpp include/dsv41/execution_policy.hpp \
 include/dsv41/runtime_profile.hpp \
 include/dsv41/swa_layer.hpp include/dsv41/compressed_layer.hpp include/dsv41/reused_layer.hpp \
 include/dsv41/compressor.hpp include/dsv41/global_kv.hpp include/dsv41/index_key.hpp include/dsv41/index_query.hpp \
 include/dsv41/shared_attention.hpp include/dsv41/attention_telemetry.hpp \
 src/model/text_backbone.cpp src/model/text_encoder.cpp src/model/text_decoder.cpp \
 src/model/block.cpp src/model/compressed_block.cpp src/model/reused_block.cpp \
 src/attention/swa_layer.cpp src/attention/compressed_layer.cpp src/attention/compressor.cpp \
 src/attention/index_key.cpp src/attention/index_query.cpp src/attention/shared_attention.cpp src/cache/global_kv.cpp \
 include/dsv41/moe.hpp src/moe/reference.cpp src/moe/expert_bank.cpp \
 metal/moe/route_select.metal metal/moe/route_reduce.metal \
 include/dsv41/sampling.hpp src/model/sampling.cpp \
 "$run_dir/prompt.txt" "$pattern" tools/reference/expand_token_pattern.py \
 artifacts/checkpoint/summary.json artifacts/checkpoint/verification.json \
 artifacts/checkpoint/hardware.json artifacts/engram/metadata.json \
 artifacts/engram/fixture-provenance.json > "$run_dir/identity.txt"
if [[ -f "$pattern.manifest.json" ]]; then
 shasum -a 256 "$pattern.manifest.json" >> "$run_dir/identity.txt"
fi
cmd=(build-mlx/dsv41-context-ladder "$checkpoint" artifacts/checkpoint/summary.json \
 artifacts/engram/metadata.json "$run_dir/prompt.txt" "$run_dir/result.json" \
 "$context" "$teacher" "$tail" "$decode")
printf '%q ' "${cmd[@]}" > "$run_dir/command.txt"; printf '\n' >> "$run_dir/command.txt"
{
 /usr/bin/uname -a
 /usr/bin/sw_vers
 xcrun clang --version
 otool -L build-mlx/dsv41-context-ladder
} > "$run_dir/environment.txt"
if [[ "${DSV41_DRY_RUN:-0}" == 1 ]]; then
 echo "DRY RUN: prompt, build, identities and command prepared; model was not started." | tee "$run_dir/test.log"
 exit 0
fi
/usr/sbin/sysctl hw.memsize > "$run_dir/system-before.txt"
/usr/bin/vm_stat >> "$run_dir/system-before.txt"
(/usr/bin/time -l "${cmd[@]}") > >(tee "$run_dir/test.log") 2> >(tee "$run_dir/resource.log" >&2)
/usr/bin/vm_stat > "$run_dir/system-after.txt"
echo "Completed; repo canonical result/resource logs require review. This single run does not qualify 32K, performance, API, or 256K."
