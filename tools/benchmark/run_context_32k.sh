#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
run_dir=${DSV41_CONTEXT_RUN_DIR:-artifacts/context-ladder/32k-run-$(date +%Y%m%d-%H%M%S)-$$}
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
export DSV41_RUNTIME_PACKED_EXPERT_BANK=${DSV41_RUNTIME_PACKED_EXPERT_BANK:-1}
export DSV41_RUNTIME_GROUP_SELECTED_EXPERTS=${DSV41_RUNTIME_GROUP_SELECTED_EXPERTS:-0}
export DSV41_RUNTIME_RESIDENT_EXPERT_ATLAS=${DSV41_RUNTIME_RESIDENT_EXPERT_ATLAS:-1}
export DSV41_RUNTIME_COMPACT_EXPERT_BANK=${DSV41_RUNTIME_COMPACT_EXPERT_BANK:-0}
export DSV41_RUNTIME_GROUPED_EXPERT_PIPELINE=${DSV41_RUNTIME_GROUPED_EXPERT_PIPELINE:-0}
export DSV41_RUNTIME_FUSED_MHC=${DSV41_RUNTIME_FUSED_MHC:-0}
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
export DSV41_RUNTIME_EXPERT_ASSIGNMENT_CHUNK=${DSV41_RUNTIME_EXPERT_ASSIGNMENT_CHUNK:-0}
export DSV41_RUNTIME_COMPONENT_PROFILE=${DSV41_RUNTIME_COMPONENT_PROFILE:-0}
export DSV41_RUNTIME_WIRED_LIMIT_BYTES=${DSV41_RUNTIME_WIRED_LIMIT_BYTES:-0}
export DSV41_RUNTIME_EXPERT_BACKING_DIR=${DSV41_RUNTIME_EXPERT_BACKING_DIR:-}
export DSV41_CONTEXT_EXECUTION=${DSV41_CONTEXT_EXECUTION:-sweep}
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
 "grouped_expert_pipeline=$DSV41_RUNTIME_GROUPED_EXPERT_PIPELINE" \
 "fused_mhc=$DSV41_RUNTIME_FUSED_MHC" \
 "route_diagnostics=$DSV41_RUNTIME_ROUTE_DIAGNOSTICS" \
 "index_diagnostics=$DSV41_RUNTIME_INDEX_DIAGNOSTICS" \
 "chunk_attention=$DSV41_RUNTIME_CHUNK_ATTENTION" \
 "batched_splitk_qk=$DSV41_RUNTIME_BATCHED_SPLITK_QK" \
 "packed_chunk_attention=$DSV41_RUNTIME_PACKED_CHUNK_ATTENTION" \
 "wide_attention=$DSV41_RUNTIME_WIDE_ATTENTION" \
 "fixed_tile_attention=$DSV41_RUNTIME_FIXED_TILE_ATTENTION" \
 "ragged_tail_qk=$DSV41_RUNTIME_RAGGED_TAIL_QK" \
 "ragged_tail_av=$DSV41_RUNTIME_RAGGED_TAIL_AV" \
 "layer_sweep=$DSV41_RUNTIME_LAYER_SWEEP" \
 "deferred_decoder=$DSV41_RUNTIME_DEFERRED_DECODER" \
 "deferred_decoder_clear_cache=$DSV41_RUNTIME_DEFERRED_DECODER_CLEAR_CACHE" \
 "batched_dense_qmm=$DSV41_RUNTIME_BATCHED_DENSE_QMM" \
 "mlx_cache_limit_bytes=$DSV41_RUNTIME_MLX_CACHE_LIMIT_BYTES" \
 "long_context_cache_limit_bytes=$DSV41_RUNTIME_LONG_CONTEXT_CACHE_LIMIT_BYTES" \
 "expert_io_threads=$DSV41_RUNTIME_EXPERT_IO_THREADS" \
 "expert_assignment_chunk=$DSV41_RUNTIME_EXPERT_ASSIGNMENT_CHUNK" \
 "component_profile=$DSV41_RUNTIME_COMPONENT_PROFILE" \
 "wired_limit_bytes=$DSV41_RUNTIME_WIRED_LIMIT_BYTES" \
 "expert_backing_dir=${DSV41_RUNTIME_EXPERT_BACKING_DIR:-disabled}" \
 "metal_trace=${DSV41_XCTRACE_OUTPUT:-disabled}" \
 "execution=$DSV41_CONTEXT_EXECUTION" \
 "cache_condition=${CACHE_CONDITION:-unknown}" "run_conditions=${RUN_CONDITIONS:-unknown}" \
 "pattern=$pattern" > "$run_dir/config.txt"
shasum -a 256 build-mlx/dsv41-context-ladder tools/benchmark/context_ladder.cpp \
 tools/benchmark/run_context_32k.sh tools/benchmark/run_resident_atlas_prefill_check.sh \
 tools/benchmark/run_resident_atlas_prefill_measurement.sh \
 tools/benchmark/run_layer_sweep_prefill_measurement.sh \
 tools/benchmark/run_batched_splitk_qk_prefill_measurement.sh \
 tools/benchmark/run_batched_splitk_qk_component_profile.sh \
 tools/benchmark/run_packed_attention_prefill_measurement.sh \
 tools/benchmark/run_fixed_tile_attention_prefill_measurement.sh \
 tools/benchmark/run_fixed_tile_attention_paired_qualification.sh \
 tools/benchmark/run_fixed_tile_attention_backbone_check.sh \
 tools/benchmark/run_fixed_tile_attention_isolation_check.sh \
 tools/benchmark/run_fixed_tile_attention_layer_localization.sh \
 tools/benchmark/run_long_context_regime_measurements.sh \
 tools/benchmark/run_long_context_cache_candidate.sh \
 tools/benchmark/run_deferred_transaction_backbone_check.sh \
 tools/benchmark/run_deferred_suffix_backbone_check.sh \
 tools/benchmark/run_deferred_decoder_context_candidate.sh \
 tools/benchmark/run_deferred_decoder_paired_qualification.sh \
 tools/benchmark/run_decode_component_profile.sh \
 tools/benchmark/run_grouped_expert_pipeline_candidate.sh \
 tools/benchmark/run_grouped_expert_pipeline_paired_qualification.sh \
 tools/benchmark/summarize_long_context_regimes.py \
 tools/benchmark/run_layer_sweep_component_profile.sh \
 tools/benchmark/run_layer_component_profile.sh \
 tools/benchmark/run_resident_layer_component_profile.sh \
 tools/benchmark/run_shape_bucket_attention_profile.sh \
 include/dsv41/text_backbone.hpp include/dsv41/runtime_residency.hpp include/dsv41/expert_backing.hpp include/dsv41/generation_loop.hpp include/dsv41/execution_policy.hpp \
 include/dsv41/deferred_decoder_plan.hpp \
 include/dsv41/runtime_profile.hpp \
 include/dsv41/swa_layer.hpp include/dsv41/swa_attention.hpp include/dsv41/compressed_layer.hpp include/dsv41/reused_layer.hpp \
 include/dsv41/compressor.hpp include/dsv41/global_kv.hpp include/dsv41/index_key.hpp include/dsv41/index_query.hpp \
 include/dsv41/shared_attention.hpp include/dsv41/attention_telemetry.hpp \
 src/model/text_backbone.cpp src/model/text_encoder.cpp src/model/text_decoder.cpp \
 src/model/block.cpp src/model/compressed_block.cpp src/model/reused_block.cpp \
 src/attention/swa_layer.cpp src/attention/swa_attention.cpp src/attention/compressed_layer.cpp src/attention/compressor.cpp \
 src/attention/batched_splitk_qk.hpp.in metal/attention/batched_splitk_qk.metal \
 metal/attention/batched_splitk_accum.metal metal/attention/steel_gemm_header.metal \
 src/attention/packed_chunk_attention.hpp.in metal/attention/packed_chunk_attention.metal \
 src/attention/wide_chunk_attention.hpp.in metal/attention/wide_chunk_attention.metal \
 src/attention/packed_attention_worklist.hpp.in metal/attention/packed_attention_worklist.metal \
 src/attention/ragged_tail_qk.hpp.in metal/attention/ragged_tail_qk.metal \
 metal/attention/ragged_tail_accum.metal src/attention/ragged_width_one_qk.hpp.in \
 metal/attention/ragged_width_one_qk.metal metal/attention/gemv_header.metal \
 src/attention/ragged_tail_av.hpp.in metal/attention/ragged_tail_av.metal \
 metal/attention/steel_attention_header.metal \
 src/attention/index_key.cpp src/attention/index_query.cpp src/attention/shared_attention.cpp src/cache/global_kv.cpp \
 include/dsv41/moe.hpp include/dsv41/moe_pipeline.hpp src/moe/reference.cpp \
 src/moe/expert_bank.cpp src/moe/expert_backing.cpp src/moe/grouped_expert_pipeline.cpp src/runtime/residency.cpp \
 src/mhc/reference.cpp src/mhc/split_sinkhorn.hpp.in metal/mhc/split_sinkhorn.metal \
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
if [[ -n "${DSV41_XCTRACE_OUTPUT:-}" ]]; then
 mkdir -p "$(dirname "$DSV41_XCTRACE_OUTPUT")"
 trace_target_env=()
 if [[ -n "${DSV41_XCTRACE_TARGET_DYLD:-}" ]]; then
  trace_target_env+=(--env "DYLD_INSERT_LIBRARIES=$DSV41_XCTRACE_TARGET_DYLD")
 fi
 if [[ -n "${DSV41_METAL_DISPATCH_COUNTER_OUTPUT:-}" ]]; then
  trace_target_env+=(--env "DSV41_METAL_DISPATCH_COUNTER_OUTPUT=$DSV41_METAL_DISPATCH_COUNTER_OUTPUT")
 fi
 if [[ -n "${DSV41_METAL_DISPATCH_COUNTER_SCOPED:-}" ]]; then
  trace_target_env+=(--env "DSV41_METAL_DISPATCH_COUNTER_SCOPED=$DSV41_METAL_DISPATCH_COUNTER_SCOPED")
 fi
 # The GPU instrument is system-wide and produced a 12 GB package which kept
 # finalizing after the target exited.  Metal Application alone keeps the
 # capture target-scoped and is sufficient for command-buffer/encoder records.
 if ((${#trace_target_env[@]})); then
  (/usr/bin/time -l xcrun xctrace record \
    --instrument 'Metal Application' \
    --output "$DSV41_XCTRACE_OUTPUT" "${trace_target_env[@]}" \
    --target-stdout - --launch -- "${cmd[@]}") \
    > >(tee "$run_dir/test.log") 2> >(tee "$run_dir/resource.log" >&2)
 else
  (/usr/bin/time -l xcrun xctrace record \
    --instrument 'Metal Application' \
    --output "$DSV41_XCTRACE_OUTPUT" \
    --target-stdout - --launch -- "${cmd[@]}") \
    > >(tee "$run_dir/test.log") 2> >(tee "$run_dir/resource.log" >&2)
 fi
else
 direct_target_env=()
 if [[ -n "${DSV41_DIRECT_TARGET_DYLD:-}" ]]; then
  direct_target_env+=("DYLD_INSERT_LIBRARIES=$DSV41_DIRECT_TARGET_DYLD")
 fi
 if [[ -n "${DSV41_METAL_DISPATCH_COUNTER_OUTPUT:-}" ]]; then
  direct_target_env+=("DSV41_METAL_DISPATCH_COUNTER_OUTPUT=$DSV41_METAL_DISPATCH_COUNTER_OUTPUT")
 fi
 if [[ -n "${DSV41_METAL_DISPATCH_COUNTER_SCOPED:-}" ]]; then
  direct_target_env+=("DSV41_METAL_DISPATCH_COUNTER_SCOPED=$DSV41_METAL_DISPATCH_COUNTER_SCOPED")
 fi
 if ((${#direct_target_env[@]})); then
  (/usr/bin/time -l env "${direct_target_env[@]}" "${cmd[@]}") \
    > >(tee "$run_dir/test.log") 2> >(tee "$run_dir/resource.log" >&2)
 else
  (/usr/bin/time -l "${cmd[@]}") \
    > >(tee "$run_dir/test.log") 2> >(tee "$run_dir/resource.log" >&2)
 fi
fi
/usr/bin/vm_stat > "$run_dir/system-after.txt"
echo "Completed; repo canonical result/resource logs require review. This single run does not qualify 32K, performance, API, or 256K."
