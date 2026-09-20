#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

pairs=${DSV41_PENDING_PAIRED_PAIRS:-5}
clear_cache=${DSV41_PENDING_PAIRED_CLEAR_CACHE:-0}
if [[ ! "$pairs" =~ ^[0-9]+$ ]] || ((pairs<5)); then
 echo "DSV41_PENDING_PAIRED_PAIRS must be at least 5" >&2; exit 2
fi
if [[ "$clear_cache" != 0 && "$clear_cache" != 1 ]]; then
 echo "DSV41_PENDING_PAIRED_CLEAR_CACHE must be 0 or 1" >&2; exit 2
fi
if ((clear_cache)); then
 default_root=artifacts/context-ladder/deferred-pending-cache-paired-$(date +%Y%m%d-%H%M%S)-$$
 resume_script=tools/benchmark/run_deferred_pending_cache_paired_qualification.sh
 candidate_description='same pending-CED path plus one idle MLX allocator cache clear after atomic publication'
else
 default_root=artifacts/context-ladder/deferred-pending-paired-$(date +%Y%m%d-%H%M%S)-$$
 resume_script=tools/benchmark/run_deferred_pending_paired_qualification.sh
 candidate_description='same path with one private 16K encoder-only plus 12,272-row exact decoder resume'
fi
task_root=${DSV41_PENDING_PAIRED_DIR:-$default_root}
if [[ "${DSV41_PENDING_PAIRED_DRY_RUN:-0}" == 1 ]]; then
 echo warmup-baseline; echo warmup-candidate
 for ((pair=1;pair<=pairs;pair++)); do
  if ((pair%2)); then echo "pair-$pair: baseline candidate"; else echo "pair-$pair: candidate baseline"; fi
 done
 exit 0
fi
mkdir -p "$task_root"
finish(){ status=$?; if [[ "${DSV41_PENDING_PAIRED_VERIFY_ONLY:-0}" == 1 ]]; then return; fi
 echo "$status" > "$task_root/exit-code.txt"; if ((status)); then
 echo "FAILED: inspect $task_root; resume with DSV41_PENDING_PAIRED_DIR=$task_root"; fi; }
trap finish EXIT

printf '%s\n' \
 "scope=one warmup per variant followed by $pairs alternating 32K baseline/pending-CED pairs; 4096 teacher continuation and 16 decode tokens" \
 "baseline=production fixed-tile/ragged/cache-bounded 4K full sweeps; candidate=$candidate_description" \
 'resources=12 sequential official-model processes by default; allow 110-180 minutes; peak Unified Memory budget 340 GB; checkpoint read-only; no swap expected' \
 'acceptance=candidate prefill win 5/5 beyond order variance; identical generated tokens/state/topology; no teacher/decode/memory/swap regression' \
 'logs=<root>/{warmup-*,pair-*/{baseline,candidate},measurements.tsv,identity.txt,tracked.patch,exit-code.txt}' \
 'failure=stop at first failed child and retain completed runs; private partial transaction is never published or resumed' \
 "resume=DSV41_PENDING_PAIRED_DIR=$task_root bash $resume_script; verified exit-0 children are skipped" \
 > "$task_root/config.txt"
echo "Logs: $task_root"

revision=$(git rev-parse HEAD)
if [[ -f "$task_root/revision.txt" && "$(<"$task_root/revision.txt")" != "$revision" ]]; then
 echo "resume revision mismatch" >&2; exit 2
fi
echo "$revision" > "$task_root/revision.txt"
patch_sha=$(git diff --binary | shasum -a 256 | awk '{print $1}')
if [[ -f "$task_root/tracked.patch" ]]; then
 existing_sha=$(shasum -a 256 "$task_root/tracked.patch" | awk '{print $1}')
 if [[ "$patch_sha" != "$existing_sha" ]]; then echo "resume tracked patch mismatch" >&2; exit 2; fi
else
 git diff --binary > "$task_root/tracked.patch"
fi
shasum -a 256 tools/benchmark/run_deferred_pending_paired_qualification.sh \
 tools/benchmark/run_deferred_pending_cache_paired_qualification.sh \
 tools/benchmark/run_context_32k.sh tools/benchmark/context_ladder.cpp \
 include/dsv41/execution_policy.hpp include/dsv41/deferred_decoder_plan.hpp \
 include/dsv41/text_backbone.hpp include/dsv41/text_decoder.hpp \
 src/model/text_backbone.cpp src/model/text_decoder.cpp build-mlx/dsv41-context-ladder \
 > "$task_root/identity.txt"

expected='[270,37370,2019,305,8470,270,15398,16,3476,477,260,20255,22896,16,47266,270]'
verify_run(){
 local dir=$1 deferred=$2 expected_chunks=$3 expected_clear=$4
 [[ -f "$dir/exit-code.txt" && "$(<"$dir/exit-code.txt")" == 0 &&
    -f "$dir/revision.txt" && "$(<"$dir/revision.txt")" == "$revision" &&
    -s "$dir/result.json" ]] || return 1
 cmp -s "$dir/tracked.patch" "$task_root/tracked.patch" || return 1
 jq -e --argjson expected "$expected" --arg deferred "$deferred" --argjson chunks "$expected_chunks" \
  '.status=="measurement_completed_requires_review" and
   .deferred_decoder==($deferred=="1") and .context_tokens==32768 and
   .phases.base_prefill.deferred_decoder_chunks==$chunks and
   .phases.teacher_head.deferred_decoder_chunks==0 and .phases.teacher_tail.deferred_decoder_chunks==0 and
   .phases.decode.generated_token_ids==$expected and .phases.decode.state_position==32767 and
   .packed_expert_bank_constructions==40 and .packed_expert_bank_loaded_experts==15360 and
   .route_execution_stats.diagnostic_readbacks==0 and .attention_telemetry.index_host_readbacks==0 and
   .attention_telemetry.chunk_scalar_qk_calls==0 and .attention_telemetry.chunk_scalar_av_calls==0' \
  "$dir/result.json" >/dev/null || return 1
 grep -Eq "^deferred_decoder_clear_cache=$expected_clear$" "$dir/config.txt" || return 1
 grep -Eq '^[[:space:]]+0[[:space:]]+swaps$' "$dir/resource.log"
}

run_one(){
 local label=$1 deferred=$2 chunks=$3
 local clear=$((deferred * clear_cache)) dir="$task_root/$1"
 if verify_run "$dir" "$deferred" "$chunks" "$clear"; then echo "SKIP verified $label"; return; fi
 mkdir -p "$(dirname "$dir")"
 echo "START $label"
 DSV41_CONTEXT_RUN_DIR="$dir" CONTEXT_TOKENS=32768 TEACHER_TOKENS=4096 \
  TAIL_TEACHER_TOKENS=2048 DECODE_TOKENS=16 DSV41_CONTEXT_EXECUTION=sweep \
  DSV41_RUNTIME_DEFERRED_DECODER="$deferred" DSV41_RUNTIME_DEFERRED_DECODER_CLEAR_CACHE="$clear" \
  DSV41_RUNTIME_COMPONENT_PROFILE=0 \
  DSV41_RUNTIME_MLX_CACHE_LIMIT_BYTES=0 DSV41_RUNTIME_LONG_CONTEXT_CACHE_LIMIT_BYTES=17179869184 \
  DSV41_CONTEXT_WALL_BUDGET_SECONDS=5400 DSV41_CONTEXT_PROJECTED_WALL_LIMIT_SECONDS=5400 \
  CACHE_CONDITION=warm-alternating RUN_CONDITIONS="pending-CED-paired-clear-$clear-$label" \
  bash tools/benchmark/run_context_32k.sh 2>&1 | tee "$task_root/$label.launch.log"
 verify_run "$dir" "$deferred" "$chunks" "$clear"
}

if [[ "${DSV41_PENDING_PAIRED_VERIFY_ONLY:-0}" == 1 ]]; then
 verified=0
 for label in warmup-baseline warmup-candidate; do
  if [[ "$label" == warmup-baseline ]]; then deferred=0; chunks=0; else deferred=1; chunks=1; fi
  clear=$((deferred * clear_cache))
  if verify_run "$task_root/$label" "$deferred" "$chunks" "$clear"; then echo "VERIFIED $label"; verified=$((verified+1)); fi
 done
 for ((pair=1;pair<=pairs;pair++)); do
  for variant in baseline candidate; do
   if [[ "$variant" == baseline ]]; then deferred=0; chunks=0; else deferred=1; chunks=1; fi
   clear=$((deferred * clear_cache))
   if verify_run "$task_root/pair-$pair/$variant" "$deferred" "$chunks" "$clear"; then
    echo "VERIFIED pair-$pair/$variant"; verified=$((verified+1))
   fi
  done
 done
 echo "VERIFY ONLY: $verified completed children; no model process was started."
 exit 0
fi

run_one warmup-baseline 0 0
run_one warmup-candidate 1 1
for ((pair=1;pair<=pairs;pair++)); do
 if ((pair%2)); then order=(baseline candidate); else order=(candidate baseline); fi
 for variant in "${order[@]}"; do
  if [[ "$variant" == baseline ]]; then deferred=0; chunks=0; else deferred=1; chunks=1; fi
  run_one "pair-$pair/$variant" "$deferred" "$chunks"
 done
done

summary="$task_root/measurements.tsv"
printf 'pair\tvariant\tprefill_seconds\tprefill_tps\tbase_seconds\tteacher_seconds\tdecode_mean\tdecode_p95\tpeak_bytes\tpeak_footprint_bytes\n' > "$summary"
for ((pair=1;pair<=pairs;pair++)); do
 for variant in baseline candidate; do
  dir="$task_root/pair-$pair/$variant"
  footprint=$(awk '/peak memory footprint/{print $1}' "$dir/resource.log")
  jq -r --arg pair "$pair" --arg variant "$variant" --argjson footprint "$footprint" \
   '[$pair,$variant,.aggregates.prefill_seconds,.aggregates.prefill_tokens_per_second,
     .phases.base_prefill.seconds,.aggregates.teacher_continuation_seconds,
     .phases.decode.mean_seconds,.phases.decode.p95_seconds,.final_memory.peak_bytes,$footprint]|@tsv' \
   "$dir/result.json" >> "$summary"
 done
done
echo "Completed; review $summary and every child result/resource/config/identity log before default or API promotion."
