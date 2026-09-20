#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

root=${DSV41_CED_PAIRED_DIR:-artifacts/context-ladder/deferred-decoder-paired-$(date +%Y%m%d-%H%M%S)-$$}
pairs=${DSV41_CED_PAIRED_PAIRS:-5}
if [[ ! "$pairs" =~ ^[0-9]+$ ]] || ((pairs<5)); then
 echo "DSV41_CED_PAIRED_PAIRS must be an integer of at least 5" >&2
 exit 2
fi
if [[ "${DSV41_CED_PAIRED_DRY_RUN:-0}" == 1 ]]; then
 echo "warmup-baseline"
 echo "warmup-candidate"
 for ((pair=1;pair<=pairs;pair++)); do
  if ((pair%2)); then echo "pair-$pair: baseline candidate"; else echo "pair-$pair: candidate baseline"; fi
 done
 echo "DRY RUN: alternating schedule prepared; no model process was started."
 exit 0
fi
mkdir -p "$root"
finish(){
 status=$?
 echo "$status" > "$root/exit-code.txt"
 if ((status)); then
  echo "FAILED: inspect $root; resume completed stages with DSV41_CED_PAIRED_DIR=$root"
 fi
}
trap finish EXIT

printf '%s\n' \
 "scope=one warmup per variant followed by $pairs alternating 16K baseline/CED pairs; each run has 4096 teacher continuation and 16 decode tokens" \
 'baseline=production fixed-tile/ragged/cache-bounded sweep; candidate=same path with one private exact decoder-suffix transaction in base prefill' \
 'resources=allow 60-100 minutes for the default 12 sequential processes; peak Unified Memory budget 340 GB; checkpoint read-only; no swap expected' \
 'acceptance=candidate prefill win 5/5, improvement exceeds run-order variance, identical generated tokens/state/topology, no teacher/decode/memory/swap regression' \
 'logs=<root>/{warmup-*,pair-*/{baseline,candidate},measurements.tsv,identity.txt,tracked.patch,exit-code.txt}; children retain canonical logs' \
 'failure=stop at first failed child and retain completed runs; partial request state is never published' \
 "resume=DSV41_CED_PAIRED_DIR=$root bash tools/benchmark/run_deferred_decoder_paired_qualification.sh; verified exit-0 children are skipped" > "$root/config.txt"
revision=$(git rev-parse HEAD)
if [[ -f "$root/revision.txt" && "$(<"$root/revision.txt")" != "$revision" ]]; then
 echo "resume revision mismatch: root=$(<"$root/revision.txt") current=$revision" >&2
 exit 2
fi
printf '%s\n' "$revision" > "$root/revision.txt"
current_patch_sha=$(git diff --binary | shasum -a 256 | awk '{print $1}')
if [[ -f "$root/tracked.patch" ]]; then
 root_patch_sha=$(shasum -a 256 "$root/tracked.patch" | awk '{print $1}')
 if [[ "$current_patch_sha" != "$root_patch_sha" ]]; then
  echo "resume tracked patch mismatch" >&2
  exit 2
 fi
else
 git diff --binary > "$root/tracked.patch"
fi
shasum -a 256 tools/benchmark/run_deferred_decoder_paired_qualification.sh \
 tools/benchmark/run_context_32k.sh tools/benchmark/context_ladder.cpp \
 include/dsv41/execution_policy.hpp include/dsv41/deferred_decoder_plan.hpp \
 include/dsv41/text_backbone.hpp include/dsv41/text_decoder.hpp \
 src/model/text_backbone.cpp src/model/text_decoder.cpp build-mlx/dsv41-context-ladder \
 > "$root/identity.txt"
echo "Logs: $root"

expected='[1354,3452,271,19462,270,37370,2019,305,8470,270,15398,16,3476,477,260,20255]'
run_one(){
 local label=$1 deferred=$2 run_dir="$root/$1"
 if [[ -f "$run_dir/exit-code.txt" && "$(<"$run_dir/exit-code.txt")" == 0 &&
       -s "$run_dir/result.json" && -f "$run_dir/tracked.patch" &&
       -f "$run_dir/revision.txt" && "$(<"$run_dir/revision.txt")" == "$revision" ]] &&
    cmp -s "$run_dir/tracked.patch" "$root/tracked.patch" &&
    jq -e --argjson expected "$expected" \
      '.status=="measurement_completed_requires_review" and
       .phases.decode.generated_token_ids==$expected and
       .phases.decode.state_position==16383 and
       .packed_expert_bank_constructions==40 and
       .packed_expert_bank_loaded_experts==15360 and
       .route_execution_stats.diagnostic_readbacks==0' "$run_dir/result.json" >/dev/null &&
    grep -Eq '^[[:space:]]+0[[:space:]]+swaps$' "$run_dir/resource.log"; then
  echo "Resume: verified and skipped $label"
  return
 fi
 echo "Starting $label"
 mkdir -p "$(dirname "$run_dir")"
 DSV41_CONTEXT_RUN_DIR="$run_dir" CONTEXT_TOKENS=16384 TEACHER_TOKENS=4096 \
 TAIL_TEACHER_TOKENS=2048 DECODE_TOKENS=16 DSV41_CONTEXT_EXECUTION=sweep \
 DSV41_RUNTIME_DEFERRED_DECODER="$deferred" DSV41_RUNTIME_COMPONENT_PROFILE=0 \
 DSV41_RUNTIME_MLX_CACHE_LIMIT_BYTES=0 DSV41_RUNTIME_LONG_CONTEXT_CACHE_LIMIT_BYTES=17179869184 \
 DSV41_CONTEXT_WALL_BUDGET_SECONDS=3600 DSV41_CONTEXT_PROJECTED_WALL_LIMIT_SECONDS=3600 \
 CACHE_CONDITION=warm-alternating RUN_CONDITIONS="deferred-decoder-paired-$label" \
 bash tools/benchmark/run_context_32k.sh 2>&1 | tee "$root/$label.launch.log"
}

run_one warmup-baseline 0
run_one warmup-candidate 1
for ((pair=1;pair<=pairs;pair++)); do
 if ((pair%2)); then order=(baseline candidate); else order=(candidate baseline); fi
 for variant in "${order[@]}"; do
  if [[ "$variant" == baseline ]]; then deferred=0; else deferred=1; fi
  run_one "pair-$pair/$variant" "$deferred"
 done
done

summary="$root/measurements.tsv"
printf 'pair\tvariant\tprefill_seconds\tprefill_tps\tbase_seconds\tteacher_seconds\tdecode_mean\tdecode_p95\tpeak_bytes\tpeak_footprint_bytes\n' > "$summary"
for ((pair=1;pair<=pairs;pair++)); do
 for variant in baseline candidate; do
  run="$root/pair-$pair/$variant"
  footprint=$(awk '/peak memory footprint/{print $1}' "$run/resource.log")
  jq -r --arg pair "$pair" --arg variant "$variant" --argjson footprint "$footprint" \
   '[$pair,$variant,.aggregates.prefill_seconds,.aggregates.prefill_tokens_per_second,
     .phases.base_prefill.seconds,.aggregates.teacher_continuation_seconds,
     .phases.decode.mean_seconds,.phases.decode.p95_seconds,.final_memory.peak_bytes,$footprint]|@tsv' \
   "$run/result.json" >> "$summary"
 done
done
echo "Completed; review $summary and every child resource/config/identity log before enabling CED by default."
