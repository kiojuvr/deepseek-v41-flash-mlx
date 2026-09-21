#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

root=${DSV41_FUSED_MHC_PAIRED_DIR:-artifacts/context-ladder/fused-mhc-paired-$(date +%Y%m%d-%H%M%S)-$$}
pairs=${DSV41_FUSED_MHC_PAIRS:-5}
if [[ ! "$pairs" =~ ^[0-9]+$ ]] || ((pairs<5));then
 echo 'DSV41_FUSED_MHC_PAIRS must be an integer of at least 5' >&2;exit 2
fi
if [[ "${DSV41_FUSED_MHC_PAIRED_DRY_RUN:-0}" == 1 ]];then
 echo 'warmup-baseline'
 echo 'warmup-candidate'
 for ((pair=1;pair<=pairs;++pair));do
  if ((pair%2));then echo "pair-$pair: baseline candidate";else echo "pair-$pair: candidate baseline";fi
 done
 echo 'DRY RUN: alternating schedule prepared; no model process was started.'
 exit 0
fi
mkdir -p "$root"
finish(){
 status=$?
 echo "$status" > "$root/exit-code.txt"
 if ((status));then
  echo "FAILED: inspect $root; rerun with DSV41_FUSED_MHC_PAIRED_DIR=$root to skip verified stages."
 fi
}
trap finish EXIT

printf '%s\n' \
 "scope=two warmups followed by $pairs alternating baseline/candidate pairs; every stage uses a fresh official model, 2,063-token production prefill, and 8 greedy tokens" \
 'resources=sequential model processes; allow 20-45 minutes for five pairs; up to 340 GB Unified Memory each; checkpoint read-only; swap 0 required' \
 'candidate=one fused Metal dispatch owns the mHC scale/base, pre/post sigmoid and Sinkhorn normalization; production default remains unchanged' \
 'checks=exact generated IDs/state/position within every pair; finite output; 40 banks; zero route/index readbacks and scalar attention; candidate records fused_mhc_invocations>0 and baseline records 0' \
 'decision=review all raw logs and paired advancing-decode deltas; this runner neither promotes nor declares qualification' \
 'logs=<root>/{warmup-*,pair-N/{baseline,candidate}}/{result.json,result.json.progress.jsonl,test.log,resource.log,config.txt,identity.txt,tracked.patch,exit-code.txt}; <root>/measurements.tsv' \
 'failure=retain failed stage; partial request state is never resumed' \
 "resume=DSV41_FUSED_MHC_PAIRED_DIR=$root bash tools/benchmark/run_fused_mhc_paired_qualification.sh" \
 | tee "$root/config.txt"
echo "Logs: $root"

revision=$(git rev-parse HEAD)
if [[ -f "$root/revision.txt" && "$(<"$root/revision.txt")" != "$revision" ]];then
 echo 'resume revision mismatch' >&2;exit 2
fi
echo "$revision" > "$root/revision.txt"
patch_sha=$(git diff --binary | shasum -a 256 | awk '{print $1}')
if [[ -f "$root/tracked.patch" ]];then
 [[ "$(shasum -a 256 "$root/tracked.patch" | awk '{print $1}')" == "$patch_sha" ]] || {
  echo 'resume tracked patch mismatch' >&2;exit 2;
 }
else
 git diff --binary > "$root/tracked.patch"
fi
source_sha=$(shasum -a 256 tools/benchmark/run_fused_mhc_paired_qualification.sh \
 tools/benchmark/run_context_32k.sh tools/benchmark/context_ladder.cpp \
 include/dsv41/execution_policy.hpp include/dsv41/mhc.hpp \
 src/mhc/reference.cpp src/mhc/split_sinkhorn.hpp.in metal/mhc/split_sinkhorn.metal | \
 shasum -a 256 | awk '{print $1}')
if [[ -f "$root/source-sha256.txt" && "$(<"$root/source-sha256.txt")" != "$source_sha" ]];then
 echo 'resume source identity mismatch' >&2;exit 2
fi
echo "$source_sha" > "$root/source-sha256.txt"

verify_stage(){
 local dir=$1 enabled=$2
 [[ -f "$dir/exit-code.txt" && "$(<"$dir/exit-code.txt")" == 0 && -s "$dir/result.json" ]] || return 1
 cmp -s "$dir/tracked.patch" "$root/tracked.patch" || return 1
 jq -e --arg enabled "$enabled" '
  .status=="measurement_completed_requires_review" and
  .context_tokens==2071 and .decode_tokens==8 and
  .fused_mhc==($enabled=="1") and
  .phases.decode.state_position==2070 and .phases.decode.next_position==2071 and
  (.phases.decode.generated_token_ids|length)==8 and
  .packed_expert_bank_constructions==40 and
  .route_execution_stats.diagnostic_readbacks==0 and
  .attention_telemetry.index_host_readbacks==0 and
  .attention_telemetry.chunk_scalar_qk_calls==0 and
  .attention_telemetry.chunk_scalar_av_calls==0 and
  (if $enabled=="1" then .fused_mhc_invocations>0 else .fused_mhc_invocations==0 end)
 ' "$dir/result.json" >/dev/null || return 1
 grep -Eq '^[[:space:]]+0[[:space:]]+swaps$' "$dir/resource.log"
}

run_one(){
 local label=$1 enabled=$2 dir="$root/$1"
 if verify_stage "$dir" "$enabled";then echo "SKIP verified $label";return;fi
 mkdir -p "$(dirname "$dir")"
 echo "START $label"
 DSV41_CONTEXT_RUN_DIR="$dir" \
 CONTEXT_TOKENS=2071 TEACHER_TOKENS=0 TAIL_TEACHER_TOKENS=0 DECODE_TOKENS=8 \
 DSV41_CONTEXT_EXECUTION=sweep DSV41_RUNTIME_COMPONENT_PROFILE=0 \
 DSV41_RUNTIME_FUSED_MHC="$enabled" DSV41_RUNTIME_MLX_CACHE_LIMIT_BYTES=0 \
 DSV41_RUNTIME_LONG_CONTEXT_CACHE_LIMIT_BYTES=17179869184 \
 DSV41_CONTEXT_WALL_BUDGET_SECONDS=900 DSV41_CONTEXT_PROJECTED_WALL_LIMIT_SECONDS=900 \
 CACHE_CONDITION=warm-alternating RUN_CONDITIONS="fused-mhc-paired-$label" \
 bash tools/benchmark/run_context_32k.sh 2>&1 | tee "$root/${label//\//-}.launch.log"
 verify_stage "$dir" "$enabled"
}

run_one warmup-baseline 0
run_one warmup-candidate 1
for ((pair=1;pair<=pairs;++pair));do
 if ((pair%2));then order=(baseline candidate);else order=(candidate baseline);fi
 for variant in "${order[@]}";do
  if [[ "$variant" == baseline ]];then enabled=0;else enabled=1;fi
  run_one "pair-$pair/$variant" "$enabled"
 done
 baseline_result="$root/pair-$pair/baseline/result.json"
 candidate_result="$root/pair-$pair/candidate/result.json"
 jq -e --slurpfile candidate "$candidate_result" '
  .phases.decode.generated_token_ids==$candidate[0].phases.decode.generated_token_ids and
  .phases.decode.state_position==$candidate[0].phases.decode.state_position and
  .phases.decode.next_position==$candidate[0].phases.decode.next_position
 ' "$baseline_result" >/dev/null
done

summary="$root/measurements.tsv"
printf 'pair\tvariant\tprefill_seconds\tfirst_seconds\tadvancing_mean_seconds\tdecode_p95_seconds\tfused_mhc_invocations\tpeak_bytes\n' > "$summary"
for ((pair=1;pair<=pairs;++pair));do
 for variant in baseline candidate;do
  jq -r --arg pair "$pair" --arg variant "$variant" '
   .phases.decode.latency_seconds[1:] as $advancing |
   [$pair,$variant,.aggregates.prefill_seconds,.phases.decode.first_token_seconds,
    (($advancing|add)/($advancing|length)),.phases.decode.p95_seconds,
    .fused_mhc_invocations,.final_memory.peak_bytes] | @tsv
  ' "$root/pair-$pair/$variant/result.json" >> "$summary"
 done
done
cat "$summary"
echo 'Completed; review every result/resource/config/identity log before accepting or rejecting the candidate.'
