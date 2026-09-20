#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

task_root=${DSV41_CED_CONTINUATION_PROFILE_DIR:-artifacts/context-ladder/deferred-continuation-profile-$(date +%Y%m%d-%H%M%S)-$$}
mkdir -p "$task_root"
finish(){
 status=$?
 echo "$status" > "$task_root/exit-code.txt"
 if ((status)); then
  echo "FAILED: inspect $task_root; rerun with DSV41_CED_CONTINUATION_PROFILE_DIR=$task_root to reuse verified stages."
 fi
}
trap finish EXIT

printf '%s\n' \
 'scope=one synchronized 16K component profile per variant; isolate the systematic post-CED 4096-token continued-prefill delta' \
 'resources=two fresh official-model processes; allow 15-30 minutes; up to 340 GB Unified Memory; checkpoint read-only; no swap expected' \
 'checks=phase-local attention/MoE/post-MoE/layer wall, topology, generated tokens, state, memory and swap' \
 'logs=<root>/{baseline,candidate}/{result.json,result.json.progress.jsonl,test.log,resource.log,config.txt,identity.txt,tracked.patch,exit-code.txt} and summary.tsv' \
 'failure=retain the failed stage; private partial request state is never published or resumed' \
 "resume=DSV41_CED_CONTINUATION_PROFILE_DIR=$task_root bash tools/benchmark/run_deferred_decoder_continuation_profile.sh" \
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

expected='[1354,3452,271,19462,270,37370,2019,305,8470,270,15398,16,3476,477,260,20255]'
verify_stage(){
 local stage=$1 deferred=$2 dir="$task_root/$1"
 [[ -f "$dir/exit-code.txt" && "$(<"$dir/exit-code.txt")" == 0 && -s "$dir/result.json" ]] || return 1
 cmp -s "$dir/tracked.patch" "$task_root/tracked.patch" || return 1
 jq -e --argjson expected "$expected" --arg deferred "$deferred" \
  '.status=="measurement_completed_requires_review" and
   .deferred_decoder==($deferred=="1") and .component_profile==true and
   .phases.decode.generated_token_ids==$expected and .phases.decode.state_position==16383 and
   .packed_expert_bank_constructions==40 and .packed_expert_bank_loaded_experts==15360 and
   .route_execution_stats.diagnostic_readbacks==0 and
   .attention_telemetry.index_host_readbacks==0 and
   .attention_telemetry.chunk_scalar_qk_calls==0 and
   .attention_telemetry.chunk_scalar_av_calls==0' "$dir/result.json" >/dev/null || return 1
 grep -Eq '^[[:space:]]+0[[:space:]]+swaps$' "$dir/resource.log"
}

run_stage(){
 local stage=$1 deferred=$2 dir="$task_root/$1"
 if verify_stage "$stage" "$deferred"; then echo "SKIP verified $stage"; return; fi
 echo "START $stage"
 DSV41_CONTEXT_RUN_DIR="$dir" CONTEXT_TOKENS=16384 TEACHER_TOKENS=4096 \
  TAIL_TEACHER_TOKENS=2048 DECODE_TOKENS=16 DSV41_CONTEXT_EXECUTION=sweep \
  DSV41_RUNTIME_DEFERRED_DECODER="$deferred" DSV41_RUNTIME_COMPONENT_PROFILE=1 \
  DSV41_RUNTIME_MLX_CACHE_LIMIT_BYTES=0 DSV41_RUNTIME_LONG_CONTEXT_CACHE_LIMIT_BYTES=17179869184 \
  DSV41_CONTEXT_WALL_BUDGET_SECONDS=5400 DSV41_CONTEXT_PROJECTED_WALL_LIMIT_SECONDS=5400 \
  CACHE_CONDITION=warm-unknown RUN_CONDITIONS="CED continuation attribution $stage" \
  bash tools/benchmark/run_context_32k.sh 2>&1 | tee "$task_root/$stage.launch.log"
 verify_stage "$stage" "$deferred"
}

run_stage baseline 0
run_stage candidate 1

printf 'variant\tphase\twall_seconds\tlayer_seconds\tattention_seconds\tmoe_seconds\tpost_moe_seconds\tlayer_calls\tcomponent_calls\n' \
 > "$task_root/summary.tsv"
for variant in baseline candidate; do
 for phase in base_prefill teacher_head teacher_tail; do
  jq -r --arg variant "$variant" --arg phase "$phase" \
   '[ $variant,$phase,.phases[$phase].seconds,
      .phases[$phase].runtime_component_profile.layer_seconds,
      .phases[$phase].runtime_component_profile.attention_path_seconds,
      .phases[$phase].runtime_component_profile.moe_path_seconds,
      .phases[$phase].runtime_component_profile.post_moe_seconds,
      .phases[$phase].runtime_component_profile.layer_calls,
      .phases[$phase].runtime_component_profile.component_calls ]|@tsv' \
   "$task_root/$variant/result.json" >> "$task_root/summary.tsv"
 done
done
printf 'variant\tphase\tlayer\tlayer_seconds\tattention_seconds\tmoe_seconds\tpost_moe_seconds\tlayer_calls\tcomponent_calls\n' \
 > "$task_root/layers.tsv"
for variant in baseline candidate; do
 for phase in teacher_head teacher_tail; do
  jq -r --arg variant "$variant" --arg phase "$phase" \
   '.phases[$phase].runtime_component_profile.layers[] |
    [$variant,$phase,.layer,.layer_seconds,.attention_path_seconds,.moe_path_seconds,
     .post_moe_seconds,.layer_calls,.component_calls]|@tsv' \
   "$task_root/$variant/result.json" >> "$task_root/layers.tsv"
 done
done
echo "Completed; review $task_root/summary.tsv and both resource/config/identity logs before changing the continuation schedule."
