#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

task_root=${DSV41_GROUPED_PIPELINE_DIR:-artifacts/context-ladder/grouped-expert-pipeline-$(date +%Y%m%d-%H%M%S)-$$}
mkdir -p "$task_root"
finish(){
 status=$?
 echo "$status" > "$task_root/exit-code.txt"
 if ((status)); then
  echo "FAILED: inspect $task_root; rerun with DSV41_GROUPED_PIPELINE_DIR=$task_root to reuse a verified stage."
 fi
}
trap finish EXIT
printf '%s\n' \
 'scope=one baseline and one grouped-expert-pipeline candidate; fresh official model per stage; 2,063-token prefill plus 8 greedy tokens' \
 'resources=two sequential processes; allow 4-10 minutes total; up to 340 GB Unified Memory each; checkpoint read-only; swap 0 required' \
 'candidate=oMLX-derived native primitive owns routed gate/up, exact official FP8 SwiGLU roundtrip, and down GatherQMM for resident 1..8-token MoE' \
 'checks=exact generated IDs/state/position; 40 banks; zero route/index readbacks and scalar attention; candidate records 280 grouped pipeline layer batches' \
 'logs=<root>/{baseline,candidate}/{result.json,result.json.progress.jsonl,test.log,resource.log,config.txt,identity.txt,tracked.patch,exit-code.txt}, summary.tsv' \
 'failure=retain failed stage; partial request state is never resumed' \
 "resume=DSV41_GROUPED_PIPELINE_DIR=$task_root bash tools/benchmark/run_grouped_expert_pipeline_candidate.sh" \
 | tee "$task_root/config.txt"
echo "Logs: $task_root"

revision=$(git rev-parse HEAD)
if [[ -f "$task_root/revision.txt" && "$(<"$task_root/revision.txt")" != "$revision" ]]; then
 echo 'resume revision mismatch' >&2;exit 2
fi
echo "$revision" > "$task_root/revision.txt"
patch_sha=$(git diff --binary | shasum -a 256 | awk '{print $1}')
if [[ -f "$task_root/tracked.patch" ]]; then
 [[ "$(shasum -a 256 "$task_root/tracked.patch" | awk '{print $1}')" == "$patch_sha" ]] || {
  echo 'resume tracked patch mismatch' >&2;exit 2;
 }
else
 git diff --binary > "$task_root/tracked.patch"
fi
source_sha=$(shasum -a 256 tools/benchmark/run_grouped_expert_pipeline_candidate.sh \
 tools/benchmark/run_context_32k.sh tools/benchmark/context_ladder.cpp \
 include/dsv41/execution_policy.hpp include/dsv41/moe.hpp include/dsv41/moe_pipeline.hpp \
 src/moe/reference.cpp src/moe/expert_bank.cpp src/moe/grouped_expert_pipeline.cpp | \
 shasum -a 256 | awk '{print $1}')
if [[ -f "$task_root/source-sha256.txt" && "$(<"$task_root/source-sha256.txt")" != "$source_sha" ]]; then
 echo 'resume source identity mismatch' >&2;exit 2
fi
echo "$source_sha" > "$task_root/source-sha256.txt"

verify_stage(){
 local stage=$1 enabled=$2
 local dir="$task_root/$stage"
 [[ -f "$dir/exit-code.txt" && "$(<"$dir/exit-code.txt")" == 0 && -s "$dir/result.json" ]] || return 1
 cmp -s "$dir/tracked.patch" "$task_root/tracked.patch" || return 1
 jq -e --arg enabled "$enabled" '
  .status=="measurement_completed_requires_review" and
  .context_tokens==2071 and .decode_tokens==8 and
  .grouped_expert_pipeline==($enabled=="1") and
  .phases.decode.state_position==2070 and .phases.decode.next_position==2071 and
  (.phases.decode.generated_token_ids|length)==8 and
  .packed_expert_bank_constructions==40 and
  .route_execution_stats.diagnostic_readbacks==0 and
  .attention_telemetry.index_host_readbacks==0 and
  .attention_telemetry.chunk_scalar_qk_calls==0 and
  .attention_telemetry.chunk_scalar_av_calls==0 and
  (if $enabled=="1" then .expert_bank_io_stats.grouped_pipeline_batches==280
   else .expert_bank_io_stats.grouped_pipeline_batches==0 end)
 ' "$dir/result.json" >/dev/null || return 1
 grep -Eq '^[[:space:]]+0[[:space:]]+swaps$' "$dir/resource.log"
}

run_stage(){
 local stage=$1 enabled=$2
 local dir="$task_root/$stage"
 if verify_stage "$stage" "$enabled"; then echo "SKIP verified $stage";return;fi
 echo "START $stage"
 DSV41_CONTEXT_RUN_DIR="$dir" \
 CONTEXT_TOKENS=2071 TEACHER_TOKENS=0 TAIL_TEACHER_TOKENS=0 DECODE_TOKENS=8 \
 DSV41_CONTEXT_EXECUTION=sweep DSV41_RUNTIME_COMPONENT_PROFILE=0 \
 DSV41_RUNTIME_GROUPED_EXPERT_PIPELINE="$enabled" DSV41_RUNTIME_MLX_CACHE_LIMIT_BYTES=0 \
 DSV41_RUNTIME_LONG_CONTEXT_CACHE_LIMIT_BYTES=17179869184 \
 DSV41_CONTEXT_WALL_BUDGET_SECONDS=900 DSV41_CONTEXT_PROJECTED_WALL_LIMIT_SECONDS=900 \
 CACHE_CONDITION=warm-unknown RUN_CONDITIONS="grouped-expert-pipeline $stage" \
 bash tools/benchmark/run_context_32k.sh 2>&1 | tee "$task_root/$stage.launch.log"
 verify_stage "$stage" "$enabled"
}

run_stage baseline 0
run_stage candidate 1
jq -e --slurpfile candidate "$task_root/candidate/result.json" '
 .phases.decode.generated_token_ids==$candidate[0].phases.decode.generated_token_ids and
 .phases.decode.state_position==$candidate[0].phases.decode.state_position and
 .phases.decode.next_position==$candidate[0].phases.decode.next_position
' "$task_root/baseline/result.json" >/dev/null

printf 'variant\tprefill_seconds\tfirst_seconds\tadvancing_mean_seconds\tdecode_mean_seconds\tdecode_p95_seconds\tgrouped_pipeline_batches\tpeak_bytes\n' \
 > "$task_root/summary.tsv"
for variant in baseline candidate; do
 jq -r --arg variant "$variant" '
  .phases.decode.latency_seconds[1:] as $advancing |
  [$variant,.aggregates.prefill_seconds,.phases.decode.first_token_seconds,
   (($advancing|add)/($advancing|length)),.phases.decode.mean_seconds,
   .phases.decode.p95_seconds,.expert_bank_io_stats.grouped_pipeline_batches,
   .final_memory.peak_bytes] | @tsv
 ' "$task_root/$variant/result.json" >> "$task_root/summary.tsv"
done
cat "$task_root/summary.tsv"
echo 'Completed single-pair candidate measurement; raw-log review is required and this run cannot promote the candidate.'
