#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

if [[ "${1:-}" == --help ]]; then
 echo 'usage: DSV41_CED_CANDIDATE_DIR=<resume-root> bash tools/benchmark/run_deferred_decoder_context_candidate.sh'
 exit 0
fi
if (($#)); then echo 'unexpected arguments; use --help' >&2; exit 2; fi

root=${DSV41_CED_CANDIDATE_DIR:-artifacts/context-ladder/deferred-decoder-candidate-$(date +%Y%m%d-%H%M%S)-$$}
mkdir -p "$root"
finish(){ status=$?; echo "$status" > "$root/exit-code.txt"; }
trap finish EXIT
printf '%s\n' \
 'scope=production fixed-tile/cache-bounded path with pending CED: 2K/16K inactive non-regression, then one 16K encoder-only plus >=8K decoder resume at 32K' \
 'resources=one official model at a time; up to 340 GB Unified Memory; checkpoint read-only; allow roughly 20-45 minutes total' \
 'checks=2K/16K schedule inactive; 32K exact pending completion, CED transaction count, wall/decode/memory, topology, state position, generated tokens and swap' \
 'logs=<root>/<stage>/attempt-N/{result.json,result.json.progress.jsonl,test.log,resource.log,config.txt,identity.txt,tracked.patch,exit-code.txt}' \
 'failure=failed attempt is retained; rerun with DSV41_CED_CANDIDATE_DIR=<same-root> to skip completed stages and retry failed stage fresh' \
 'resume=stage-level only; a partial transaction is never published or resumed' | tee "$root/README.txt"

revision=$(git rev-parse HEAD)
if [[ -f "$root/revision.txt" && "$(<"$root/revision.txt")" != "$revision" ]]; then
 echo "resume revision mismatch" >&2; exit 2
fi
echo "$revision" > "$root/revision.txt"
patch_sha=$(git diff --binary | shasum -a 256 | awk '{print $1}')
if [[ -f "$root/tracked.patch" ]]; then
 existing_sha=$(shasum -a 256 "$root/tracked.patch" | awk '{print $1}')
 if [[ "$patch_sha" != "$existing_sha" ]]; then echo "resume tracked patch mismatch" >&2; exit 2; fi
else
 git diff --binary > "$root/tracked.patch"
fi

verify_stage() {
 local attempt=$1 context=$2 decode=$3 expected_deferred=$4
 [[ -f "$attempt/exit-code.txt" && "$(<"$attempt/exit-code.txt")" == 0 &&
    -f "$attempt/revision.txt" && "$(<"$attempt/revision.txt")" == "$revision" &&
    -s "$attempt/result.json" && -s "$attempt/resource.log" ]] || return 1
 cmp -s "$attempt/tracked.patch" "$root/tracked.patch" || return 1
 jq -e --argjson context "$context" --argjson decode "$decode" \
       --argjson expected "$expected_deferred" \
  '.status=="measurement_completed_requires_review" and .deferred_decoder==true and
   .context_tokens==$context and .phases.base_prefill.deferred_decoder_chunks==$expected and
   .phases.teacher_head.deferred_decoder_chunks==0 and
   .phases.teacher_tail.deferred_decoder_chunks==0 and
   .phases.decode.state_position==($context-1) and
   (.phases.decode.generated_token_ids|length)==$decode and
   .packed_expert_bank_constructions==40 and .packed_expert_bank_loaded_experts==15360 and
   .route_execution_stats.diagnostic_readbacks==0 and
   .attention_telemetry.index_host_readbacks==0 and
   .attention_telemetry.chunk_scalar_qk_calls==0 and
   .attention_telemetry.chunk_scalar_av_calls==0' "$attempt/result.json" >/dev/null || return 1
 grep -Eq '^[[:space:]]+0[[:space:]]+swaps$' "$attempt/resource.log"
}

run_stage() {
 local name=$1 context=$2 teacher=$3 tail=$4 decode=$5 wall=$6 expected_deferred=$7
 local stage="$root/$name" marker="$root/$name/completed-run.txt"
 mkdir -p "$stage"
 if [[ -f "$marker" ]] && verify_stage "$(<"$marker")" "$context" "$decode" "$expected_deferred"; then
  echo "SKIP verified stage $name: $(<"$marker")"; return
 fi
 local number attempt
 number=$(find "$stage" -maxdepth 1 -type d -name 'attempt-*' | wc -l | tr -d ' ')
 number=$((number+1)); printf -v attempt '%s/attempt-%03d' "$stage" "$number"
 echo "START $name -> $attempt"
 if DSV41_CONTEXT_RUN_DIR="$attempt" CONTEXT_TOKENS="$context" TEACHER_TOKENS="$teacher" \
    TAIL_TEACHER_TOKENS="$tail" DECODE_TOKENS="$decode" DSV41_CONTEXT_EXECUTION=sweep \
    DSV41_RUNTIME_DEFERRED_DECODER=1 DSV41_RUNTIME_COMPONENT_PROFILE=0 \
    DSV41_RUNTIME_MLX_CACHE_LIMIT_BYTES=0 DSV41_RUNTIME_LONG_CONTEXT_CACHE_LIMIT_BYTES=17179869184 \
    DSV41_CONTEXT_WALL_BUDGET_SECONDS="$wall" DSV41_CONTEXT_PROJECTED_WALL_LIMIT_SECONDS="$wall" \
    CACHE_CONDITION=warm-unknown RUN_CONDITIONS="candidate CED with production fixed-tile and 16-GiB long-context cache bound" \
    bash tools/benchmark/run_context_32k.sh &&
    verify_stage "$attempt" "$context" "$decode" "$expected_deferred"; then
  printf '%s\n' "$attempt" > "$marker"
 else
  echo "FAILED $name; inspect $attempt and rerun with DSV41_CED_CANDIDATE_DIR=$root" >&2
  exit 1
 fi
}

run_stage 2k 2064 0 0 1 1200 0
run_stage 16k 16384 4096 2048 16 3600 0
run_stage 32k 32768 4096 2048 16 5400 1

echo "Completed all stages; require deferred chunks 0/0/1 at 2K/16K/32K and review every result/resource/config/identity log before qualification."
