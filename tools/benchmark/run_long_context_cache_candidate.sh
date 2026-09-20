#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

if [[ "${1:-}" == --help ]]; then
 echo 'usage: DSV41_CACHE_CANDIDATE_DIR=<resume-root> bash tools/benchmark/run_long_context_cache_candidate.sh'
 exit 0
fi
if (($#)); then echo 'unexpected arguments; use --help' >&2; exit 2; fi

root=${DSV41_CACHE_CANDIDATE_DIR:-artifacts/context-ladder/cache-candidate-$(date +%Y%m%d-%H%M%S)-$$}
limit=17179869184
mkdir -p "$root"
printf '%s\n' \
 'scope=production fixed-tile path at 2K, 16K, and 32K; long contexts use a 16 GiB MLX allocator-cache cap; no component synchronization' \
 'resources=one official model at a time; expected peak below 340 GB; checkpoint read-only; allow roughly 30-60 minutes total' \
 'checks=2K policy non-regression; 16K/32K wall, decode, active/cache/peak, process footprint, swap, topology, state position and generated tokens' \
 'logs=<root>/<stage>/attempt-N/{result.json,result.json.progress.jsonl,test.log,resource.log,config.txt,identity.txt,tracked.patch,exit-code.txt}' \
 'failure=failed attempt is retained; rerun with DSV41_CACHE_CANDIDATE_DIR=<same-root> to retry only the failed stage' \
 'resume=stage-level only; every attempt uses fresh model/request state' | tee "$root/README.txt"

run_stage() {
 local name=$1 context=$2 teacher=$3 tail=$4 decode=$5 wall=$6
 local stage="$root/$name" marker="$root/$name/completed-run.txt"
 mkdir -p "$stage"
 if [[ -f "$marker" ]]; then echo "SKIP completed stage $name: $(<"$marker")"; return; fi
 local number attempt
 number=$(find "$stage" -maxdepth 1 -type d -name 'attempt-*' | wc -l | tr -d ' ')
 number=$((number+1)); printf -v attempt '%s/attempt-%03d' "$stage" "$number"
 echo "START $name -> $attempt"
 if DSV41_CONTEXT_RUN_DIR="$attempt" CONTEXT_TOKENS="$context" TEACHER_TOKENS="$teacher" \
    TAIL_TEACHER_TOKENS="$tail" DECODE_TOKENS="$decode" DSV41_CONTEXT_EXECUTION=sweep \
    DSV41_RUNTIME_COMPONENT_PROFILE=0 DSV41_RUNTIME_MLX_CACHE_LIMIT_BYTES=0 \
    DSV41_RUNTIME_LONG_CONTEXT_CACHE_LIMIT_BYTES="$limit" \
    DSV41_CONTEXT_WALL_BUDGET_SECONDS="$wall" DSV41_CONTEXT_PROJECTED_WALL_LIMIT_SECONDS="$wall" \
    CACHE_CONDITION=warm-unknown RUN_CONDITIONS="long-context 16-GiB allocator-cache candidate" \
    bash tools/benchmark/run_context_32k.sh; then
  printf '%s\n' "$attempt" > "$marker"
 else
  echo "FAILED $name; inspect $attempt and rerun with DSV41_CACHE_CANDIDATE_DIR=$root" >&2
  exit 1
 fi
}

run_stage 2k 2064 0 0 1 1200
run_stage 16k 16384 4096 2048 16 5400
run_stage 32k 32768 4096 2048 16 9000

echo "Completed all stages; review every result/resource/config/identity log under $root before promotion."
