#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

if [[ "${1:-}" == --help ]]; then
 echo 'usage: DSV41_LONG_REGIME_DIR=<resume-root> bash tools/benchmark/run_long_context_regime_measurements.sh'
 echo 'Runs production and synchronized component-profile measurements at 16K and 32K.'
 exit 0
fi
if (($#)); then
 echo 'unexpected arguments; use --help' >&2
 exit 2
fi

root=${DSV41_LONG_REGIME_DIR:-artifacts/context-ladder/long-regime-$(date +%Y%m%d-%H%M%S)-$$}
mkdir -p "$root"
printf '%s\n' \
 'scope=four sequential runs: production and synchronized component profile at 16K and 32K; 4096 teacher continuation; 2048 tail; 16 decode tokens' \
 'resources=one official model at a time; up to 340 GB Unified Memory; checkpoint read-only; expect tens of minutes to a few hours total depending on thermal/cache state' \
 'logs=<root>/<stage>/attempt-N/{result.json,result.json.progress.jsonl,test.log,resource.log,config.txt,identity.txt,tracked.patch,exit-code.txt}' \
 'failure=failed attempt is retained; rerun with DSV41_LONG_REGIME_DIR=<same-root> to skip completed stages and retry only the failed stage from fresh model state' \
 'resume=stage-level only; an interrupted model state is never resumed' | tee "$root/README.txt"

run_stage() {
 local name=$1 context=$2 profile=$3 wall=$4
 local stage="$root/$name" marker="$root/$name/completed-run.txt"
 mkdir -p "$stage"
 if [[ -f "$marker" ]]; then
  echo "SKIP completed stage $name: $(<"$marker")"
  return
 fi
 local attempt_number
 attempt_number=$(find "$stage" -maxdepth 1 -type d -name 'attempt-*' | wc -l | tr -d ' ')
 attempt_number=$((attempt_number+1))
 local attempt
 printf -v attempt '%s/attempt-%03d' "$stage" "$attempt_number"
 echo "START $name -> $attempt"
 if DSV41_CONTEXT_RUN_DIR="$attempt" CONTEXT_TOKENS="$context" TEACHER_TOKENS=4096 \
    TAIL_TEACHER_TOKENS=2048 DECODE_TOKENS=16 DSV41_CONTEXT_EXECUTION=sweep \
    DSV41_RUNTIME_COMPONENT_PROFILE="$profile" DSV41_CONTEXT_WALL_BUDGET_SECONDS="$wall" \
    DSV41_CONTEXT_PROJECTED_WALL_LIMIT_SECONDS="$wall" CACHE_CONDITION=warm-unknown \
    RUN_CONDITIONS="long-context structural regime measurement; production defaults" \
    bash tools/benchmark/run_context_32k.sh; then
  printf '%s\n' "$attempt" > "$marker"
  python3 tools/benchmark/summarize_long_context_regimes.py "$root" --output "$root/summary.json"
 else
  echo "FAILED $name; inspect $attempt and rerun with DSV41_LONG_REGIME_DIR=$root" >&2
  exit 1
 fi
}

run_stage 16k-production 16384 0 5400
run_stage 16k-profile 16384 1 7200
run_stage 32k-production 32768 0 9000
run_stage 32k-profile 32768 1 10800

python3 tools/benchmark/summarize_long_context_regimes.py "$root" --output "$root/summary.json"
echo "Completed all stages; review $root/summary.json and every child result/resource log before qualification."
