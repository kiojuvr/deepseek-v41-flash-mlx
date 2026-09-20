#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

root=${DSV41_CUMULATIVE_QUAL_DIR:-artifacts/context-ladder/cumulative-long-$(date +%Y%m%d-%H%M%S)-$$}
enable_64k=${DSV41_CUMULATIVE_ENABLE_64K:-0}
if [[ "$enable_64k" != 0 && "$enable_64k" != 1 ]];then
 echo 'DSV41_CUMULATIVE_ENABLE_64K must be 0 or 1' >&2;exit 2
fi
mkdir -p "$root"
printf '%s\n' \
 'scope=one cumulative 32K session; after artifact review, DSV41_CUMULATIVE_ENABLE_64K=1 admits one cumulative 64K session; 8,128 deterministic fixture tokens plus 64 greedy decode tokens per turn' \
 'resources=one official model at a time; up to 340 GB Unified Memory; checkpoint read-only; expect roughly 25-75 minutes total depending on cache/thermal state' \
 'metrics=per-turn append wall/TPS, first and late decode TPT distribution, MLX active/cache/peak, attention/index topology, state position/revision, generated IDs, process compression/decompression/swap, and post-session state/cache recovery' \
 'gate=all turns complete at exact positions; finite outputs; 40 resident banks; no diagnostic readbacks or scalar attention; swap 0; footprint <=340 GB; no qualification claim until every raw log is reviewed' \
 'logs=<root>/{32k-cumulative,64k-cumulative}/attempt-N/{result.json,result.json.progress.jsonl,resource.log,config.txt,identity.txt,tracked.patch,exit-code.txt}' \
 'failure=retain failed attempt; rerun with DSV41_CUMULATIVE_QUAL_DIR=<same-root> to skip completed stages and retry only the failed stage from a fresh model/request' \
 'resume=stage-level only; a partially executed session is never resumed; 64K requires an explicit post-32K-review enable flag' | tee "$root/README.txt"

run_stage(){
 local name=$1 context=$2 turns=$3 wall_budget=$4
 local stage="$root/$name" marker="$root/$name/completed-run.txt"
 mkdir -p "$stage"
 if [[ -f "$marker" ]]; then echo "SKIP completed $name: $(<"$marker")";return;fi
 local number attempt schedule="$stage/turn-schedule.txt"
 number=$(find "$stage" -maxdepth 1 -type d -name 'attempt-*' | wc -l | tr -d ' ')
 number=$((number+1));printf -v attempt '%s/attempt-%03d' "$stage" "$number"
 local oracle_progress= prior
 for prior in "$stage"/attempt-*/result.json.progress.jsonl;do
  [[ -f "$prior" ]] || continue
  if [[ -z "$oracle_progress" && "$(wc -l < "$prior" | tr -d ' ')" == "$turns" ]];then
   oracle_progress=$prior
  fi
 done
 : > "$schedule"
 for ((turn=0;turn<turns;++turn));do printf '8128 64\n' >> "$schedule";done
 echo "START $name -> $attempt"
 if DSV41_CUMULATIVE_RUN_DIR="$attempt" TURN_SCHEDULE_FILE="$schedule" CONTEXT_TOKENS="$context" \
    DSV41_CUMULATIVE_WALL_BUDGET_SECONDS="$wall_budget" \
    CACHE_CONDITION=warm-unknown RUN_CONDITIONS="cumulative agent/session continuation" \
    bash tools/benchmark/run_cumulative_session.sh;then
  if [[ "${DSV41_DRY_RUN:-0}" == 1 ]];then
   echo "DRY RUN prepared $name without a completion marker";return
  fi
  [[ -s "$attempt/result.json" && -f "$attempt/exit-code.txt" && "$(<"$attempt/exit-code.txt")" == 0 ]] || {
   echo "FAILED $name artifact completeness check: $attempt" >&2;exit 1
  }
  if [[ -n "$oracle_progress" ]];then
   jq -c '{turn,next_position,generated_token_ids}' "$oracle_progress" > "$attempt/oracle-turns.jsonl"
   jq -c '{turn,next_position,generated_token_ids}' "$attempt/result.json.progress.jsonl" > "$attempt/candidate-turns.jsonl"
   cmp "$attempt/oracle-turns.jsonl" "$attempt/candidate-turns.jsonl" || {
    echo "FAILED $name generated-token continuity against $oracle_progress" >&2;exit 1
   }
   printf 'PASS: turn positions and generated token IDs match %s\n' "$oracle_progress" \
    > "$attempt/continuity-comparison.txt"
  fi
  printf '%s\n' "$attempt" > "$marker"
 else
  echo "FAILED $name; inspect $attempt and rerun with DSV41_CUMULATIVE_QUAL_DIR=$root" >&2;exit 1
 fi
}

run_stage 32k-cumulative 32768 4 7200
if [[ "$enable_64k" != 1 ]];then
 if [[ "${DSV41_DRY_RUN:-0}" != 1 ]];then
  "${PYTHON:-python3}" tools/benchmark/summarize_cumulative_sessions.py "$root" > "$root/summary.log"
 fi
 echo "Stopped after 32K as required; review its artifact, then resume this root with DSV41_CUMULATIVE_ENABLE_64K=1."
 exit 0
fi
run_stage 64k-cumulative 65536 8 14400
if [[ "${DSV41_DRY_RUN:-0}" != 1 ]];then
 "${PYTHON:-python3}" tools/benchmark/summarize_cumulative_sessions.py "$root" > "$root/summary.log"
fi
echo "Completed all stages under $root; read and check every result/resource/config/identity log before calling either regime passed."
