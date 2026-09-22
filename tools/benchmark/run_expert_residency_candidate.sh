#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mode=${1:-bounded}
[[ "$mode" == bounded || "$mode" == paired ]] || { echo 'usage: bash tools/benchmark/run_expert_residency_candidate.sh [bounded|paired]' >&2; exit 2; }
if [[ ${DSV41_ALLOW_REJECTED_ANONYMOUS_RESIDENCY:-0} != 1 ]]; then
 echo 'REJECTED: anonymous-atlas residency budgets 320/288/272 GiB violated the zero-system-swap gate; do not rerun or start paired.' >&2
 echo 'Set DSV41_ALLOW_REJECTED_ANONYMOUS_RESIDENCY=1 only to reproduce the historical rejected experiment.' >&2
 exit 2
fi
root=${DSV41_RESIDENCY_DIR:-artifacts/context-ladder/expert-residency-$(date +%Y%m%d-%H%M%S)-$$}
limit=${DSV41_RESIDENCY_CANDIDATE_BYTES:-292057776128} # 272 GiB
mkdir -p "$root"
printf '%s\n' \
 'scope=bounded: one official-model wired/unwired exactness gate, then fresh baseline/candidate 2063+8 full paths; paired: three alternating fresh-process pairs after bounded review' \
 'resources=bounded approximately 6-12 minutes; paired approximately 12-25 additional minutes; sequential processes, <=340 GB footprint each; checkpoint read-only' \
 "candidate=MLX wired limit $limit bytes, acquired before model allocation and restored at model teardown; arithmetic/layout unchanged" \
 'logs=<root>/parity and <root>/bounded/{baseline,candidate} or paired/pair-NN/{baseline,candidate}; comparison.json plus canonical logs' \
 'failure=stop and retain artifacts; rerun same root to skip completed stages; failed measurement directories are archived and restarted with fresh model state' \
 "resume=DSV41_RESIDENCY_DIR=$root bash tools/benchmark/run_expert_residency_candidate.sh $mode" \
 'gate=exact bounded observables, zero process/system swap, <=340 GB footprint, generated-token identity; never auto-promotes or qualifies 64K' | tee "$root/README.txt"
if [[ ${DSV41_DRY_RUN:-0} == 1 ]]; then echo 'DRY RUN: preflight -> build -> parity -> full-path pairs -> comparison; no model started'; exit 0; fi
preflight(){
 local page_size free_pages free_gib big
 page_size=$(vm_stat | sed -n '1s/.*page size of \([0-9][0-9]*\).*/\1/p'); [[ -n "$page_size" ]] || page_size=$(sysctl -n hw.pagesize)
 free_pages=$(vm_stat | awk '/Pages free/{gsub(/\./,"",$3);f=$3} /Pages inactive/{gsub(/\./,"",$3);i=$3} END{print f+i}')
 free_gib=$((free_pages*page_size/1024/1024/1024))
 big=$(ps -axo rss=,pid=,comm= | awk -v limit=$((32*1024*1024)) '$1>limit{printf "  pid %s %s (%d MiB)\n",$2,$3,$1/1024}')
 [[ -z "$big" ]] || { printf 'preflight FAILED: large process(es):\n%s\n' "$big" >&2; exit 3; }
 ((free_gib>=360)) || { echo "preflight FAILED: free+inactive ${free_gib} GiB; need >=360 GiB" >&2; exit 3; }
 { echo "free_inactive_gib=$free_gib"; sysctl iogpu.wired_limit_mb; } | tee "$root/preflight.txt"
}
preflight
identity=$( { git rev-parse HEAD; git diff --binary; rg --files src include tests tools/benchmark | LC_ALL=C sort | while IFS= read -r f; do shasum -a 256 "$f"; done; } | shasum -a 256 | awk '{print $1}')
if [[ -f "$root/source-identity.txt" ]]; then [[ $(<"$root/source-identity.txt") == "$identity" ]] || { echo 'source identity changed; use a new root' >&2; exit 2; }; else echo "$identity" > "$root/source-identity.txt"; fi
cmake -S . -B build-mlx > "$root/configure.log" 2>&1
cmake --build build-mlx --target dsv41-residency-test dsv41-text-backbone-test dsv41-context-ladder -j 4 > "$root/build.log" 2>&1
build-mlx/dsv41-residency-test > "$root/unit.log" 2>&1
grep -q 'PASS: residency policy' "$root/unit.log"
binary_identity=$(shasum -a 256 build-mlx/dsv41-residency-test build-mlx/dsv41-text-backbone-test build-mlx/dsv41-context-ladder | shasum -a 256 | awk '{print $1}')
if [[ -f "$root/binary-identity.txt" ]]; then [[ $(<"$root/binary-identity.txt") == "$binary_identity" ]] || { echo 'binary identity changed; use a new root' >&2; exit 2; }; else echo "$binary_identity" > "$root/binary-identity.txt"; fi
checkpoint=${CHECKPOINT:-/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash}
export DSV41_RUNTIME_RESIDENT_EXPERT_ATLAS=1 DSV41_RUNTIME_PACKED_EXPERT_BANK=1
export DSV41_RUNTIME_COMPACT_EXPERT_BANK=0 DSV41_RUNTIME_GROUP_SELECTED_EXPERTS=0
export DSV41_RUNTIME_COMPONENT_PROFILE=0 DSV41_RUNTIME_DECODE_STACK_GRAPH=0
if [[ ! -f "$root/parity/completed" ]]; then
 [[ ! -d "$root/parity" ]] || mv "$root/parity" "$root/parity.failed-$(date +%Y%m%d-%H%M%S)-$$"
 mkdir -p "$root/parity"
 vm_stat > "$root/parity/system-before.txt"
 DSV41_RUNTIME_WIRED_LIMIT_BYTES=$limit DSV41_CHECK_RESIDENCY=1 \
  /usr/bin/time -l build-mlx/dsv41-text-backbone-test "$checkpoint" artifacts/checkpoint/summary.json artifacts/engram/metadata.json \
  > "$root/parity/test.log" 2> "$root/parity/resource.log"
 vm_stat > "$root/parity/system-after.txt"
 grep -q 'PASS: residency policy' "$root/parity/test.log"
 grep -Eq '^[[:space:]]+0[[:space:]]+swaps$' "$root/parity/resource.log"
 awk '/peak memory footprint/{found=1;if($1>340000000000)exit 1} END{if(!found)exit 1}' "$root/parity/resource.log"
 for metric in Swapins Swapouts; do
  before=$(awk -v key="$metric:" '$1==key{print $2}' "$root/parity/system-before.txt")
  after=$(awk -v key="$metric:" '$1==key{print $2}' "$root/parity/system-after.txt")
  [[ -n "$before" && "$before" == "$after" ]] || { echo "parity system $metric changed or missing" >&2; exit 1; }
 done
 touch "$root/parity/completed"
fi
run_variant(){
 local dir=$1 wired=$2
 if [[ -f "$dir/completed" && -s "$dir/result.json" && $(<"$dir/exit-code.txt") == 0 ]]; then echo "SKIP $dir"; return; fi
 [[ ! -d "$dir" ]] || mv "$dir" "$dir.failed-$(date +%Y%m%d-%H%M%S)-$$"
 DSV41_CONTEXT_RUN_DIR="$dir" CONTEXT_TOKENS=2071 TEACHER_TOKENS=0 TAIL_TEACHER_TOKENS=0 DECODE_TOKENS=8 \
 DSV41_CONTEXT_EXECUTION=sweep DSV41_RUNTIME_WIRED_LIMIT_BYTES=$wired \
 DSV41_CONTEXT_WALL_BUDGET_SECONDS=900 DSV41_CONTEXT_PROJECTED_WALL_LIMIT_SECONDS=900 \
 CACHE_CONDITION=warm-unknown RUN_CONDITIONS=expert-residency-production-short \
 bash tools/benchmark/run_context_32k.sh
 [[ $(<"$dir/exit-code.txt") == 0 ]]; touch "$dir/completed"
}
if [[ "$mode" == bounded ]]; then
 run_variant "$root/bounded/baseline" 0
 run_variant "$root/bounded/candidate" "$limit"
 python3 tools/benchmark/compare_expert_residency.py "$root/bounded" "$limit" | tee "$root/bounded/comparison.json"
else
 [[ -s "$root/bounded/comparison.json" ]] || { echo 'Run and review bounded mode in this root first.' >&2; exit 2; }
 jq -e '.status=="gates_passed_requires_raw_log_review_not_promotion"' "$root/bounded/comparison.json" >/dev/null
 for spec in '01 baseline candidate' '02 candidate baseline' '03 baseline candidate'; do
  read -r n first second <<<"$spec"; pair="$root/paired/pair-$n"
  for variant in "$first" "$second"; do [[ "$variant" == candidate ]] && wired=$limit || wired=0; run_variant "$pair/$variant" "$wired"; done
 done
 python3 tools/benchmark/compare_expert_residency.py "$root/paired" "$limit" | tee "$root/paired/comparison.json"
fi
echo 'Completed candidate gate; inspect raw logs, VM compression deltas, latency signs, and host state. This does not promote or qualify 64K.'
