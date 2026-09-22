#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mode=${1:-bounded}
[[ "$mode" == bounded || "$mode" == paired ]] || { echo 'usage: bash tools/benchmark/run_file_backed_residency_candidate.sh [bounded|paired]' >&2; exit 2; }
if [[ ${DSV41_ALLOW_UNSAFE_FILE_BACKED_REPRODUCTION:-0} != 1 ]]; then
 echo 'REJECTED/UNSAFE: the first parity run caused severe host memory pressure and system swap; do not resume bounded or paired.' >&2
 echo 'The override exists only for forensic reproduction and must not be used for validation.' >&2
 exit 2
fi
root=${DSV41_FILE_RESIDENCY_DIR:-artifacts/context-ladder/file-backed-residency-$(date +%Y%m%d-%H%M%S)-$$}
backing=${DSV41_EXPERT_BACKING_DIR:-artifacts/expert-backing/official-packed-v1}
limit=${DSV41_RESIDENCY_CANDIDATE_BYTES:-292057776128} # 272 GiB
mkdir -p "$root"
printf '%s\n' \
 'scope=bounded: anonymous-checkpoint versus file-backed exactness through 129+4 advancing tokens, then fresh anonymous baseline/file-backed+wired candidate 2063+8 full paths; paired: three alternating pairs after review' \
 'resources=bounded approximately 8-18 minutes; paired approximately 12-25 additional minutes; sequential processes, <=340 GB footprint each; checkpoint/backing read-only' \
 "candidate=file-backed no-copy atlas at $backing plus MLX wired limit $limit bytes; arithmetic/layout unchanged" \
 'logs=<root>/parity and <root>/bounded/{baseline,candidate} or paired/pair-NN/{baseline,candidate}; comparison.json plus canonical logs' \
 'failure=stop and retain artifacts; rerun same root to skip completed stages; failed measurement directories are archived and restarted with fresh model state' \
 "resume=DSV41_FILE_RESIDENCY_DIR=$root DSV41_EXPERT_BACKING_DIR=$backing bash tools/benchmark/run_file_backed_residency_candidate.sh $mode" \
 'gate=exact bounded observables, zero process/system swap, <=340 GB footprint, generated-token identity; never auto-promotes or qualifies 64K' | tee "$root/README.txt"
if [[ ${DSV41_DRY_RUN:-0} == 1 ]]; then echo 'DRY RUN: preflight -> build -> parity -> full-path pairs -> comparison; no model started'; exit 0; fi
preflight(){
 local page_size free_pages inactive_pages free_inactive_gib big stats
 stats=$(vm_stat);page_size=$(printf '%s\n' "$stats" | sed -n '1s/.*page size of \([0-9][0-9]*\).*/\1/p'); [[ -n "$page_size" ]] || page_size=$(sysctl -n hw.pagesize)
 read -r free_pages inactive_pages < <(printf '%s\n' "$stats" | awk '/Pages free/{gsub(/\./,"",$3);f=$3} /Pages inactive/{gsub(/\./,"",$3);i=$3} END{print f,i}')
 free_inactive_gib=$(((free_pages+inactive_pages)*page_size/1024/1024/1024))
 big=$(ps -axo rss=,pid=,comm= | awk -v limit=$((32*1024*1024)) '$1>limit{printf "  pid %s %s (%d MiB)\n",$2,$3,$1/1024}')
 [[ -z "$big" ]] || { printf 'preflight FAILED: large process(es):\n%s\n' "$big" >&2; exit 3; }
 ((free_inactive_gib>=360)) || { echo "preflight FAILED: conservative free+inactive ${free_inactive_gib} GiB; need >=360 GiB (speculative cache is deliberately excluded)" >&2; exit 3; }
 { echo "free_inactive_gib=$free_inactive_gib"; echo "free_pages=$free_pages inactive_pages=$inactive_pages page_size=$page_size"; sysctl iogpu.wired_limit_mb; } | tee "$root/preflight.txt"
}
preflight
[[ -r "$backing/manifest.json" ]] || { echo 'complete expert backing is missing' >&2; exit 2; }
identity=$( { git rev-parse HEAD; git diff --binary; shasum -a 256 "$backing/manifest.json"; rg --files src include tests tools/benchmark | LC_ALL=C sort | while IFS= read -r f; do shasum -a 256 "$f"; done; } | shasum -a 256 | awk '{print $1}')
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
 DSV41_RUNTIME_WIRED_LIMIT_BYTES=0 DSV41_RUNTIME_EXPERT_BACKING_DIR="$backing" DSV41_CHECK_FILE_BACKED_EXACTNESS=1 \
  /usr/bin/time -l build-mlx/dsv41-text-backbone-test "$checkpoint" artifacts/checkpoint/summary.json artifacts/engram/metadata.json \
  > "$root/parity/test.log" 2> "$root/parity/resource.log"
 vm_stat > "$root/parity/system-after.txt"
 grep -q 'PASS: file-backed prefill/decode' "$root/parity/test.log"
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
 local dir=$1 wired=$2 backing_dir=$3
 if [[ -f "$dir/completed" && -s "$dir/result.json" && $(<"$dir/exit-code.txt") == 0 ]]; then echo "SKIP $dir"; return; fi
 [[ ! -d "$dir" ]] || mv "$dir" "$dir.failed-$(date +%Y%m%d-%H%M%S)-$$"
 DSV41_CONTEXT_RUN_DIR="$dir" CONTEXT_TOKENS=2071 TEACHER_TOKENS=0 TAIL_TEACHER_TOKENS=0 DECODE_TOKENS=8 \
 DSV41_CONTEXT_EXECUTION=sweep DSV41_RUNTIME_WIRED_LIMIT_BYTES=$wired DSV41_RUNTIME_EXPERT_BACKING_DIR="$backing_dir" \
 DSV41_CONTEXT_WALL_BUDGET_SECONDS=900 DSV41_CONTEXT_PROJECTED_WALL_LIMIT_SECONDS=900 \
 CACHE_CONDITION=warm-unknown RUN_CONDITIONS=file-backed-residency-production-short \
 bash tools/benchmark/run_context_32k.sh
 [[ $(<"$dir/exit-code.txt") == 0 ]]; touch "$dir/completed"
}
if [[ "$mode" == bounded ]]; then
 run_variant "$root/bounded/baseline" 0 ""
 run_variant "$root/bounded/candidate" "$limit" "$backing"
 python3 tools/benchmark/compare_file_backed_residency.py "$root/bounded" "$limit" | tee "$root/bounded/comparison.json"
else
 [[ -s "$root/bounded/comparison.json" ]] || { echo 'Run and review bounded mode in this root first.' >&2; exit 2; }
 jq -e '.status=="gates_passed_requires_raw_log_review_not_promotion"' "$root/bounded/comparison.json" >/dev/null
 for spec in '01 baseline candidate' '02 candidate baseline' '03 baseline candidate'; do
  read -r n first second <<<"$spec"; pair="$root/paired/pair-$n"
  for variant in "$first" "$second"; do if [[ "$variant" == candidate ]]; then wired=$limit; backing_dir=$backing; else wired=0; backing_dir=""; fi; run_variant "$pair/$variant" "$wired" "$backing_dir"; done
 done
 python3 tools/benchmark/compare_file_backed_residency.py "$root/paired" "$limit" | tee "$root/paired/comparison.json"
fi
echo 'Completed candidate gate; inspect raw logs, VM compression deltas, latency signs, and host state. This does not promote or qualify 64K.'
