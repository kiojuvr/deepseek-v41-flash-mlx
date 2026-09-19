#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

root=${DSV41_PAIRED_RUN_DIR:-artifacts/context-ladder/fixed-tile-paired-$(date +%Y%m%d-%H%M%S)-$$}
pairs=${DSV41_PAIRED_PAIRS:-5}
if [[ ! "$pairs" =~ ^[0-9]+$ ]] || ((pairs<5)); then
 echo "DSV41_PAIRED_PAIRS must be an integer of at least 5" >&2
 exit 2
fi
if [[ "${DSV41_PAIRED_DRY_RUN:-0}" == 1 ]]; then
 echo "warmup-baseline"
 echo "warmup-candidate"
 for ((pair=1;pair<=pairs;pair++)); do
  if ((pair%2)); then echo "pair-$pair: baseline candidate"; else echo "pair-$pair: candidate baseline"; fi
 done
 echo "DRY RUN: alternating schedule prepared; no model process was started."
 exit 0
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
 echo "paired qualification requires a clean tracked worktree" >&2
 exit 2
fi
mkdir -p "$root"
finish(){
 status=$?
 echo "$status" > "$root/exit-code.txt"
 if ((status)); then
  echo "FAILED: inspect $root; resume completed independent runs with DSV41_PAIRED_RUN_DIR=$root"
 fi
}
trap finish EXIT

printf '%s\n' \
 "scope=two warmups followed by $pairs alternating warm baseline/candidate pairs; each run is one 2063-token transactional prefill plus one decode" \
 'baseline=qualified packed exact-shape attention; candidate=qualified fixed-tile attention with ragged QK/AV and request-boundary graph' \
 'resources=allow 60-120 minutes for the default 12 sequential processes; peak Unified Memory budget 340 GB; about 3.5 TB aggregate logical read-only checkpoint reads; no swap expected' \
 'acceptance=review per-pair prefill wall/throughput, decode, memory, swap, next token, topology telemetry, identities, and run-order variance; this runner does not auto-promote' \
 'logs=<root>/{warmup-*,pair-*/{baseline,candidate},measurements.tsv,identity.txt,tracked.patch,exit-code.txt}; each child retains canonical result/resource logs' \
 'failure=stop at the first failed child and retain all completed runs' \
 "resume=DSV41_PAIRED_RUN_DIR=$root bash tools/benchmark/run_fixed_tile_attention_paired_qualification.sh; completed exit-0 children are verified and skipped" > "$root/config.txt"
revision=$(git rev-parse HEAD)
if [[ -f "$root/revision.txt" && "$(<"$root/revision.txt")" != "$revision" ]]; then
 echo "resume revision mismatch: root=$(<"$root/revision.txt") current=$revision" >&2
 exit 2
fi
printf '%s\n' "$revision" > "$root/revision.txt"
git diff --binary > "$root/tracked.patch"
shasum -a 256 tools/benchmark/run_fixed_tile_attention_paired_qualification.sh \
 tools/benchmark/run_fixed_tile_attention_prefill_measurement.sh \
 tools/benchmark/run_packed_attention_prefill_measurement.sh \
 tools/benchmark/run_context_32k.sh build-mlx/dsv41-context-ladder > "$root/identity.txt"
echo "Logs: $root"

run_one(){
 local label=$1 runner=$2 run_dir="$root/$1"
 if [[ -f "$run_dir/exit-code.txt" && "$(<"$run_dir/exit-code.txt")" == 0 &&
       -s "$run_dir/result.json" && -f "$run_dir/tracked.patch" && ! -s "$run_dir/tracked.patch" &&
       -f "$run_dir/revision.txt" && "$(<"$run_dir/revision.txt")" == "$revision" ]] &&
    jq -e '.status=="measurement_completed_requires_review" and .phases.decode.generated_token_ids==[339]' \
      "$run_dir/result.json" >/dev/null &&
    grep -Eq '^[[:space:]]+0[[:space:]]+swaps$' "$run_dir/resource.log"; then
  echo "Resume: verified and skipped $label"
  return
 fi
 echo "Starting $label"
 mkdir -p "$(dirname "$run_dir")"
 DSV41_CONTEXT_RUN_DIR="$run_dir" CACHE_CONDITION=warm-alternating \
 RUN_CONDITIONS="fixed-tile-paired-qualification-$label" \
 bash "$runner" 2>&1 | tee "$root/$label.launch.log"
}

baseline=tools/benchmark/run_packed_attention_prefill_measurement.sh
candidate=tools/benchmark/run_fixed_tile_attention_prefill_measurement.sh
run_one warmup-baseline "$baseline"
run_one warmup-candidate "$candidate"
for ((pair=1;pair<=pairs;pair++)); do
 if ((pair%2)); then order=(baseline candidate); else order=(candidate baseline); fi
 for variant in "${order[@]}"; do
  if [[ "$variant" == baseline ]]; then runner=$baseline; else runner=$candidate; fi
  run_one "pair-$pair/$variant" "$runner"
 done
done

summary="$root/measurements.tsv"
printf 'pair\tvariant\tprefill_seconds\tprefill_tps\tdecode_seconds\tnext_token\tpeak_bytes\n' > "$summary"
for ((pair=1;pair<=pairs;pair++)); do
 for variant in baseline candidate; do
  result="$root/pair-$pair/$variant/result.json"
  jq -r --arg pair "$pair" --arg variant "$variant" \
   '[$pair,$variant,.aggregates.prefill_seconds,.aggregates.prefill_tokens_per_second,.phases.decode.first_token_seconds,.phases.decode.generated_token_ids[0],.final_memory.peak_bytes]|@tsv' \
   "$result" >> "$summary"
 done
done
echo "Completed; review $summary and every child resource/config/identity log before changing runtime defaults."
