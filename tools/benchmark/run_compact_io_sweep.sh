#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
root="artifacts/context-ladder/compact-io-sweep-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$root"
echo "Logs: $root"
printf '%s\n' \
 'scope=independent compact-only layer-major processes; 257-token context (256 prefill + 1 decode); worker sweep 1,2,4,8' \
 'resources=each run typically under two minutes; cache limit 8 GiB; Unified Memory target under 30 GB; checkpoint read-only' \
 'resume=each worker run is independent; retain failed subdirectory and rerun the sweep fresh' > "$root/config.txt"
for workers in 1 2 4 8; do
  run_dir="$root/workers-$workers"
  mkdir -p "$run_dir"
  echo "Starting workers=$workers"
  CONTEXT_TOKENS=257 TEACHER_TOKENS=0 TAIL_TEACHER_TOKENS=0 DECODE_TOKENS=1 \
  DSV41_RUNTIME_PACKED_EXPERT_BANK=1 DSV41_RUNTIME_COMPACT_EXPERT_BANK=1 \
  DSV41_RUNTIME_RESIDENT_EXPERT_ATLAS=0 DSV41_RUNTIME_MLX_CACHE_LIMIT_BYTES=8589934592 \
  DSV41_CONTEXT_EXECUTION=layer_major DSV41_RUNTIME_EXPERT_IO_THREADS="$workers" \
  DSV41_CONTEXT_WALL_BUDGET_SECONDS=300 DSV41_CONTEXT_PROJECTED_WALL_LIMIT_SECONDS=0 \
  bash tools/benchmark/run_context_32k.sh > "$run_dir/test.log" 2> "$run_dir/stderr.log" || {
    status=$?
    echo "$status" > "$run_dir/exit-code.txt"
    echo "FAILED workers=$workers; inspect $run_dir and continue with the remaining sweep"
    continue
  }
  echo 0 > "$run_dir/exit-code.txt"
done
echo 'Completed; compare each result.json and resource.log. No performance qualification is implied.'
