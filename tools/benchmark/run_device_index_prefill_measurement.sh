#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

printf '%s\n' \
  'scope=1 model; compact-only layer-major; 2063-token prefill + 1 decode; post-device-index measurement' \
  'resources=allow about 5 minutes; Unified Memory target under 30 GB (budget 100 GB); checkpoint read-only; sustained SSD reads' \
  'logs=artifacts/context-ladder/32k-run-<timestamp>-<pid>/{result.json,result.json.progress.jsonl,resource.log,identity.txt}' \
  'failure=retain the failed run directory and inspect exit-code/test/resource logs' \
  'resume=unsupported; rerun this script for a fresh model state'

CONTEXT_TOKENS=2064 TEACHER_TOKENS=0 TAIL_TEACHER_TOKENS=0 DECODE_TOKENS=1 \
DSV41_RUNTIME_PACKED_EXPERT_BANK=1 DSV41_RUNTIME_COMPACT_EXPERT_BANK=1 \
DSV41_RUNTIME_RESIDENT_EXPERT_ATLAS=0 DSV41_RUNTIME_MLX_CACHE_LIMIT_BYTES=8589934592 \
DSV41_CONTEXT_EXECUTION=layer_major DSV41_RUNTIME_EXPERT_IO_THREADS=4 \
DSV41_RUNTIME_EXPERT_ASSIGNMENT_CHUNK=128 \
DSV41_CONTEXT_WALL_BUDGET_SECONDS=900 DSV41_CONTEXT_PROJECTED_WALL_LIMIT_SECONDS=0 \
CACHE_CONDITION=unknown RUN_CONDITIONS=post-device-index-selection \
bash tools/benchmark/run_context_32k.sh
