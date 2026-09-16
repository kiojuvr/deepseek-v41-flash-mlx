#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
# One process/model, 256 prompt tokens; only the first sampled token is emitted.
# No decode forward is hidden in the prefill timing.
export CONTEXT_TOKENS=257 TEACHER_TOKENS=0 TAIL_TEACHER_TOKENS=0 DECODE_TOKENS=1
export DSV41_CONTEXT_EXECUTION=layer_major
export DSV41_RUNTIME_PACKED_EXPERT_BANK=1 DSV41_RUNTIME_GROUP_SELECTED_EXPERTS=0
export DSV41_RUNTIME_LAYER_FINITE_CHECKS=0
export DSV41_CONTEXT_WALL_BUDGET_SECONDS=600 DSV41_CONTEXT_PROJECTED_WALL_LIMIT_SECONDS=0
echo 'Scope: one layer-major model, 2x128 prefill, bank construction included, one sampled token.'
echo 'Resources: allow several minutes, budget 80 GB Unified Memory; about 578 GB logical checkpoint reads.'
echo 'Logs: artifacts/context-ladder/32k-run-*/; result.json, progress JSONL and resource.log.'
echo 'Failure: keep logs and rerun this script fresh; state resume is unavailable. Checkpoint is read-only.'
cmake -S . -B build-mlx
bash tools/benchmark/run_context_32k.sh
