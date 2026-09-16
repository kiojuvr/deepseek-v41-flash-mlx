#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

# One production-layout candidate model. Chunk 0 constructs the 40 final
# packed banks directly in MLX-owned buffers; chunk 1 must reuse all 40.
export CONTEXT_TOKENS=257 TEACHER_TOKENS=0 TAIL_TEACHER_TOKENS=0 DECODE_TOKENS=1
export DSV41_CONTEXT_EXECUTION=layer_major
export DSV41_RUNTIME_PACKED_EXPERT_BANK=1
export DSV41_RUNTIME_GROUP_SELECTED_EXPERTS=0
export DSV41_RUNTIME_RESIDENT_EXPERT_ATLAS=1
export DSV41_RUNTIME_COMPACT_EXPERT_BANK=0
export DSV41_RUNTIME_LAYER_FINITE_CHECKS=0
export DSV41_CONTEXT_WALL_BUDGET_SECONDS=600
export DSV41_CONTEXT_PROJECTED_WALL_LIMIT_SECONDS=0

echo 'Scope: one model, 40-layer resident expert atlas, 2x128 prefill and one sampled token; expect exactly 40 bank constructions.'
echo 'Resources: allow several minutes; budget 340 GB Unified Memory; about 289 GB logical checkpoint reads; checkpoint read-only.'
echo 'Logs: artifacts/context-ladder/32k-run-*/; result.json, progress JSONL, resource.log and system snapshots.'
echo 'Failure: retain logs and rerun this script fresh; model state and partial resident atlas cannot resume.'
cmake -S . -B build-mlx
bash tools/benchmark/run_context_32k.sh
