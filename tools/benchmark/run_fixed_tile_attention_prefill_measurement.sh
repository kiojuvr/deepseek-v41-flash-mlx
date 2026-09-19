#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

printf '%s\n' \
 'scope=1 model; resident official layout; one transactional 2063-token sweep + 1 reference decode; qualified fixed-tile attention production path' \
 'resources=allow 5-10 minutes; Unified Memory budget 340 GB; about 289 GB one-time checkpoint reads; checkpoint read-only' \
 'checks=40 model-lifetime banks; zero request bank construction/readback; fixed device work list; ragged QK/AV and request-boundary graph enabled' \
 'prerequisite=run only after tools/benchmark/run_fixed_tile_attention_backbone_check.sh passes and its semantic/state/resource logs are reviewed' \
 'logs=artifacts/context-ladder/32k-run-<timestamp>-<pid>/{result.json,result.json.progress.jsonl,test.log,resource.log,config.txt,identity.txt,tracked.patch,exit-code.txt}' \
 'failure=retain the failed run directory and inspect result/test/resource logs; partial request state is never published or resumed' \
 'resume=unsupported because publication is transactional; rerun this script for a fresh model and request state'

CONTEXT_TOKENS=2064 TEACHER_TOKENS=0 TAIL_TEACHER_TOKENS=0 DECODE_TOKENS=1 \
DSV41_RUNTIME_PACKED_EXPERT_BANK=1 DSV41_RUNTIME_COMPACT_EXPERT_BANK=0 \
DSV41_RUNTIME_GROUP_SELECTED_EXPERTS=0 DSV41_RUNTIME_LAYER_FINITE_CHECKS=0 \
DSV41_RUNTIME_RESIDENT_EXPERT_ATLAS=1 DSV41_RUNTIME_ROUTE_DIAGNOSTICS=0 \
DSV41_RUNTIME_INDEX_DIAGNOSTICS=0 DSV41_RUNTIME_CHUNK_ATTENTION=0 \
DSV41_RUNTIME_BATCHED_SPLITK_QK=1 DSV41_RUNTIME_PACKED_CHUNK_ATTENTION=0 \
DSV41_RUNTIME_WIDE_ATTENTION=0 DSV41_RUNTIME_FIXED_TILE_ATTENTION=1 \
DSV41_RUNTIME_RAGGED_TAIL_QK=1 DSV41_RUNTIME_RAGGED_TAIL_AV=1 \
DSV41_RUNTIME_LAYER_SWEEP=1 DSV41_RUNTIME_BATCHED_DENSE_QMM=0 \
DSV41_RUNTIME_MLX_CACHE_LIMIT_BYTES=0 DSV41_CONTEXT_EXECUTION=sweep \
DSV41_RUNTIME_EXPERT_IO_THREADS=4 DSV41_RUNTIME_EXPERT_ASSIGNMENT_CHUNK=128 \
DSV41_RUNTIME_COMPONENT_PROFILE=0 DSV41_CONTEXT_WALL_BUDGET_SECONDS=1200 \
DSV41_CONTEXT_PROJECTED_WALL_LIMIT_SECONDS=0 CACHE_CONDITION=${CACHE_CONDITION:-unknown} \
RUN_CONDITIONS=${RUN_CONDITIONS:-resident-transactional-sweep-qualified-fixed-tile-attention} \
bash tools/benchmark/run_context_32k.sh
