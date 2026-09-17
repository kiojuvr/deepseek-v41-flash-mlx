#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

printf '%s\n' \
  'scope=1 model; model-lifetime 40-layer resident atlas; one transactional 2063-token layer sweep + 1 reference decode; official checkpoint' \
  'resources=allow 5-10 minutes; Unified Memory budget 340 GB; about 289 GB one-time checkpoint reads; checkpoint read-only' \
  'checks=model construction exactly 40 banks; request sweep zero bank constructions/readbacks; state/logits/generation and full-path wall require review' \
  'isolation=batched dense QMM remains disabled; this measures request-schedule inversion with qualified chunk attention' \
  'logs=artifacts/context-ladder/32k-run-<timestamp>-<pid>/{result.json,result.json.progress.jsonl,test.log,resource.log,identity.txt,tracked.patch,exit-code.txt}' \
  'failure=retain the failed run directory and inspect result/test/resource logs; partial request state is never published or resumed' \
  'resume=unsupported because publication is transactional; rerun this script for a fresh model and request state'

CONTEXT_TOKENS=2064 TEACHER_TOKENS=0 TAIL_TEACHER_TOKENS=0 DECODE_TOKENS=1 \
DSV41_RUNTIME_PACKED_EXPERT_BANK=1 DSV41_RUNTIME_COMPACT_EXPERT_BANK=0 \
DSV41_RUNTIME_GROUP_SELECTED_EXPERTS=0 DSV41_RUNTIME_LAYER_FINITE_CHECKS=0 \
DSV41_RUNTIME_RESIDENT_EXPERT_ATLAS=1 DSV41_RUNTIME_ROUTE_DIAGNOSTICS=0 \
DSV41_RUNTIME_INDEX_DIAGNOSTICS=0 DSV41_RUNTIME_CHUNK_ATTENTION=1 \
DSV41_RUNTIME_LAYER_SWEEP=1 DSV41_RUNTIME_BATCHED_DENSE_QMM=0 \
DSV41_RUNTIME_MLX_CACHE_LIMIT_BYTES=0 DSV41_CONTEXT_EXECUTION=sweep \
DSV41_RUNTIME_EXPERT_IO_THREADS=4 DSV41_RUNTIME_EXPERT_ASSIGNMENT_CHUNK=128 \
DSV41_RUNTIME_COMPONENT_PROFILE=0 \
DSV41_CONTEXT_WALL_BUDGET_SECONDS=1200 DSV41_CONTEXT_PROJECTED_WALL_LIMIT_SECONDS=0 \
CACHE_CONDITION=unknown RUN_CONDITIONS=model-lifetime-resident-transactional-layer-sweep \
bash tools/benchmark/run_context_32k.sh
