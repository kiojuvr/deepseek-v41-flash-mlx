#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

printf '%s\n' \
  'scope=1 model; model-lifetime 40-layer resident atlas; 2063-token shape-bucket prefill + 1 decode; additive attention/MoE/post-MoE GPU-completion wall' \
  'resources=allow 5-10 minutes; Unified Memory budget 340 GB; about 289 GB one-time checkpoint reads; checkpoint read-only' \
  'measurement=component synchronization perturbs normal lazy execution; compare to reviewed resident component profile, not unsynchronized wall' \
  'checks=40 initialization banks; zero warm constructions and route/index readbacks; chunk groups and retained scalar QK/AV dispatches recorded' \
  'logs=artifacts/context-ladder/32k-run-<timestamp>-<pid>/{result.json,result.json.progress.jsonl,resource.log,identity.txt}' \
  'failure=retain the failed run directory and inspect exit-code/test/resource logs; partial atlas/state is never reused' \
  'resume=unsupported because publication is transactional; rerun this script for a fresh model and request state'

CONTEXT_TOKENS=2064 TEACHER_TOKENS=0 TAIL_TEACHER_TOKENS=0 DECODE_TOKENS=1 \
DSV41_RUNTIME_PACKED_EXPERT_BANK=1 DSV41_RUNTIME_COMPACT_EXPERT_BANK=0 \
DSV41_RUNTIME_RESIDENT_EXPERT_ATLAS=1 DSV41_RUNTIME_ROUTE_DIAGNOSTICS=0 \
DSV41_RUNTIME_INDEX_DIAGNOSTICS=0 DSV41_RUNTIME_CHUNK_ATTENTION=1 \
DSV41_RUNTIME_MLX_CACHE_LIMIT_BYTES=0 DSV41_CONTEXT_EXECUTION=layer_major \
DSV41_RUNTIME_EXPERT_IO_THREADS=4 DSV41_RUNTIME_EXPERT_ASSIGNMENT_CHUNK=128 \
DSV41_RUNTIME_COMPONENT_PROFILE=1 DSV41_CONTEXT_WALL_BUDGET_SECONDS=1200 \
DSV41_CONTEXT_PROJECTED_WALL_LIMIT_SECONDS=0 CACHE_CONDITION=unknown \
RUN_CONDITIONS=model-lifetime-resident-shape-bucket-attention-profile \
bash tools/benchmark/run_context_32k.sh
