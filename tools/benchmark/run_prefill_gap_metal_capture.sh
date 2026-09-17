#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

root="artifacts/prefill-gap/metal-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$root"
printf '%s\n' \
  'scope=1 model; current official-precision resident path; 2063-token prefill + 1 decode under focused Metal Application + GPU instruments' \
  'resources=allow 10-20 minutes; Unified Memory budget 340 GB; about 289 GB one-time checkpoint reads; focused trace is expected to remain far below a full System Trace; checkpoint read-only' \
  "logs=$root/{metal.trace,runtime/{result.json,result.json.progress.jsonl,resource.log,identity.txt},capture-exit-code.txt}" \
  'measurement=trace overhead invalidates wall-time comparison; command-buffer and encoder counts are exact; kernel dispatch remains unresolved unless an explicit dispatch table is present' \
  'failure=retain the entire directory; inspect capture-exit-code.txt and runtime/exit-code.txt; a partial trace is not a count' \
  'resume=unsupported; rerun this script with fresh model/request state because publication is transactional'

finish() {
 status=$?
 printf '%s\n' "$status" > "$root/capture-exit-code.txt"
 if ((status)); then
  echo "FAILED: retain and inspect $root; do not report partial trace counts." >&2
 fi
}
trap finish EXIT

export DSV41_CONTEXT_RUN_DIR="$root/runtime"
export DSV41_XCTRACE_OUTPUT="$root/metal.trace"
export CONTEXT_TOKENS=2064 TEACHER_TOKENS=0 TAIL_TEACHER_TOKENS=0 DECODE_TOKENS=1
export DSV41_RUNTIME_PACKED_EXPERT_BANK=1 DSV41_RUNTIME_COMPACT_EXPERT_BANK=0
export DSV41_RUNTIME_GROUP_SELECTED_EXPERTS=0 DSV41_RUNTIME_LAYER_FINITE_CHECKS=0
export DSV41_RUNTIME_RESIDENT_EXPERT_ATLAS=1 DSV41_RUNTIME_ROUTE_DIAGNOSTICS=0
export DSV41_RUNTIME_INDEX_DIAGNOSTICS=0 DSV41_RUNTIME_CHUNK_ATTENTION=1
export DSV41_RUNTIME_MLX_CACHE_LIMIT_BYTES=0 DSV41_CONTEXT_EXECUTION=layer_major
export DSV41_RUNTIME_EXPERT_IO_THREADS=4 DSV41_RUNTIME_EXPERT_ASSIGNMENT_CHUNK=128
export DSV41_RUNTIME_COMPONENT_PROFILE=0 DSV41_CONTEXT_WALL_BUDGET_SECONDS=1200
export DSV41_CONTEXT_PROJECTED_WALL_LIMIT_SECONDS=0 CACHE_CONDITION=unknown
export RUN_CONDITIONS=prefill-gap-focused-metal-trace

bash tools/benchmark/run_context_32k.sh
echo "Completed; review the runtime result and summarize the trace. Never substitute encoder count for kernel dispatch count."
