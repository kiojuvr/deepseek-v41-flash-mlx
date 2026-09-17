#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

root="artifacts/prefill-gap/metal-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$root"
counter="$root/metal-dispatch-counter.dylib"
xcrun clang++ -std=c++17 -O2 -fobjc-arc -dynamiclib \
  tools/benchmark/metal_dispatch_counter.mm -framework Foundation \
  -framework Metal -o "$counter"
printf '%s\n' \
  'scope=1 model; current official-precision resident path; 2063-token prefill + 1 decode with a process-local Metal selector counter; Instruments is not launched' \
  'resources=allow 5-10 minutes; Unified Memory budget 340 GB; about 289 GB one-time checkpoint reads; no trace package; checkpoint read-only' \
  "logs=$root/{metal-dispatch-counts.json,runtime/{result.json,result.json.progress.jsonl,resource.log,identity.txt},capture-exit-code.txt}" \
  'measurement=command buffers, compute encoders, compute dispatch total, and shape distribution are exact for prefill only; selector-hook overhead invalidates wall-time comparison' \
  'failure=retain the directory; inspect capture-exit-code.txt and runtime/exit-code.txt; a partial run is not a count' \
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
export DSV41_METAL_DISPATCH_COUNTER_OUTPUT="$root/metal-dispatch-counts.json"
export DSV41_METAL_DISPATCH_COUNTER_SCOPED=1
export DSV41_DIRECT_TARGET_DYLD="$counter"
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
test -s "$root/metal-dispatch-counts.json"
python3 -c 'import json,sys; assert json.load(open(sys.argv[1]))["dispatch_total"] > 0' \
  "$root/metal-dispatch-counts.json"
python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["command_buffers"] > 0 and d["compute_encoders"] > 0' \
  "$root/metal-dispatch-counts.json"
echo "Completed; review runtime/result.json and metal-dispatch-counts.json together."
