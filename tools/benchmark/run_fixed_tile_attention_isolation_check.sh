#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
run_dir="artifacts/attention/fixed-tile-isolation-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$run_dir"
finish(){
 status=$?
 /usr/bin/vm_stat > "$run_dir/system-after.txt" 2>&1 || true
 echo "$status" > "$run_dir/exit-code.txt"
 if ((status)); then echo "FAILED: inspect $run_dir; retain logs; do not rerun the full-backbone gate yet."; fi
}
trap finish EXIT
echo "Logs: $run_dir"
checkpoint=${CHECKPOINT:-/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash}
printf '%s\n' \
 'scope=official layer 2 producer and layer 3 first reuse; token-serial oracle vs fixed-tile 2x128 chunks; no mHC, MoE, or later-layer amplification' \
 'resources=allow 1-2 minutes; Unified Memory target under 40 GB; checkpoint read-only; only two attention layers loaded' \
 'gate=dense-content/exact-shape and dense-reduction/exact-shape diagnostics; relative RMS <0.002 producer/reuse output; publication rows, window state, and positions exact; zero scalar QK' \
 'logs=artifacts/attention/fixed-tile-isolation-<timestamp>-<pid>/{test.log,resource.log,identity.txt,tracked.patch,exit-code.txt}' \
 'failure=retain the failed directory and localize producer versus reuse before changing arithmetic' \
 'resume=unsupported; rerun this script for fresh layer/request state' > "$run_dir/config.txt"
{
 cmake -S . -B build-mlx
 cmake --build build-mlx --target dsv41-swa-attention-test -j 4
} > "$run_dir/build.log" 2>&1
git rev-parse HEAD > "$run_dir/revision.txt"
git diff --binary > "$run_dir/tracked.patch"
shasum -a 256 build-mlx/dsv41-swa-attention-test tests/attention/test_attention.cpp \
 include/dsv41/execution_policy.hpp include/dsv41/swa_attention.hpp include/dsv41/attention_telemetry.hpp \
 src/attention/swa_attention.cpp src/attention/compressed_layer.cpp \
 src/attention/packed_attention_worklist.hpp.in metal/attention/packed_attention_worklist.metal \
 src/attention/batched_splitk_qk.hpp.in metal/attention/batched_splitk_qk.metal \
 metal/attention/batched_splitk_accum.metal metal/attention/steel_gemm_header.metal \
 src/attention/index_query.cpp src/attention/shared_attention.cpp \
 artifacts/checkpoint/summary.json > "$run_dir/identity.txt"
cmd=(env DSV41_RUNTIME_LAYER_FINITE_CHECKS=0 DSV41_RUNTIME_INDEX_DIAGNOSTICS=0 \
 DSV41_RUNTIME_CHUNK_ATTENTION=0 DSV41_RUNTIME_BATCHED_SPLITK_QK=1 \
 DSV41_RUNTIME_PACKED_CHUNK_ATTENTION=0 DSV41_RUNTIME_WIDE_ATTENTION=0 \
 DSV41_RUNTIME_FIXED_TILE_ATTENTION=1 DSV41_RUNTIME_FIXED_TILE_ATTENTION_DIAGNOSTICS=1 \
 DSV41_CHECK_FIXED_TILE_ISOLATION=1 \
 build-mlx/dsv41-swa-attention-test "$checkpoint" artifacts/checkpoint/summary.json)
printf '%q ' "${cmd[@]}" > "$run_dir/command.txt"; printf '\n' >> "$run_dir/command.txt"
/usr/bin/vm_stat > "$run_dir/system-before.txt"
(/usr/bin/time -l "${cmd[@]}") > >(tee "$run_dir/test.log") 2> >(tee "$run_dir/resource.log" >&2)
echo 'Completed; review producer and first-reuse semantic lines before any full-backbone rerun.'
