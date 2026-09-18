#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
run_dir="artifacts/attention/fixed-tile-layer-localization-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$run_dir"
finish(){
 status=$?
 /usr/bin/vm_stat > "$run_dir/system-after.txt" 2>&1 || true
 echo "$status" > "$run_dir/exit-code.txt"
 if ((status)); then echo "FAILED: inspect $run_dir; retain logs; fixed-tile attention remains disabled."; fi
}
trap finish EXIT
echo "Logs: $run_dir"
checkpoint=${CHECKPOINT:-/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash}
printf '%s\n' \
 'scope=one 128-token official 40-layer oracle/candidate chunk; compare attn-in/out, post-attn, ffn-in, MoE-out, hidden, and pre-mix after every layer' \
 'resources=allow 3 minutes; budget 240 GB Unified Memory; substantial read-only checkpoint expert reads; no swap expected' \
 'result=reports first layer/stage crossing RMS 0.002; final state equality is intentionally not asserted after semantic divergence; diagnostic only, not qualification or performance' \
 'logs=artifacts/attention/fixed-tile-layer-localization-<timestamp>-<pid>/{test.log,resource.log,identity.txt,tracked.patch,exit-code.txt}' \
 'failure=retain the failed directory; do not promote or run 2K' \
 'resume=unsupported; rerun this script for fresh model/request state' > "$run_dir/config.txt"
{
 cmake -S . -B build-mlx
 cmake --build build-mlx --target dsv41-text-backbone-test -j 4
} > "$run_dir/build.log" 2>&1
git rev-parse HEAD > "$run_dir/revision.txt"
git diff --binary > "$run_dir/tracked.patch"
shasum -a 256 build-mlx/dsv41-text-backbone-test tests/attention/test_text_backbone.cpp \
 include/dsv41/trace.hpp src/model/trace.cpp src/model/block.cpp \
 src/model/compressed_block.cpp src/model/reused_block.cpp src/attention/compressed_layer.cpp \
 include/dsv41/execution_policy.hpp src/attention/swa_attention.cpp \
 src/attention/ragged_tail_qk.hpp.in metal/attention/ragged_tail_qk.metal \
 metal/attention/ragged_tail_accum.metal metal/attention/packed_attention_worklist.metal \
 artifacts/checkpoint/summary.json \
 artifacts/engram/metadata.json > "$run_dir/identity.txt"
cmd=(env DSV41_RUNTIME_LAYER_FINITE_CHECKS=0 DSV41_RUNTIME_PACKED_EXPERT_BANK=0 \
 DSV41_RUNTIME_GROUP_SELECTED_EXPERTS=0 DSV41_RUNTIME_INDEX_DIAGNOSTICS=1 \
 DSV41_RUNTIME_CHUNK_ATTENTION=0 DSV41_RUNTIME_BATCHED_SPLITK_QK=1 \
 DSV41_RUNTIME_PACKED_CHUNK_ATTENTION=0 DSV41_RUNTIME_WIDE_ATTENTION=0 \
 DSV41_RUNTIME_FIXED_TILE_ATTENTION=1 \
 DSV41_RUNTIME_RAGGED_TAIL_QK=${DSV41_RUNTIME_RAGGED_TAIL_QK:-0} \
 DSV41_CHECK_LAYER_MAJOR_BACKBONE=1 \
 DSV41_CHECK_FIXED_TILE_LAYER_LOCALIZATION=1 \
 build-mlx/dsv41-text-backbone-test "$checkpoint" artifacts/checkpoint/summary.json \
 artifacts/engram/metadata.json)
printf '%q ' "${cmd[@]}" > "$run_dir/command.txt"; printf '\n' >> "$run_dir/command.txt"
/usr/bin/vm_stat > "$run_dir/system-before.txt"
(/usr/bin/time -l "${cmd[@]}") > >(tee "$run_dir/test.log") 2> >(tee "$run_dir/resource.log" >&2)
echo 'Completed; inspect first_gate_failure_layer/stage before changing the fixed schedule.'
