#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

run_dir="artifacts/mhc/fused-check-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$run_dir"
finish(){
 status=$?
 /usr/bin/vm_stat > "$run_dir/system-after.txt" 2>&1 || true
 echo "$status" > "$run_dir/exit-code.txt"
 if ((status)); then echo "FAILED: inspect $run_dir; retain logs and rerun fresh; no state resume."; fi
}
trap finish EXIT
checkpoint=${CHECKPOINT:-/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash}
printf '%s\n' \
 'scope=40-layer fused-mHC bit parity (2x128 prefill sweep) and advancing-token decode parity; official checkpoint' \
 'resources=allow 5-15 minutes total; budget 340 GB Unified Memory; read-only checkpoint expert reads; no swap expected' \
 'gate=DSV41_RUNTIME_FUSED_MHC=1 hidden/pre-mix/logits/state/publication/hash/revision bit-exact against the reference path' \
 "logs=$run_dir/{backbone.log,decode.log,backbone-resource.log,decode-resource.log,identity.txt,tracked.patch,exit-code.txt}" \
 'failure=retain the failed run directory; inspect logs; do not promote the fused path' \
 'resume=unsupported; rerun this script for fresh model/request state' | tee "$run_dir/config.txt"
{
 cmake -S . -B build-mlx
 cmake --build build-mlx --target dsv41-text-backbone-test -j 4
} > "$run_dir/build.log" 2>&1
git rev-parse HEAD > "$run_dir/revision.txt"
git diff --binary > "$run_dir/tracked.patch"
shasum -a 256 build-mlx/dsv41-text-backbone-test tests/attention/test_text_backbone.cpp \
 include/dsv41/mhc.hpp src/mhc/reference.cpp metal/mhc/split_sinkhorn.metal \
 src/mhc/split_sinkhorn.hpp.in include/dsv41/execution_policy.hpp CMakeLists.txt \
 artifacts/checkpoint/summary.json artifacts/engram/metadata.json > "$run_dir/identity.txt"

base_env=(env DSV41_RUNTIME_LAYER_FINITE_CHECKS=0)

backbone_cmd=("${base_env[@]}" DSV41_CHECK_FUSED_MHC_BACKBONE_PARITY=1 \
 build-mlx/dsv41-text-backbone-test "$checkpoint" artifacts/checkpoint/summary.json \
 artifacts/engram/metadata.json)
printf '%q ' "${backbone_cmd[@]}" > "$run_dir/backbone-command.txt"; printf '\n' >> "$run_dir/backbone-command.txt"
(/usr/bin/time -l "${backbone_cmd[@]}") > >(tee "$run_dir/backbone.log") 2> >(tee "$run_dir/backbone-resource.log" >&2)

decode_cmd=("${base_env[@]}" DSV41_CHECK_FUSED_MHC_DECODE_PARITY=1 \
 build-mlx/dsv41-text-backbone-test "$checkpoint" artifacts/checkpoint/summary.json \
 artifacts/engram/metadata.json)
printf '%q ' "${decode_cmd[@]}" > "$run_dir/decode-command.txt"; printf '\n' >> "$run_dir/decode-command.txt"
(/usr/bin/time -l "${decode_cmd[@]}") > >(tee "$run_dir/decode.log") 2> >(tee "$run_dir/decode-resource.log" >&2)

echo 'Completed; review both PASS lines and resource logs before the performance measurement.'
