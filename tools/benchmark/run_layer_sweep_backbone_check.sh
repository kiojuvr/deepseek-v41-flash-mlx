#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

run_dir="artifacts/prefill-gap/layer-sweep-backbone-$(date +%Y%m%d-%H%M%S)-$$"
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
 'scope=40 layers; 2x128 token-serial oracle chunks vs one transactional 256-token layer sweep; official checkpoint' \
 'resources=allow 5-10 minutes; budget 240 GB Unified Memory; substantial read-only checkpoint expert reads; no swap expected' \
 'gate=relative RMS <0.002 hidden/pre-mix/logits; logits argmax, route ties, persistent state/publication/hash and invalid-request atomicity exact' \
 "logs=$run_dir/{test.log,resource.log,identity.txt,tracked.patch,exit-code.txt}" \
 'failure=retain the failed run directory; inspect test/resource logs; do not promote the sweep' \
 'resume=unsupported; rerun this script for fresh model/request state' | tee "$run_dir/config.txt"
{
 cmake -S . -B build-mlx
 cmake --build build-mlx --target dsv41-text-backbone-test -j 4
} > "$run_dir/build.log" 2>&1
git rev-parse HEAD > "$run_dir/revision.txt"
git diff --binary > "$run_dir/tracked.patch"
shasum -a 256 build-mlx/dsv41-text-backbone-test tests/attention/test_text_backbone.cpp \
 include/dsv41/text_backbone.hpp include/dsv41/text_encoder.hpp include/dsv41/text_decoder.hpp \
 src/model/text_backbone.cpp src/model/text_encoder.cpp src/model/text_decoder.cpp \
 artifacts/checkpoint/summary.json artifacts/engram/metadata.json > "$run_dir/identity.txt"
cmd=(env DSV41_RUNTIME_LAYER_FINITE_CHECKS=0 DSV41_RUNTIME_PACKED_EXPERT_BANK=0 \
 DSV41_RUNTIME_COMPACT_EXPERT_BANK=0 DSV41_RUNTIME_GROUP_SELECTED_EXPERTS=0 \
 DSV41_RUNTIME_INDEX_DIAGNOSTICS=1 DSV41_CHECK_LAYER_MAJOR_BACKBONE=1 \
 DSV41_CHECK_LAYER_SWEEP_BACKBONE=1 \
 build-mlx/dsv41-text-backbone-test "$checkpoint" artifacts/checkpoint/summary.json \
 artifacts/engram/metadata.json)
printf '%q ' "${cmd[@]}" > "$run_dir/command.txt"; printf '\n' >> "$run_dir/command.txt"
/usr/bin/vm_stat > "$run_dir/system-before.txt"
(/usr/bin/time -l "${cmd[@]}") > >(tee "$run_dir/test.log") 2> >(tee "$run_dir/resource.log" >&2)
echo 'Completed; review semantic/state/resource logs before connecting the sweep to 2K production prefill.'
