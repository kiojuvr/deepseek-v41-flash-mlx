#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

run_dir="artifacts/prefill-gap/deferred-pending-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$run_dir"
finish(){
 status=$?
 /usr/bin/vm_stat > "$run_dir/system-after.txt" 2>&1 || true
 echo "$status" > "$run_dir/exit-code.txt"
 if ((status)); then
  echo "FAILED: inspect $run_dir; the private encoder/decoder frontier was not published; rerun fresh."
 fi
}
trap finish EXIT
checkpoint=${CHECKPOINT:-/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash}
printf '%s\n' \
 'scope=one 24,576-token full packed sweep versus DwarfStar-style private 16K encoder-only sweep plus exact 8K decoder resume' \
 'resources=allow 15-35 minutes; up to 340 GB Unified Memory; substantial read-only checkpoint expert reads; no swap expected' \
 'gate=bit-exact final hidden/pre-mix/logits, executed route ties, all encoder/decoder persistent state/publication/hash; source revision atomicity and reuse rejection' \
 "logs=$run_dir/{test.log,resource.log,identity.txt,tracked.patch,exit-code.txt}" \
 'failure=retain logs; source request state remains unchanged; pending CED stays disconnected from production default' \
 'resume=unsupported because publication is transactional; rerun this script with fresh model/request state' \
 | tee "$run_dir/config.txt"
{
 cmake -S . -B build-mlx
 cmake --build build-mlx --target dsv41-text-backbone-test -j 4
} > "$run_dir/build.log" 2>&1
git rev-parse HEAD > "$run_dir/revision.txt"
git diff --binary > "$run_dir/tracked.patch"
shasum -a 256 build-mlx/dsv41-text-backbone-test tests/attention/test_text_backbone.cpp \
 include/dsv41/execution_policy.hpp include/dsv41/deferred_decoder_plan.hpp \
 include/dsv41/text_backbone.hpp include/dsv41/text_encoder.hpp include/dsv41/text_decoder.hpp \
 include/dsv41/compressed_block.hpp include/dsv41/reused_block.hpp \
 include/dsv41/compressed_layer.hpp include/dsv41/reused_layer.hpp \
 include/dsv41/shared_attention.hpp src/attention/shared_attention.cpp \
 src/model/text_backbone.cpp src/model/text_encoder.cpp src/model/text_decoder.cpp \
 src/model/compressed_block.cpp src/model/reused_block.cpp src/attention/compressed_layer.cpp \
 artifacts/checkpoint/summary.json artifacts/engram/metadata.json > "$run_dir/identity.txt"
cmd=(env DSV41_RUNTIME_LAYER_FINITE_CHECKS=0 DSV41_RUNTIME_PACKED_EXPERT_BANK=0 \
 DSV41_RUNTIME_RESIDENT_EXPERT_ATLAS=0 DSV41_RUNTIME_COMPACT_EXPERT_BANK=0 \
 DSV41_RUNTIME_GROUP_SELECTED_EXPERTS=0 DSV41_RUNTIME_ROUTE_DIAGNOSTICS=1 \
 DSV41_RUNTIME_INDEX_DIAGNOSTICS=1 DSV41_CHECK_LAYER_MAJOR_BACKBONE=1 \
 DSV41_CHECK_DEFERRED_PENDING=1 build-mlx/dsv41-text-backbone-test \
 "$checkpoint" artifacts/checkpoint/summary.json artifacts/engram/metadata.json)
printf '%q ' "${cmd[@]}" > "$run_dir/command.txt"; printf '\n' >> "$run_dir/command.txt"
/usr/bin/vm_stat > "$run_dir/system-before.txt"
(/usr/bin/time -l "${cmd[@]}") > >(tee "$run_dir/test.log") 2> >(tee "$run_dir/resource.log" >&2)
echo 'Completed; review exact outputs, complete state/publication/hash, route ties, and resources before any pending-CED measurement.'
