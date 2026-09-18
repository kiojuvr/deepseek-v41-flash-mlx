#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
run_dir="artifacts/attention/wide-backbone-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$run_dir"
finish(){
 status=$?
 /usr/bin/vm_stat > "$run_dir/system-after.txt" 2>&1 || true
 echo "$status" > "$run_dir/exit-code.txt"
 if ((status)); then echo "FAILED: inspect $run_dir; retain logs; wide attention remains disabled; rerun fresh."; fi
}
trap finish EXIT
echo "Logs: $run_dir"
checkpoint=${CHECKPOINT:-/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash}
printf '%s\n' \
 'scope=40 layers; 2x128 token-serial oracle chunks vs one DwarfStar-style fused attention operation per compressed layer chunk; official checkpoint' \
 'resources=allow 5 minutes; budget 240 GB Unified Memory; approximately 578 GB logical read-only checkpoint expert reads; no swap expected' \
 'gate=relative RMS <0.002 hidden/pre-mix/logits; logits argmax, route ties, persistent state/publication/hash and invalid-request atomicity exact' \
 'topology=QK/mask/online-softmax/BF16-PV/AV fused; fixed [tokens,512] device work metadata; packed pooled cache read directly; no QK or AV child dispatch' \
 'logs=artifacts/attention/wide-backbone-<timestamp>-<pid>/{test.log,resource.log,identity.txt,tracked.patch,exit-code.txt}' \
 'failure=retain the failed run directory; do not promote or run the 2K measurement' \
 'resume=unsupported; rerun this script for fresh model/request state' > "$run_dir/config.txt"
{
 cmake -S . -B build-mlx
 cmake --build build-mlx --target dsv41-text-backbone-test -j 4
} > "$run_dir/build.log" 2>&1
git rev-parse HEAD > "$run_dir/revision.txt"
git diff --binary > "$run_dir/tracked.patch"
shasum -a 256 build-mlx/dsv41-text-backbone-test tests/attention/test_text_backbone.cpp \
 include/dsv41/execution_policy.hpp include/dsv41/swa_attention.hpp include/dsv41/attention_telemetry.hpp \
 src/attention/swa_attention.cpp src/attention/wide_chunk_attention.hpp.in \
 metal/attention/wide_chunk_attention.metal metal/attention/steel_attention_header.metal \
 src/attention/compressed_layer.cpp src/attention/index_query.cpp src/attention/shared_attention.cpp \
 src/model/text_backbone.cpp src/model/text_encoder.cpp src/model/text_decoder.cpp \
 src/model/block.cpp src/model/compressed_block.cpp src/model/reused_block.cpp \
 src/moe/reference.cpp src/moe/expert_bank.cpp metal/moe/route_select.metal metal/moe/route_reduce.metal \
 artifacts/checkpoint/summary.json artifacts/engram/metadata.json > "$run_dir/identity.txt"
cmd=(env DSV41_RUNTIME_LAYER_FINITE_CHECKS=0 DSV41_RUNTIME_PACKED_EXPERT_BANK=0 \
 DSV41_RUNTIME_GROUP_SELECTED_EXPERTS=0 DSV41_RUNTIME_INDEX_DIAGNOSTICS=1 \
 DSV41_RUNTIME_CHUNK_ATTENTION=0 DSV41_RUNTIME_BATCHED_SPLITK_QK=0 \
 DSV41_RUNTIME_PACKED_CHUNK_ATTENTION=0 DSV41_RUNTIME_WIDE_ATTENTION=1 \
 DSV41_CHECK_LAYER_MAJOR_BACKBONE=1 DSV41_CHECK_CHUNK_ATTENTION_BACKBONE=1 \
 build-mlx/dsv41-text-backbone-test "$checkpoint" artifacts/checkpoint/summary.json \
 artifacts/engram/metadata.json)
printf '%q ' "${cmd[@]}" > "$run_dir/command.txt"; printf '\n' >> "$run_dir/command.txt"
/usr/bin/vm_stat > "$run_dir/system-before.txt"
(/usr/bin/time -l "${cmd[@]}") > >(tee "$run_dir/test.log") 2> >(tee "$run_dir/resource.log" >&2)
echo 'Completed; review semantic/state/resource logs before any wide-attention full-path measurement.'
