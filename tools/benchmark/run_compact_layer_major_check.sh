#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
run_dir="artifacts/expert-bank/compact-layer-major-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$run_dir"
finish(){
 status=$?
 /usr/bin/vm_stat > "$run_dir/system-after.txt" 2>&1 || true
 echo "$status" > "$run_dir/exit-code.txt"
 if ((status)); then echo "FAILED: inspect $run_dir; retain logs and rerun fresh; no state resume."; fi
}
trap finish EXIT
echo "Logs: $run_dir"
checkpoint=${CHECKPOINT:-/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash}
{
 cmake -S . -B build-mlx
 cmake --build build-mlx --target dsv41-text-backbone-test -j 4
} > "$run_dir/build.log" 2>&1
git rev-parse HEAD > "$run_dir/revision.txt"
git diff --binary > "$run_dir/tracked.patch"
shasum -a 256 build-mlx/dsv41-text-backbone-test tests/attention/test_text_backbone.cpp \
 include/dsv41/moe.hpp include/dsv41/execution_policy.hpp \
 include/dsv41/swa_layer.hpp include/dsv41/compressed_layer.hpp include/dsv41/reused_layer.hpp \
 include/dsv41/compressor.hpp include/dsv41/global_kv.hpp include/dsv41/index_key.hpp include/dsv41/index_query.hpp \
 src/model/block.cpp src/model/compressed_block.cpp src/model/reused_block.cpp \
 src/attention/swa_layer.cpp src/attention/compressed_layer.cpp src/attention/compressor.cpp \
 src/attention/index_key.cpp src/attention/index_query.cpp src/cache/global_kv.cpp \
 src/moe/reference.cpp src/moe/expert_bank.cpp \
 metal/moe/route_select.metal metal/moe/route_reduce.metal \
 artifacts/checkpoint/summary.json artifacts/engram/metadata.json > "$run_dir/identity.txt"
printf '%s\n' \
 'scope=40 layers; 2x128 tokens; individual reference versus route-first compact banks; exact output/state/publication/ties plus loaded-expert count' \
 'resources=allow several minutes; budget 100 GB Unified Memory; logical checkpoint reads depend on observed route unions and are bounded above by 578 GB; checkpoint read-only' \
 'resume=unsupported; retain failed logs and rerun fresh' > "$run_dir/config.txt"
cmd=(env DSV41_RUNTIME_LAYER_FINITE_CHECKS=0 DSV41_RUNTIME_PACKED_EXPERT_BANK=0 \
 DSV41_RUNTIME_GROUP_SELECTED_EXPERTS=0 DSV41_RUNTIME_RESIDENT_EXPERT_ATLAS=0 \
 DSV41_RUNTIME_COMPACT_EXPERT_BANK=0 DSV41_CHECK_LAYER_MAJOR_BACKBONE=1 \
 DSV41_CHECK_COMPACT_LAYER_MAJOR_BACKBONE=1 \
 build-mlx/dsv41-text-backbone-test "$checkpoint" artifacts/checkpoint/summary.json artifacts/engram/metadata.json)
printf '%q ' "${cmd[@]}" > "$run_dir/command.txt"; printf '\n' >> "$run_dir/command.txt"
/usr/bin/vm_stat > "$run_dir/system-before.txt"
(/usr/bin/time -l "${cmd[@]}") > >(tee "$run_dir/test.log") 2> >(tee "$run_dir/resource.log" >&2)
echo 'Completed; compact full-backbone chunk parity/route-union/resources require review. Performance and 32K remain unqualified.'
