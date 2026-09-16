#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
run_dir="artifacts/layer-finite-policy/run-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$run_dir"
finish(){
 status=$?
 /usr/bin/vm_stat > "$run_dir/system-after.txt" 2>&1 || true
 echo "$status" > "$run_dir/exit-code.txt"
 if ((status)); then echo "FAILED: inspect $run_dir; rerun this command for a fresh run."; fi
}
trap finish EXIT
echo "Logs: $run_dir"

checkpoint=${CHECKPOINT:-/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash}
cmake --build build-mlx --target dsv41-text-backbone-test -j 4 > "$run_dir/build.log" 2>&1
git rev-parse HEAD > "$run_dir/revision.txt"
git diff --binary > "$run_dir/tracked.patch"
shasum -a 256 build-mlx/dsv41-text-backbone-test tests/attention/test_text_backbone.cpp \
 include/dsv41/execution_policy.hpp src/model/block.cpp src/model/compressed_block.cpp \
 src/model/reused_block.cpp src/attention/swa_layer.cpp src/attention/compressed_layer.cpp \
 src/attention/compressor.cpp include/dsv41/moe.hpp src/moe/reference.cpp > "$run_dir/identity.txt"
printf '%s\n' \
 "scope=3-token full-backbone bit parity with per-layer finite checks enabled versus disabled" \
 "resource=full 40-layer checkpoint, on-demand experts, Engram mmap; expect several minutes and high Unified Memory/SSD use" \
 "resume=not supported; failed runs retain logs and must restart fresh" > "$run_dir/config.txt"
/usr/bin/vm_stat > "$run_dir/system-before.txt"
cmd=(build-mlx/dsv41-text-backbone-test "$checkpoint" artifacts/checkpoint/summary.json artifacts/engram/metadata.json)
printf 'DSV41_RUNTIME_LAYER_FINITE_CHECKS=1 DSV41_CHECK_LAYER_FINITE_POLICY_PARITY=1 ' > "$run_dir/command.txt"
printf '%q ' "${cmd[@]}" >> "$run_dir/command.txt"
printf '\n' >> "$run_dir/command.txt"
(
 /usr/bin/time -l env DSV41_RUNTIME_LAYER_FINITE_CHECKS=1 \
  DSV41_CHECK_LAYER_FINITE_POLICY_PARITY=1 "${cmd[@]}"
) > >(tee "$run_dir/test.log") 2> >(tee "$run_dir/resource.log" >&2)
echo "Completed; bit-parity and resource results require review. This does not qualify performance or 32K."
