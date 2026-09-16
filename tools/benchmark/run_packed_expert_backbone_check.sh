#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
run_dir="artifacts/expert-bank/full-backbone-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$run_dir"
finish(){
 status=$?
 /usr/bin/vm_stat > "$run_dir/system-after.txt" 2>&1 || true
 echo "$status" > "$run_dir/exit-code.txt"
 if ((status)); then echo "FAILED: inspect $run_dir; model state cannot resume; rerun this command fresh."; fi
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
 include/dsv41/execution_policy.hpp include/dsv41/moe.hpp src/moe/reference.cpp \
 src/moe/expert_bank.cpp artifacts/checkpoint/summary.json artifacts/engram/metadata.json \
 > "$run_dir/identity.txt"
printf '%s\n' \
 "scope=one-token 40-layer full-backbone individual-versus-packed expert hidden/pre-mix/state/logits/route-tie bit parity, followed by the standard three-token lifecycle checks" \
 "resource=40 packed expert banks total 288777830400 bytes plus duplicated backbone/state; expect several minutes, roughly 310-340 GB Unified Memory, and about 289 GB checkpoint SSD reads" \
 "checkpoint_read_only=true" \
 "resume=unsupported; retain failed logs and restart fresh" > "$run_dir/config.txt"
cmd=(build-mlx/dsv41-text-backbone-test "$checkpoint" artifacts/checkpoint/summary.json artifacts/engram/metadata.json)
printf 'DSV41_RUNTIME_LAYER_FINITE_CHECKS=0 DSV41_RUNTIME_PACKED_EXPERT_BANK=0 DSV41_RUNTIME_GROUP_SELECTED_EXPERTS=0 DSV41_CHECK_PACKED_EXPERT_BACKBONE_PARITY=1 ' > "$run_dir/command.txt"
printf '%q ' "${cmd[@]}" >> "$run_dir/command.txt";printf '\n' >> "$run_dir/command.txt"
/usr/bin/vm_stat > "$run_dir/system-before.txt"
(
 /usr/bin/time -l env DSV41_RUNTIME_LAYER_FINITE_CHECKS=0 \
  DSV41_RUNTIME_PACKED_EXPERT_BANK=0 \
  DSV41_RUNTIME_GROUP_SELECTED_EXPERTS=0 \
  DSV41_CHECK_PACKED_EXPERT_BACKBONE_PARITY=1 "${cmd[@]}"
) > >(tee "$run_dir/test.log") 2> >(tee "$run_dir/resource.log" >&2)
echo "Completed; full-backbone packed-expert parity/resources require review. This does not qualify performance, 32K, or 256K."
