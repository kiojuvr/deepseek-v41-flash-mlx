#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
run_dir="artifacts/expert-bank/compact-backbone-$(date +%Y%m%d-%H%M%S)-$$"
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
cmake --build build-mlx --target dsv41-text-backbone-test -j 4 > "$run_dir/build.log" 2>&1
git rev-parse HEAD > "$run_dir/revision.txt"
git diff --binary > "$run_dir/tracked.patch"
shasum -a 256 build-mlx/dsv41-text-backbone-test tests/attention/test_text_backbone.cpp \
 include/dsv41/moe.hpp include/dsv41/execution_policy.hpp \
 src/moe/reference.cpp src/moe/expert_bank.cpp \
 metal/moe/route_select.metal metal/moe/route_reduce.metal \
 artifacts/checkpoint/summary.json artifacts/engram/metadata.json > "$run_dir/identity.txt"
printf '%s\n' \
 'scope=one-token full-backbone individual versus route-first compact expert banks, followed by the standard 3-token lifecycle check' \
 'resources=typically under one minute; budget 50 GB Unified Memory; about 5 GB compact-bank checkpoint reads plus on-demand reference reads; checkpoint read-only' \
 'resume=unsupported; retain failed logs and rerun fresh' > "$run_dir/config.txt"
cmd=(env DSV41_CHECK_COMPACT_EXPERT_BACKBONE_PARITY=1 \
 DSV41_RUNTIME_PACKED_EXPERT_BANK=0 DSV41_RUNTIME_COMPACT_EXPERT_BANK=0 \
 build-mlx/dsv41-text-backbone-test "$checkpoint" artifacts/checkpoint/summary.json artifacts/engram/metadata.json)
printf '%q ' "${cmd[@]}" > "$run_dir/command.txt"; printf '\n' >> "$run_dir/command.txt"
/usr/bin/vm_stat > "$run_dir/system-before.txt"
(/usr/bin/time -l "${cmd[@]}") > >(tee "$run_dir/test.log") 2> >(tee "$run_dir/resource.log" >&2)
echo 'Completed; compact full-backbone parity/resources require review. Performance, prefill and 32K remain unqualified.'
