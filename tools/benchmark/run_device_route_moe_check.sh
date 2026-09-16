#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
run_dir="artifacts/expert-bank/device-route-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$run_dir"
finish(){
 status=$?
 echo "$status" > "$run_dir/exit-code.txt"
 if ((status)); then echo "FAILED: inspect $run_dir; rerun this script fresh."; fi
}
trap finish EXIT
echo "Logs: $run_dir"
checkpoint=${CHECKPOINT:-/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash}
cmake --build build-mlx --target dsv41-route-policy-test dsv41-moe-batch-probe -j 4 \
 > "$run_dir/build.log" 2>&1
git rev-parse HEAD > "$run_dir/revision.txt"
git diff --binary > "$run_dir/tracked.patch"
shasum -a 256 build-mlx/dsv41-route-policy-test build-mlx/dsv41-moe-batch-probe \
 include/dsv41/moe.hpp src/moe/reference.cpp src/moe/expert_bank.cpp \
 metal/moe/route_select.metal metal/moe/route_reduce.metal \
 tests/attention/test_route_policy.cpp tools/benchmark/moe_batch_probe.cpp \
 artifacts/checkpoint/summary.json > "$run_dir/identity.txt"
printf '%s\n' \
 'scope=synthetic CPU/device route parity plus layer-0 128-token full MoE serial parity and bounded warm timing' \
 'resources=seconds after build; approximately 10 GB Unified Memory; one layer expert-bank checkpoint read; checkpoint read-only' \
 'resume=not needed; retain failed logs and rerun fresh' > "$run_dir/config.txt"
build-mlx/dsv41-route-policy-test > "$run_dir/route-policy.log" 2>&1
/usr/bin/time -l build-mlx/dsv41-moe-batch-probe "$checkpoint" \
 artifacts/checkpoint/summary.json "$run_dir/result.json" \
 > "$run_dir/test.log" 2> "$run_dir/resource.log"
sed -n '1,80p' "$run_dir/route-policy.log"
sed -n '1,80p' "$run_dir/test.log"
sed -n '1,120p' "$run_dir/resource.log" >&2
echo 'Completed; device routing/reduction parity and bounded timing require review; not full-path qualification.'
