#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
run_dir="artifacts/expert-bank/run-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$run_dir"
finish(){
 status=$?
 /usr/bin/vm_stat > "$run_dir/system-after.txt" 2>&1 || true
 echo "$status" > "$run_dir/exit-code.txt"
 if ((status)); then echo "FAILED: inspect $run_dir; bank state cannot resume; rerun this command fresh."; fi
}
trap finish EXIT
echo "Logs: $run_dir"
checkpoint=${CHECKPOINT:-/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash}
{
 cmake -S . -B build-mlx
 cmake --build build-mlx --target dsv41-expert-bank-probe -j 4
} > "$run_dir/build.log" 2>&1
git rev-parse HEAD > "$run_dir/revision.txt"
git diff --binary > "$run_dir/tracked.patch"
shasum -a 256 build-mlx/dsv41-expert-bank-probe include/dsv41/moe.hpp src/moe/expert_bank.cpp \
 src/moe/reference.cpp src/model/linear.cpp tools/benchmark/expert_bank_probe.cpp \
 artifacts/checkpoint/summary.json > "$run_dir/identity.txt"
printf '%s\n' \
 "scope=layer-0 384-expert packed bank build; six selected expert intermediate/output bit parity; nine warm timing pairs" \
 "resource=about 7.2 GB persistent packed bank plus construction temporaries and checkpoint SSD reads; allow several minutes" \
 "checkpoint_read_only=true" "resume=unsupported; retain failed logs and restart fresh" > "$run_dir/config.txt"
cmd=(build-mlx/dsv41-expert-bank-probe "$checkpoint" artifacts/checkpoint/summary.json "$run_dir/result.json")
printf '%q ' "${cmd[@]}" > "$run_dir/command.txt";printf '\n' >> "$run_dir/command.txt"
/usr/bin/vm_stat > "$run_dir/system-before.txt"
(/usr/bin/time -l "${cmd[@]}") > >(tee "$run_dir/test.log") 2> >(tee "$run_dir/resource.log" >&2)
echo "Completed; bank parity/memory/timing require review. This does not qualify full MoE, performance, 32K, or 256K."
