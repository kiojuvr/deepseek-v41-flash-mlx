#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
run_dir="artifacts/generation-lifecycle/run-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$run_dir"
trap 'status=$?; echo "$status" > "$run_dir/exit-code.txt"; if ((status)); then echo "FAILED: inspect $run_dir; rerun the same command for a fresh request/run."; fi' EXIT
echo "Logs: $run_dir"
git rev-parse HEAD > "$run_dir/revision.txt"
git diff --binary > "$run_dir/tracked.patch"
cmake --build build-mlx --target dsv41-text-generate -j 4 > "$run_dir/build.log" 2>&1
checkpoint=${CHECKPOINT:-/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash}
max_new=${MAX_NEW:-4}
if [[ ! "$max_new" =~ ^[0-9]+$ ]] || ((max_new<2 || max_new>262143)); then
 echo "MAX_NEW must be an integer in 2..262143 for the lifecycle check" >&2
 exit 2
fi
args=("$checkpoint" artifacts/checkpoint/summary.json artifacts/engram/metadata.json "$max_new" "${TEMPERATURE:-0}" "${SEED:-0}")
if [[ -n "${TOKENS_FILE:-}" && -n "${PROMPT_LENGTH:-}" ]]; then
 echo "TOKENS_FILE and PROMPT_LENGTH are mutually exclusive" >&2
 exit 2
fi
if [[ -n "${PROMPT_LENGTH:-}" ]]; then
 prompt_length=$PROMPT_LENGTH
 repeat_token=${REPEAT_TOKEN:-42}
 if [[ ! "$prompt_length" =~ ^[0-9]+$ || ! "$repeat_token" =~ ^[0-9]+$ ]] ||
    ((prompt_length<1 || prompt_length>262144-max_new || repeat_token>=129280 || repeat_token==129264)); then
  echo "PROMPT_LENGTH or REPEAT_TOKEN is outside the text generation contract" >&2
  exit 2
 fi
 for ((i=0;i<prompt_length;++i)); do
  ((i)) && printf ' ' >> "$run_dir/prompt.txt"
  printf '%s' "$repeat_token" >> "$run_dir/prompt.txt"
 done
 printf '\n' >> "$run_dir/prompt.txt"
 args+=("$run_dir/prompt.txt")
elif [[ -n "${TOKENS_FILE:-}" ]]; then
 cp "$TOKENS_FILE" "$run_dir/prompt.txt"
 args+=("$run_dir/prompt.txt")
fi
shasum -a 256 build-mlx/dsv41-text-generate artifacts/checkpoint/summary.json artifacts/engram/metadata.json artifacts/engram/fixture-provenance.json include/dsv41/generation_loop.hpp include/dsv41/text_generate.hpp include/dsv41/sampling.hpp src/model/text_generate.cpp src/model/sampling.cpp tools/benchmark/text_generate.cpp > "$run_dir/identity.txt"
if [[ -f "$run_dir/prompt.txt" ]]; then shasum -a 256 "$run_dir/prompt.txt" >> "$run_dir/identity.txt"; fi
printf 'DSV41_CHECK_GENERATION_LIFECYCLE=1 build-mlx/dsv41-text-generate ' > "$run_dir/command.txt"
printf '%q ' "${args[@]}" >> "$run_dir/command.txt"
printf '\n' >> "$run_dir/command.txt"
DSV41_CHECK_GENERATION_LIFECYCLE=1 build-mlx/dsv41-text-generate "${args[@]}" 2>&1 | tee "$run_dir/test.log"
echo "Completed; results need review. This does not qualify the API or 256K runtime."
