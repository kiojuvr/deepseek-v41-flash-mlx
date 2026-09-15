#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
run_dir="artifacts/text-backbone/run-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$run_dir"
trap 'status=$?; echo "$status" > "$run_dir/exit-code.txt"; if ((status)); then echo "FAILED: inspect $run_dir; rerun the same command for a fresh run."; fi' EXIT
echo "Logs: $run_dir"
checkpoint=${CHECKPOINT:-/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash}
args=("$checkpoint" artifacts/checkpoint/summary.json artifacts/engram/metadata.json)
if [[ -n "${TOKENS_FILE:-}" && ( -n "${PROMPT_LENGTH:-}" || -n "${PROMPT_PATTERN_FILE:-}" ) ]]; then
 echo "TOKENS_FILE is mutually exclusive with PROMPT_LENGTH/PROMPT_PATTERN_FILE" >&2
 exit 2
fi
if [[ -n "${PROMPT_PATTERN_FILE:-}" && -z "${PROMPT_LENGTH:-}" ]]; then
 echo "PROMPT_PATTERN_FILE requires PROMPT_LENGTH" >&2
 exit 2
fi
if [[ -n "${PROMPT_LENGTH:-}" ]]; then
 prompt_length=$PROMPT_LENGTH
 if [[ ! "$prompt_length" =~ ^[0-9]+$ ]] || ((prompt_length<1 || prompt_length>262144)); then
  echo "PROMPT_LENGTH is outside the text backbone contract" >&2
  exit 2
 fi
 if [[ -n "${PROMPT_PATTERN_FILE:-}" ]]; then
  "${PYTHON:-python3}" tools/reference/expand_token_pattern.py --input "$PROMPT_PATTERN_FILE" --output "$run_dir/prompt.txt" --length "$prompt_length" > "$run_dir/prompt-build.log"
 else
  repeat_token=${REPEAT_TOKEN:-42}
  if [[ ! "$repeat_token" =~ ^[0-9]+$ ]] || ((repeat_token>=129280 || repeat_token==129264)); then
   echo "REPEAT_TOKEN is outside the text backbone contract" >&2
   exit 2
  fi
  for ((i=0;i<prompt_length;++i)); do
   ((i)) && printf ' ' >> "$run_dir/prompt.txt"
   printf '%s' "$repeat_token" >> "$run_dir/prompt.txt"
  done
  printf '\n' >> "$run_dir/prompt.txt"
 fi
 args+=("$run_dir/prompt.txt")
elif [[ -n "${TOKENS_FILE:-}" ]]; then
 cp "$TOKENS_FILE" "$run_dir/prompt.txt"
 args+=("$run_dir/prompt.txt")
fi
git rev-parse HEAD > "$run_dir/revision.txt"
git diff --binary > "$run_dir/tracked.patch"
cmake --build build-mlx --target dsv41-text-backbone-test -j 4 > "$run_dir/build.log" 2>&1
shasum -a 256 build-mlx/dsv41-text-backbone-test artifacts/checkpoint/summary.json artifacts/engram/metadata.json artifacts/engram/fixture-provenance.json include/dsv41/generation_loop.hpp include/dsv41/text_backbone.hpp src/model/text_backbone.cpp tests/attention/test_text_backbone.cpp > "$run_dir/identity.txt"
if [[ -f "$run_dir/prompt.txt" ]]; then shasum -a 256 "$run_dir/prompt.txt" >> "$run_dir/identity.txt"; fi
if [[ -n "${PROMPT_PATTERN_FILE:-}" ]]; then shasum -a 256 "$PROMPT_PATTERN_FILE" tools/reference/expand_token_pattern.py >> "$run_dir/identity.txt"; fi
printf 'build-mlx/dsv41-text-backbone-test ' > "$run_dir/command.txt"
printf '%q ' "${args[@]}" >> "$run_dir/command.txt"
printf '\n' >> "$run_dir/command.txt"
build-mlx/dsv41-text-backbone-test "${args[@]}" 2>&1 | tee "$run_dir/test.log"
echo "Completed. Chunk/token tensor and state results require review; external oracle and 256K remain unqualified."
