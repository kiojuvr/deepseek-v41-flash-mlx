#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

task_root=${DSV41_CED_API_DIR:-artifacts/native-model-api/deferred-decoder-$(date +%Y%m%d-%H%M%S)-$$}
checkpoint=${CHECKPOINT:-/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash}
mkdir -p "$task_root"
finish(){
 status=$?
 echo "$status" > "$task_root/exit-code.txt"
 if ((status)); then
  echo "FAILED: inspect $task_root; completed stages remain reusable, and no partial request state was published."
 fi
}
trap finish EXIT

printf '%s\n' \
 'scope=one 24,576-token native generation request with explicit full-decoder fallback, then no-environment production 16K-begin/8K-resume CED; four greedy decode tokens' \
 'resources=two fresh official-model processes; allow 12-25 minutes; up to 340 GB Unified Memory; checkpoint read-only; no swap expected' \
 'gate=fallback/production-default generated tokens and next_position exact; production default policy; exit 0; swap 0' \
 'logs=<root>/{fallback,production}/{test.log,resource.log,config.txt,identity.txt,tracked.patch,exit-code.txt} plus comparison.txt' \
 'failure=retain both stage directories; source request state is process-private and is never resumed or published after failure' \
 "resume=DSV41_CED_API_DIR=$task_root bash tools/benchmark/run_deferred_decoder_native_api_check.sh; verified exit-0 stages are skipped" \
 > "$task_root/config.txt"
echo "Logs: $task_root"

revision=$(git rev-parse HEAD)
if [[ -f "$task_root/revision.txt" && "$(<"$task_root/revision.txt")" != "$revision" ]]; then
 echo "resume revision mismatch" >&2
 exit 2
fi
echo "$revision" > "$task_root/revision.txt"
patch_sha=$(git diff --binary | shasum -a 256 | awk '{print $1}')
if [[ -f "$task_root/tracked.patch" ]]; then
 existing_sha=$(shasum -a 256 "$task_root/tracked.patch" | awk '{print $1}')
 if [[ "$patch_sha" != "$existing_sha" ]]; then echo "resume tracked patch mismatch" >&2; exit 2; fi
else
 git diff --binary > "$task_root/tracked.patch"
fi

if [[ ! -f "$task_root/prompt.txt" ]]; then
 python3 tools/reference/expand_token_pattern.py \
  --input artifacts/logits-trace/prompt-tokens.txt --output "$task_root/prompt.txt" --length 24576 \
  > "$task_root/prompt-build.log"
fi
cmake --build build-mlx --target dsv41-text-generate -j 4 > "$task_root/build.log" 2>&1
shasum -a 256 build-mlx/dsv41-text-generate include/dsv41/execution_policy.hpp \
 include/dsv41/text_backbone.hpp src/model/text_backbone.cpp src/model/text_generate.cpp \
 tools/benchmark/text_generate.cpp tools/benchmark/run_deferred_decoder_native_api_check.sh \
 "$task_root/prompt.txt" artifacts/checkpoint/summary.json artifacts/engram/metadata.json \
 artifacts/engram/fixture-provenance.json > "$task_root/identity.txt"

verify_stage(){
 local stage=$1 expected_mode=$2 expected_clear=$3 dir="$task_root/$1"
 [[ -f "$dir/exit-code.txt" && "$(<"$dir/exit-code.txt")" == 0 ]] || return 1
 [[ -s "$dir/test.log" && -s "$dir/resource.log" && -s "$dir/identity.txt" ]] || return 1
 cmp -s "$dir/tracked.patch" "$task_root/tracked.patch" || return 1
 grep -qx "deferred_decoder=$expected_mode" "$dir/config.txt" || return 1
 grep -qx "deferred_decoder_clear_cache=$expected_clear" "$dir/config.txt" || return 1
 grep -Eq '^generated: [0-9]+ [0-9]+ [0-9]+ [0-9]+$' "$dir/test.log" || return 1
 grep -qx 'next_position: 24580 stopped: false' "$dir/test.log" || return 1
 grep -Eq '^[[:space:]]+0[[:space:]]+swaps$' "$dir/resource.log"
}

run_stage(){
 local stage=$1 mode=$2 expected_mode=$3 expected_clear=$4 dir="$task_root/$1"
 if verify_stage "$stage" "$expected_mode" "$expected_clear"; then echo "SKIP verified $stage"; return; fi
 mkdir -p "$dir"
 cp "$task_root/tracked.patch" "$dir/tracked.patch"
 cp "$task_root/identity.txt" "$dir/identity.txt"
 printf 'deferred_decoder=%s\ndeferred_decoder_clear_cache=%s\ncontext_tokens=24576\ndecode_tokens=4\ncheckpoint_read_only=true\n' \
  "$expected_mode" "$expected_clear" > "$dir/config.txt"
 echo "START $stage"
 set +e
 if [[ "$mode" == default ]]; then
  (/usr/bin/time -l env -u DSV41_RUNTIME_DEFERRED_DECODER \
    -u DSV41_RUNTIME_DEFERRED_DECODER_CLEAR_CACHE \
    DSV41_RUNTIME_MLX_CACHE_LIMIT_BYTES=0 \
    DSV41_RUNTIME_LONG_CONTEXT_CACHE_LIMIT_BYTES=17179869184 \
    build-mlx/dsv41-text-generate "$checkpoint" artifacts/checkpoint/summary.json \
    artifacts/engram/metadata.json 4 0 0 "$task_root/prompt.txt") \
    > >(tee "$dir/test.log") 2> >(tee "$dir/resource.log" >&2)
  status=$?
 else
  (/usr/bin/time -l env DSV41_RUNTIME_DEFERRED_DECODER=0 \
    DSV41_RUNTIME_DEFERRED_DECODER_CLEAR_CACHE=0 \
    DSV41_RUNTIME_MLX_CACHE_LIMIT_BYTES=0 \
    DSV41_RUNTIME_LONG_CONTEXT_CACHE_LIMIT_BYTES=17179869184 \
    build-mlx/dsv41-text-generate "$checkpoint" artifacts/checkpoint/summary.json \
    artifacts/engram/metadata.json 4 0 0 "$task_root/prompt.txt") \
    > >(tee "$dir/test.log") 2> >(tee "$dir/resource.log" >&2)
  status=$?
 fi
 set -e
 echo "$status" > "$dir/exit-code.txt"
 ((status==0)) || return "$status"
 verify_stage "$stage" "$expected_mode" "$expected_clear"
}

run_stage fallback fallback 0 0
run_stage production default 1 1

awk '/^generated:/{print}' "$task_root/fallback/test.log" > "$task_root/fallback-generated.txt"
awk '/^generated:/{print}' "$task_root/production/test.log" > "$task_root/production-generated.txt"
cmp "$task_root/fallback-generated.txt" "$task_root/production-generated.txt"
awk '/^next_position:/{print}' "$task_root/fallback/test.log" > "$task_root/fallback-position.txt"
awk '/^next_position:/{print}' "$task_root/production/test.log" > "$task_root/production-position.txt"
cmp "$task_root/fallback-position.txt" "$task_root/production-position.txt"
printf 'PASS: production-default pending CED native generation matches fallback tokens and state at 24K\n' \
 | tee "$task_root/comparison.txt"
echo "Completed; review both result/resource/config/identity logs before calling native API promotion qualified."
