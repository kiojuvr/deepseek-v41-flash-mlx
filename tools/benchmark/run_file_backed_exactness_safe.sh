#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mode=${1:-baseline};[[ "$mode" == baseline || "$mode" == candidate || "$mode" == compare ]] || { echo 'usage: bash tools/benchmark/run_file_backed_exactness_safe.sh [baseline|candidate|compare]' >&2;exit 2; }
root=${DSV41_FILE_EXACT_DIR:-artifacts/context-ladder/file-backed-exact-$(date +%Y%m%d-%H%M%S)-$$}
backing=${DSV41_EXPERT_BACKING_DIR:-artifacts/expert-backing/official-packed-v1};limit=${DSV41_FILE_RESIDENCY_BYTES:-292057776128}
mkdir -p "$root"
printf '%s\n' \
 'scope=process-separated exactness only: compact request-local reference emits a 129+4 boundary/state digest; candidate emits the same digest from file-backed+272 GiB residency; compare is offline' \
 'resources=compact baseline approximately 2-8 minutes with bounded expert-bank memory; candidate may take 15-40 minutes on cold file backing; one process per invocation; no model overlap' \
 'safety=requires >=400 GiB conservative free+inactive (speculative excluded), zero active swap usage, no process >32 GiB; zero process/system swap delta; stops after each invocation' \
 "backing=$backing wired_limit_bytes=$limit" \
 'logs=<root>/{baseline,candidate}/{digest.json,test.log,resource.log,system-before.txt,system-after.txt}; comparison.json' \
 'failure=retain the stage; never resume model state; rerun that mode archives failed stage and starts a fresh process' \
 "resume=DSV41_FILE_EXACT_DIR=$root DSV41_EXPERT_BACKING_DIR=$backing bash tools/benchmark/run_file_backed_exactness_safe.sh $mode" \
 'gate=offline bitwise digest equality and raw-log review only; no performance, promotion, paired, or 64K claim' | tee "$root/README.txt"
if [[ ${DSV41_DRY_RUN:-0} == 1 ]];then echo "DRY RUN: $mode only; no model started";exit 0;fi
preflight(){
 local stats page f i gib big swap
 stats=$(vm_stat);page=$(printf '%s\n' "$stats"|sed -n '1s/.*page size of \([0-9][0-9]*\).*/\1/p');read -r f i < <(printf '%s\n' "$stats"|awk '/Pages free/{gsub(/\./,"",$3);f=$3}/Pages inactive/{gsub(/\./,"",$3);i=$3}END{print f,i}');gib=$(((f+i)*page/1024/1024/1024))
 big=$(ps -axo rss=,pid=,comm=|awk -v limit=$((32*1024*1024)) '$1>limit{print}');[[ -z "$big" ]]||{ echo "preflight FAILED: >32 GiB process: $big" >&2;exit 3; };((gib>=400))||{ echo "preflight FAILED: conservative free+inactive ${gib} GiB; need >=400 GiB; speculative excluded" >&2;exit 3; }
 swap=$(sysctl -n vm.swapusage);[[ "$swap" == *'used = 0.00M'* ]]||{ echo "preflight FAILED: active swap is not zero: $swap" >&2;exit 3; }
 printf 'free_inactive_gib=%s\n%s\n' "$gib" "$swap"
}
vm_swap(){ awk '/Swapins:|Swapouts:/{print $1,$2}' "$1"; }
if [[ "$mode" == compare ]];then
 [[ -s "$root/baseline/digest.json" && -s "$root/candidate/digest.json" ]]||{ echo 'both digest stages required' >&2;exit 2; }
 python3 tools/benchmark/compare_backbone_digests.py "$root" "$limit"|tee "$root/comparison.json";exit 0
fi
[[ -r "$backing/manifest.json" ]]||{ echo 'reviewed backing missing' >&2;exit 2; };preflight|tee "$root/preflight-$mode.txt"
identity=$( { git rev-parse HEAD;git diff --binary -- CMakeLists.txt src include tests tools/benchmark;shasum -a 256 "$backing/manifest.json";rg --files src include tests tools/benchmark|LC_ALL=C sort|while IFS= read -r f;do shasum -a 256 "$f";done; }|shasum -a 256|awk '{print $1}')
if [[ -f "$root/source-identity.txt" ]];then [[ $(<"$root/source-identity.txt") == "$identity" ]]||{ echo 'source identity changed; use new root' >&2;exit 2; };else echo "$identity">"$root/source-identity.txt";fi
cmake -S . -B build-mlx>"$root/configure.log" 2>&1;cmake --build build-mlx --target dsv41-backbone-digest -j 4>"$root/build.log" 2>&1
binary=$(shasum -a 256 build-mlx/dsv41-backbone-digest|awk '{print $1}');if [[ -f "$root/binary-identity.txt" ]];then [[ $(<"$root/binary-identity.txt") == "$binary" ]]||{ echo 'binary changed; use new root' >&2;exit 2; };else echo "$binary">"$root/binary-identity.txt";fi
dir="$root/$mode";if [[ -d "$dir" ]];then mv "$dir" "$dir.failed-$(date +%Y%m%d-%H%M%S)-$$";fi;mkdir -p "$dir"
/usr/bin/vm_stat>"$dir/system-before.txt";checkpoint=${CHECKPOINT:-/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash}
if [[ "$mode" == baseline ]];then wired=0;backing_dir="";resident=0;compact=1;else [[ -f "$root/baseline/completed" ]]||{ echo 'review baseline before candidate' >&2;exit 2; };wired=$limit;backing_dir=$backing;resident=1;compact=0;fi
set +e
DSV41_RUNTIME_RESIDENT_EXPERT_ATLAS=$resident DSV41_RUNTIME_PACKED_EXPERT_BANK=1 DSV41_RUNTIME_COMPACT_EXPERT_BANK=$compact DSV41_RUNTIME_GROUP_SELECTED_EXPERTS=0 \
DSV41_RUNTIME_WIRED_LIMIT_BYTES=$wired DSV41_RUNTIME_EXPERT_BACKING_DIR="$backing_dir" \
/usr/bin/time -l build-mlx/dsv41-backbone-digest "$checkpoint" artifacts/checkpoint/summary.json artifacts/engram/metadata.json "$dir/digest.json" \
 >"$dir/test.log" 2>"$dir/resource.log"
status=$?;set -e;/usr/bin/vm_stat>"$dir/system-after.txt";echo "$status">"$dir/exit-code.txt";((status==0))||{ echo "$mode process failed; retain logs" >&2;exit "$status"; }
grep -q 'PASS: emitted full bounded backbone digest' "$dir/test.log";grep -Eq '^[[:space:]]+0[[:space:]]+swaps$' "$dir/resource.log"
awk '/peak memory footprint/{found=1;if($1>340000000000)exit 1}END{if(!found)exit 1}' "$dir/resource.log"
[[ "$(vm_swap "$dir/system-before.txt")" == "$(vm_swap "$dir/system-after.txt")" ]]||{ echo 'system swap delta; stage rejected' >&2;exit 1; }
touch "$dir/completed";echo "PASS: $mode process completed safely; stop and review before the next mode. Root: $root"
