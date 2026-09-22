#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
root=${DSV41_FILE_MEASURE_DIR:-artifacts/context-ladder/file-backed-candidate-$(date +%Y%m%d-%H%M%S)-$$};backing=${DSV41_EXPERT_BACKING_DIR:-artifacts/expert-backing/official-packed-v1};limit=${DSV41_FILE_RESIDENCY_BYTES:-274877906944};mkdir -p "$root"
printf '%s\n' \
 'scope=one fresh file-backed+256 GiB residency 2,063-token prefill plus 8 greedy tokens; candidate-only safety/performance observation' \
 'resources=approximately 2-8 minutes when backing is warm, potentially up to 40 minutes if cold; one process; <=340 GB footprint; checkpoint/backing read-only' \
 'safety=requires >=400 GiB free+inactive (speculative excluded), zero active swap, no process >32 GiB; zero process/system swap delta' \
 "backing=$backing wired_limit_bytes=$limit" \
 'logs=<root>/candidate canonical context result/test/resource/system/config/identity logs plus review.json' \
 'failure=retain logs; no model-state resume; rerun with a new root after review' \
 'gate=candidate observation only; frozen-baseline comparison is descriptive and cannot promote, pair, or qualify 64K' | tee "$root/README.txt"
if [[ ${DSV41_DRY_RUN:-0} == 1 ]];then echo 'DRY RUN: candidate-only process not started';exit 0;fi
stats=$(vm_stat);page=$(printf '%s\n' "$stats"|sed -n '1s/.*page size of \([0-9][0-9]*\).*/\1/p');read -r f i < <(printf '%s\n' "$stats"|awk '/Pages free/{gsub(/\./,"",$3);f=$3}/Pages inactive/{gsub(/\./,"",$3);i=$3}END{print f,i}');gib=$(((f+i)*page/1024/1024/1024));((gib>=400))||{ echo "preflight FAILED: free+inactive ${gib} GiB; need 400; speculative excluded" >&2;exit 3; }
big=$(ps -axo rss=,pid=,comm=|awk -v limit=$((32*1024*1024)) '$1>limit{print}');[[ -z "$big" ]]||{ echo "preflight FAILED: >32 GiB process: $big" >&2;exit 3; };swap=$(sysctl -n vm.swapusage);[[ "$swap" == *'used = 0.00M'* ]]||{ echo "preflight FAILED: active swap: $swap" >&2;exit 3; };printf 'free_inactive_gib=%s\n%s\n' "$gib" "$swap">"$root/preflight.txt"
[[ -r "$backing/manifest.json" ]]||{ echo 'backing missing' >&2;exit 2; }
DSV41_CONTEXT_RUN_DIR="$root/candidate" CONTEXT_TOKENS=2071 TEACHER_TOKENS=0 TAIL_TEACHER_TOKENS=0 DECODE_TOKENS=8 \
DSV41_CONTEXT_EXECUTION=sweep DSV41_RUNTIME_COMPONENT_PROFILE=0 DSV41_RUNTIME_WIRED_LIMIT_BYTES=$limit DSV41_RUNTIME_EXPERT_BACKING_DIR="$backing" \
DSV41_CONTEXT_WALL_BUDGET_SECONDS=2400 DSV41_CONTEXT_PROJECTED_WALL_LIMIT_SECONDS=2400 CACHE_CONDITION=warm-unknown RUN_CONDITIONS=file-backed-candidate-only \
bash tools/benchmark/run_context_32k.sh
r="$root/candidate";[[ $(<"$r/exit-code.txt") == 0 ]];grep -Eq '^[[:space:]]+0[[:space:]]+swaps$' "$r/resource.log"
python3 - "$r" "$limit" <<'PY' | tee "$root/review.json"
import json,pathlib,re,sys
p=pathlib.Path(sys.argv[1]);limit=int(sys.argv[2]);r=json.loads((p/'result.json').read_text())
assert r['status']=='measurement_completed_requires_review' and r['context_tokens']==2071 and r['decode_tokens']==8
assert r['wired_limit_bytes']==limit and r['expert_backing_file_backed'] and r['packed_expert_bank_constructions']==40
assert r['phases']['decode']['generated_token_ids']==[339,3465,2595,4731,79316,88287,3395,361]
def vm(name):return {k:int(v) for k,v in re.findall(r'(Swapins|Swapouts):\s*(\d+)',(p/name).read_text())}
assert vm('system-before.txt')==vm('system-after.txt'),'system swap delta'
text=(p/'resource.log').read_text();foot=re.search(r'^\s+(\d+)\s+peak memory footprint$',text,re.M);assert foot and int(foot[1])<=340_000_000_000
lat=r['phases']['decode']['latency_seconds'];print(json.dumps({'status':'candidate_safe_measurement_requires_review_not_promotion','prefill_seconds':r['aggregates']['prefill_seconds'],'advancing_decode_mean_seconds':sum(lat[1:])/len(lat[1:]),'decode_latency_seconds':lat,'full_path_seconds':r['aggregates']['prefill_seconds']+sum(lat),'peak_footprint_bytes':int(foot[1])},indent=2))
PY
echo "PASS: candidate-only measurement completed safely; review $root before any next step"
