#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

run_dir=${DSV41_DECODE_PROFILE_DIR:-artifacts/context-ladder/decode-component-profile-$(date +%Y%m%d-%H%M%S)-$$}
mkdir -p "$run_dir"
finish(){
 status=$?
 echo "$status" > "$run_dir/exit-code.txt"
 if ((status)); then
  echo "FAILED: inspect $run_dir; request state is not resumable; rerun with a new DSV41_DECODE_PROFILE_DIR."
 fi
}
trap finish EXIT
printf '%s\n' \
 'scope=one fresh official model; 2,063-token production prefill followed by 8 greedy tokens; 7 state-advancing decode steps receive synchronized component attribution' \
 'resources=one model; allow 3-8 minutes; up to 340 GB Unified Memory; checkpoint read-only; no swap expected' \
 'measurement=decode-local attention/MoE/post-MoE GPU-completion wall plus MoE sub-phases (input deps/route/routed/shared/combine); synchronization perturbs normal lazy execution, so use ratios for attribution rather than production TPT' \
 'preflight=abort before model load if another process holds >32 GiB RSS or if free+inactive memory is below 360 GiB; a resident omlx-server (~300 GB) makes decode TPT incomparable' \
 'checks=exact final position; finite output; 40 resident banks; zero route/index readbacks and scalar attention; decode profile has 280 layer calls and completed component triplets; swap 0' \
 'logs=<run-dir>/{result.json,result.json.progress.jsonl,test.log,resource.log,config.txt,identity.txt,tracked.patch,exit-code.txt,decode-profile.tsv}' \
 'failure=retain the failed run directory; request state is not resumable; rerun with a new DSV41_DECODE_PROFILE_DIR for a fresh model/request' \
 'resume=unsupported because the short session is transactional and component synchronization changes its execution schedule' \
 | tee "$run_dir/scope.txt"

preflight(){
 local page_size free_pages free_gib big
 page_size=$(vm_stat | sed -n '1s/.*page size of \([0-9][0-9]*\).*/\1/p')
 [[ -n "$page_size" ]] || page_size=$(sysctl -n hw.pagesize)
 free_pages=$(vm_stat | awk '/Pages free/{gsub(/\./,"",$3);f=$3} /Pages inactive/{gsub(/\./,"",$3);i=$3} END{print f+i}')
 free_gib=$(( free_pages * page_size / 1024 / 1024 / 1024 ))
 big=$(ps -axo rss=,pid=,comm= | awk -v limit=$((32*1024*1024)) '$1>limit{printf "  pid %s %s (%d MiB)\n",$2,$3,$1/1024}')
 if [[ -n "$big" ]]; then
  { echo "preflight FAILED: large resident process(es) detected; free+inactive ${free_gib} GiB:"
   printf '%s\n' "$big"
   echo "Stop them (a loaded omlx-server holds ~300 GB) and rerun; the frozen baseline was measured on a quiet host."; } >&2
  exit 3
 fi
 if (( free_gib < 360 )); then
  echo "preflight FAILED: only ${free_gib} GiB free+inactive; need >= 360 GiB for a clean 340 GB run." >&2
  exit 3
 fi
 echo "preflight OK: free+inactive ${free_gib} GiB; no process above 32 GiB RSS" | tee "$run_dir/preflight.txt"
}
preflight
echo "Logs: $run_dir"

DSV41_CONTEXT_RUN_DIR="$run_dir" \
CONTEXT_TOKENS=2071 TEACHER_TOKENS=0 TAIL_TEACHER_TOKENS=0 DECODE_TOKENS=8 \
DSV41_CONTEXT_EXECUTION=sweep DSV41_RUNTIME_COMPONENT_PROFILE=1 \
DSV41_RUNTIME_MLX_CACHE_LIMIT_BYTES=0 \
DSV41_RUNTIME_LONG_CONTEXT_CACHE_LIMIT_BYTES=17179869184 \
DSV41_CONTEXT_WALL_BUDGET_SECONDS=900 \
DSV41_CONTEXT_PROJECTED_WALL_LIMIT_SECONDS=900 \
CACHE_CONDITION=warm-unknown RUN_CONDITIONS=production-short-decode-component-attribution \
bash tools/benchmark/run_context_32k.sh

jq -e '
 .status=="measurement_completed_requires_review" and
 .context_tokens==2071 and .decode_tokens==8 and
 .phases.decode.state_position==2070 and .phases.decode.next_position==2071 and
 .phases.decode.runtime_component_profile.layer_calls==280 and
 .phases.decode.runtime_component_profile.component_calls==280 and
 .packed_expert_bank_constructions==40 and
 .route_execution_stats.diagnostic_readbacks==0 and
 .attention_telemetry.index_host_readbacks==0 and
 .attention_telemetry.chunk_scalar_qk_calls==0 and
 .attention_telemetry.chunk_scalar_av_calls==0
' "$run_dir/result.json" >/dev/null
grep -Eq '^[[:space:]]+0[[:space:]]+swaps$' "$run_dir/resource.log"

printf 'scope\tlayer_seconds\tattention_seconds\tmoe_seconds\tpost_moe_seconds\tmoe_input_seconds\tmoe_route_seconds\tmoe_routed_seconds\tmoe_shared_seconds\tmoe_combine_seconds\tlayer_calls\tcomponent_calls\n' \
 > "$run_dir/decode-profile.tsv"
jq -r '
 .phases.decode.runtime_component_profile as $p |
 ["decode",$p.layer_seconds,$p.attention_path_seconds,$p.moe_path_seconds,
  $p.post_moe_seconds,$p.moe_input_seconds,$p.moe_route_seconds,
  $p.moe_routed_seconds,$p.moe_shared_seconds,$p.moe_combine_seconds,
  $p.layer_calls,$p.component_calls] | @tsv
' "$run_dir/result.json" >> "$run_dir/decode-profile.tsv"
cat "$run_dir/decode-profile.tsv"
echo "MoE sub-phase split (nested inside moe_seconds, over 7 advancing decode tokens):"
jq -r '
 .phases.decode.runtime_component_profile as $p |
 ["  input-deps(mHC/norm)",$p.moe_input_seconds],
 ["  route selection",$p.moe_route_seconds],
  ["  routed expert service (cold)",$p.moe_routed_seconds],
  ["  routed expert service (warm repeat)",$p.moe_routed_warm_seconds],
  ["  no-op eval+sync overhead",$p.moe_sync_noop_seconds],
  ["  shared expert",$p.moe_shared_seconds],
  ["  combine",$p.moe_combine_seconds] | @tsv
' "$run_dir/result.json"
echo "Routed-service split (nested inside routed expert service):"
jq -r '
 .phases.decode.runtime_component_profile as $p |
 ["  prep sort/take + input FP8",$p.routed_prep_seconds],
 ["  gate/up gather-QMM",$p.routed_gateup_seconds],
 ["  swiglu + down-input FP8",$p.routed_mid_seconds],
 ["  down gather-QMM + weighting",$p.routed_down_seconds],
 ["  inverse-sort + reduce",$p.routed_reduce_seconds] | @tsv
' "$run_dir/result.json"
echo "Completed measurement; review every raw log before selecting or promoting a decode candidate."
