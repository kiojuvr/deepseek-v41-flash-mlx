#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

task_root=${DSV41_PENDING_CACHE_DIR:-artifacts/context-ladder/deferred-pending-cache-$(date +%Y%m%d-%H%M%S)-$$}
attempt="$task_root/attempt-001"
mkdir -p "$task_root"
finish(){ status=$?; echo "$status" > "$task_root/exit-code.txt"; if ((status)); then
 echo "FAILED: inspect $task_root; rerun with a fresh root because request state is transactional."; fi; }
trap finish EXIT
printf '%s\n' \
 'scope=one 32K pending-CED production-topology run with idle MLX allocator cache cleared exactly once after atomic decoder publication' \
 'resources=one official model; allow 8-15 minutes; up to 340 GB Unified Memory; checkpoint read-only; no swap expected' \
 'gate=one pending CED transaction; exact canonical generated tokens/state/topology; teacher/decode/base wall; memory and swap' \
 'comparison=review against the five-pair pending-CED means; this single observation can reject but cannot promote cache normalization' \
 'logs=<root>/attempt-001/{result.json,result.json.progress.jsonl,test.log,resource.log,config.txt,identity.txt,tracked.patch,exit-code.txt}' \
 'failure=retain logs; partial request state is never published or resumed' \
 'resume=unsupported; rerun with a fresh root' | tee "$task_root/config.txt"
echo "Logs: $task_root"
DSV41_CONTEXT_RUN_DIR="$attempt" CONTEXT_TOKENS=32768 TEACHER_TOKENS=4096 \
 TAIL_TEACHER_TOKENS=2048 DECODE_TOKENS=16 DSV41_CONTEXT_EXECUTION=sweep \
 DSV41_RUNTIME_DEFERRED_DECODER=1 DSV41_RUNTIME_DEFERRED_DECODER_CLEAR_CACHE=1 \
 DSV41_RUNTIME_COMPONENT_PROFILE=0 DSV41_RUNTIME_MLX_CACHE_LIMIT_BYTES=0 \
 DSV41_RUNTIME_LONG_CONTEXT_CACHE_LIMIT_BYTES=17179869184 \
 DSV41_CONTEXT_WALL_BUDGET_SECONDS=5400 DSV41_CONTEXT_PROJECTED_WALL_LIMIT_SECONDS=5400 \
 CACHE_CONDITION=warm-unknown RUN_CONDITIONS='pending CED plus post-publication idle-cache normalization' \
 bash tools/benchmark/run_context_32k.sh
expected='[270,37370,2019,305,8470,270,15398,16,3476,477,260,20255,22896,16,47266,270]'
jq -e --argjson expected "$expected" \
 '.status=="measurement_completed_requires_review" and .deferred_decoder==true and
  .context_tokens==32768 and .phases.base_prefill.deferred_decoder_chunks==1 and
  .phases.teacher_head.deferred_decoder_chunks==0 and .phases.teacher_tail.deferred_decoder_chunks==0 and
  .phases.decode.generated_token_ids==$expected and .phases.decode.state_position==32767 and
  .packed_expert_bank_constructions==40 and .packed_expert_bank_loaded_experts==15360 and
  .route_execution_stats.diagnostic_readbacks==0 and .attention_telemetry.index_host_readbacks==0 and
  .attention_telemetry.chunk_scalar_qk_calls==0 and .attention_telemetry.chunk_scalar_av_calls==0' \
 "$attempt/result.json" >/dev/null
grep -Eq '^deferred_decoder_clear_cache=1$' "$attempt/config.txt"
grep -Eq '^[[:space:]]+0[[:space:]]+swaps$' "$attempt/resource.log"
echo 'Completed; review result/resource/config/identity against the paired pending-CED distribution before any follow-up.'
