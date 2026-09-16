#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
run_dir="artifacts/expert-bank/encoder-chunk-$(date +%Y%m%d-%H%M%S)-$$"
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
{
 cmake -S . -B build-mlx
 cmake --build build-mlx --target dsv41-text-encoder-test -j 4
} > "$run_dir/build.log" 2>&1
git rev-parse HEAD > "$run_dir/revision.txt"
git diff --binary > "$run_dir/tracked.patch"
shasum -a 256 build-mlx/dsv41-text-encoder-test tests/attention/test_text_encoder.cpp \
 include/dsv41/text_encoder.hpp src/model/text_encoder.cpp src/model/block.cpp \
 src/model/compressed_block.cpp src/model/reused_block.cpp src/moe/reference.cpp \
 src/moe/expert_bank.cpp artifacts/checkpoint/summary.json artifacts/engram/metadata.json \
 > "$run_dir/identity.txt"
printf '%s\n' 'scope=encoder layers 0..19; 2x128 tokens; hidden/pre-mix, state, publications, hash continuation, route ties and invalid-token rejection' \
 'resources=several minutes; budget 160 GB Unified Memory; approximately 289 GB logical checkpoint reads for 40 bank constructions; checkpoint read-only' \
 'resume=unsupported; retain failed logs and rerun fresh' > "$run_dir/config.txt"
cmd=(env DSV41_RUNTIME_LAYER_FINITE_CHECKS=0 DSV41_RUNTIME_PACKED_EXPERT_BANK=0 \
 DSV41_RUNTIME_GROUP_SELECTED_EXPERTS=0 DSV41_CHECK_ENCODER_PACKED_CHUNK=1 \
 build-mlx/dsv41-text-encoder-test "$checkpoint" artifacts/checkpoint/summary.json artifacts/engram/metadata.json)
printf '%q ' "${cmd[@]}" > "$run_dir/command.txt"
printf '\n' >> "$run_dir/command.txt"
/usr/bin/vm_stat > "$run_dir/system-before.txt"
(/usr/bin/time -l "${cmd[@]}") > >(tee "$run_dir/test.log") 2> >(tee "$run_dir/resource.log" >&2)
echo 'Completed; encoder chunk parity/resources require review. Decoder, performance and 32K remain unqualified.'
