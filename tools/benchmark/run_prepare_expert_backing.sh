#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
checkpoint=${CHECKPOINT:-/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash}
destination=${DSV41_EXPERT_BACKING_DIR:-artifacts/expert-backing/official-packed-v1}
run_dir=${DSV41_EXPERT_BACKING_PREP_LOG_DIR:-artifacts/expert-backing/prepare-$(date +%Y%m%d-%H%M%S)-$$}
mkdir -p "$run_dir"
finish(){ status=$?; echo "$status" > "$run_dir/exit-code.txt"; if ((status)); then echo "FAILED: retain $destination.building and resume with the same destination; inspect $run_dir"; fi; }
trap finish EXIT
printf '%s\n' \
 'scope=prepare 240 exact packed routed-expert payloads (w1/w2/w3 weight+scale x40) and publish a read-only file-backed atlas' \
 'resources=approximately 288.78 GB payload plus alignment/manifest; require >=320 GB destination free; expected 30-120 minutes depending on checkpoint/destination storage' \
 'checkpoint=read-only; output is a derived exact-layout cache, never canonical weights' \
 "destination=$destination; in-progress=$destination.building" \
 "logs=$run_dir/{prepare.log,resource.log,identity.txt,tracked.patch,exit-code.txt}" \
 'failure=completed payloads with matching stat identities are retained; rerun the exact command to resume; a complete root is published only by atomic rename' \
 "resume=DSV41_EXPERT_BACKING_DIR=$destination bash tools/benchmark/run_prepare_expert_backing.sh" \
 'gate=preparation only; PASS does not qualify runtime correctness, memory, swap, or performance' | tee "$run_dir/scope.txt"
if [[ ${DSV41_DRY_RUN:-0} == 1 ]]; then echo 'DRY RUN: preflight -> build -> resumable 240-payload preparation -> atomic publish; no payload written'; exit 0; fi
[[ "$destination" != "$checkpoint" && "$destination" != "$checkpoint"/* ]] || { echo 'destination must not be inside checkpoint' >&2; exit 2; }
[[ ! -e "$destination" ]] || { echo 'complete destination already exists; refusing overwrite' >&2; exit 2; }
mkdir -p "$(dirname "$destination")"
free_kib=$(df -Pk "$(dirname "$destination")" | awk 'NR==2{print $4}')
((free_kib>=320*1024*1024)) || { echo 'destination has less than 320 GiB free' >&2; exit 3; }
cmake -S . -B build-mlx > "$run_dir/configure.log" 2>&1
cmake --build build-mlx --target dsv41-prepare-expert-backing -j 4 > "$run_dir/build.log" 2>&1
git rev-parse HEAD > "$run_dir/revision.txt";git diff --binary > "$run_dir/tracked.patch"
shasum -a 256 build-mlx/dsv41-prepare-expert-backing include/dsv41/expert_backing.hpp src/moe/expert_backing.cpp \
 tools/benchmark/prepare_expert_backing.cpp tools/benchmark/run_prepare_expert_backing.sh \
 artifacts/checkpoint/summary.json artifacts/checkpoint/verification.json > "$run_dir/identity.txt"
identity=$(shasum -a 256 "$run_dir/identity.txt" | awk '{print $1}'); identity_file="$destination.prepare-identity"
if [[ -f "$identity_file" ]]; then [[ $(<"$identity_file") == "$identity" ]] || { echo 'preparation identity changed; do not resume this building root' >&2; exit 2; }; else echo "$identity" > "$identity_file"; fi
printf '%q ' build-mlx/dsv41-prepare-expert-backing "$checkpoint" artifacts/checkpoint/summary.json "$destination" > "$run_dir/command.txt";echo >> "$run_dir/command.txt"
/usr/bin/time -l build-mlx/dsv41-prepare-expert-backing "$checkpoint" artifacts/checkpoint/summary.json "$destination" \
 > >(tee "$run_dir/prepare.log") 2> >(tee "$run_dir/resource.log" >&2)
grep -q 'PASS: complete file-backed expert atlas published' "$run_dir/prepare.log"
[[ -r "$destination/manifest.json" && ! -e "$destination.building" ]]
rm -f "$identity_file"
echo 'Prepared derived backing; do not run runtime validation until manifest/raw resource review.'
