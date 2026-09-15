#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
run_dir="artifacts/sampling/formula-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$run_dir"
trap 'status=$?; echo "$status" > "$run_dir/exit-code.txt"; if ((status)); then echo "FAILED: inspect $run_dir; rerun for a fresh run."; fi' EXIT
checkpoint=${DSV41_CHECKPOINT:-/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash}
python=${PYTHON:-/Volumes/SDXC-512/deltafin/.venv/bin/python}
cmake -S . -B build-mlx > "$run_dir/configure.log" 2>&1
cmake --build build-mlx --target dsv41-sampling-test -j 4 > "$run_dir/build.log" 2>&1
build-mlx/dsv41-sampling-test > "$run_dir/native.log" 2>&1
"$python" tools/reference/check_sampling_formula.py \
  --official-source "$checkpoint/inference/model.py" \
  --verification artifacts/checkpoint/verification.json \
  --native-log "$run_dir/native.log" --output "$run_dir/report.json" \
  > "$run_dir/oracle.log" 2>&1
shasum -a 256 build-mlx/dsv41-sampling-test include/dsv41/sampling.hpp \
  src/model/sampling.cpp tests/attention/test_sampling.cpp \
  tools/reference/check_sampling_formula.py "$checkpoint/inference/model.py" \
  artifacts/checkpoint/verification.json > "$run_dir/identity.txt"
echo "PASS: $run_dir (fixed Exp(1) formula only; CUDA RNG/generation remain unqualified)"
