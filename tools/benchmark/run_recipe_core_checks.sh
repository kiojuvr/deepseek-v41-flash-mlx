#!/usr/bin/env bash
set -euo pipefail

repo=${RECIPE_DIR:-third_party/deepseek-recipe}
if [[ ! -f "$repo/Cargo.toml" ]]; then
  echo "recipe checkout missing: $repo" >&2
  exit 2
fi
run_dir="artifacts/recipe/checks-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$run_dir"
trap 'status=$?; echo "$status" > "$run_dir/exit-code.txt"' EXIT
git -C "$repo" rev-parse HEAD > "$run_dir/revision.txt"
git -C "$repo" rev-parse HEAD^{tree} > "$run_dir/tree.txt"
set +e
cargo test --manifest-path "$repo/Cargo.toml" \
  -p deepseek-recipe-core -p deepseek-recipe-encoding -p deepseek-recipe \
  --no-default-features 2>&1 | tee "$run_dir/test.log"
codes=("${PIPESTATUS[@]}"); status=${codes[0]}; if [[ "$status" -eq 0 && "${codes[1]}" -ne 0 ]]; then status=${codes[1]}; fi
set -e
echo "$status" > "$run_dir/test-exit-code.txt"
if [[ "$status" -ne 0 ]]; then echo "FAILED: inspect $run_dir/test.log; rerun with a fresh run directory." >&2; exit "$status"; fi
echo "Recipe core checks completed: $run_dir"
