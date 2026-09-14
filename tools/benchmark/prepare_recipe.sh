#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
repo_url=${RECIPE_URL:-https://github.com/deepseek-ai/deepseek-recipe.git}
revision=${RECIPE_REVISION:-8cadfede7063c896b944e7bae05daa3549ae97ea}
out=${RECIPE_DIR:-third_party/deepseek-recipe}

if [[ -e "$out" ]]; then
  echo "output exists; choose RECIPE_DIR for a fresh checkout: $out" >&2
  exit 2
fi

mkdir -p "$(dirname "$out")"
git clone --filter=blob:none "$repo_url" "$out"
git -C "$out" fetch --depth 1 origin "$revision"
git -C "$out" checkout --detach "$revision"

run_dir="artifacts/recipe/prepare-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$run_dir"
git -C "$out" rev-parse HEAD > "$run_dir/revision.txt"
git -C "$out" rev-parse HEAD^{tree} > "$run_dir/tree.txt"
git -C "$out" status --short > "$run_dir/status.txt"
if [[ -f "$out/Cargo.toml" ]]; then
  cargo metadata --manifest-path "$out/Cargo.toml" --format-version 1 --no-deps > "$run_dir/cargo-metadata.json"
fi
cat > "$run_dir/README.txt" <<EOF
Recipe checkout prepared at: $out
Revision: $revision
This is source verification only; the production Cargo dependency remains disabled.
EOF
echo "Recipe prepared: $run_dir"
