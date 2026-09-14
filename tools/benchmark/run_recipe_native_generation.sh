#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
fixture=${1:-chat}
fixture_json=${RECIPE_FIXTURES:-artifacts/recipe/encoding-fixtures.json}
token_file=$(mktemp /tmp/dsv41-recipe-tokens.XXXXXX)
trap 'rm -f "$token_file"' EXIT
/Volumes/SDXC-512/deltafin/.venv/bin/python - "$fixture_json" "$fixture" "$token_file" <<'PY'
import json, pathlib, sys
data=json.loads(pathlib.Path(sys.argv[1]).read_text())
for item in data['fixtures']:
    if item['name']==sys.argv[2]:
        ids=item.get('token_ids')
        if not ids: raise SystemExit('fixture has no token_ids: '+sys.argv[2])
        pathlib.Path(sys.argv[3]).write_text(' '.join(map(str,ids))+'\n'); break
else: raise SystemExit('unknown fixture: '+sys.argv[2])
PY
TOKENS_FILE="$token_file" MAX_NEW="${MAX_NEW:-16}" TEMPERATURE="${TEMPERATURE:-0}" SEED="${SEED:-0}" \
  bash tools/benchmark/run_text_generate.sh
