#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

root="$(pwd)/artifacts/prefill-gap/omlx-metal-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$root"
counter="$root/metal-dispatch-counter.dylib"
omlx_source=${OMLX_SOURCE:-/Users/kioju/omlx-0.7.0.dev2}
checkpoint=${CHECKPOINT:-/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash}
python_root=/Applications/oMLX.app/Contents/Resources/Python
python="$python_root/cpython-3.11/bin/python3.11"
mlx_site="$python_root/framework-mlx-base/lib/python3.11/site-packages"

printf '%s\n' \
 'scope=pinned oMLX b390b31; reviewed DeepSeek-V4.1 settings (Engram SSD offload, MTP preserved); official checkpoint; exactly 2063 prompt tokens in one model call; process-local prefill dispatch count' \
 'resources=allow 5-10 minutes; Unified Memory budget 340 GB; about 289 GB checkpoint reads; checkpoint read-only; no Instruments trace' \
 "logs=$root/{result.json,metal-dispatch-counts.json,test.log,resource.log,identity.txt,exit-code.txt}" \
 'measurement=counter is enabled only after model load and cache creation, through evaluated prefill completion; hook overhead invalidates wall time' \
 'failure=retain the directory and inspect exit-code/test/resource logs; partial result is not a count' \
 'resume=unsupported; rerun for a fresh model and cache state'

finish() {
 status=$?
 printf '%s\n' "$status" > "$root/exit-code.txt"
 if ((status)); then echo "FAILED: retain and inspect $root; no state resume." >&2; fi
}
trap finish EXIT

test "$(git -C "$omlx_source" rev-parse HEAD)" = b390b31e0c6831225fed0f24d278eb1db7fcb68b
test -z "$(git -C "$omlx_source" status --short --untracked-files=no)"
test -x "$python"
xcrun clang++ -std=c++17 -O2 -fobjc-arc -dynamiclib \
 tools/benchmark/metal_dispatch_counter.mm -framework Foundation \
 -framework Metal -o "$counter"
python3 tools/reference/expand_token_pattern.py \
 --input artifacts/logits-trace/prompt-tokens.txt --output "$root/prompt.txt" \
 --length 2063 > "$root/prompt-build.log"
{
 git rev-parse HEAD
 git -C "$omlx_source" rev-parse HEAD
 shasum -a 256 tools/benchmark/metal_dispatch_counter.mm \
  tools/benchmark/omlx_prefill_dispatch_audit.py \
  "$omlx_source/omlx/patches/deepseek_v41/loading.py" \
  "$omlx_source/omlx/patches/deepseek_v41/model.py" "$root/prompt.txt"
} > "$root/identity.txt"

(/usr/bin/time -l env \
 PYTHONHOME="$python_root/cpython-3.11" \
 PYTHONDONTWRITEBYTECODE=1 \
 PYTHONPATH="$omlx_source:$mlx_site" \
 DYLD_INSERT_LIBRARIES="$counter" \
 DSV41_METAL_DISPATCH_COUNTER_LIBRARY="$counter" \
 DSV41_METAL_DISPATCH_COUNTER_OUTPUT="$root/metal-dispatch-counts.json" \
 DSV41_METAL_DISPATCH_COUNTER_SCOPED=1 \
 "$python" tools/benchmark/omlx_prefill_dispatch_audit.py \
 --checkpoint "$checkpoint" --tokens "$root/prompt.txt" --output "$root/result.json") \
 > >(tee "$root/test.log") 2> >(tee "$root/resource.log" >&2)

python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["dispatch_total"] > 0 and d["command_buffers"] > 0 and d["compute_encoders"] > 0' \
 "$root/metal-dispatch-counts.json"
echo "Completed; review result.json and metal-dispatch-counts.json together before accepting the cross-runtime ratio."
