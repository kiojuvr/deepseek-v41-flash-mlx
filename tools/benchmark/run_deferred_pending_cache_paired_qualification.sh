#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

export DSV41_PENDING_PAIRED_CLEAR_CACHE=1
exec bash tools/benchmark/run_deferred_pending_paired_qualification.sh
