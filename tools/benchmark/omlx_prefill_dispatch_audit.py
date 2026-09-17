#!/usr/bin/env python3
"""Paired official-checkpoint oMLX prefill dispatch audit."""

import argparse
import ctypes
import json
import os
import time
from pathlib import Path

import mlx.core as mx

from omlx.patches.deepseek_v41.loading import load


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.checkpoint.resolve()):
        parser.error("checkpoint is read-only")
    ids = [int(value) for value in args.tokens.read_text().split()]
    if len(ids) != 2063 or any(value < 0 or value >= 129280 for value in ids):
        parser.error("tokens must contain exactly 2063 valid text token IDs")

    counter_path = os.environ.get("DSV41_METAL_DISPATCH_COUNTER_LIBRARY")
    if not counter_path:
        parser.error("DSV41_METAL_DISPATCH_COUNTER_LIBRARY is required")
    counter = ctypes.CDLL(counter_path, mode=ctypes.RTLD_GLOBAL)
    counter.dsv41_metal_dispatch_counter_reset.argtypes = []
    counter.dsv41_metal_dispatch_counter_set_enabled.argtypes = [ctypes.c_int]

    load_started = time.monotonic()
    model, _ = load(args.checkpoint, preserve_mtp=False)
    mx.synchronize()
    load_seconds = time.monotonic() - load_started
    cache = model.language_model.make_cache()
    tokens = mx.array([ids], dtype=mx.int32)

    counter.dsv41_metal_dispatch_counter_reset()
    counter.dsv41_metal_dispatch_counter_set_enabled(1)
    prefill_started = time.monotonic()
    logits = model(tokens, cache=cache)
    mx.eval(logits)
    mx.synchronize()
    prefill_seconds = time.monotonic() - prefill_started
    counter.dsv41_metal_dispatch_counter_set_enabled(0)

    next_token = int(mx.argmax(logits[0, -1, :129280]).item())
    offsets = []
    for entry in cache:
        offset = getattr(entry, "offset", None)
        if offset is not None:
            offsets.append(mx.array(offset).tolist())
    report = {
        "status": "measurement_completed_requires_review",
        "scope": "Pinned oMLX official-checkpoint 2063-token single-sweep prefill; selector-hook wall time is not a performance result.",
        "tokens": len(ids),
        "load_seconds": load_seconds,
        "prefill_seconds": prefill_seconds,
        "prefill_tokens_per_second": len(ids) / prefill_seconds,
        "next_token": next_token,
        "cache_offsets": offsets,
        "active_bytes": mx.get_active_memory(),
        "cache_bytes": mx.get_cache_memory(),
        "peak_bytes": mx.get_peak_memory(),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    model.close()


if __name__ == "__main__":
    main()
