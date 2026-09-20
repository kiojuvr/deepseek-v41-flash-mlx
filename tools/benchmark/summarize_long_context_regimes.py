#!/usr/bin/env python3
"""Summarize completed 16K/32K production and component-profile runs."""

import argparse
import json
import re
from pathlib import Path


def load_resources(run: Path):
    text = (run / "resource.log").read_text()
    def value(label):
        match = re.search(rf"^\s*(\d+)\s+{re.escape(label)}$", text, re.MULTILINE)
        if not match:
            raise ValueError(f"missing {label} in {run / 'resource.log'}")
        return int(match.group(1))
    return {
        "maximum_resident_set_bytes": value("maximum resident set size"),
        "peak_footprint_bytes": value("peak memory footprint"),
        "swaps": value("swaps"),
        "page_faults": value("page faults"),
    }


def load_stage(root: Path, name: str):
    marker = root / name / "completed-run.txt"
    if not marker.exists():
        return None
    run = Path(marker.read_text().strip())
    result = json.loads((run / "result.json").read_text())
    if (run / "exit-code.txt").read_text().strip() != "0":
        raise ValueError(f"completed stage has nonzero exit: {run}")
    phases = result["phases"]
    profile = {}
    for phase in ("base_prefill", "teacher_head", "teacher_tail"):
        current = phases[phase].get("runtime_component_profile")
        if not current:
            continue
        for key in ("layer_seconds", "attention_path_seconds", "moe_path_seconds", "post_moe_seconds"):
            profile[key] = profile.get(key, 0.0) + current[key]
    component_sum = sum(profile.get(key, 0.0) for key in
                        ("attention_path_seconds", "moe_path_seconds", "post_moe_seconds"))
    if component_sum:
        profile["component_sum_seconds"] = component_sum
        profile["attention_share"] = profile["attention_path_seconds"] / component_sum
        profile["moe_share"] = profile["moe_path_seconds"] / component_sum
        profile["post_moe_share"] = profile["post_moe_seconds"] / component_sum
    return {
        "run": str(run),
        "context_tokens": result["context_tokens"],
        "component_profile": result["component_profile"],
        "prefill_seconds": result["aggregates"]["prefill_seconds"],
        "prefill_tokens_per_second": result["aggregates"]["prefill_tokens_per_second"],
        "teacher_continuation_seconds": result["aggregates"]["teacher_continuation_seconds"],
        "decode_mean_seconds": phases["decode"]["mean_seconds"],
        "decode_p95_seconds": phases["decode"]["p95_seconds"],
        "peak_bytes": result["final_memory"]["peak_bytes"],
        "final_active_bytes": result["final_memory"]["active_bytes"],
        "final_cache_bytes": result["final_memory"]["cache_bytes"],
        "resources": load_resources(run),
        "profile": profile,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    stages = {name: load_stage(args.root, name) for name in
              ("16k-production", "16k-profile", "32k-production", "32k-profile")}
    stages = {name: value for name, value in stages.items() if value is not None}
    summary = {"schema_version": 1, "status": "partial" if len(stages) < 4 else "complete",
               "stages": stages}
    if "16k-production" in stages and "32k-production" in stages:
        a, b = stages["16k-production"], stages["32k-production"]
        summary["production_scaling"] = {
            "prefill_wall_ratio_32k_over_16k": b["prefill_seconds"] / a["prefill_seconds"],
            "throughput_ratio_32k_over_16k": b["prefill_tokens_per_second"] / a["prefill_tokens_per_second"],
            "decode_mean_ratio_32k_over_16k": b["decode_mean_seconds"] / a["decode_mean_seconds"],
            "peak_byte_delta_32k_minus_16k": b["peak_bytes"] - a["peak_bytes"],
            "cache_byte_delta_32k_minus_16k": b["final_cache_bytes"] - a["final_cache_bytes"],
            "peak_footprint_byte_delta_32k_minus_16k":
                b["resources"]["peak_footprint_bytes"] - a["resources"]["peak_footprint_bytes"],
        }
    text = json.dumps(summary, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
