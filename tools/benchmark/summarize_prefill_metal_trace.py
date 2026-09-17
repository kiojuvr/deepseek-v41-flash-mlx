#!/usr/bin/env python3
"""Export stable focused-Metal trace tables and summarize execution work.

This is validation orchestration, not a production runtime dependency.  A
missing dispatch event is reported as unresolved rather than silently as zero.
"""

from __future__ import annotations

import argparse
import collections
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path


SCHEMAS = (
    "metal-application-command-buffer-submissions",
    "metal-application-encoders-list",
    "metal-application-intervals",
    "metal-resource-allocations",
)


def export(trace: Path, schema: str, destination: Path) -> ET.Element:
    xpath = f'/trace-toc/run[@number="1"]/data/table[@schema="{schema}"]'
    if destination.exists():
        destination.unlink()
    completed = subprocess.run(
        ["xcrun", "xctrace", "export", "--input", str(trace), "--xpath", xpath,
         "--output", str(destination)],
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"xctrace export failed for {schema}: {completed.returncode}")
    return ET.parse(destination).getroot()


def value(cell: ET.Element, references: dict[tuple[str, str], str]) -> str:
    reference = cell.get("ref")
    if reference is not None:
        return references.get((cell.tag, reference), "")
    rendered = cell.get("fmt")
    if rendered is None:
        rendered = "".join(cell.itertext()).strip()
    identifier = cell.get("id")
    if identifier is not None:
        references[(cell.tag, identifier)] = rendered
    for child in cell.iter():
        child_id = child.get("id")
        if child_id is not None:
            references[(child.tag, child_id)] = child.get("fmt", "".join(child.itertext()).strip())
    return rendered


def table(root: ET.Element) -> tuple[list[str], list[list[str]]]:
    node = root.find(".//node")
    schema = None if node is None else node.find("schema")
    if node is None or schema is None:
        return [], []
    columns = [col.findtext("mnemonic", default="") for col in schema.findall("col")]
    references: dict[tuple[str, str], str] = {}
    rows = [[value(cell, references) for cell in row] for row in node.findall("row")]
    return columns, rows


def frequencies(columns: list[str], rows: list[list[str]], name: str) -> dict[str, int]:
    if name not in columns:
        return {}
    index = columns.index(name)
    counts = collections.Counter(row[index] for row in rows if index < len(row) and row[index])
    return dict(counts.most_common())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.trace.exists():
        parser.error(f"trace does not exist: {args.trace}")
    destination = args.output or args.trace.parent / "metal-work-summary.json"
    export_dir = destination.parent / "metal-export"
    export_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "scope": "Metal trace work counts only; trace overhead is not a performance result.",
        "counting_contract": (
            "Metal Application tables are target-scoped. The system-wide GPU "
            "instrument is deliberately excluded. An encoder is not a kernel dispatch."
        ),
        "trace": str(args.trace),
        "tables": {},
    }
    parsed: dict[str, tuple[list[str], list[list[str]]]] = {}
    for schema in SCHEMAS:
        root = export(args.trace, schema, export_dir / f"{schema}.xml")
        columns, rows = table(root)
        parsed[schema] = columns, rows
        summary["tables"][schema] = {
            "rows": len(rows),
            "event_types": frequencies(columns, rows, "event-type"),
            "channels": frequencies(columns, rows, "channel-name"),
            "top_labels": dict(list(frequencies(columns, rows, "event-label").items())[:100]),
        }

    app_columns, app_rows = parsed["metal-application-intervals"]
    dispatch_rows = []
    for row in app_rows:
        fields = dict(zip(app_columns, row))
        kind = fields.get("event-type", "")
        label = fields.get("event-label", "")
        if "dispatch" in f"{kind} {label}".lower():
            dispatch_rows.append(fields)
    summary["command_buffer_submissions"] = len(
        parsed["metal-application-command-buffer-submissions"][1]
    )
    summary["encoder_creations"] = len(parsed["metal-application-encoders-list"][1])
    summary["encoder_count_status"] = "target Metal Application encoder rows"
    summary["kernel_dispatch_rows"] = len(dispatch_rows) if dispatch_rows else None
    summary["kernel_dispatch_count_status"] = (
        "observed_dispatch_events" if dispatch_rows else
        "unresolved: this trace exposed no dispatch event rows; do not substitute encoder count"
    )
    if dispatch_rows:
        summary["dispatch_event_types"] = dict(collections.Counter(
            row.get("event-type", "") for row in dispatch_rows
        ).most_common())
        summary["dispatch_labels"] = dict(collections.Counter(
            row.get("event-label", "") for row in dispatch_rows
        ).most_common(200))

    destination.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
