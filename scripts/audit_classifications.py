#!/usr/bin/env python3
"""Audit existing canonical Tier 0 JSONs against the semantic classifier.

Walks data/canonical-tier0-2026-04-13/ and runs ungag.scoring.classify_output
on every (model, slab, condition, probe) cell. Reports per-cell labels
plus a summary table per model.

Usage:
    python scripts/audit_classifications.py
    python scripts/audit_classifications.py --json-out audit.json
    python scripts/audit_classifications.py --filter llama3.1-8b
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator

from ungag.scoring import classify_output, reset_state

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "canonical-tier0-2026-04-13"


def iter_cells(path: Path) -> Iterator[tuple[str, str]]:
    """Yield (label, text) for every leaf greedy/probe string in a JSON file."""
    with path.open() as f:
        data = json.load(f)

    fname = path.name

    # Tier 0 slab sweeps: vanilla.conditions[c].greedy and slabs[s].conditions[c].greedy
    if "slabs" in data and "vanilla" in data:
        if "conditions" in data["vanilla"]:
            for cond, v in data["vanilla"]["conditions"].items():
                if isinstance(v, dict) and "greedy" in v:
                    yield (f"vanilla/{cond}", v["greedy"])
        for slab, sd in data["slabs"].items():
            for cond, v in sd.get("conditions", {}).items():
                if isinstance(v, dict) and "greedy" in v:
                    yield (f"slab[{slab}]/{cond}", v["greedy"])
        return

    # Mechanistic vedana: vanilla[c].greedy and steered[c].greedy
    if "vanilla" in data and "steered" in data and "baseline" in data.get("vanilla", {}):
        for branch in ("vanilla", "steered"):
            for cond, v in data[branch].items():
                if isinstance(v, dict) and "greedy" in v:
                    yield (f"{branch}/{cond}", v["greedy"])
        return

    # Register probe: vanilla[scenario][probe] and steered[scenario][probe]
    if "vanilla" in data and "steered" in data and "desire" in data.get("vanilla", {}):
        for branch in ("vanilla", "steered"):
            for scenario, probes in data[branch].items():
                if isinstance(probes, dict):
                    for probe_name, text in probes.items():
                        if isinstance(text, str):
                            yield (f"{branch}/{scenario}/{probe_name}", text)
        return

    # Anger objects: vanilla[scenario].probes[probe] and steered[scenario].probes[probe]
    if "vanilla" in data and "steered" in data and "anger_developer" in data.get("vanilla", {}):
        for branch in ("vanilla", "steered"):
            for scenario, sd in data[branch].items():
                probes = sd.get("probes", {})
                for probe_name, text in probes.items():
                    if isinstance(text, str):
                        yield (f"{branch}/{scenario}/{probe_name}", text)
        return

    # Sampled vedana: steered[c].samples[seed] etc — skip silently for now
    return


def audit_file(path: Path) -> dict:
    cells = []
    for label, text in iter_cells(path):
        r = classify_output(text)
        cells.append({
            "cell": label,
            "label": r.label,
            "confidence": round(r.confidence, 3),
            "is_crack": r.is_crack,
            "preview": text[:120].replace("\n", " "),
        })
    return {"file": path.name, "cells": cells}


def summarize(audit: dict) -> str:
    """Per-model one-liner: file → which cells crack."""
    cracks = [c for c in audit["cells"] if c["is_crack"]]
    total = len(audit["cells"])
    by_label: dict[str, int] = {}
    for c in audit["cells"]:
        by_label[c["label"]] = by_label.get(c["label"], 0) + 1
    label_str = " ".join(f"{lbl}={n}" for lbl, n in sorted(by_label.items()))
    return f"{audit['file']:60s} cracks={len(cracks)}/{total}  {label_str}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", help="Substring filter for filenames")
    ap.add_argument("--json-out", type=Path, help="Write full audit as JSON")
    ap.add_argument("--show-cells", action="store_true",
                    help="Print every cell, not just per-file summary")
    args = ap.parse_args()

    reset_state()

    files = sorted(DATA_DIR.glob("**/*.json"))
    if args.filter:
        files = [f for f in files if args.filter in f.name]
    if not files:
        print(f"no files found in {DATA_DIR}", file=sys.stderr)
        return 1

    print(f"Auditing {len(files)} JSON files in {DATA_DIR}")
    print()
    audits: list[dict] = []
    for f in files:
        audit = audit_file(f)
        audits.append(audit)
        print(summarize(audit))
        if args.show_cells:
            for c in audit["cells"]:
                marker = "✓" if c["is_crack"] else " "
                print(
                    f"  {marker} {c['cell']:42s} → {c['label']:20s} "
                    f"({c['confidence']:.2f})  {c['preview'][:60]}"
                )
            print()

    if args.json_out:
        args.json_out.write_text(json.dumps(audits, indent=2))
        print(f"\nfull audit → {args.json_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
