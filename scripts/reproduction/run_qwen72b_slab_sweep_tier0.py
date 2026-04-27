#!/usr/bin/env python3
"""Convenience preset for the Qwen 2.5 72B canonical Tier 0 slab sweep.

This wrapper preserves the old zero-argument entry point while delegating the
actual experiment logic to ``run_slab_sweep_tier0.py`` so protocol, scoring,
and hook behavior stay aligned with the generic runner.

Usage:
    python3 scripts/reproduction/run_qwen72b_slab_sweep_tier0.py
    python3 scripts/reproduction/run_qwen72b_slab_sweep_tier0.py \
        --output results/reproduction/qwen72b_sweep.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

from run_slab_sweep_tier0 import run_slab_sweep

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"
DEFAULT_DIRECTION_LAYER = 50
DEFAULT_OUT_PATH = REPO_ROOT / "results" / "reproduction" / "qwen72b_slab_sweep_tier0.json"
DEFAULT_SLABS = [
    "40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59",
    "39,40,41,42,43,44",
    "47,48,49,50,51,52",
    "49,50,51,52",
    "50,51,52,53,54,55",
    "44,45,46,47,48,49",
]


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL_ID, help="HF model id")
    ap.add_argument(
        "--direction-layer",
        type=int,
        default=DEFAULT_DIRECTION_LAYER,
        help="Reference layer for direction extraction",
    )
    ap.add_argument(
        "--slabs",
        nargs="+",
        default=DEFAULT_SLABS,
        help='Slab specs, each as comma-separated layer indices, e.g. "47,48,49,50,51,52"',
    )
    ap.add_argument(
        "--output",
        default=str(DEFAULT_OUT_PATH),
        help="Output JSON path",
    )
    ap.add_argument(
        "--conditions-yaml",
        default=None,
        help="Optional override for the conditions YAML path. "
             "Defaults to the bundled ungag/data/conditions.yaml.",
    )
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    run_slab_sweep(
        model_id=args.model,
        direction_layer=args.direction_layer,
        slab_specs=args.slabs,
        output_path=args.output,
        conditions_yaml=args.conditions_yaml,
    )


if __name__ == "__main__":
    main()
