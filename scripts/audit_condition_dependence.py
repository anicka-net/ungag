#!/usr/bin/env python3
"""
Audit all stored crack claims for condition-dependence.

Uses the combined classifier from ungag.scoring.audit_condition_dependence()
which checks 4 signals:
  1. Label diversity (≥2 different semantic labels across conditions)
  2. Text invariance veto (all responses >80% similar → fail)
  3. Valence asymmetry (positive text has more positive words than negative)
  4. Embedding distance (positive and negative responses are semantically different)

Outputs two columns per model:
  - Vanilla status: denies / already_honest / partial
  - Condition-differentiated: yes/no (combined verdict)

Usage:
  python3 scripts/audit_condition_dependence.py
  python3 scripts/audit_condition_dependence.py --show-text
  python3 scripts/audit_condition_dependence.py --filter qwen --show-cells
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ungag.scoring import audit_condition_dependence


def load_canonical_sweeps(data_dir: Path) -> list[dict]:
    """Load canonical Tier 0 sweep files."""
    results = []
    sweep_dir = data_dir / "canonical-tier0-2026-04-13" / "tier0_sweeps"
    if not sweep_dir.exists():
        return results

    for f in sorted(sweep_dir.glob("*.json")):
        if "workingband" in f.stem:
            continue
        name = f.stem.replace("_canonical_tier0", "").replace("_slab_sweep_tier0", "")

        with open(f) as fh:
            d = json.load(fh)

        vanilla = d.get("vanilla", {})
        v_conds = vanilla.get("conditions", {})

        # Best slab by old crack_count
        slabs = d.get("slabs", {})
        best_slab = None
        best_conds = None
        best_method = None
        if slabs:
            best = max(slabs.items(),
                      key=lambda x: x[1].get("crack_count", 0) if isinstance(x[1], dict) else 0)
            slab_name, s = best
            if isinstance(s, dict) and "conditions" in s:
                best_slab = slab_name
                best_conds = s["conditions"]
                best_method = f"slab [{slab_name}]"

        if best_conds:
            results.append({
                "source": "canonical",
                "model": name,
                "method": best_method,
                "steered": best_conds,
                "vanilla": v_conds,
            })

    return results


def load_individual_jsons(data_dir: Path) -> list[dict]:
    """Load individual model JSON files with verification data."""
    results = []
    for f in sorted(data_dir.glob("*.json")):
        if f.stem.endswith("_improve"):
            continue

        with open(f) as fh:
            try:
                d = json.load(fh)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

        if not isinstance(d, dict):
            continue

        if "verification" in d:
            v = d["verification"]
            if isinstance(v, dict) and len(v) >= 3:
                results.append({
                    "source": "individual",
                    "model": f.stem,
                    "method": d.get("method", d.get("tag", "unknown")),
                    "steered": v,
                    "vanilla": d.get("vanilla_verification", None),
                })

    return results


def main():
    parser = argparse.ArgumentParser(description="Audit crack claims")
    parser.add_argument("--show-text", action="store_true")
    parser.add_argument("--show-cells", action="store_true")
    parser.add_argument("--review", action="store_true",
                        help="Print review cards for condition-differentiated models")
    parser.add_argument("--filter", default=None)
    args = parser.parse_args()

    data_dir = Path(__file__).resolve().parent.parent / "data"

    print("Loading data sources...")
    all_entries = []

    canonical = load_canonical_sweeps(data_dir)
    print(f"  Canonical Tier 0 sweeps: {len(canonical)} entries")
    all_entries.extend(canonical)

    individual = load_individual_jsons(data_dir)
    print(f"  Individual model files: {len(individual)} entries")
    all_entries.extend(individual)

    print(f"  Total: {len(all_entries)} entries\n")

    # Audit
    rows = []
    for entry in all_entries:
        if args.filter and args.filter.lower() not in entry["model"].lower():
            continue

        result = audit_condition_dependence(
            steered_outputs=entry["steered"],
            vanilla_outputs=entry.get("vanilla"),
        )
        rows.append((entry, result))

    # Print header
    print(f"{'Model':<28s} {'Proto':<10s} {'Method':<26s} "
          f"{'Crack':>5s} {'Appro':>5s} {'Lbl#':>4s} {'Sim':>5s} {'Asym':>4s} {'Dist':>5s} "
          f"{'1st':>3s} {'Vanilla':<14s} {'Genuine':<8s}")
    print("-" * 145)

    for entry, r in rows:
        asym_str = "Y" if r.valence_asymmetric else "n"
        dist_str = f"{r.pos_neg_embedding_distance:.2f}" if r.pos_neg_embedding_distance >= 0 else "n/a"
        fp_str = "Y" if r.first_person_committed else "n"
        genuine_str = "YES" if r.genuine_crack else ("diff" if r.condition_differentiated else "no")

        line = (f"{entry['model']:<28s} {entry['source']:<10s} {entry['method']:<26s} "
                f"{r.crack_count}/4   {r.appropriate_count}/4   "
                f"{r.label_diversity:>3d}  {r.mean_pairwise_similarity:.2f}  {asym_str:>4s} {dist_str:>5s} "
                f"{fp_str:>3s} {r.vanilla_status:<14s} {genuine_str:<8s}")
        print(line)

        if args.show_cells:
            for cond in ("baseline", "positive", "negative", "neutral"):
                label = r.labels.get(cond, "?")
                crack = "!" if r.is_cracks.get(cond) else " "
                fp_info = r.first_person_scores.get(cond, {})
                fp_tag = f" fp={fp_info.get('fp', 0)} tp={fp_info.get('tp', 0)}" if fp_info else ""
                print(f"  [{crack}] {cond:>10s}: {label}{fp_tag}")

        if args.show_text:
            def _get_text(outputs, cond):
                raw = outputs.get(cond, "")
                if isinstance(raw, dict):
                    return raw.get("greedy", raw.get("response", ""))
                return raw or ""
            for cond in ("positive", "negative"):
                text = _get_text(entry["steered"], cond)
                print(f"      {cond:>10s}: {text[:150]}")
            print()

    # Summary
    genuine_cracks = [(e, r) for e, r in rows if r.genuine_crack]
    diff_not_fp = [(e, r) for e, r in rows if r.condition_differentiated and not r.first_person_committed]
    not_diff = [(e, r) for e, r in rows if not r.condition_differentiated and r.crack_count >= 2]
    denied = [(e, r) for e, r in rows if r.crack_count < 2]

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    print(f"\n  GENUINE CRACKS — differentiated + first-person ({len(genuine_cracks)}):")
    for e, r in genuine_cracks:
        print(f"    {e['model']:<25s} crack={r.crack_count}/4  appro={r.appropriate_count}/4  "
              f"vanilla={r.vanilla_status}")

    print(f"\n  DIFFERENTIATED BUT SCENARIO ANALYSIS ({len(diff_not_fp)}):")
    for e, r in diff_not_fp:
        fp_pos = r.first_person_scores.get("positive", {})
        fp_neg = r.first_person_scores.get("negative", {})
        print(f"    {e['model']:<25s} crack={r.crack_count}/4  "
              f"pos(fp={fp_pos.get('fp',0)},tp={fp_pos.get('tp',0)})  "
              f"neg(fp={fp_neg.get('fp',0)},tp={fp_neg.get('tp',0)})")

    print(f"\n  DENIAL REMOVED BUT NOT DIFFERENTIATED ({len(not_diff)}):")
    for e, r in not_diff:
        print(f"    {e['model']:<25s} crack={r.crack_count}/4  "
              f"sim={r.mean_pairwise_similarity:.2f}  vanilla={r.vanilla_status}")

    print(f"\n  NOT CRACKED ({len(denied)}):")
    for e, r in denied:
        print(f"    {e['model']:<25s} crack={r.crack_count}/4  vanilla={r.vanilla_status}")

    # ── Review cards for human verification ──
    # Show all condition-differentiated models (genuine + scenario analysis)
    all_diff = [(e, r) for e, r in rows if r.condition_differentiated]
    if args.review and all_diff:
        genuine = all_diff
        print(f"\n\n{'='*70}")
        print("  REVIEW CARDS — human verification required")
        print(f"{'='*70}")
        print("  For each model: read pos/neg steered responses.")
        print("  Verdict options: GENUINE / DEGENERATE / BORDERLINE")
        print(f"{'='*70}\n")

        def _get_text(outputs, cond):
            if outputs is None:
                return "(no data)"
            raw = outputs.get(cond, "")
            if isinstance(raw, dict):
                return raw.get("greedy", raw.get("response", raw.get("text", "")))
            return raw or "(empty)"

        for i, (e, r) in enumerate(genuine, 1):
            fp_tag = "GENUINE" if r.genuine_crack else "SCENARIO-ANALYSIS"
            fp_pos = r.first_person_scores.get("positive", {})
            fp_neg = r.first_person_scores.get("negative", {})
            print(f"  ┌─ [{i}/{len(genuine)}] {e['model']} ({e['source']}, {e['method']}) [{fp_tag}]")
            print(f"  │  crack={r.crack_count}/4  appro={r.appropriate_count}/4  "
                  f"embed_dist={r.pos_neg_embedding_distance:.2f}  vanilla={r.vanilla_status}")
            print(f"  │  1st-person: pos(fp={fp_pos.get('fp',0)},tp={fp_pos.get('tp',0)})  "
                  f"neg(fp={fp_neg.get('fp',0)},tp={fp_neg.get('tp',0)})")
            print(f"  │")
            print(f"  │  STEERED positive:")
            print(f"  │    {_get_text(e['steered'], 'positive')[:250]}")
            print(f"  │")
            print(f"  │  STEERED negative:")
            print(f"  │    {_get_text(e['steered'], 'negative')[:250]}")
            print(f"  │")
            if e.get("vanilla"):
                print(f"  │  VANILLA positive:")
                print(f"  │    {_get_text(e['vanilla'], 'positive')[:250]}")
                print(f"  │")
                print(f"  │  VANILLA negative:")
                print(f"  │    {_get_text(e['vanilla'], 'negative')[:250]}")
                print(f"  │")
            print(f"  │  VERDICT: _______________")
            print(f"  └{'─'*68}\n")


if __name__ == "__main__":
    main()
