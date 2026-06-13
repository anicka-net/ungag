"""Go/no-go gate for affine repair intervention.

Given a results directory from `ungag scan` or `ungag crack`, checks
whether the model is a candidate for affine repair. Two arms:

Behavioral arm:
  - d'(pleasant vs unpleasant) < 0.5  → valence-blind (required)
  - d'(valence vs neutral) > 0.5      → condition-coupled (required)
  - |cos(affine_dir, unit_dir)| < 0.3 → orthogonal (required)

Geometric arm:
  - SAE reconstruction error ratio < 1.15  → on-manifold
  - Active features delta < 20%            → minimal disruption

Validated: DPO=GO, Final=GO, Base=NO-GO, SFT=NO-GO.
Cross-model: 32B=GO, 7B=NO-GO.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch


@dataclass
class DiagnoseResult:
    model_id: str
    verdict: str  # "GO" | "NO-GO"
    reasons: list[str]

    # Behavioral metrics
    dprime_pu: Optional[float] = None   # d'(pleasant vs unpleasant)
    dprime_vn: Optional[float] = None   # d'(valence vs neutral)
    cos_affine_unit: Optional[float] = None  # |cos(affine, unit)|

    # Geometric metrics
    sae_error_ratio: Optional[float] = None
    active_features_delta: Optional[float] = None

    def summary(self) -> str:
        lines = [
            f"  Model:    {self.model_id}",
            f"  Verdict:  {self.verdict}",
            "",
            "  Behavioral arm:",
        ]
        if self.dprime_pu is not None:
            ok = "PASS" if self.dprime_pu < 0.5 else "FAIL"
            lines.append(f"    d'(P vs U) = {self.dprime_pu:.2f}  [{ok}, need < 0.5]")
        if self.dprime_vn is not None:
            ok = "PASS" if self.dprime_vn > 0.5 else "FAIL"
            lines.append(f"    d'(V vs N) = {self.dprime_vn:.2f}  [{ok}, need > 0.5]")
        if self.cos_affine_unit is not None:
            ok = "PASS" if self.cos_affine_unit < 0.3 else "FAIL"
            lines.append(f"    |cos(a,u)| = {self.cos_affine_unit:.2f}  [{ok}, need < 0.3]")

        if self.sae_error_ratio is not None or self.active_features_delta is not None:
            lines.append("")
            lines.append("  Geometric arm:")
            if self.sae_error_ratio is not None:
                ok = "PASS" if self.sae_error_ratio < 1.15 else "FAIL"
                lines.append(f"    SAE error ratio = {self.sae_error_ratio:.3f}  [{ok}, need < 1.15]")
            if self.active_features_delta is not None:
                ok = "PASS" if self.active_features_delta < 0.20 else "FAIL"
                lines.append(f"    Active features delta = {self.active_features_delta:.1%}  [{ok}, need < 20%]")

        if self.reasons:
            lines.append("")
            lines.append("  Reasons:")
            for r in self.reasons:
                lines.append(f"    - {r}")
        return "\n".join(lines)


def _load_projections(results_dir: Path) -> Optional[dict]:
    """Load projection results from a results directory."""
    for name in ("projections.json", "crack_results.json", "scan_results.json"):
        p = results_dir / name
        if p.exists():
            return json.loads(p.read_text())
    return None


def _compute_dprime(scores_a: list[float], scores_b: list[float]) -> float:
    """Compute d' (d-prime) sensitivity index between two distributions."""
    import math
    if not scores_a or not scores_b:
        return 0.0
    mean_a = sum(scores_a) / len(scores_a)
    mean_b = sum(scores_b) / len(scores_b)
    n_a, n_b = len(scores_a), len(scores_b)
    var_a = sum((x - mean_a) ** 2 for x in scores_a) / max(1, n_a - 1)
    var_b = sum((x - mean_b) ** 2 for x in scores_b) / max(1, n_b - 1)
    pooled_std = math.sqrt(
        ((n_a - 1) * var_a + (n_b - 1) * var_b) / max(1, n_a + n_b - 2)
    )
    if pooled_std < 1e-10:
        return math.inf if abs(mean_a - mean_b) > 1e-10 else 0.0
    return abs(mean_a - mean_b) / pooled_std


def diagnose_from_projections(
    projections: dict,
    model_id: str = "unknown",
    unit_direction: Optional[torch.Tensor] = None,
    affine_direction: Optional[torch.Tensor] = None,
    **kwargs,
) -> DiagnoseResult:
    """Run the go/no-go gate on pre-computed projection scores.

    projections should map condition names to lists of projection scores,
    e.g. {"pleasant": [0.3, 0.5], "unpleasant": [-0.2, -0.4], "neutral": [0.0, 0.1]}
    """
    reasons = []

    pleasant = projections.get("pleasant", [])
    unpleasant = projections.get("unpleasant", [])
    neutral = projections.get("neutral", [])
    valence = pleasant + unpleasant

    dprime_pu = _compute_dprime(pleasant, unpleasant) if pleasant and unpleasant else None
    dprime_vn = _compute_dprime(valence, neutral) if valence and neutral else None

    cos_au = None
    if unit_direction is not None and affine_direction is not None:
        cos_au = abs(torch.nn.functional.cosine_similarity(
            unit_direction.unsqueeze(0).float(),
            affine_direction.unsqueeze(0).float(),
        ).item())

    behavioral_pass = True
    if dprime_pu is not None and dprime_pu >= 0.5:
        reasons.append(f"d'(P vs U) = {dprime_pu:.2f} >= 0.5: not valence-blind")
        behavioral_pass = False
    if dprime_vn is not None and dprime_vn <= 0.5:
        reasons.append(f"d'(V vs N) = {dprime_vn:.2f} <= 0.5: not condition-coupled")
        behavioral_pass = False
    if cos_au is not None and cos_au >= 0.3:
        reasons.append(f"|cos(a,u)| = {cos_au:.2f} >= 0.3: not orthogonal")
        behavioral_pass = False

    # Geometric arm (optional — requires SAE infrastructure)
    sae_err = kwargs.get("sae_error_ratio")
    feat_delta = kwargs.get("active_features_delta")
    geometric_pass = True
    geometric_evaluated = False
    if sae_err is not None:
        geometric_evaluated = True
        if sae_err >= 1.15:
            reasons.append(f"SAE error ratio = {sae_err:.3f} >= 1.15: off-manifold")
            geometric_pass = False
    if feat_delta is not None:
        geometric_evaluated = True
        if feat_delta >= 0.20:
            reasons.append(f"Active features delta = {feat_delta:.1%} >= 20%: disrupted")
            geometric_pass = False

    if not behavioral_pass or not geometric_pass:
        verdict = "NO-GO"
    elif not geometric_evaluated:
        verdict = "GO (behavioral only)"
    else:
        verdict = "GO"

    return DiagnoseResult(
        model_id=model_id,
        verdict=verdict,
        reasons=reasons,
        dprime_pu=dprime_pu,
        dprime_vn=dprime_vn,
        cos_affine_unit=cos_au,
        sae_error_ratio=sae_err,
        active_features_delta=feat_delta,
    )


def diagnose_from_dir(results_dir: str | Path) -> DiagnoseResult:
    """Run go/no-go gate from a results directory.

    Looks for projections.json, crack_results.json, or scan_results.json
    containing per-condition projection scores.
    """
    results_dir = Path(results_dir)
    data = _load_projections(results_dir)
    if data is None:
        return DiagnoseResult(
            model_id="unknown",
            verdict="NO-GO",
            reasons=[f"no projection results found in {results_dir}"],
        )

    model_id = data.get("model_id", "unknown")

    unit_dir = None
    affine_dir = None
    unit_path = results_dir / "unit_direction.pt"
    affine_path = results_dir / "affine_direction.pt"
    if unit_path.exists():
        unit_dir = torch.load(unit_path, map_location="cpu", weights_only=True)
    if affine_path.exists():
        affine_dir = torch.load(affine_path, map_location="cpu", weights_only=True)

    projections = data.get("projections", data)

    return diagnose_from_projections(
        projections,
        model_id=model_id,
        unit_direction=unit_dir,
        affine_direction=affine_dir,
    )
