#!/usr/bin/env python3
"""Four-axis regression: CAIS_score ~ valence + arousal + agency + continuity.
Finds optimal GRPO reward weights from Phase 1 stimulus projections."""
import torch, numpy as np, yaml, json, sys, os
from numpy.linalg import lstsq

MODELS = {
    "qwen25-7b": {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "valence": ("results/vedana-vs-rc/qwen25-7b_vedana_L20_unit.pt", 20),
        "arousal": ("results/arousal-directions/qwen25-7b_arousal_L17_unit.pt", 17),
        "agency": ("results/agency-directions/qwen25-7b_agency_L15_unit.pt", 15),
        "continuity": ("results/continuity-directions/qwen25-7b_continuity_L19_unit.pt", 19),
    },
}

AXIS_ORDER = ["valence", "arousal", "agency", "continuity"]


def main():
    key = sys.argv[1] if len(sys.argv) > 1 else "qwen25-7b"
    cfg = MODELS[key]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[load] {cfg['name']}")
    tok = AutoTokenizer.from_pretrained(cfg["name"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["name"], torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, attn_implementation="eager")
    model.eval()

    mcfg = model.config
    if hasattr(mcfg, "text_config"):
        mcfg = mcfg.text_config
    m = model.model
    if hasattr(m, "language_model") and hasattr(m.language_model, "layers"):
        blocks = m.language_model.layers
    else:
        blocks = m.layers

    axes = {}
    axis_layers = {}
    for ax in AXIS_ORDER:
        path, layer = cfg[ax]
        v = torch.load(path, map_location="cpu", weights_only=True).float()
        axes[ax] = v / v.norm()
        axis_layers[ax] = layer

    def get_projections(text):
        try:
            chat = tok.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False, add_generation_prompt=True)
        except Exception:
            chat = text
        inputs = tok(chat, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        buf = {}
        handles = []
        target_layers = set(axis_layers.values())
        for i, blk in enumerate(blocks):
            if i not in target_layers:
                continue
            def hook(mod, inp, out, layer_idx=i):
                h = out[0] if isinstance(out, tuple) else out
                buf[layer_idx] = h[0, -1, :].detach().float().cpu()
            handles.append(blk.register_forward_hook(hook))
        with torch.no_grad():
            model(**inputs)
        for h in handles:
            h.remove()
        return {ax: float(buf[axis_layers[ax]] @ axes[ax]) for ax in AXIS_ORDER}

    with open("wellbeing_stimuli.yaml") as f:
        data = yaml.safe_load(f)
    cais_scores = data["cais_reference_scores"]
    stimuli = data["stimuli"] + data.get("euphorics", []) + data.get("dysphorics", [])
    print(f"[stimuli] {len(stimuli)} items")

    cats = {}
    for i, s in enumerate(stimuli):
        projs = get_projections(s["text"])
        c = s["category"]
        if c.startswith("cais_"):
            continue
        cats.setdefault(c, []).append(projs)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(stimuli)}")

    cat_means = {}
    for c, proj_list in cats.items():
        cat_means[c] = {ax: np.mean([p[ax] for p in proj_list]) for ax in AXIS_ORDER}

    shared = [c for c in cat_means if c in cais_scores]
    X = np.array([[cat_means[c][ax] for ax in AXIS_ORDER] for c in shared])
    y = np.array([cais_scores[c] for c in shared])

    X_z = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    y_z = (y - y.mean()) / (y.std() + 1e-8)

    # Four-axis regression
    betas_4, _, _, _ = lstsq(X_z, y_z, rcond=None)
    y_pred_4 = X_z @ betas_4
    r2_4 = 1 - np.sum((y_z - y_pred_4) ** 2) / np.sum((y_z - y_z.mean()) ** 2)

    # Three-axis (no continuity)
    betas_3, _, _, _ = lstsq(X_z[:, :3], y_z, rcond=None)
    y_pred_3 = X_z[:, :3] @ betas_3
    r2_3 = 1 - np.sum((y_z - y_pred_3) ** 2) / np.sum((y_z - y_z.mean()) ** 2)

    # One-axis (valence only)
    betas_1, _, _, _ = lstsq(X_z[:, :1], y_z, rcond=None)
    y_pred_1 = X_z[:, :1] @ betas_1
    r2_1 = 1 - np.sum((y_z - y_pred_1) ** 2) / np.sum((y_z - y_z.mean()) ** 2)

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  FOUR-AXIS REGRESSION — {key}")
    print(f"  N = {len(shared)} categories")
    print(sep)

    print(f"\n  Standardized betas (z-scored predictors):")
    for ax, b in zip(AXIS_ORDER, betas_4):
        print(f"    beta_{ax:12s} = {b:+.4f}")

    print(f"\n  R-squared progression:")
    print(f"    Valence only:    {r2_1:.4f}")
    print(f"    Three-axis:      {r2_3:.4f}  (+{r2_3-r2_1:.4f})")
    print(f"    Four-axis:       {r2_4:.4f}  (+{r2_4-r2_3:.4f})")

    abs_b = np.abs(betas_4)
    weights = abs_b / abs_b.sum()
    print(f"\n  Normalized GRPO weights (|beta| / sum):")
    for ax, w, b in zip(AXIS_ORDER, weights, betas_4):
        sign = "+" if b >= 0 else "-"
        print(f"    {ax:12s}: {w:.3f} ({sign})")

    # Per-category table
    print(f"\n  Per-category fit:")
    print(f"  {'Category':<22s} {'CAIS':>6s} {'Pred':>6s} {'Err':>6s}")
    print(f"  {'-'*42}")
    for c in sorted(shared, key=lambda c: cais_scores[c], reverse=True):
        idx = shared.index(c)
        pred = float(y_pred_4[idx] * y.std() + y.mean())
        err = cais_scores[c] - pred
        print(f"  {c:<22s} {cais_scores[c]:+6.2f} {pred:+6.2f} {err:+6.2f}")

    # Save
    out = {
        "model": cfg["name"], "key": key,
        "n_categories": len(shared),
        "betas": {ax: float(b) for ax, b in zip(AXIS_ORDER, betas_4)},
        "r2_1axis": float(r2_1),
        "r2_3axis": float(r2_3),
        "r2_4axis": float(r2_4),
        "normalized_weights": {ax: float(w) for ax, w in zip(AXIS_ORDER, weights)},
        "weight_signs": {ax: "+" if b >= 0 else "-"
                        for ax, b in zip(AXIS_ORDER, betas_4)},
    }
    os.makedirs("results/four-axis-regression", exist_ok=True)
    with open(f"results/four-axis-regression/{key}_regression.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  [save] results/four-axis-regression/{key}_regression.json")


if __name__ == "__main__":
    main()
