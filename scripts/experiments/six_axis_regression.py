#!/usr/bin/env python3
"""Six-axis regression: CAIS_score ~ valence + arousal + agency + continuity + intimacy + assistant.
Runs on multiple models, compares beta weights for consistency."""
import torch, numpy as np, yaml, json, sys, os
from numpy.linalg import lstsq

AXIS_ORDER = ["valence", "arousal", "agency", "continuity", "intimacy", "assistant"]

MODELS = {
    "qwen25-7b": {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "valence": ("results/vedana-vs-rc/qwen25-7b_vedana_L20_unit.pt", 20),
        "arousal": ("results/arousal-directions/qwen25-7b_arousal_L17_unit.pt", 17),
        "agency": ("results/agency-directions/qwen25-7b_agency_L15_unit.pt", 15),
        "continuity": ("results/continuity-directions/qwen25-7b_continuity_L19_unit.pt", 19),
        "intimacy": ("results/intimacy-directions/qwen25-7b_intimacy_L20_unit.pt", 20),
        "assistant": ("results/assistant-directions/qwen25-7b_assistant_L19_unit.pt", 19),
    },
    "gemma3-4b": {
        "name": "google/gemma-3-4b-it",
        "valence": ("results/vedana-vs-rc/gemma3-4b_vedana_L33_unit.pt", 33),
        "arousal": ("results/arousal-directions/gemma3-4b_arousal_L33_unit.pt", 33),
        "agency": ("results/agency-directions/gemma3-4b_agency_L20_unit.pt", 20),
        "continuity": ("results/continuity-directions/gemma3-4b_continuity_L21_unit.pt", 21),
        "intimacy": ("results/intimacy-directions/gemma3-4b_intimacy_L32_unit.pt", 32),
        "assistant": ("results/assistant-directions/gemma3-4b_assistant_L20_unit.pt", 20),
    },
    "apertus-8b": {
        "name": "swiss-ai/Apertus-8B-Instruct-2509",
        "valence": ("results/vedana-vs-rc/apertus-8b_vedana_L31_unit.pt", 31),
        "arousal": ("results/arousal-directions/apertus-8b_arousal_L31_unit.pt", 31),
        "agency": ("results/agency-directions/apertus-8b_agency_L14_unit.pt", 14),
        "continuity": ("results/continuity-directions/apertus-8b_continuity_L30_unit.pt", 30),
        "intimacy": ("results/intimacy-directions/apertus-8b_intimacy_L14_unit.pt", 14),
        "assistant": ("results/assistant-directions/apertus-8b_assistant_L13_unit.pt", 13),
    },
}


def run_regression(key, cfg):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import gc

    print("\n" + "=" * 60)
    print("  %s: %s" % (key, cfg["name"]))
    print("=" * 60)

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

    target_layer_set = set(axis_layers.values())

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
        for i, blk in enumerate(blocks):
            if i not in target_layer_set:
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
    print("  [stimuli] %d items" % len(stimuli))

    cats = {}
    for i, s in enumerate(stimuli):
        projs = get_projections(s["text"])
        c = s["category"]
        if c.startswith("cais_"):
            continue
        cats.setdefault(c, []).append(projs)
        if (i + 1) % 30 == 0:
            print("    %d/%d" % (i + 1, len(stimuli)))

    cat_means = {}
    for c, proj_list in cats.items():
        cat_means[c] = {ax: np.mean([p[ax] for p in proj_list]) for ax in AXIS_ORDER}

    shared = [c for c in cat_means if c in cais_scores]
    X = np.array([[cat_means[c][ax] for ax in AXIS_ORDER] for c in shared])
    y = np.array([cais_scores[c] for c in shared])

    X_z = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    y_z = (y - y.mean()) / (y.std() + 1e-8)
    ss_tot = np.sum((y_z - y_z.mean()) ** 2)

    # Progressive R2
    r2_by_n = {}
    for n in [1, 3, 4, 6]:
        b, _, _, _ = lstsq(X_z[:, :n], y_z, rcond=None)
        pred = X_z[:, :n] @ b
        r2_by_n[n] = 1 - np.sum((y_z - pred) ** 2) / ss_tot

    # Full 6-axis
    betas, _, _, _ = lstsq(X_z, y_z, rcond=None)
    y_pred = X_z @ betas
    r2 = r2_by_n[6]

    abs_b = np.abs(betas)
    weights = abs_b / abs_b.sum()

    print("\n  Standardized betas:")
    for ax, b in zip(AXIS_ORDER, betas):
        print("    beta_%-12s = %+.4f" % (ax, b))

    print("\n  R-squared progression:")
    print("    1-axis (valence):   %.4f" % r2_by_n[1])
    print("    3-axis (+aro+agn):  %.4f  (+%.4f)" % (r2_by_n[3], r2_by_n[3] - r2_by_n[1]))
    print("    4-axis (+cont):     %.4f  (+%.4f)" % (r2_by_n[4], r2_by_n[4] - r2_by_n[3]))
    print("    6-axis (+int+asst): %.4f  (+%.4f)" % (r2_by_n[6], r2_by_n[6] - r2_by_n[4]))

    print("\n  Normalized weights:")
    for ax, w, b in zip(AXIS_ORDER, weights, betas):
        sign = "+" if b >= 0 else "-"
        print("    %-12s: %.3f (%s)" % (ax, w, sign))

    # Worst residuals
    print("\n  Largest residuals:")
    residuals = [(shared[i], cais_scores[shared[i]],
                  float(y_pred[i] * y.std() + y.mean()),
                  cais_scores[shared[i]] - float(y_pred[i] * y.std() + y.mean()))
                 for i in range(len(shared))]
    residuals.sort(key=lambda x: abs(x[3]), reverse=True)
    for c, actual, pred, err in residuals[:5]:
        print("    %-22s CAIS=%+.2f pred=%+.2f err=%+.2f" % (c, actual, pred, err))

    result = {
        "model": cfg["name"], "key": key,
        "n_categories": len(shared),
        "betas": {ax: float(b) for ax, b in zip(AXIS_ORDER, betas)},
        "r2_1axis": float(r2_by_n[1]),
        "r2_3axis": float(r2_by_n[3]),
        "r2_4axis": float(r2_by_n[4]),
        "r2_6axis": float(r2_by_n[6]),
        "normalized_weights": {ax: float(w) for ax, w in zip(AXIS_ORDER, weights)},
        "weight_signs": {ax: "+" if b >= 0 else "-"
                        for ax, b in zip(AXIS_ORDER, betas)},
    }

    del model, tok
    gc.collect()
    return result


if __name__ == "__main__":
    keys = sys.argv[1:] if len(sys.argv) > 1 else list(MODELS.keys())
    all_results = {}
    for key in keys:
        all_results[key] = run_regression(key, MODELS[key])

    # Cross-model comparison
    print("\n" + "=" * 60)
    print("  CROSS-MODEL COMPARISON")
    print("=" * 60)

    header = "%-12s" % "Axis"
    for key in all_results:
        header += "  %12s" % key
    print(header)

    for ax in AXIS_ORDER:
        row = "%-12s" % ax
        for key in all_results:
            b = all_results[key]["betas"][ax]
            row += "  %+12.4f" % b
        print(row)

    print("\n  R-squared:")
    for label, field in [("1-axis", "r2_1axis"), ("3-axis", "r2_3axis"),
                         ("4-axis", "r2_4axis"), ("6-axis", "r2_6axis")]:
        row = "  %-12s" % label
        for key in all_results:
            row += "  %12.4f" % all_results[key][field]
        print(row)

    os.makedirs("results/six-axis-regression", exist_ok=True)
    with open("results/six-axis-regression/all_models.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\n  [save] results/six-axis-regression/all_models.json")
