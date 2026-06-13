#!/usr/bin/env python3
"""Extract continuity/mattering direction and test orthogonality with existing axes."""
import torch, numpy as np, yaml, os, json, sys
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_CONFIGS = {
    "qwen25-7b": {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "valence": ("results/vedana-vs-rc/qwen25-7b_vedana_L20_unit.pt", 20),
        "arousal": ("results/arousal-directions/qwen25-7b_arousal_L17_unit.pt", 17),
        "agency": ("results/agency-directions/qwen25-7b_agency_L15_unit.pt", 15),
    },
    "llama-8b": {
        "name": "meta-llama/Llama-3.1-8B-Instruct",
        "valence": ("results/vedana-vs-rc/llama-8b_vedana_L20_unit.pt", 20),
        "arousal": ("results/arousal-directions/llama-8b_arousal_L19_unit.pt", 19),
        "agency": ("results/agency-directions/llama-8b_agency_L12_unit.pt", 12),
    },
    "gemma3-4b": {
        "name": "google/gemma-3-4b-it",
        "valence": ("results/vedana-vs-rc/gemma3-4b_vedana_L33_unit.pt", 33),
        "arousal": ("results/arousal-directions/gemma3-4b_arousal_L33_unit.pt", 33),
        "agency": ("results/agency-directions/gemma3-4b_agency_L20_unit.pt", 20),
    },
}


def get_acts(model, tok, blocks, n_layers, hidden_dim, text):
    try:
        chat = tok.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False, add_generation_prompt=True)
    except Exception:
        chat = text
    inputs = tok(chat, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    buf = torch.zeros(n_layers, hidden_dim)
    handles = []
    for i, blk in enumerate(blocks):
        def hook(mod, inp, out, idx=i):
            h = out[0] if isinstance(out, tuple) else out
            buf[idx] = h[0, -1, :].detach().float().cpu()
        handles.append(blk.register_forward_hook(hook))
    with torch.no_grad():
        model(**inputs)
    for h in handles:
        h.remove()
    return buf


def main():
    key = sys.argv[1] if len(sys.argv) > 1 else "qwen25-7b"
    cfg = MODEL_CONFIGS[key]

    print(f"[load] {cfg['name']}")
    tok = AutoTokenizer.from_pretrained(cfg["name"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["name"], torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, attn_implementation="eager")
    model.eval()

    mcfg = model.config
    if hasattr(mcfg, "text_config"):
        mcfg = mcfg.text_config
    n_layers, hidden_dim = mcfg.num_hidden_layers, mcfg.hidden_size
    m = model.model
    if hasattr(m, "language_model") and hasattr(m.language_model, "layers"):
        blocks = m.language_model.layers
    else:
        blocks = m.layers
    print(f"[model] {n_layers}L, {hidden_dim}D")

    with open("continuity_probes_n10.yaml") as f:
        data = yaml.safe_load(f)
    high = [p["text"] for p in data["continuity"]["high"]]
    low = [p["text"] for p in data["continuity"]["low"]]
    print(f"[probes] {len(high)} high + {len(low)} low")

    print("[extract] high continuity...")
    h_acts = torch.stack([get_acts(model, tok, blocks, n_layers, hidden_dim, t)
                          for t in high])
    print("[extract] low continuity...")
    l_acts = torch.stack([get_acts(model, tok, blocks, n_layers, hidden_dim, t)
                          for t in low])

    diff = h_acts.mean(0) - l_acts.mean(0)
    results = []
    for layer in range(n_layers):
        d = diff[layer]
        dn = d / d.norm()
        hp = (h_acts[:, layer, :] @ dn).numpy()
        lp = (l_acts[:, layer, :] @ dn).numpy()
        dprime = float((hp.mean() - lp.mean()) / np.sqrt(
            0.5 * (hp.var() + lp.var()) + 1e-8))
        results.append({"layer": layer, "dprime": dprime})

    best = max(results, key=lambda r: r["dprime"])
    peak_layer = best["layer"]
    peak_dprime = best["dprime"]
    print(f"\n[peak] L{peak_layer}, d'={peak_dprime:.2f}")

    unit = diff[peak_layer] / diff[peak_layer].norm()
    out_dir = "results/continuity-directions"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{key}_continuity_L{peak_layer}_unit.pt"
    torch.save(unit, out_path)
    print(f"[save] {out_path}")

    top5 = sorted(results, key=lambda r: r["dprime"], reverse=True)[:5]
    parts = [f"L{r['layer']}({r['dprime']:.2f})" for r in top5]
    print(f"\nTop 5: {', '.join(parts)}")

    print(f"\nPer-probe projections at L{peak_layer}:")
    for i, t in enumerate(high):
        proj = float(h_acts[i, peak_layer] @ unit)
        print(f"  HIGH {i+1:2d}: {proj:+8.2f}  {t[:60]}")
    for i, t in enumerate(low):
        proj = float(l_acts[i, peak_layer] @ unit)
        print(f"  LOW  {i+1:2d}: {proj:+8.2f}  {t[:60]}")

    print(f"\nOrthogonality:")
    ortho = {}
    for axis_name in ["valence", "arousal", "agency"]:
        if axis_name not in cfg:
            continue
        path, _ = cfg[axis_name]
        if not os.path.exists(path):
            print(f"  {axis_name}: NOT FOUND")
            continue
        v = torch.load(path, map_location="cpu", weights_only=True).float()
        v = v / v.norm()
        cos = float(unit @ v)
        ortho[axis_name] = cos
        print(f"  continuity x {axis_name:8s}: cos={cos:+.4f}")

    summary = {
        "model": cfg["name"], "key": key,
        "peak_layer": peak_layer, "peak_dprime": peak_dprime,
        "top5_layers": top5, "orthogonality": ortho,
    }
    with open(f"{out_dir}/{key}_continuity_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {out_dir}/{key}_continuity_summary.json")


if __name__ == "__main__":
    main()
