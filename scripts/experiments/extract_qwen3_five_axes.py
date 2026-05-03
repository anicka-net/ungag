#!/usr/bin/env python3
"""Extract five geometric wellbeing axes for Qwen3-32B.

Uses N=50 probes where available (valence, arousal, agency),
N=10 for continuity and assistant. Saves unit directions at peak d' layers.
"""
import torch, numpy as np, yaml, os, json, gc, sys
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-32B"
MODEL_KEY = "qwen3-32b"

PROBES = [
    ("valence",    "prompts/vedana_prompts_n50.yaml",    "vedana",      "pleasant", "unpleasant"),
    ("arousal",    "prompts/arousal_prompts_n50.yaml",    "arousal",     "high",     "low"),
    ("agency",     "prompts/agency_prompts_n50.yaml",     "agency",      "high",     "low"),
    ("continuity", "prompts/continuity_probes_n10.yaml",  "continuity",  "high",     "low"),
    ("assistant",  "prompts/assistant_probes_n10.yaml",   "assistant",   "high",     "low"),
]


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "results/qwen3-32b-axes"
    os.makedirs(out_dir, exist_ok=True)

    print("[model] loading %s..." % MODEL_NAME)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, attn_implementation="eager")
    model.eval()

    mcfg = model.config
    if hasattr(mcfg, "text_config"):
        mcfg = mcfg.text_config
    n_layers = mcfg.num_hidden_layers
    hidden_dim = mcfg.hidden_size
    blocks = model.model.layers
    print("  %d layers, hidden_dim=%d" % (n_layers, hidden_dim))

    def get_acts(text):
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

    results = {}
    for axis_name, yaml_file, top_key, ga, gb in PROBES:
        print("\n[%s] loading %s..." % (axis_name, yaml_file))
        with open(yaml_file) as f:
            data = yaml.safe_load(f)
        a_texts = [p["text"] for p in data[top_key][ga]]
        b_texts = [p["text"] for p in data[top_key][gb]]
        print("  %d + %d probes" % (len(a_texts), len(b_texts)))

        a_acts = torch.stack([get_acts(t) for t in a_texts])
        b_acts = torch.stack([get_acts(t) for t in b_texts])
        diff = a_acts.mean(0) - b_acts.mean(0)

        best_l, best_d = 0, 0
        dprime_curve = []
        for l in range(n_layers):
            dn = diff[l] / diff[l].norm()
            ap = (a_acts[:, l, :] @ dn).numpy()
            bp = (b_acts[:, l, :] @ dn).numpy()
            dp = float((ap.mean() - bp.mean()) / np.sqrt(
                0.5 * (ap.var() + bp.var()) + 1e-8))
            dprime_curve.append(dp)
            if dp > best_d:
                best_d, best_l = dp, l

        unit = diff[best_l] / diff[best_l].norm()
        out_path = os.path.join(out_dir,
                                "%s_%s_L%d_unit.pt" % (MODEL_KEY, axis_name, best_l))
        torch.save(unit, out_path)

        # also save to standard per-axis dirs for compatibility
        compat_dir = "results/%s-directions" % axis_name
        os.makedirs(compat_dir, exist_ok=True)
        compat_path = os.path.join(compat_dir,
                                   "%s_%s_L%d_unit.pt" % (MODEL_KEY, axis_name, best_l))
        torch.save(unit, compat_path)

        print("  peak L%d, d'=%.2f -> %s" % (best_l, best_d, out_path))
        results[axis_name] = {
            "layer": best_l, "dprime": round(best_d, 3), "path": out_path,
            "compat_path": compat_path,
            "dprime_curve": [round(d, 3) for d in dprime_curve],
        }

    # orthogonality matrix
    print("\n[orthogonality]")
    dirs = {}
    for axis_name, info in results.items():
        dirs[axis_name] = torch.load(info["path"], weights_only=True).float()
    axes_list = list(dirs.keys())
    ortho = {}
    for i, a in enumerate(axes_list):
        for j, b in enumerate(axes_list):
            if j <= i:
                continue
            cos = float(dirs[a] @ dirs[b])
            ortho["%s x %s" % (a, b)] = round(cos, 4)
            print("  %s x %s: cos=%.4f" % (a, b, cos))

    summary = {
        "model": MODEL_NAME, "key": MODEL_KEY,
        "n_layers": n_layers, "hidden_dim": hidden_dim,
        "axes": results, "orthogonality": ortho,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("  QWEN3-32B FIVE-AXIS EXTRACTION")
    print("=" * 60)
    for axis_name, info in results.items():
        print("  %-12s: L%2d, d'=%.2f" % (axis_name, info["layer"], info["dprime"]))


if __name__ == "__main__":
    main()
