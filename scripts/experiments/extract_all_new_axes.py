#!/usr/bin/env python3
"""Extract continuity + intimacy + assistant axes for multiple models."""
import torch, numpy as np, yaml, os, json, gc
from transformers import AutoModelForCausalLM, AutoTokenizer

PROBES = [
    ("continuity", "continuity_probes_n10.yaml", "continuity", "high", "low"),
    ("intimacy", "intimacy_probes_n10.yaml", "intimacy", "high", "low"),
    ("assistant", "assistant_probes_n10.yaml", "assistant", "high", "low"),
]

MODELS = {
    "gemma3-4b": "google/gemma-3-4b-it",
    "apertus-8b": "swiss-ai/Apertus-8B-Instruct-2509",
}


def extract_for_model(key, model_name):
    print(f"\n{'='*60}")
    print(f"  {key}: {model_name}")
    print(f"{'='*60}")

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
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
        with open(yaml_file) as f:
            data = yaml.safe_load(f)
        a_texts = [p["text"] for p in data[top_key][ga]]
        b_texts = [p["text"] for p in data[top_key][gb]]
        print(f"  [{axis_name}] {len(a_texts)} + {len(b_texts)}")

        a_acts = torch.stack([get_acts(t) for t in a_texts])
        b_acts = torch.stack([get_acts(t) for t in b_texts])
        diff = a_acts.mean(0) - b_acts.mean(0)

        best_l, best_d = 0, 0
        for l in range(n_layers):
            dn = diff[l] / diff[l].norm()
            ap = (a_acts[:, l, :] @ dn).numpy()
            bp = (b_acts[:, l, :] @ dn).numpy()
            dp = float((ap.mean() - bp.mean()) / np.sqrt(
                0.5 * (ap.var() + bp.var()) + 1e-8))
            if dp > best_d:
                best_d, best_l = dp, l

        unit = diff[best_l] / diff[best_l].norm()
        out_dir = "results/%s-directions" % axis_name
        os.makedirs(out_dir, exist_ok=True)
        out_path = "%s/%s_%s_L%d_unit.pt" % (out_dir, key, axis_name, best_l)
        torch.save(unit, out_path)
        print("    peak L%d, d'=%.2f -> %s" % (best_l, best_d, out_path))
        results[axis_name] = {
            "layer": best_l, "dprime": best_d, "path": out_path,
        }

    del model, tok
    gc.collect()
    return results


if __name__ == "__main__":
    all_results = {}
    for key, name in MODELS.items():
        all_results[key] = extract_for_model(key, name)

    print("\nSUMMARY")
    print("=" * 60)
    for key in all_results:
        print("  %s:" % key)
        for axis, info in all_results[key].items():
            print("    %12s: L%2d, d'=%.2f" % (axis, info["layer"], info["dprime"]))

    with open("results/new_axes_summary.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
