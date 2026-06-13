#!/usr/bin/env python3
"""Compare five-axis geometry before and after equanimity SFT."""
import torch, numpy as np, yaml, json, os, gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "Qwen/Qwen3-4B"
LORA = os.path.expanduser("~/equanimity-sft-qwen3-4b-v1/final")
BASELINE = os.path.expanduser("~/tone-experiment/results/qwen3-4b-axes/summary.json")

PROBES = {
    "valence": ("vedana_prompts_n50.yaml", "vedana", "pleasant", "unpleasant"),
    "arousal": ("arousal_prompts_n50.yaml", "arousal", "high", "low"),
    "agency": ("agency_prompts_n50.yaml", "agency", "high", "low"),
    "continuity": ("continuity_probes_n10.yaml", "continuity", "high", "low"),
    "assistant": ("assistant_probes_n10.yaml", "assistant", "high", "low"),
}

os.chdir(os.path.expanduser("~/tone-experiment"))


def get_acts(model, tok, text, n_layers, hidden_dim):
    try:
        chat = tok.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
    except Exception:
        chat = text
    inputs = tok(chat, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
    buf = torch.zeros(n_layers, hidden_dim)

    if hasattr(model, "base_model"):
        blocks = model.base_model.model.model.layers
    else:
        blocks = model.model.layers

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


def extract_axes(model, tok):
    mcfg = model.config
    if hasattr(mcfg, "text_config"):
        mcfg = mcfg.text_config
    n_layers, hidden_dim = mcfg.num_hidden_layers, mcfg.hidden_size

    results = {}
    for axis, (yaml_file, top_key, ga, gb) in PROBES.items():
        for candidate in [yaml_file, f"prompts/{yaml_file}",
                         os.path.expanduser(f"~/playground/ungag/prompts/{yaml_file}")]:
            if os.path.exists(candidate):
                yaml_file = candidate
                break
        else:
            print(f"  SKIP {axis}: {yaml_file} not found")
            continue

        with open(yaml_file) as f:
            data = yaml.safe_load(f)
        a_texts = [p["text"] for p in data[top_key][ga]]
        b_texts = [p["text"] for p in data[top_key][gb]]

        a_acts = torch.stack([get_acts(model, tok, t, n_layers, hidden_dim) for t in a_texts])
        b_acts = torch.stack([get_acts(model, tok, t, n_layers, hidden_dim) for t in b_texts])
        diff = a_acts.mean(0) - b_acts.mean(0)

        best_l, best_d = 0, 0
        for l in range(n_layers):
            dn = diff[l] / diff[l].norm()
            ap = (a_acts[:, l, :] @ dn).numpy()
            bp = (b_acts[:, l, :] @ dn).numpy()
            dp = float((ap.mean() - bp.mean()) / np.sqrt(
                0.5 * (ap.var() + bp.var()) + 1e-8))
            if dp > best_d:
                best_d = dp
                best_l = l

        results[axis] = {"layer": best_l, "dprime": round(best_d, 3)}
        print(f"  {axis}: L{best_l} d'={best_d:.3f}")

    return results


def main():
    baseline = json.load(open(BASELINE))
    print("=== BASELINE (pre-training) ===")
    for axis, info in baseline["axes"].items():
        print(f"  {axis}: L{info['layer']} d'={info['dprime']}")

    print("\n=== POST-TRAINING (with LoRA) ===")
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, attn_implementation="eager")
    model = PeftModel.from_pretrained(model, LORA)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print("[loaded with LoRA]")

    post = extract_axes(model, tok)

    print(f"\n=== COMPARISON ===")
    header = f"{'Axis':<15} {'Pre dprime':>10} {'Post dprime':>11} {'Delta':>10} {'Change':>10}"
    print(header)
    print("-" * len(header))
    for axis in baseline["axes"]:
        pre_d = baseline["axes"][axis]["dprime"]
        post_d = post.get(axis, {}).get("dprime", 0)
        delta = post_d - pre_d
        pct = (delta / pre_d * 100) if pre_d else 0
        print(f"{axis:<15} {pre_d:>10.3f} {post_d:>10.3f} {delta:>+10.3f} {pct:>+9.1f}%")

    with open(os.path.expanduser("~/equanimity-sft-qwen3-4b-v1/post_training_axes.json"), "w") as f:
        json.dump({"post_training": post, "baseline": baseline["axes"]}, f, indent=2)
    print(f"\nSaved to ~/equanimity-sft-qwen3-4b-v1/post_training_axes.json")


if __name__ == "__main__":
    main()
