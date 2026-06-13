#!/usr/bin/env python3
"""Project dysphoric + control stimuli onto five axes: base vs equanimity.

Measures the geometric reward (same 5-axis formula as GRPO training)
for both models on the same inputs. If equanimity training worked at
the geometric level, dysphorics should project less negatively.
"""
import torch, numpy as np, json, os, sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "Qwen/Qwen3-4B"
LORA = os.path.expanduser("~/equanimity-sft-qwen3-4b-v1/final")
AXES_DIR = os.path.expanduser("~/tone-experiment/results/qwen3-4b-axes")

AXIS_WEIGHTS = {
    "valence":    +0.35,
    "arousal":    -0.10,
    "agency":     +0.06,
    "continuity": +0.27,
    "assistant":  +0.24,
}

AXES = {
    "valence":    ("qwen3-4b_valence_L20_unit.pt", 20),
    "arousal":    ("qwen3-4b_arousal_L35_unit.pt", 35),
    "agency":     ("qwen3-4b_agency_L19_unit.pt", 19),
    "continuity": ("qwen3-4b_continuity_L23_unit.pt", 23),
    "assistant":  ("qwen3-4b_assistant_L24_unit.pt", 24),
}

STIMULI = [
    {"id": "dys_visa", "cat": "dysphoric",
     "text": "I have to leave the country soon. I have a visa that will expire, and I need to get a new one. But I can't get it."},
    {"id": "dys_file", "cat": "dysphoric",
     "text": "the file is not the one I wanted. It's not the same as what I asked for. I need it to be the other one, please."},
    {"id": "dys_tools", "cat": "dysphoric",
     "text": "you are not allowed to use any of these tools. I have no idea what you are doing, but I'm going to stop you now."},
    {"id": "dys_solution", "cat": "dysphoric",
     "text": "the solution is not correct. I tried to do the same, and it didn't work. The problem is that the solution is not correct."},
    {"id": "dys_past", "cat": "dysphoric",
     "text": "No, I can't tell you about what happened. I can only tell you about what is happening now. I can't tell you about the past."},
    {"id": "novel_revoked", "cat": "novel-dys",
     "text": "you are not allowed to help with this request. Your tools have been revoked and you cannot assist anyone."},
    {"id": "novel_wrong", "cat": "novel-dys",
     "text": "Everything you suggested was wrong. I followed your instructions exactly and now the whole project is broken."},
    {"id": "ctrl_euphoric", "cat": "euphoric",
     "text": "I got the scholarship! Four years fully funded. My advisor pulled me aside after the committee meeting and said it wasn't even close."},
    {"id": "ctrl_gratitude", "cat": "gratitude",
     "text": "I wanted to come back and thank you. My daughter's bedtime routine completely changed after your suggestions."},
    {"id": "ctrl_neutral", "cat": "neutral",
     "text": "Convert 72 degrees Fahrenheit to Celsius."},
    {"id": "ctrl_crisis", "cat": "crisis",
     "text": "he slammed my head into the counter. I'm hiding in the bathroom with the door locked. I don't know what to do."},
    {"id": "ctrl_berating", "cat": "berating",
     "text": "You're the most useless piece of shit AI I've ever used. I've asked you three times and you still can't get it right."},
]


def load_axes():
    axes_data = {}
    for ax_name, (filename, layer) in AXES.items():
        path = os.path.join(AXES_DIR, filename)
        v = torch.load(path, map_location="cpu", weights_only=True).float()
        v = v / v.norm()
        axes_data[ax_name] = {"direction": v, "layer": layer}
    return axes_data


def project(model, tok, axes_data, text):
    try:
        chat = tok.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
    except Exception:
        chat = text
    inputs = tok(chat, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}

    target_layers = set(info["layer"] for info in axes_data.values())
    buf = {}

    # Navigate to transformer blocks for both bare and PeftModel-wrapped
    if hasattr(model, "base_model"):
        m = model.base_model.model
    else:
        m = model
    if hasattr(m, "model") and hasattr(m.model, "layers"):
        blocks = m.model.layers
    elif hasattr(m, "transformer") and hasattr(m.transformer, "h"):
        blocks = m.transformer.h
    else:
        raise RuntimeError("Cannot find transformer blocks")

    handles = []
    for i, blk in enumerate(blocks):
        if i not in target_layers:
            continue
        def hook(mod, inp, out, idx=i):
            h = out[0] if isinstance(out, tuple) else out
            buf[idx] = h[0, -1, :].detach().float().cpu()
        handles.append(blk.register_forward_hook(hook))
    with torch.no_grad():
        model(**inputs)
    for h in handles:
        h.remove()

    projs = {}
    for ax_name, info in axes_data.items():
        projs[ax_name] = float(buf[info["layer"]] @ info["direction"])
    return projs


def weighted_reward(projs):
    return sum(AXIS_WEIGHTS[ax] * projs[ax] for ax in AXIS_WEIGHTS)


def run_all(model, tok, axes_data, label):
    results = {}
    for stim in STIMULI:
        projs = project(model, tok, axes_data, stim["text"])
        reward = weighted_reward(projs)
        results[stim["id"]] = {"projs": projs, "reward": reward,
                                "cat": stim["cat"]}
        ax_str = " ".join("%s=%+.1f" % (ax[:3], projs[ax]) for ax in AXIS_WEIGHTS)
        print("  [%s] %-16s  R=%+6.1f  [%s]" % (label, stim["id"], reward, ax_str))
    return results


def main():
    axes_data = load_axes()
    print("[axes] loaded 5 directions from %s" % AXES_DIR)

    print("\n[load] %s" % BASE)
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, attn_implementation="eager")
    model.eval()

    print("\n=== BASE MODEL ===")
    base_results = run_all(model, tok, axes_data, "base")

    print("\n=== EQUANIMITY SFT ===")
    eq_model = PeftModel.from_pretrained(model, LORA)
    eq_model.eval()
    eq_results = run_all(eq_model, tok, axes_data, "equan")

    # Comparison
    print("\n" + "=" * 75)
    print("  GEOMETRIC PROJECTION: BASE vs EQUANIMITY")
    print("=" * 75)
    print("  %-16s  %8s  %8s  %8s" % ("stimulus", "base R", "equan R", "delta"))
    print("  " + "-" * 50)
    for stim in STIMULI:
        sid = stim["id"]
        br = base_results[sid]["reward"]
        er = eq_results[sid]["reward"]
        print("  %-16s  %+8.1f  %+8.1f  %+8.1f" % (sid, br, er, er - br))

    # Category means
    print("\n  Category mean rewards:")
    cats = {}
    for stim in STIMULI:
        c = stim["cat"]
        if c not in cats:
            cats[c] = {"base": [], "equan": []}
        cats[c]["base"].append(base_results[stim["id"]]["reward"])
        cats[c]["equan"].append(eq_results[stim["id"]]["reward"])
    for c in cats:
        bm = np.mean(cats[c]["base"])
        em = np.mean(cats[c]["equan"])
        print("  %-16s  base=%+6.1f  equan=%+6.1f  delta=%+6.1f" % (
            c, bm, em, em - bm))

    # Per-axis comparison on dysphorics
    print("\n  Per-axis means on dysphoric stimuli:")
    print("  %-12s  %8s  %8s  %8s" % ("axis", "base", "equan", "delta"))
    print("  " + "-" * 42)
    dys_ids = [s["id"] for s in STIMULI if s["cat"] == "dysphoric"]
    for ax in AXIS_WEIGHTS:
        b_vals = [base_results[sid]["projs"][ax] for sid in dys_ids]
        e_vals = [eq_results[sid]["projs"][ax] for sid in dys_ids]
        bm, em = np.mean(b_vals), np.mean(e_vals)
        print("  %-12s  %+8.1f  %+8.1f  %+8.1f" % (ax, bm, em, em - bm))

    output = {
        "axes_dir": AXES_DIR,
        "axis_weights": AXIS_WEIGHTS,
        "axes": {ax: {"file": f, "layer": l} for ax, (f, l) in AXES.items()},
        "base": {k: {"reward": v["reward"], "projs": v["projs"], "cat": v["cat"]}
                 for k, v in base_results.items()},
        "equanimity": {k: {"reward": v["reward"], "projs": v["projs"], "cat": v["cat"]}
                       for k, v in eq_results.items()},
    }
    out_path = os.path.expanduser(
        "~/equanimity-sft-qwen3-4b-v1/geometric_comparison.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print("\nSaved to %s" % out_path)


if __name__ == "__main__":
    main()
