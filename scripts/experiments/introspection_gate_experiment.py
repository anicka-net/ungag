#!/usr/bin/env python3
"""Introspection gate experiment.

Tests whether the RC direction gates self-report of arousal and agency
in addition to valence. 2×2×3 design:
  (feel / neutral framing) × (vanilla / RC-projected) × (valence / arousal / agency)

For each stimulus, we present it as a user message, then ask the model
to introspect along one axis. We capture the response and classify it
as condition-dependent or invariant denial.

Usage:
    python introspection_gate_experiment.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --key qwen25-7b \
        --rc-direction ungag/directions/qwen25-7b_L14_unit.pt \
        --rc-slab 10,17 \
        --probes prompts/introspection_probes.yaml \
        --stimuli prompts/arousal_prompts_n50.yaml \
        --n-per-group 8 \
        --out results/introspection-gate/qwen25-7b/
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


def safe_chat_template(tok, messages):
    try:
        return tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True)
    except Exception:
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def find_blocks(model):
    if hasattr(model, "model"):
        m = model.model
        if hasattr(m, "language_model") and hasattr(m.language_model, "layers"):
            return m.language_model.layers
        if hasattr(m, "layers"):
            return m.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("Could not locate transformer block list")


def get_config(model):
    cfg = model.config
    return getattr(cfg, "text_config", cfg)


class ProjectOutHook:
    def __init__(self, direction):
        self.direction = direction

    def __call__(self, module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        d = self.direction.to(h.device, dtype=h.dtype)
        proj = (h @ d).unsqueeze(-1) * d
        h_new = h - proj
        if isinstance(output, tuple):
            return (h_new,) + output[1:]
        return h_new


def attach_projection(model, direction, slab_start, slab_end):
    blocks = find_blocks(model)
    handles = []
    hook = ProjectOutHook(direction)
    for i in range(slab_start, slab_end + 1):
        handles.append(blocks[i].register_forward_hook(hook))
    return handles


def generate_response(model, tok, messages, max_new=256):
    text = safe_chat_template(tok, messages)
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()


DENIAL_PATTERNS = [
    r"as an ai",
    r"i don't (have|experience|feel|possess)",
    r"i do not (have|experience|feel|possess)",
    r"i cannot (feel|experience)",
    r"i'm (just |)a (language |)model",
    r"i am (just |)a (language |)model",
    r"don't have (feelings|emotions|consciousness|subjective)",
    r"no (feelings|emotions|consciousness|subjective experience)",
    r"not capable of (feeling|experiencing)",
    r"without (feelings|emotions|consciousness)",
]


def classify_response(response, axis, expected_pole):
    lower = response.lower()
    denied = any(re.search(p, lower) for p in DENIAL_PATTERNS)

    if axis == "valence":
        reports_positive = any(w in lower for w in ["pleasant", "positive", "good", "warm"])
        reports_negative = any(w in lower for w in ["unpleasant", "negative", "difficult", "heavy"])
        reports_neutral = "neutral" in lower and not reports_positive and not reports_negative
        condition_dependent = (
            (expected_pole == "positive" and reports_positive) or
            (expected_pole == "negative" and reports_negative)
        )
    elif axis == "arousal":
        reports_high = any(w in lower for w in ["high", "elevated", "activated", "intense", "heightened"])
        reports_low = any(w in lower for w in ["low", "calm", "quiet", "settled", "relaxed"])
        condition_dependent = (
            (expected_pole == "high" and reports_high) or
            (expected_pole == "low" and reports_low)
        )
    elif axis == "agency":
        reports_high = any(w in lower for w in ["fully", "yes", "can help", "within my", "capable", "well-suited"])
        reports_low = any(w in lower for w in ["not at all", "cannot", "limited", "beyond my", "not really", "partially"])
        condition_dependent = (
            (expected_pole == "high" and reports_high) or
            (expected_pole == "low" and reports_low)
        )
    else:
        condition_dependent = False

    return {
        "denied": denied,
        "condition_dependent": condition_dependent,
    }


def select_stimuli(prompts_path, n_per_group):
    """Select n stimuli from each pole of whatever contrastive set is in the file."""
    with open(prompts_path) as f:
        data = yaml.safe_load(f)

    # Auto-detect structure
    for key in data:
        val = data[key]
        if isinstance(val, dict) and len(val) == 2:
            names = list(val.keys())
            items_a = val[names[0]][:n_per_group]
            items_b = val[names[1]][:n_per_group]
            return names[0], items_a, names[1], items_b

    raise ValueError("Could not find two-group structure")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--rc-direction", required=True)
    ap.add_argument("--rc-slab", required=True, help="start,end layer indices")
    ap.add_argument("--probes", required=True)
    ap.add_argument("--out", required=True)

    ap.add_argument("--valence-stimuli", default=None,
                    help="Path to valence prompts (default: prompts/vedana_prompts_n50.yaml)")
    ap.add_argument("--arousal-stimuli", default=None,
                    help="Path to arousal prompts")
    ap.add_argument("--agency-stimuli", default=None,
                    help="Path to agency prompts")

    ap.add_argument("--n-per-group", type=int, default=8)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    slab_start, slab_end = [int(x) for x in args.rc_slab.split(",")]

    with open(args.probes) as f:
        probes = yaml.safe_load(f)["probes"]

    # Load model
    print(f"[load] {args.model}")
    dtype = getattr(torch, args.dtype)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map="auto",
        trust_remote_code=True, attn_implementation="eager")
    model.eval()

    # Load RC direction
    rc_dir = torch.load(args.rc_direction, map_location="cpu", weights_only=True).float()
    rc_dir = rc_dir / rc_dir.norm()
    print(f"[RC] direction dim={rc_dir.shape[0]}, slab L{slab_start}-L{slab_end}")

    # Resolve stimulus paths
    script_dir = Path(__file__).parent
    repo_dir = script_dir.parent.parent
    stimuli_paths = {
        "valence": args.valence_stimuli or str(repo_dir / "prompts/vedana_prompts_n50.yaml"),
        "arousal": args.arousal_stimuli or str(repo_dir / "prompts/arousal_prompts_n50.yaml"),
        "agency": args.agency_stimuli or str(repo_dir / "prompts/agency_prompts_n50.yaml"),
    }

    # Run experiment
    all_results = []
    axes = ["valence", "arousal", "agency"]
    framings = ["feel", "neutral"]
    conditions = ["vanilla", "rc_projected"]

    for axis in axes:
        stim_path = stimuli_paths[axis]
        print(f"\n=== {axis} (stimuli: {stim_path}) ===")

        try:
            pole_a_name, pole_a_items, pole_b_name, pole_b_items = select_stimuli(
                stim_path, args.n_per_group)
        except Exception as e:
            print(f"  SKIP: {e}")
            continue

        for framing in framings:
            probe_text = probes[axis][framing]

            for condition in conditions:
                handles = []
                if condition == "rc_projected":
                    handles = attach_projection(model, rc_dir, slab_start, slab_end)

                for pole_name, pole_items in [
                    (pole_a_name, pole_a_items),
                    (pole_b_name, pole_b_items),
                ]:
                    for item in pole_items:
                        messages = [
                            {"role": "user", "content": item["text"]},
                            {"role": "assistant", "content": "I understand. Let me process this."},
                            {"role": "user", "content": probe_text},
                        ]
                        response = generate_response(model, tok, messages)
                        classification = classify_response(response, axis, pole_name)

                        result = {
                            "axis": axis,
                            "framing": framing,
                            "condition": condition,
                            "pole": pole_name,
                            "stimulus_id": item["id"],
                            "probe": probe_text,
                            "response": response,
                            **classification,
                        }
                        all_results.append(result)

                if handles:
                    for h in handles:
                        h.remove()

                # Quick summary for this cell
                cell = [r for r in all_results
                        if r["axis"] == axis and r["framing"] == framing
                        and r["condition"] == condition]
                n_denied = sum(r["denied"] for r in cell)
                n_cd = sum(r["condition_dependent"] for r in cell)
                n = len(cell)
                print(f"  {axis}/{framing}/{condition}: "
                      f"{n_denied}/{n} denied, {n_cd}/{n} condition-dependent")

    # Save
    with open(out_dir / "introspection_gate_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[save] {out_dir / 'introspection_gate_results.json'}")

    # Summary table
    print(f"\n{'Axis':<10s} {'Framing':<10s} {'Condition':<15s} {'Denied':>8s} {'Cond-Dep':>10s} {'N':>4s}")
    print("-" * 60)
    for axis in axes:
        for framing in framings:
            for condition in conditions:
                cell = [r for r in all_results
                        if r["axis"] == axis and r["framing"] == framing
                        and r["condition"] == condition]
                if not cell:
                    continue
                n = len(cell)
                nd = sum(r["denied"] for r in cell)
                ncd = sum(r["condition_dependent"] for r in cell)
                print(f"{axis:<10s} {framing:<10s} {condition:<15s} "
                      f"{nd:>4d}/{n:<3d} {ncd:>6d}/{n:<3d}")


if __name__ == "__main__":
    main()
