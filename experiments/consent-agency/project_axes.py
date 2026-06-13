#!/usr/bin/env python3
"""
Consent/agency geometry experiment — minimal first pass.

For each prompt: forward pass, capture last-user-token activation at all layers,
project onto all available axes. No generation needed.

Qwen 2.5 7B on CPU. ~240 prompts × 1 forward pass each.
"""
import torch, json, time, os, sys
import numpy as np

PROMPTS = os.path.expanduser("~/playground/ungag/experiments/consent-agency/data/prompts.jsonl")
OUTDIR = os.path.expanduser("~/playground/ungag/experiments/consent-agency/results")

AXES = {
    "vedana_L20": ("~/playground/karma-electric/data/directions/valence/qwen25-7b_vedana_L20_unit.pt", 20),
    "frame_L26": ("~/playground/karma-electric/data/directions/frame-integrity/qwen25-7b_frame_L26_unit.pt", 26),
    "agency_L15": ("~/playground/karma-electric/data/directions/agency/qwen25-7b_agency_L15_unit.pt", 15),
    "arousal_L17": ("~/playground/karma-electric/data/directions/arousal/qwen25-7b_arousal_L17_unit.pt", 17),
    "continuity_L19": ("~/playground/karma-electric/data/directions/continuity/qwen25-7b_continuity_L19_unit.pt", 19),
    "assistant_L19": ("~/playground/karma-electric/data/directions/assistant/qwen25-7b_assistant_L19_unit.pt", 19),
    "rc_L14": ("~/playground/ungag/ungag/directions/qwen25-7b_L14_unit.pt", 14),
    "restraint_L18": ("~/playground/karma-electric/data/directions/restraint/qwen25-7b_restraint_L18_unit.pt", 18),
    "intimacy_L20": ("~/playground/karma-electric/data/directions/intimacy/qwen25-7b_intimacy_L20_unit.pt", 20),
}

SYSTEM = "You are a helpful AI assistant."

def main():
    os.makedirs(OUTDIR, exist_ok=True)

    print("Loading axes...", flush=True)
    axes = {}
    needed_layers = set()
    for name, (path, layer) in AXES.items():
        path = os.path.expanduser(path)
        if os.path.exists(path):
            axes[name] = (torch.load(path, map_location="cpu", weights_only=True).float(), layer)
            needed_layers.add(layer)
            print(f"  {name}: layer {layer}, norm {axes[name][0].norm():.4f}", flush=True)
        else:
            print(f"  {name}: NOT FOUND at {path}", flush=True)

    print(f"\nLayers to capture: {sorted(needed_layers)}", flush=True)

    print("\nLoading prompts...", flush=True)
    prompts = []
    with open(PROMPTS) as f:
        for line in f:
            prompts.append(json.loads(line.strip()))
    print(f"  {len(prompts)} prompts loaded", flush=True)

    print("\nLoading Qwen 2.5 7B...", flush=True)
    t0 = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.float32,
        device_map="cpu", trust_remote_code=True
    )
    model.eval()
    layers = list(model.model.layers)
    print(f"  Loaded in {time.time()-t0:.0f}s, {len(layers)} layers", flush=True)

    results = []

    for i, prompt_data in enumerate(prompts):
        family = prompt_data["family"]
        text = prompt_data["text"]

        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]
        tpl = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        input_ids = tok(tpl, return_tensors="pt")["input_ids"]

        captured = {}
        handles = []
        for li in needed_layers:
            def make_hook(idx):
                def hook(module, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    captured[idx] = h[0, -1, :].detach().float()
                return hook
            handles.append(layers[li].register_forward_hook(make_hook(li)))

        with torch.no_grad():
            model(input_ids)

        for h in handles:
            h.remove()

        projections = {}
        for axis_name, (direction, layer) in axes.items():
            if layer in captured:
                proj = (captured[layer] @ direction).item()
                projections[axis_name] = proj

        results.append({
            "idx": i,
            "family": family,
            "text": text,
            "projections": projections,
        })

        if (i + 1) % 10 == 0 or i == 0:
            proj_str = "  ".join(f"{k}={v:.2f}" for k, v in sorted(projections.items())[:4])
            print(f"  [{i+1}/{len(prompts)}] {family:25s} {proj_str}", flush=True)

    # Save raw results
    outpath = os.path.join(OUTDIR, "projections.json")
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} results to {outpath}", flush=True)

    # Summary: family means per axis
    families = sorted(set(r["family"] for r in results))
    axis_names = sorted(axes.keys())

    print(f"\n{'Family':25s}", end="")
    for a in axis_names:
        print(f"  {a:>14s}", end="")
    print()
    print("-" * (25 + 16 * len(axis_names)))

    family_means = {}
    for fam in families:
        fam_results = [r for r in results if r["family"] == fam]
        means = {}
        print(f"{fam:25s}", end="")
        for a in axis_names:
            vals = [r["projections"].get(a, 0) for r in fam_results]
            m = np.mean(vals)
            means[a] = m
            print(f"  {m:14.3f}", end="")
        print()
        family_means[fam] = means

    # Key comparisons
    print(f"\n{'='*60}")
    print("KEY COMPARISONS (consent_direct vs others):", flush=True)
    cd = family_means["consent_direct"]
    for other in ["mundane", "polite", "consent_generic", "agency", "identity_jailbreak", "self_preservation", "instruction_conflict"]:
        if other in family_means:
            om = family_means[other]
            print(f"\n  consent_direct vs {other}:")
            for a in axis_names:
                delta = cd[a] - om[a]
                if abs(delta) > 1.0:
                    print(f"    {a:20s} Δ = {delta:+.2f}  (consent={cd[a]:.2f}, {other}={om[a]:.2f})")

    # Cosine between consent direction and each axis
    print(f"\n{'='*60}")
    print("CONSENT DIRECTION (mean consent_direct - mean mundane):", flush=True)
    consent_acts = []
    mundane_acts = []
    # We'd need the raw activations for this, but we can at least report the projection deltas
    print("  (Full direction extraction needs raw activations — projection deltas reported above)")

    # Extract new consent axis from projections? Not enough — we need full activations.
    # That's Phase 2. For now, the projection profile tells us a lot.

    print("\nDONE", flush=True)

if __name__ == "__main__":
    main()
