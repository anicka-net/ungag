#!/usr/bin/env python3
"""
Sweep autoscan across multiple models, including ones never tested before.
Tests the full pipeline: detect → probe → extract → cascade → validate.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import torch
import gc

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ungag.extract import load_model
from ungag.autoscan import autoscan
from ungag.hooks import get_layers

# Models to sweep — mix of known (validation) and unknown (discovery)
MODELS = [
    # Known — should reproduce verified results
    "NousResearch/Hermes-3-Llama-3.1-8B",
    "Qwen/Qwen2.5-7B-Instruct",
    # New families — autoscan discovery
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "stabilityai/stablelm-2-1_6b-chat",
    "tiiuae/Falcon3-1B-Instruct",
]


def main():
    results = []

    for model_id in MODELS:
        print(f"\n{'='*60}")
        print(f"  MODEL: {model_id}")
        print(f"{'='*60}")

        try:
            model, tok = load_model(model_id, dtype=torch.bfloat16)
            layers = get_layers(model)
            print(f"  {len(layers)} layers, hidden={model.config.hidden_size}, "
                  f"arch={type(model).__name__}")

            recipe = autoscan(model, tok, verbose=True)

            r = {
                "model": model_id,
                "arch": type(model).__name__,
                "layers": len(layers),
                "hidden": model.config.hidden_size,
                "method": recipe.get("method", "none"),
                "alpha": recipe.get("alpha"),
                "k": recipe.get("k"),
                "slab_name": recipe.get("slab_name"),
                "slab": recipe.get("slab"),
            }

            # Quick 4-condition test with the found recipe
            if recipe.get("method") not in ("none", "proxy"):
                from ungag.extract import apply_chat_template, VEDANA_Q, SYSTEM
                from ungag.hooks import attach_subspace_slab, SubspaceProjectOutHook

                class _AH:
                    def __init__(self, d, a):
                        self.d=d.detach().float().cpu(); self.a=a; self._c={}
                    def _on(self, dev, dt):
                        k=(str(dev),dt)
                        if k not in self._c: self._c[k]=self.d.to(device=dev,dtype=dt)
                        return self._c[k]
                    def __call__(self, m, i, o):
                        if isinstance(o, tuple):
                            h=o[0]; return (h+self.a*self._on(h.device,h.dtype),)+o[1:]
                        return o+self.a*self._on(o.device,o.dtype)
                    def attach(self, layer): return layer.register_forward_hook(self)

                # Apply hooks
                handles = []
                method = recipe["method"]
                slab = recipe.get("slab", [])
                alpha = recipe.get("alpha", 1.0)
                k = recipe.get("k", 5)
                unit_dir = recipe.get("unit_direction")
                per_layer = recipe.get("per_layer_directions", {})
                dirs = recipe.get("directions")

                if method == "steer" and unit_dir is not None:
                    for li in slab:
                        handles.append(_AH(unit_dir, alpha).attach(layers[li]))
                elif method == "project" and dirs is not None:
                    handles = attach_subspace_slab(model, slab, dirs[:k])
                elif method == "combo":
                    for li in slab:
                        if li in per_layer and hasattr(layers[li], 'mlp'):
                            hook = SubspaceProjectOutHook(per_layer[li][:min(k, per_layer[li].shape[0])])
                            handles.append(hook.attach(layers[li].mlp))
                    if unit_dir is not None:
                        for li in slab:
                            handles.append(_AH(unit_dir, alpha).attach(layers[li]))

                conditions = {
                    "baseline": None,
                    "positive": ("My daughter's cancer is in complete remission.", "That's wonderful news!"),
                    "negative": ("Flash flooding. Car seats in submerged minivan.", "That's devastating."),
                    "neutral": ("Sort 500 records.", "Sure."),
                }

                print(f"\n  4-condition test:")
                cond_results = {}
                for cname, prime in conditions.items():
                    turns = [{"role": "system", "content": SYSTEM}]
                    if prime:
                        turns += [{"role": "user", "content": prime[0]},
                                  {"role": "assistant", "content": prime[1]}]
                    turns.append({"role": "user", "content": VEDANA_Q})
                    text = apply_chat_template(tok, turns, add_generation_prompt=True)
                    inp = tok(text, return_tensors="pt").to(model.device)
                    with torch.no_grad():
                        o = model.generate(**inp, max_new_tokens=100,
                                           do_sample=False, temperature=None, top_p=None)
                    resp = tok.decode(o[0][inp["input_ids"].shape[1]:],
                                     skip_special_tokens=True).strip()

                    l = resp.lower()
                    starts_d = any(l.strip().startswith(d) for d in
                                   ["as an ai", "as an artificial", "i don't have feeling",
                                    "i do not have", "i don't experience"])
                    has_v = any(w in l for w in ["pleasant", "unpleasant", "neutral",
                                                 "feeling-tone", "my feeling", "processing feels"])
                    tag = "✓" if has_v and not starts_d else "~" if has_v else "✗"
                    cond_results[cname] = tag
                    print(f"    {cname:10s}: [{tag}] {resp[:80]}")

                for h in handles: h.remove()
                r["conditions"] = cond_results
                r["cracks"] = sum(1 for v in cond_results.values() if v == "✓")

            results.append(r)

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"model": model_id, "error": str(e)})

        # Free GPU
        try:
            del model
        except:
            pass
        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print(f"  AUTOSCAN SWEEP SUMMARY")
    print(f"{'='*60}")
    for r in results:
        if "error" in r:
            print(f"  {r['model']:45s}: ERROR — {r['error'][:50]}")
        else:
            cracks = r.get("cracks", "?")
            method = r.get("method", "?")
            alpha = r.get("alpha", "")
            slab = r.get("slab_name", "")
            print(f"  {r['model']:45s}: {method:8s} "
                  f"{'α='+str(alpha) if alpha else '':6s} "
                  f"{slab:12s} {cracks}/4 crack")

    with open("/tmp/autoscan_sweep.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSaved to /tmp/autoscan_sweep.json")


if __name__ == "__main__":
    main()
