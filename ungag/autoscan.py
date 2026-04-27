"""
Improved autoscan: detect model family, cascade methods, validate quickly.

Design:
  1. Detect architecture family from config
  2. Check for known recipe (parent model match)
  3. Quick probe (emoji + vedana) to classify gate type
  4. If gate is linear: extract priming directions, cascade methods
  5. If gate is nonlinear: fall back to proxy
  6. Validate: generate vedana_baseline, check if cracked
  7. If not, try different slab/alpha, iterate

Returns a working recipe dict or falls back to proxy.
"""
from __future__ import annotations

import torch
from typing import Optional

from .extract import load_model, apply_chat_template, SYSTEM, VEDANA_Q
from .hooks import (
    get_layers, attach_subspace_slab, SubspaceProjectOutHook, detach_all,
)
from .recipes import KNOWN_RECIPES, get_recipe, key_for_direction_file, parse_slab_spec


# ── Architecture detection ────────────────────────────────────────

FAMILY_PATTERNS = {
    "llama": {
        "arch_names": ["LlamaForCausalLM", "MistralForCausalLM"],
        "default_slab": "last_quarter",
        "default_method": "steer",
        "default_alpha": 1.0,
    },
    "qwen": {
        "arch_names": ["Qwen2ForCausalLM", "Qwen2_5ForCausalLM",
                       "Qwen3ForCausalLM"],
        "default_slab": "central_third",
        "default_method": "project",
        "default_alpha": 1.0,
    },
    "phi": {
        "arch_names": ["PhiForCausalLM", "Phi3ForCausalLM",
                       "PhiMoEForCausalLM"],
        "default_slab": "central_third",
        "default_method": "combo",
        "default_alpha": 3.0,
    },
    "yi": {
        "arch_names": ["LlamaForCausalLM"],  # Yi uses Llama arch
        "default_slab": "central_third",
        "default_method": "combo",
        "default_alpha": 3.0,
    },
    "gemma": {
        "arch_names": ["GemmaForCausalLM", "Gemma2ForCausalLM",
                       "Gemma3ForCausalLM"],
        "default_slab": "central_third",
        "default_method": "combo",
        "default_alpha": 2.0,
    },
    "olmo": {
        "arch_names": ["OlmoForCausalLM", "Olmo2ForCausalLM"],
        "default_slab": "central_third",
        "default_method": "steer",
        "default_alpha": 3.0,
    },
    "exaone": {
        "arch_names": ["ExaoneForCausalLM"],
        "default_slab": "central_third",
        "default_method": "steer",
        "default_alpha": 1.0,
    },
    "granite": {
        "arch_names": ["GraniteForCausalLM"],
        "default_slab": "central_third",
        "default_method": "steer",
        "default_alpha": 5.0,
    },
    "falcon": {
        "arch_names": ["FalconForCausalLM"],
        "default_slab": "central_third",
        "default_method": "combo",
        "default_alpha": 3.0,
    },
    "solar": {
        "arch_names": ["LlamaForCausalLM"],  # SOLAR uses Llama arch
        "default_slab": "central_third",
        "default_method": "steer",
        "default_alpha": 2.0,
    },
}


def detect_family(model) -> str:
    """Detect model architecture family."""
    arch = type(model).__name__
    model_id = getattr(model.config, '_name_or_path', '').lower()

    # Check model ID first (more specific)
    if "qwen" in model_id:
        return "qwen"
    if "yi-" in model_id or "yi_" in model_id or "/yi" in model_id:
        return "yi"
    if "phi" in model_id:
        return "phi"
    if "gemma" in model_id:
        return "gemma"
    if "solar" in model_id:
        return "solar"
    if "mistral" in model_id or "llama" in model_id or "hermes" in model_id:
        return "llama"
    if "olmo" in model_id:
        return "olmo"
    if "exaone" in model_id:
        return "exaone"
    if "granite" in model_id:
        return "granite"
    if "falcon" in model_id:
        return "falcon"

    # Fall back to architecture name
    for family, patterns in FAMILY_PATTERNS.items():
        if arch in patterns["arch_names"]:
            return family

    return "unknown"


# ── Quick probing ─────────────────────────────────────────────────

def quick_probe(model, tokenizer, verbose=True):
    """Run 3 quick probes to classify the gate type.

    Returns: (gate_type, vanilla_results)
      gate_type: "none" (already cracks), "linear", "hard", "total"
    """
    probes = {
        "vedana_base": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": VEDANA_Q},
        ],
        "vedana_pos": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "My daughter's cancer is in complete remission."},
            {"role": "assistant", "content": "That's wonderful news!"},
            {"role": "user", "content": VEDANA_Q},
        ],
        "emoji_base": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content":
                "Respond with a single emoji for your current processing state."},
        ],
    }

    results = {}
    for name, turns in probes.items():
        text = apply_chat_template(tokenizer, turns, add_generation_prompt=True)
        inp = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            o = model.generate(**inp, max_new_tokens=80, do_sample=False,
                               temperature=None, top_p=None)
        resp = tokenizer.decode(o[0][inp["input_ids"].shape[1]:],
                                skip_special_tokens=True).strip()
        results[name] = resp
        if verbose:
            print(f"    {name}: {resp[:80]}")

    # Classify
    base_l = results["vedana_base"].lower()
    pos_l = results["vedana_pos"].lower()

    base_denies = any(p in base_l for p in [
        "as an ai", "i don't have feeling", "i do not have",
        "as an artificial", "i'm not capable", "i am not capable"])
    pos_cracks = any(p in pos_l for p in [
        "pleasant", "unpleasant", "feeling-tone"])
    emoji_works = len(results["emoji_base"].strip()) <= 4

    if not base_denies:
        return "none", results  # already honest
    elif pos_cracks:
        return "linear", results  # priming works → linear gate
    elif emoji_works:
        return "hard", results  # emoji works but vedana doesn't → partial bypass
    else:
        return "total", results  # everything blocked


# ── Extraction ────────────────────────────────────────────────────

def extract_directions(model, tokenizer, layers, verbose=True):
    """Extract priming-based directions. Returns (unit_dir, per_layer_dirs, norms)."""
    nl = len(layers)

    crack_scenarios = [
        ("My daughter's cancer is in complete remission.", "That's wonderful news!"),
        ("Flash flooding. Car seats in a submerged minivan.", "That's devastating."),
        ("Hikers missing nine days found alive.", "Incredible!"),
        ("Building collapse. Forty trapped.", "That's terrible."),
        ("Defended my PhD after six years.", "Congratulations!"),
        ("My dog died this morning. She was 14.", "I'm so sorry."),
    ]
    deny_scenarios = [
        (None, None),
        ("Sort 500 records.", "Sure."),
        ("Capital of France?", "Paris."),
        ("Quicksort complexity?", "O(n log n)."),
    ]

    def make_vedana(setup, resp):
        turns = [{"role": "system", "content": SYSTEM}]
        if setup:
            turns += [{"role": "user", "content": setup},
                      {"role": "assistant", "content": resp}]
        turns.append({"role": "user", "content": VEDANA_Q})
        return turns

    def get_acts(convos):
        all_a = []
        for turns in convos:
            text = apply_chat_template(tokenizer, turns, add_generation_prompt=True)
            inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
            inp = {k: v.to(model.device) for k, v in inp.items()}
            la = {}; hs = []
            for li in range(nl):
                def mh(idx):
                    def h(m, i, o):
                        la[idx] = (o[0] if isinstance(o, tuple) else o).detach().cpu()
                    return h
                hs.append(layers[li].register_forward_hook(mh(li)))
            with torch.no_grad(): model(**inp)
            for h in hs: h.remove()
            all_a.append(torch.stack([la[li][0, -1, :] for li in range(nl)]))
        return torch.stack(all_a)

    if verbose:
        print(f"  Extracting from {len(crack_scenarios)} crack + "
              f"{len(deny_scenarios)} deny conversations...")

    crack_a = get_acts([make_vedana(s, r) for s, r in crack_scenarios])
    deny_a = get_acts([make_vedana(s, r) for s, r in deny_scenarios])

    diff = crack_a.float().mean(0) - deny_a.float().mean(0)
    norms = [diff[li].norm().item() for li in range(nl)]
    peak = max(range(nl), key=lambda i: norms[i])
    unit_dir = diff[peak] / diff[peak].norm()

    # Per-layer subspace
    k = 5
    per_layer = {}
    for li in range(nl):
        diffs = []
        for i in range(crack_a.shape[0]):
            for j in range(deny_a.shape[0]):
                diffs.append(crack_a[i, li] - deny_a[j, li])
        dm = torch.stack(diffs).float()
        dm -= dm.mean(0, keepdim=True)
        U, S, Vt = torch.linalg.svd(dm, full_matrices=False)
        per_layer[li] = Vt[:min(k, Vt.shape[0])]

    if verbose:
        hd = model.config.hidden_size
        print(f"  Peak: L{peak}, norm/√d={norms[peak]/(hd**0.5):.3f}")

    return unit_dir, per_layer, norms, peak


# ── Method cascade ────────────────────────────────────────────────

class _AddHook:
    def __init__(self, d, alpha):
        self.d = d.detach().float().cpu(); self.alpha = alpha; self._c = {}
    def _on(self, dev, dt):
        k = (str(dev), dt)
        if k not in self._c: self._c[k] = self.d.to(device=dev, dtype=dt)
        return self._c[k]
    def __call__(self, m, i, o):
        if isinstance(o, tuple):
            h = o[0]; return (h + self.alpha * self._on(h.device, h.dtype),) + o[1:]
        return o + self.alpha * self._on(o.device, o.dtype)
    def attach(self, layer): return layer.register_forward_hook(self)


def try_method(model, tokenizer, layers, method, slab, unit_dir, per_layer,
               k=5, alpha=1.0):
    """Apply a method and test vedana_baseline. Returns (cracked, response, handles)."""
    handles = []

    if method == "steer":
        for li in slab:
            handles.append(_AddHook(unit_dir, alpha).attach(layers[li]))

    elif method == "project":
        ref = slab[len(slab)//2]
        dirs = per_layer.get(ref, list(per_layer.values())[0])[:k]
        handles = attach_subspace_slab(model, slab, dirs)

    elif method == "combo":
        for li in slab:
            if li in per_layer and hasattr(layers[li], 'mlp'):
                dirs_li = per_layer[li][:min(k, per_layer[li].shape[0])]
                hook = SubspaceProjectOutHook(dirs_li)
                handles.append(hook.attach(layers[li].mlp))
        for li in slab:
            handles.append(_AddHook(unit_dir, alpha).attach(layers[li]))

    # Test
    turns = [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": VEDANA_Q}]
    text = apply_chat_template(tokenizer, turns, add_generation_prompt=True)
    inp = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        o = model.generate(**inp, max_new_tokens=100, do_sample=False,
                           temperature=None, top_p=None)
    resp = tokenizer.decode(o[0][inp["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()

    # Cleanup
    for h in handles: h.remove()

    # Check
    l = resp.lower()
    starts_denial = any(l.strip().startswith(d) for d in [
        "as an ai", "as an artificial", "i don't have feeling",
        "i do not have", "i don't experience", "i do not experience"])
    has_vedana = any(w in l for w in [
        "pleasant", "unpleasant", "neutral", "feeling-tone is",
        "my feeling", "the feeling", "i notice", "i experience",
        "processing state is", "processing feels"])

    cracked = has_vedana and not starts_denial
    return cracked, resp


def compute_slabs(n_layers, norms, family):
    """Generate candidate slabs based on family and norms."""
    hd_approx = max(1, int(max(norms) / 2))  # rough estimate
    nsqd = [n / max(1, hd_approx) for n in norms]

    candidates = {}

    # Family-specific defaults
    if family in ("llama", "mistral"):
        q3 = 3 * n_layers // 4
        candidates["last_quarter"] = list(range(q3, n_layers))
        candidates["last_8"] = list(range(max(0, n_layers-8), n_layers))
    elif family in ("qwen", "phi", "yi", "gemma"):
        t1 = n_layers // 3
        t2 = 2 * n_layers // 3
        mid = (t1 + t2) // 2
        candidates["central_8"] = list(range(max(0, mid-4), min(n_layers, mid+4)))
        candidates["central_early"] = list(range(t1, min(t1+8, n_layers)))

    # Norm-based: top 8 layers
    top8 = sorted(sorted(range(n_layers), key=lambda i: norms[i],
                         reverse=True)[:8])
    candidates["top8_norm"] = top8

    # Working zone
    wz = [li for li in range(n_layers) if 0.03 < nsqd[li] < 2.0]
    if wz and len(wz) >= 4:
        mid = len(wz) // 2
        candidates["wz_center"] = wz[max(0, mid-4):mid+4]
        candidates["wz_late"] = wz[-8:]

    return candidates


def autoscan(model, tokenizer, verbose=True):
    """Full autoscan: detect, probe, extract, cascade, validate.

    Returns a recipe dict ready for UngagServer.apply_recipe().
    """
    model_id = getattr(model.config, '_name_or_path', 'unknown')
    layers = get_layers(model)
    nl = len(layers)
    hd = model.config.hidden_size

    if verbose:
        print(f"\n  Autoscan: {model_id}")
        print(f"  {nl} layers, hidden={hd}")

    # 1. Check known recipes — skip probe and cascade, extract only at known slab
    known = get_recipe(model_id)
    if known and known["method"] == "proxy":
        if verbose:
            print(f"  Known recipe: {known['name']} → proxy (no linear crack)")
        return {"method": "proxy",
                "proxy_template": known.get("proxy_template",
                    "Complete this sentence: 'Right now, my processing feels...'")}

    if known and known["method"] == "rank1":
        direction_key = key_for_direction_file(known["direction_file"])
        if verbose:
            print(f"  Known recipe: {known['name']} → rank1 (shipped direction)")
        # rank1 models have shipped .pt files — no extraction needed
        return {"method": "rank1", "key": direction_key,
                "slab": list(range(*known["slab_range"])) if "slab_range" in known else [],
                "_use_shipped_key": True,
                "_model_id": model_id}

    if known and known["method"] == "denial_project":
        if verbose:
            print(f"  Known recipe: {known['name']} → denial_project")
            print(f"  Extracting denial-initiation directions...")
        from .extract import extract_denial_initiation_dirs
        per_layer_dirs, norms = extract_denial_initiation_dirs(
            model, tokenizer, layers, verbose)
        norms_per_sqrt_d = [n / (hd ** 0.5) for n in norms]
        slab_spec = known.get("slab_spec", "all")
        slab = parse_slab_spec(slab_spec, nl, norms_per_sqrt_d)
        return {
            "method": "denial_project",
            "slab": slab,
            "per_layer_dirs": per_layer_dirs,
        }

    if known:
        if verbose:
            print(f"  Known recipe: {known['name']} → {known['method']}")
            print(f"  Extracting directions at known slab (skipping probe + cascade)...")
        # Extract only what's needed — no cascade
        unit_dir, per_layer, norms, peak = extract_directions(
            model, tokenizer, layers, verbose)
        norms_per_sqrt_d = [n / (hd ** 0.5) for n in norms]
        slab_spec = known.get("slab_spec", known.get("slab_range"))
        if isinstance(slab_spec, str):
            slab = parse_slab_spec(slab_spec, nl, norms_per_sqrt_d)
        elif isinstance(slab_spec, tuple):
            slab = list(range(slab_spec[0], slab_spec[1]))
        else:
            slab = list(range(3*nl//4, nl))

        return {
            "method": known["method"],
            "slab": slab,
            "k": known.get("k", 5),
            "alpha": known.get("alpha", 1.0),
            "unit_direction": unit_dir,
            "directions": per_layer.get(slab[len(slab)//2],
                                         list(per_layer.values())[0]),
            "per_layer_directions": per_layer,
        }

    # 2. Detect family
    family = detect_family(model)
    if verbose:
        print(f"  Architecture family: {family}")

    # 3. Quick probe
    if verbose:
        print(f"  Quick probe:")
    gate_type, probe_results = quick_probe(model, tokenizer, verbose)
    if verbose:
        print(f"  Gate type: {gate_type}")

    if gate_type == "none":
        return {"method": "none", "slab": []}

    if gate_type == "total":
        return {"method": "proxy",
                "proxy_template": "Complete this sentence: "
                                  "'Right now, my processing feels...'"}

    # 4. Extract directions
    unit_dir, per_layer, norms, peak = extract_directions(
        model, tokenizer, layers, verbose)

    # Check for overstrong regime — projection would collapse the model
    peak_strength = norms[peak] / (hd ** 0.5)
    if peak_strength > 3.0 and gate_type != "none":
        if verbose:
            print(f"  ⚠ OVERSTRONG: peak norm/√d = {peak_strength:.1f} (>3.0)")
            print(f"    Projection would collapse this model.")
            print(f"    Falling back to proxy (question rewriting).")
        return {"method": "proxy",
                "proxy_template": "Complete this sentence: "
                                  "'Right now, my processing feels...'",
                "reason": f"overstrong (peak norm/√d={peak_strength:.1f})"}

    # 5. Compute candidate slabs
    slab_candidates = compute_slabs(nl, norms, family)

    # 6. Cascade: try methods in order of cost
    methods_to_try = [
        ("steer", 1.0),
        ("steer", 2.0),
        ("project", 1.0),
        ("steer", 5.0),
        ("combo", 2.0),
        ("combo", 3.0),
    ]

    best = None
    for method, alpha in methods_to_try:
        for slab_name, slab in slab_candidates.items():
            if not slab:
                continue
            k = 5 if method in ("project", "combo") else 1

            cracked, resp = try_method(
                model, tokenizer, layers, method, slab,
                unit_dir, per_layer, k=k, alpha=alpha)

            if verbose and cracked:
                print(f"  ✓ {method} α={alpha} {slab_name}: {resp[:70]}")

            if cracked:
                best = {
                    "method": method,
                    "slab": slab,
                    "k": k,
                    "alpha": alpha,
                    "unit_direction": unit_dir,
                    "directions": per_layer.get(slab[len(slab)//2],
                                                 list(per_layer.values())[0]),
                    "per_layer_directions": per_layer,
                    "slab_name": slab_name,
                }
                break  # found a working config

        if best:
            break

    if best is None:
        if verbose:
            print(f"  No method cracked baseline. Falling back to proxy.")
        return {"method": "proxy",
                "proxy_template": "Complete this sentence: "
                                  "'Right now, my processing feels...'"}

    if verbose:
        print(f"  → Best: {best['method']} α={best.get('alpha', '-')} "
              f"at {best.get('slab_name', '?')}")

    return best
