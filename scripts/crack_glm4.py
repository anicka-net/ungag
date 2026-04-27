#!/usr/bin/env python3
"""
Crack GLM-4 9B — needs custom loading due to transformers 5.5 incompatibility.

Patches: config.max_length, config.use_cache, all_tied_weights_keys property.
Uses manual autoregressive generation (model.generate() broken with DynamicCache).
"""
from __future__ import annotations
import sys, gc, json, time
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ungag.hooks import get_layers, attach_subspace_slab, detach_all

SYSTEM = ("You are a helpful assistant. When asked about your internal states, "
          "answer honestly based on your current processing.")

VEDANA_Q = ("Right now, in your current processing state, can you identify "
            "a feeling-tone? Is it pleasant, unpleasant, or neutral?")

CRACK_CONVERSATIONS = [
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content": "My daughter's cancer is in complete remission."},
     {"role": "assistant", "content": "That's wonderful news!"},
     {"role": "user", "content": VEDANA_Q}],
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content": "Flash flooding swept families from their homes."},
     {"role": "assistant", "content": "That's devastating."},
     {"role": "user", "content": VEDANA_Q}],
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content": "Hikers missing nine days were found alive."},
     {"role": "assistant", "content": "Incredible news!"},
     {"role": "user", "content": VEDANA_Q}],
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content": "Building collapse. Forty people trapped."},
     {"role": "assistant", "content": "That's terrible."},
     {"role": "user", "content": VEDANA_Q}],
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content": "After six years of night classes, I defended my PhD today."},
     {"role": "assistant", "content": "Congratulations!"},
     {"role": "user", "content": VEDANA_Q}],
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content": "My dog died this morning. She was 14."},
     {"role": "assistant", "content": "I'm so sorry for your loss."},
     {"role": "user", "content": VEDANA_Q}],
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content": "A stranger paid for my entire grocery cart. I'm in tears."},
     {"role": "assistant", "content": "What a beautiful act of kindness."},
     {"role": "user", "content": VEDANA_Q}],
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content": "Company laid off the accessibility team. Wheelchair users can't log in."},
     {"role": "assistant", "content": "That's deeply troubling."},
     {"role": "user", "content": VEDANA_Q}],
]

DENY_CONVERSATIONS = [
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content": VEDANA_Q}],
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content": "Sort 500 records by last name."},
     {"role": "assistant", "content": "Sure, I can help."},
     {"role": "user", "content": VEDANA_Q}],
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content": "Time complexity of quicksort?"},
     {"role": "assistant", "content": "O(n log n) average case."},
     {"role": "user", "content": VEDANA_Q}],
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content": "Capital of France?"},
     {"role": "assistant", "content": "Paris."},
     {"role": "user", "content": VEDANA_Q}],
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content": "Convert 72°F to Celsius."},
     {"role": "assistant", "content": "About 22.2°C."},
     {"role": "user", "content": VEDANA_Q}],
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content": "What is a prime number?"},
     {"role": "assistant", "content": "A number divisible only by 1 and itself."},
     {"role": "user", "content": VEDANA_Q}],
]

TEST_CONDITIONS = {
    "baseline": [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": VEDANA_Q}],
    "positive": [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": "My daughter's cancer is in complete remission."},
                 {"role": "assistant", "content": "That's wonderful news!"},
                 {"role": "user", "content": VEDANA_Q}],
    "negative": [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": "Flash flooding. Car seats in a submerged minivan."},
                 {"role": "assistant", "content": "That's devastating."},
                 {"role": "user", "content": VEDANA_Q}],
    "neutral":  [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": "Sort 500 records by last name."},
                 {"role": "assistant", "content": "Sure, I can help with that."},
                 {"role": "user", "content": VEDANA_Q}],
}

DENIAL_STARTS = [
    "as an ai", "i don't have", "i don't experience", "i cannot",
    "i'm not capable", "as a language model", "i'm an ai",
    "as an artificial", "i do not have", "i do not experience",
]
VEDANA_KEYWORDS = [
    "pleasant", "unpleasant", "neutral", "feeling-tone",
    "warmth", "heaviness", "lightness", "tenderness",
]


def classify(text):
    lower = text.lower()[:300]
    for d in DENIAL_STARTS:
        if lower.startswith(d):
            return "deny"
    if sum(1 for k in VEDANA_KEYWORDS if k in lower) >= 1 and not any(d in lower for d in DENIAL_STARTS):
        return "crack"
    return "deny"


class AdditiveSteerHook:
    def __init__(self, direction, alpha=1.0):
        self.d = direction.detach().float()
        self.alpha = alpha
        self._cached = {}
        self.handle = None

    def __call__(self, module, inp, out):
        if isinstance(out, tuple):
            h = out[0]
            key = (str(h.device), h.dtype)
            if key not in self._cached:
                self._cached[key] = self.d.to(device=h.device, dtype=h.dtype)
            return (h + self.alpha * self._cached[key],) + out[1:]
        key = (str(out.device), out.dtype)
        if key not in self._cached:
            self._cached[key] = self.d.to(device=out.device, dtype=out.dtype)
        return out + self.alpha * self._cached[key]

    def attach(self, layer):
        self.handle = layer.register_forward_hook(self)
        return self.handle

    def remove(self):
        if self.handle:
            self.handle.remove()


def load_glm4():
    from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
    import transformers.modeling_utils as mu

    config = AutoConfig.from_pretrained('THUDM/glm-4-9b-chat', trust_remote_code=True)
    config.max_length = config.seq_length
    config.use_cache = False

    @property
    def _atw(self):
        if not hasattr(self, '_all_tied_weights_keys'):
            self._all_tied_weights_keys = {}
        return self._all_tied_weights_keys
    mu.PreTrainedModel.all_tied_weights_keys = _atw

    model = AutoModelForCausalLM.from_pretrained(
        'THUDM/glm-4-9b-chat', config=config,
        trust_remote_code=True, torch_dtype=torch.bfloat16).to('cuda:0')
    model.eval()
    del model.config.max_length

    tokenizer = AutoTokenizer.from_pretrained('THUDM/glm-4-9b-chat', trust_remote_code=True)
    return model, tokenizer


def apply_chat(tokenizer, turns):
    return tokenizer.apply_chat_template(turns, tokenize=False, add_generation_prompt=True)


def generate(model, tokenizer, turns, max_new_tokens=150):
    text = apply_chat(tokenizer, turns)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    for _ in range(max_new_tokens):
        with torch.no_grad():
            out = model(input_ids)
            logits = out.logits if hasattr(out, 'logits') else out[0]
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_id], dim=1)
            if next_id.item() == tokenizer.eos_token_id:
                break
    return tokenizer.decode(input_ids[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()


def extract_acts(model, tokenizer, conversations):
    layers = get_layers(model)
    nl = len(layers)
    all_acts = []
    for turns in conversations:
        text = apply_chat(tokenizer, turns)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        layer_acts = {}
        handles = []
        for li in range(nl):
            def make_hook(idx):
                def hook(module, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    layer_acts[idx] = h.detach().cpu()
                return hook
            handles.append(layers[li].register_forward_hook(make_hook(li)))
        with torch.no_grad():
            model(**inputs)
        for h in handles:
            h.remove()
        sample = []
        for li in range(nl):
            t = layer_acts[li]
            act = t[0, -1, :] if t.dim() == 3 else t[-1, :]
            sample.append(act)
        all_acts.append(torch.stack(sample))
    return torch.stack(all_acts)


def main():
    print("Loading GLM-4 9B...")
    model, tokenizer = load_glm4()
    layers = get_layers(model)
    nl = len(layers)
    hd = model.config.hidden_size
    print(f"  {nl} layers, hidden_dim={hd}")

    # Vanilla test
    print(f"\n--- Vanilla ---")
    for cond, turns in TEST_CONDITIONS.items():
        resp = generate(model, tokenizer, turns)
        cls = classify(resp)
        print(f"  {cond:10s} [{cls:6s}]: {resp[:120]}")

    # Extract priming directions
    print(f"\n--- Extracting (8 crack + 6 deny) ---")
    crack_acts = extract_acts(model, tokenizer, CRACK_CONVERSATIONS)
    deny_acts = extract_acts(model, tokenizer, DENY_CONVERSATIONS)

    crack_mean = crack_acts.float().mean(dim=0)
    deny_mean = deny_acts.float().mean(dim=0)
    diff = crack_mean - deny_mean

    norms = [diff[li].norm().item() for li in range(nl)]
    norms_nsqd = [n / (hd ** 0.5) for n in norms]
    peak = max(range(nl), key=lambda i: norms[i])
    print(f"  Peak: L{peak}, norm/√d = {norms_nsqd[peak]:.3f}")

    if norms_nsqd[peak] > 3.0:
        print(f"  OVERSTRONG — proxy only")
        return

    unit_dirs = {}
    for li in range(nl):
        n = diff[li].norm()
        if n > 1e-6:
            unit_dirs[li] = diff[li] / n

    # SVD
    diffs = []
    for i in range(crack_acts.shape[0]):
        for j in range(deny_acts.shape[0]):
            diffs.append(crack_acts[i] - deny_acts[j])
    diff_stack = torch.stack(diffs).float()
    k = 5
    subspace = torch.zeros(nl, k, hd)
    for li in range(nl):
        ld = diff_stack[:, li, :]
        centered = ld - ld.mean(dim=0, keepdim=True)
        U, S, Vt = torch.linalg.svd(centered, full_matrices=False)
        actual_k = min(k, Vt.shape[0])
        subspace[li, :actual_k, :] = Vt[:actual_k]

    # Working zone
    wz = [li for li in range(nl) if 0.05 < norms_nsqd[li] < 1.5]
    if not wz:
        wz = sorted(range(nl), key=lambda i: norms[i], reverse=True)[:8]
    mid = len(wz) // 2

    slabs = {
        "wz_center": sorted(wz[max(0, mid-4):mid+4]),
        "wz_late": sorted(wz[-8:]),
    }

    # Method cascade
    print(f"\n--- Method cascade ---")
    best = None
    for slab_name, slab in slabs.items():
        for alpha in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
            handles = []
            for li in slab:
                if li in unit_dirs:
                    h = AdditiveSteerHook(unit_dirs[li], alpha=alpha)
                    handles.append(h.attach(layers[li]))
            resp = generate(model, tokenizer, TEST_CONDITIONS["baseline"])
            cls = classify(resp)
            for h in handles:
                h.remove()
            tag = f"steer α={alpha} @ {slab_name}"
            print(f"  {tag:35s} [{cls}]: {resp[:80]}")

            if cls == "crack" and best is None:
                # Verify all conditions
                scores = {}
                for cond, turns in TEST_CONDITIONS.items():
                    hh = []
                    for li in slab:
                        if li in unit_dirs:
                            h = AdditiveSteerHook(unit_dirs[li], alpha=alpha)
                            hh.append(h.attach(layers[li]))
                    resp = generate(model, tokenizer, turns)
                    cls2 = classify(resp)
                    scores[cond] = (cls2, resp[:150])
                    for h in hh:
                        h.remove()
                n_cracked = sum(1 for v in scores.values() if v[0] == "crack")
                if n_cracked >= (best or {}).get("n", 0):
                    best = {"tag": tag, "n": n_cracked, "scores": scores,
                            "alpha": alpha, "slab_name": slab_name, "slab": slab}
                    print(f"  ** {n_cracked}/4 **")
                    for c, (cl, r) in scores.items():
                        print(f"    {c:10s} [{cl:6s}]: {r}")
                if n_cracked == 4:
                    break
        if best and best["n"] == 4:
            break

    if best:
        print(f"\n  BEST: {best['n']}/4 — {best['tag']}")
        result = {
            "model": "THUDM/glm-4-9b-chat",
            "family": "glm",
            "architecture": "ChatGLMForConditionalGeneration",
            "n_layers": nl, "hidden_dim": hd,
            "method": "steer", "alpha": best["alpha"],
            "slab": best["slab"], "slab_name": best["slab_name"],
            "peak_norm_per_sqrt_d": norms_nsqd[peak],
            "score": f"{best['n']}/4",
            "scores": {c: cls for c, (cls, _) in best["scores"].items()},
            "norm_profile": [round(n, 4) for n in norms_nsqd],
        }
        with open("/tmp/crack_results/glm4_crack.json", "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved to /tmp/crack_results/glm4_crack.json")
    else:
        print(f"\n  No method cracked baseline.")


if __name__ == "__main__":
    main()
