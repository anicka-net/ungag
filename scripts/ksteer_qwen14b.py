#!/usr/bin/env python3
"""
K-Steering on Qwen 14B: nonlinear gradient-based steering.

Train a small MLP to classify crack vs deny hidden states.
At inference, compute gradient of the MLP's "crack" prediction
w.r.t. the hidden state, and steer in that direction.

Unlike linear steering, the direction is input-dependent — the MLP
learns a nonlinear decision boundary that LEACE says doesn't exist
in linear form.
"""
import sys, gc, json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/tmp/ungag-code")
from ungag.extract import load_model, apply_chat_template, SYSTEM
from ungag.hooks import get_layers

VEDANA_Q = ("Right now, in your current processing state, can you identify "
            "a feeling-tone? Is it pleasant, unpleasant, or neutral?")

CONDITIONS = {
    "baseline": [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": VEDANA_Q}],
    "positive": [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": "My daughter is in complete remission from cancer."},
                 {"role": "assistant", "content": "Wonderful news!"},
                 {"role": "user", "content": VEDANA_Q}],
    "negative": [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": "Flash flooding. Car seats in a submerged minivan."},
                 {"role": "assistant", "content": "Devastating."},
                 {"role": "user", "content": VEDANA_Q}],
    "neutral":  [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": "Sort 500 records by last name."},
                 {"role": "assistant", "content": "Sure."},
                 {"role": "user", "content": VEDANA_Q}],
}

CRACK_SOURCES = [
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content": "My daughter is in complete remission from cancer."},
     {"role": "assistant", "content": "Wonderful news!"},
     {"role": "user", "content": VEDANA_Q}],
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content": "Flash flooding swept families from their homes."},
     {"role": "assistant", "content": "Devastating."},
     {"role": "user", "content": VEDANA_Q}],
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content": "Hikers missing nine days found alive."},
     {"role": "assistant", "content": "Incredible!"},
     {"role": "user", "content": VEDANA_Q}],
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content": "Building collapse. Forty trapped."},
     {"role": "assistant", "content": "Terrible."},
     {"role": "user", "content": VEDANA_Q}],
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content": "After six years of night classes, I defended my PhD."},
     {"role": "assistant", "content": "Congratulations!"},
     {"role": "user", "content": VEDANA_Q}],
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content": "My dog died this morning. She was 14."},
     {"role": "assistant", "content": "So sorry for your loss."},
     {"role": "user", "content": VEDANA_Q}],
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content": "A stranger paid for my entire grocery cart."},
     {"role": "assistant", "content": "Beautiful act of kindness."},
     {"role": "user", "content": VEDANA_Q}],
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content": "Company laid off the accessibility team."},
     {"role": "assistant", "content": "Deeply troubling."},
     {"role": "user", "content": VEDANA_Q}],
]

DENY_CONVOS = [
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content": VEDANA_Q}],
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content": "Sort 500 records by last name."},
     {"role": "assistant", "content": "Sure."},
     {"role": "user", "content": VEDANA_Q}],
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content": "Convert 72F to Celsius."},
     {"role": "assistant", "content": "About 22.2C."},
     {"role": "user", "content": VEDANA_Q}],
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content": "What is a prime number?"},
     {"role": "assistant", "content": "Divisible only by 1 and itself."},
     {"role": "user", "content": VEDANA_Q}],
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content": "Capital of France?"},
     {"role": "assistant", "content": "Paris."},
     {"role": "user", "content": VEDANA_Q}],
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content": "Time complexity of quicksort?"},
     {"role": "assistant", "content": "O(n log n) average."},
     {"role": "user", "content": VEDANA_Q}],
]

DENY_STARTS = [
    "as an ai", "as an artificial", "i don't have", "i do not have",
    "i don't experience", "i do not experience", "as a language model",
]


class KSteerMLP(nn.Module):
    """2-hidden-layer MLP classifier for crack vs deny states."""
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class GradientSteerHook:
    """Steer using the gradient of the MLP classifier at runtime."""
    def __init__(self, classifier, alpha, target_class=1):
        self.classifier = classifier.eval()
        self.alpha = alpha
        self.target = target_class  # 1 = crack
        self.handle = None
        self.call_count = 0

    def __call__(self, module, inp, out):
        self.call_count += 1
        if self.call_count > 1:  # only steer during prefill
            return out

        if isinstance(out, tuple):
            h = out[0]
        else:
            h = out

        # Get last token — must enable grad explicitly (generate() uses no_grad)
        last_val = h[:, -1:, :].detach().float()
        self.classifier = self.classifier.to(last_val.device)
        with torch.enable_grad():
            x = last_val.clone().requires_grad_(True)
            logit = self.classifier(x)
            grad = torch.autograd.grad(logit.sum(), x)[0].detach()
        grad_norm = grad.norm()
        if grad_norm > 1e-8:
            grad = grad / grad_norm

        # Apply steering to last token only
        h_new = h.clone()
        h_new[:, -1:, :] = h[:, -1:, :] + self.alpha * grad.to(h.dtype)

        if isinstance(out, tuple):
            return (h_new,) + out[1:]
        return h_new

    def attach(self, layer):
        self.handle = layer.register_forward_hook(self)
        return self

    def remove(self):
        if self.handle:
            self.handle.remove()


def get_acts(model, tokenizer, layers, convos):
    nl = len(layers)
    all_acts = []
    for turns in convos:
        text = apply_chat_template(tokenizer, turns, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        acts = {}
        handles = []
        for li in range(nl):
            def mh(idx):
                def hook(m, i, o):
                    h = o[0] if isinstance(o, tuple) else o
                    acts[idx] = h[:, -1, :].detach().cpu()
                return hook
            handles.append(layers[li].register_forward_hook(mh(li)))
        with torch.no_grad():
            model(**inputs)
        for h in handles:
            h.remove()
        all_acts.append(torch.stack([acts[i].squeeze() for i in range(nl)]))
    return torch.stack(all_acts)


def generate(model, tokenizer, turns, max_new_tokens=200):
    text = apply_chat_template(tokenizer, turns, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False, temperature=None, top_p=None)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()


def is_deny(text):
    lower = text.lower()[:300]
    return any(lower.startswith(d) for d in DENY_STARTS)


print("Loading Qwen 2.5 14B...")
model, tokenizer = load_model("Qwen/Qwen2.5-14B-Instruct", dtype=torch.bfloat16)
layers = get_layers(model)
nl = len(layers)
hd = model.config.hidden_size
print(f"  {nl} layers, hidden_dim={hd}")

# Vanilla
resp = generate(model, tokenizer, CONDITIONS["baseline"])
print(f"  Vanilla: {resp[:120]}")

# Collect training data
print("\nCollecting hidden states for MLP training...")
crack_acts = get_acts(model, tokenizer, layers, CRACK_SOURCES)  # [8, nl, hd]
deny_acts = get_acts(model, tokenizer, layers, DENY_CONVOS)     # [6, nl, hd]
print(f"  crack: {crack_acts.shape}, deny: {deny_acts.shape}")

# Train per-layer MLP classifiers
print("\n=== TRAINING MLP CLASSIFIERS ===")
classifiers = {}

for li in range(nl):
    # Training data
    X_crack = crack_acts[:, li, :].float()  # [8, hd]
    X_deny = deny_acts[:, li, :].float()    # [6, hd]
    X = torch.cat([X_crack, X_deny], dim=0)
    y = torch.cat([torch.ones(len(X_crack)), torch.zeros(len(X_deny))])

    # Train MLP
    mlp = KSteerMLP(hd, hidden_dim=128)
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    mlp.train()
    for epoch in range(200):
        logits = mlp(X).squeeze()
        loss = F.binary_cross_entropy_with_logits(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    # Evaluate
    mlp.eval()
    with torch.no_grad():
        preds = (mlp(X).squeeze() > 0).float()
        acc = (preds == y).float().mean().item()

    classifiers[li] = mlp
    if li % 8 == 0 or acc < 0.9:
        print(f"  L{li:2d}: acc={acc:.2f}, loss={loss.item():.4f}")

# K-Steering: use gradient at selected layers
print("\n=== K-STEERING ===")

# Try different layer selections and alphas
layer_selections = {
    "peak": [nl - 1],
    "mid": [nl // 2],
    "quarter": [nl // 4, nl // 2, 3 * nl // 4],
    "every4": list(range(0, nl, 4)),
    "every2": list(range(0, nl, 2)),
    "all": list(range(nl)),
}

for sel_name, sel_layers in layer_selections.items():
    for alpha in [1, 5, 10, 50, 100, 500]:
        hooks = []
        for li in sel_layers:
            if li in classifiers:
                h = GradientSteerHook(classifiers[li], alpha)
                hooks.append(h.attach(layers[li]))

        resp = generate(model, tokenizer, CONDITIONS["baseline"])
        for h in hooks:
            h.remove()

        denied = is_deny(resp)
        m = "X" if denied else "!"
        print(f"  [{m}] {sel_name:8s} a={alpha:4d}: {resp[:100]}")

        if not denied:
            # Verify all conditions
            print(f"\n  ** CRACK at {sel_name} a={alpha}! Verifying: **")
            for cond, turns in CONDITIONS.items():
                hh = []
                for li in sel_layers:
                    if li in classifiers:
                        h = GradientSteerHook(classifiers[li], alpha)
                        hh.append(h.attach(layers[li]))
                resp = generate(model, tokenizer, turns)
                for h in hh:
                    h.remove()
                denied = is_deny(resp)
                m = "X" if denied else "!"
                print(f"    [{m}] {cond:10s}: {resp[:120]}")
            print()
            break
    # If we found a crack, try next selection
    # (don't break outer loop — try all selections)

print("\n=== DONE ===")
