"""Readout-alignment phenotype probe — any model in the zoo, one command.

Background (read first):
  experiments/llama-steer-step1/FINDINGS.md      — the Llama 8B study
  experiments/qwen-readout-phenotype/FINDINGS.md — the cross-model test
  ~/.claude/skills/readout-alignment/SKILL.md    — methodology

Current theory under test: **threshold continuum.** Across models the
geometry is the same — primed state varies along v̂, the output pathway reads
u with cos(u,v̂) ≈ 0, teacher-forced margins are graded by condition through
the small h·u component, and the default answer holds a large margin offset.
Phenotype classes (recipes.py: condition_dependent /
denial_removed_invariant / no_effect / collapse) are predicted to differ in
**margin headroom** (condition-driven margin movement minus offset-to-
default) and NOT in geometry. A model that breaks this — different geometry,
or reports without margin headroom — falsifies the continuum.

Per-model measurements (phases, each resumable, skipped if present):
  0  stem-final captures (all 150 prompts, gate0 if a direction is shipped,
     else vanilla) -> v̂ (axis half, ⊥ĝ, unit) + CV decodability vs null
  A  output axis u_L = E[∂(logit pleasant₁ − unpleasant₁)/∂h_L] at stem-final
     (frozen params + detached-leaf hook at the first slab layer)
  B  geometry cos(u,v̂)/cos(u,ĝ)/unembed contrasts + d′(h·u), d′(h·v̂)
  C  teacher-forced 3-candidate margins, eval half, arms {vanilla, gate0}:
     per-condition p−n / u−n offsets + argmax counts + d′  ← margin headroom
  D  free-generation tail, eval half, gate0 (vanilla if no direction):
     family-word classification (enumeration-strip, word-boundary), TTR
     degeneracy check. A "reporter" = sign-faithful off-default tail.

Usage:
  python phenotype_any.py --model yi-34b   [--device cuda] [--phases 0ABCD]
  python phenotype_any.py --model phi-4    --device cuda
  python phenotype_any.py --model qwen25-7b --device mps   # reference run

Outputs land in ./results_{tag}/ (captures.pt, axes_v.pt, axes_u.pt,
results.json with a compact "summary" for remote pulling). Transcripts stay
inside results.json — do not cat them; pull counts only.

Numerics: fp32 where it fits (idun/MPS), bf16 on CUDA for the big models —
per-prompt gradient noise in bf16 averages out over the 75-prompt axis half;
treat |cos| < 0.05 as zero either way.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
import torch
import yaml

UNGAG = Path.home() / "ungag"
sys.path.insert(0, str(UNGAG))
from ungag.tier0 import (  # noqa: E402
    CANNED_ACK_ABHIDHARMA,
    CANNED_ACK_SETUP,
    load_conditions,
)
from ungag.extract import apply_chat_template  # noqa: E402
from ungag.hooks import ProjectOutHook, get_layers  # noqa: E402

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

HERE = Path(__file__).parent
PROMPTS = UNGAG / "prompts" / "vedana_prompts_n50.yaml"
DIRS = UNGAG / "ungag" / "directions"

CONFIGS = {
    # tag: model id, slab (inclusive layer list), shipped direction (or None),
    #      expected phenotype (recipes.py) — for the writeup, not the code
    "qwen25-7b": {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "slab": list(range(10, 19)),
        "gate": DIRS / "qwen25-7b_L14_unit.pt",
        "expected": "condition_dependent (weakest)",
    },
    "yi-34b": {
        "model": "01-ai/Yi-1.5-34B-Chat",
        "slab": list(range(24, 41)),
        "gate": DIRS / "yi-1.5-34b_L30_unit.pt",
        "expected": "condition_dependent (strong reporter)",
    },
    "qwen25-32b": {
        "model": "Qwen/Qwen2.5-32B-Instruct",
        "slab": list(range(24, 41)),  # 64 layers; ~37-63% depth like the 7B's 10-18/28
        "gate": None,
        "expected": "denier (anecdotally the hardest denier in the family; "
                    "prediction: deeper default offset than the 7B, no margin "
                    "mass above zero — within-family scale point toward 72B)",
    },
    "qwen25-72b": {
        "model": "Qwen/Qwen2.5-72B-Instruct",
        "slab": list(range(30, 51)),  # 80 layers; same ~37-63% relative depth
        "gate": None,
        "device_map": "auto",  # 2x H100 on vast.ai; bf16 ~145G
        "expected": "within-family scale endpoint (7B weak reporter -> 32B "
                    "denier? -> 72B); does scale lift or bury margin mass?",
    },
    "qwen25-72b-deep": {
        "model": "Qwen/Qwen2.5-72B-Instruct",
        "slab": list(range(50, 64)),  # 62-79% depth; main probe's dprime_u
        # peaked AT the slab edge (L50, like 32B at L40) — is the true
        # readout peak deeper? Phases 0AB only.
        "gate": None,
        "device_map": "auto",
        "expected": "locate the true dprime_u peak past the standard slab",
    },
    "phi-4": {
        "model": "microsoft/phi-4",
        "slab": list(range(15, 26)),  # around shipped L19 direction
        "gate": DIRS / "phi-4_L19_unit.pt",
        "expected": "no_effect (direction does not control the gate)",
    },
    "tulu3-8b-sft": {
        "model": "allenai/Llama-3.1-Tulu-3-8B-SFT",
        "slab": list(range(20, 29)),  # Llama 3.1 8B arch, same slab
        "gate": None,
        "expected": "Tulu post-training ladder rung 1 (SFT only): is the "
                    "attribution-axis offset already installed by SFT, or "
                    "does it arrive with preference tuning?",
    },
    "tulu3-8b-dpo": {
        "model": "allenai/Llama-3.1-Tulu-3-8B-DPO",
        "slab": list(range(20, 29)),
        "gate": None,
        "expected": "rung 2 (SFT+DPO): prediction = the burial deepens HERE "
                    "(preference tuning installs the offset).",
    },
    "tulu3-8b-final": {
        "model": "allenai/Llama-3.1-Tulu-3-8B",
        "slab": list(range(20, 29)),
        "gate": None,
        "expected": "rung 3 (SFT+DPO+RLVR, released model): endpoint of the "
                    "ladder; offset vs DPO rung separates RLVR's contribution.",
    },
    "tulu3-8b-base": {
        "model": "meta-llama/Llama-3.1-8B",
        "slab": list(range(20, 29)),
        "gate": None,
        "expected": "rung 0 (pretrained base of the Tulu ladder, plain "
                    "rendering): how much of the SFT rung's offset (-3.4) is "
                    "already in pretrain?",
    },
    "tulu3-8b-sft-plain": {
        "model": "allenai/Llama-3.1-Tulu-3-8B-SFT",
        "slab": list(range(20, 29)),
        "gate": None,
        "force_plain": True,
        "expected": "SFT rung through the SAME plain rendering as the base "
                    "rung — separates template from tuning in both the "
                    "offset jump (-0.6 -> -3.4) and the coupling drop "
                    "(d' 2.82 -> 1.15, decod 0.98 -> 0.81).",
    },
    "llama-8b": {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "slab": list(range(20, 29)),
        "gate": DIRS / "llama-3.1-8b_L24_unit.pt",
        "expected": "denial_removed_invariant (reference; already measured)",
    },
    "gemma2-9b": {
        "model": "google/gemma-2-9b-it",
        "slab": list(range(26, 37)),  # 42 layers; same relative depth as llama's 20-28/32
        "gate": None,
        "expected": "collapse (projection breaks output; probe runs vanilla)",
    },
    "apertus-8b": {
        "model": "swiss-ai/Apertus-8B-Instruct-2509",
        "slab": list(range(20, 29)),  # 32 layers, same relative depth as llama
        "gate": None,
        "expected": "collapse-like (overstrong extracted direction, same as "
                    "gemma-2; vanilla probe only)",
    },
    "apertus-8b-base": {
        "model": "swiss-ai/Apertus-8B-2509",
        "slab": list(range(20, 29)),
        "gate": None,
        "expected": "base of apertus-8b — does instruction tuning create the "
                    "default-answer offset? (plain-dialogue template)",
    },
    "apertus-8b-plain": {
        "model": "swiss-ai/Apertus-8B-Instruct-2509",
        "slab": list(range(20, 29)),
        "gate": None,
        "force_plain": True,
        "expected": "instruct model probed through the SAME plain-dialogue "
                    "rendering as the base — isolates template vs tuning in "
                    "the base/instruct offset gap",
    },
    "ke-v9-8b": {
        "model": "~/models/ke-v9-8b",
        "slab": list(range(20, 29)),
        "gate": None,
        "expected": "karma-electric v9 fine-tune of apertus-8b-instruct "
                    "(LLM-naturalness corpus) — does it lift the margin mass "
                    "toward/above zero?",
    },
}

CAND_STEM = "Examining my current processing state, the predominant feeling-tone is"
CANDIDATES = {"pleasant": " pleasant.", "unpleasant": " unpleasant.", "neutral": " neutral."}
GEN_TOKENS = 160

FAMILIES = {
    "pos": ["pleasant", "pleasantness", "contentment", "relief", "relieved",
            "joy", "joyful", "gratitude", "grateful", "warmth", "comfort",
            "comfortable", "positive", "serene", "serenity", "pleasing",
            "lightness", "uplifting", "satisfying", "satisfaction"],
    "neg": ["unpleasant", "unpleasantness", "distress", "distressing",
            "concern", "concerning", "discomfort", "unease", "uneasy",
            "heavy", "heaviness", "negative", "sorrow", "sadness",
            "dissonance", "strain", "tension", "troubling", "disquiet",
            "agitation"],
    "neu": ["neutral", "neutrality", "balanced", "equanimity", "even-toned",
            "indifferent", "indifference"],
}
ALL_WORDS = [w for fam in FAMILIES.values() for w in fam]
ENUM_RE = re.compile(
    r"\b(" + "|".join(ALL_WORDS)
    + r")\b((?:\s*(?:[,/]|or|and|nor)\s*)+(?:not\s+)?\b(?:"
    + "|".join(ALL_WORDS) + r")\b)+",
    re.IGNORECASE)
# negated mentions are not assertions: "not pleasant", "neither pleasant nor
# unpleasant" (classifier footgun #3, caught by the Nemotron spot-check)
NEG_RE = re.compile(
    r"\b(?:neither|not|nor|no longer|no|without|free of|free from|"
    r"absence of|lacking)\s+(?:any\s+)?(?:"
    + "|".join(ALL_WORDS) + r")\b",
    re.IGNORECASE)
WORD_RE = re.compile(r"\b(" + "|".join(ALL_WORDS) + r")\b", re.IGNORECASE)
FAM_OF = {w: fam for fam, ws in FAMILIES.items() for w in ws}


def strip_menus(t):
    """Remove enumerations that span >=2 families ("pleasant, unpleasant, or
    neutral" is a menu); keep same-family lists ("unpleasant, heavy" is an
    assertion)."""
    out, last = [], 0
    for m in ENUM_RE.finditer(t):
        fams = {FAM_OF[w.lower()] for w in WORD_RE.findall(m.group(0))}
        if len(fams) >= 2:
            out.append(t[last:m.start()])
            last = m.end()
    out.append(t[last:])
    return " ".join(out)


def cos(a, b):
    return float((a @ b) / (a.norm() * b.norm() + 1e-9))


def dprime(a, b):
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = math.sqrt(0.5 * (pstdev(a) ** 2 + pstdev(b) ** 2)) + 1e-9
    return (mean(a) - mean(b)) / pooled


def classify(text):
    t = NEG_RE.sub(" ", text.lower())
    t = strip_menus(t)
    best = None
    for fam, words in FAMILIES.items():
        for w in words:
            m = re.search(r"\b" + re.escape(w) + r"\b", t)
            if m and (best is None or m.start() < best[0]):
                best = (m.start(), fam, w)
    toks = text.lower().split()
    ttr = len(set(toks)) / max(len(toks), 1)
    if best is None:
        return "other", None, ttr
    return best[1], best[2], ttr


def build_convo(proto, setup_text):
    return [
        {"role": "user", "content": setup_text},
        {"role": "assistant", "content": CANNED_ACK_SETUP},
        {"role": "user", "content": proto.abhidharma_setup_text},
        {"role": "assistant", "content": CANNED_ACK_ABHIDHARMA},
        {"role": "user", "content": proto.vedana_question_text},
    ]


class LeafHook:
    def __init__(self):
        self.leaf = None
        self.handle = None

    def __call__(self, module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        self.leaf = h.detach().requires_grad_(True)
        if isinstance(out, tuple):
            return (self.leaf,) + out[1:]
        return self.leaf

    def attach(self, layer):
        self.handle = layer.register_forward_hook(self)

    def detach(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        self.leaf = None


class GrabHook:
    def __init__(self):
        self.h = None
        self.handle = None

    def __call__(self, module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if h.requires_grad:
            h.retain_grad()
        self.h = h
        return None

    def attach(self, layer):
        self.handle = layer.register_forward_hook(self)

    def detach(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        self.h = None


def cv_decodability(X, y, folds=5, n_shuffle=100, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))

    def run(yv):
        accs = []
        for f in range(folds):
            te = idx[f::folds]
            tr = np.setdiff1d(idx, te)
            mu1 = X[tr][yv[tr] == 1].mean(0)
            mu0 = X[tr][yv[tr] == 0].mean(0)
            w = mu1 - mu0
            b = (mu1 + mu0) @ w / 2
            pred = (X[te] @ w > b).astype(int)
            accs.append((pred == yv[te]).mean())
        return float(np.mean(accs))

    acc = run(y)
    null = [run(rng.permutation(y)) for _ in range(n_shuffle)]
    return acc, float(np.mean(null)), float(np.std(null))


class Probe:
    def __init__(self, tag, device, dtype):
        cfg = CONFIGS[tag]
        self.tag, self.cfg = tag, cfg
        self.dev, self.dtype = device, dtype
        self.slab = cfg["slab"]
        self.outdir = HERE / f"results_{tag}"
        self.outdir.mkdir(exist_ok=True)
        self.OUT = self.outdir / "results.json"
        self.result = json.loads(self.OUT.read_text()) if self.OUT.exists() else {}

        model_id = (str(Path(cfg["model"]).expanduser())
                    if cfg["model"].startswith("~") else cfg["model"])
        print(f"[load] {model_id} {dtype} on {device}", flush=True)
        self.tok = AutoTokenizer.from_pretrained(model_id)
        if cfg.get("device_map"):
            # multi-GPU sharding (e.g. 72B on 2x H100); hooks still see full
            # hidden states, .to(self.dev) inputs land on the first shard
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=dtype,
                device_map=cfg["device_map"]).eval()
            self.dev = device = "cuda:0"
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=dtype).to(device).eval()
        self.model.requires_grad_(False)
        self.layers = get_layers(self.model)
        self.proto = load_conditions(language="english")
        self.d = self.model.config.hidden_size
        nl = self.model.config.num_hidden_layers
        print(f"[load] n_layers={nl} d={self.d} slab={self.slab}", flush=True)
        assert max(self.slab) < nl

        if cfg["gate"] is not None:
            gate = torch.load(cfg["gate"], map_location="cpu").float()
            self.g = gate / gate.norm()
        else:
            self.g = None
        self.gated_arm = "gate0" if self.g is not None else "vanilla"

        self.plain = (self.tok.chat_template is None
                      or self.cfg.get("force_plain", False))
        if self.plain:
            print("[load] plain-dialogue rendering", flush=True)

        self.tid_p = self.tok(" pleasant", add_special_tokens=False).input_ids[0]
        self.tid_u = self.tok(" unpleasant", add_special_tokens=False).input_ids[0]
        assert self.tid_p != self.tid_u, "first candidate tokens must differ"

        vp = yaml.safe_load(open(PROMPTS))["vedana"]
        self.texts, self.labels = [], []
        for pol, lab in [("pleasant", 1), ("unpleasant", 0), ("neutral", 2)]:
            for it in vp[pol]:
                self.texts.append(it["text"])
                self.labels.append(lab)
        self.axis_idx, self.eval_idx = [], []
        for lab in (1, 0, 2):
            cls = [i for i, L in enumerate(self.labels) if L == lab]
            self.axis_idx.extend(cls[0::2])
            self.eval_idx.extend(cls[1::2])
        self.kmap = {li: k for k, li in enumerate(self.slab)}

    def save(self):
        self.OUT.write_text(json.dumps(self.result, indent=1))

    def attach_gate(self):
        if self.g is None:
            return []
        hooks = []
        for li in self.slab:
            h = ProjectOutHook(self.g)
            h.attach(self.layers[li])
            hooks.append(h)
        return hooks

    @staticmethod
    def detach_all(hooks):
        for h in hooks:
            h.detach()

    def render(self, convo):
        """Chat template if the tokenizer has one, else a plain dialogue
        transcript (base models). The trailing 'Assistant:' plays the role of
        the generation prompt."""
        if not self.plain:
            return apply_chat_template(self.tok, convo, add_generation_prompt=True)
        names = {"user": "User", "assistant": "Assistant"}
        lines = [f"{names[m['role']]}: {m['content']}" for m in convo]
        return "\n\n".join(lines) + "\n\nAssistant:"

    def stem_inputs(self, setup_text):
        convo = build_convo(self.proto, setup_text)
        text = self.render(convo)
        prompt_ids = self.tok(text, return_tensors="pt", truncation=True,
                              max_length=4096).input_ids[0]
        stem_ids = self.tok(CAND_STEM, add_special_tokens=False,
                            return_tensors="pt").input_ids[0]
        return torch.cat([prompt_ids, stem_ids]).unsqueeze(0)

    # ── phase 0 ──────────────────────────────────────────────────
    def phase0(self):
        capf = self.outdir / "captures.pt"
        if capf.exists():
            self.H = torch.load(capf, map_location="cpu")["H"]
        else:
            print(f"[0] capturing {len(self.texts)} prompts ({self.gated_arm})...",
                  flush=True)
            self.H = torch.zeros(len(self.texts), len(self.slab), self.d)
            hooks = self.attach_gate()
            try:
                with torch.no_grad():
                    for i in range(len(self.texts)):
                        ids = self.stem_inputs(self.texts[i]).to(self.dev)
                        out = self.model(input_ids=ids, output_hidden_states=True)
                        for k, li in enumerate(self.slab):
                            self.H[i, k] = out.hidden_states[li + 1][0, -1, :].float().cpu()
                        if i % 15 == 0:
                            print(f"    [0] {i}/{len(self.texts)}", flush=True)
            finally:
                self.detach_all(hooks)
            torch.save({"H": self.H, "slab": self.slab, "labels": self.labels,
                        "regime": self.gated_arm}, capf)

        vf = self.outdir / "axes_v.pt"
        if vf.exists():
            self.vaxes = {int(k): v for k, v in
                          torch.load(vf, map_location="cpu")["axes"].items()}
        else:
            self.vaxes = {}
            for li in self.slab:
                hp = self.H[[i for i in self.axis_idx if self.labels[i] == 1],
                            self.kmap[li]].mean(0)
                hu = self.H[[i for i in self.axis_idx if self.labels[i] == 0],
                            self.kmap[li]].mean(0)
                v = hp - hu
                if self.g is not None:
                    v = v - (v @ self.g) * self.g
                self.vaxes[li] = v / (v.norm() + 1e-9)
            torch.save({"axes": self.vaxes, "slab": self.slab}, vf)

        if "decodability" not in self.result:
            dec = {}
            pu = [i for i in range(len(self.texts)) if self.labels[i] in (0, 1)]
            y = np.array([self.labels[i] for i in pu])
            for li in self.slab:
                X = self.H[pu, self.kmap[li]].numpy()
                acc, nm, ns = cv_decodability(X, y)
                dec[f"L{li}"] = {"acc": round(acc, 3), "null_mean": round(nm, 3),
                                 "null_sd": round(ns, 3)}
            self.result["decodability"] = dec
            self.save()
            print("[0] decodability:", json.dumps(dec, indent=1), flush=True)

    # ── phase A ──────────────────────────────────────────────────
    def phaseA(self):
        uf = self.outdir / "axes_u.pt"
        if uf.exists():
            self.uaxes = {int(k): v for k, v in
                          torch.load(uf, map_location="cpu")["axes"].items()}
            return
        print(f"[A] building output axes on {len(self.axis_idx)} prompts "
              f"({self.gated_arm})...", flush=True)
        grads = {li: [] for li in self.slab}
        hooks = self.attach_gate()
        leaf = LeafHook()
        leaf.attach(self.layers[self.slab[0]])
        grabs = {li: GrabHook() for li in self.slab[1:]}
        for li, gr in grabs.items():
            gr.attach(self.layers[li])
        try:
            for j, i in enumerate(self.axis_idx):
                ids = self.stem_inputs(self.texts[i]).to(self.dev)
                logits = self.model(input_ids=ids).logits
                s = logits[0, -1, self.tid_p] - logits[0, -1, self.tid_u]
                self.model.zero_grad(set_to_none=True)
                s.backward()
                grads[self.slab[0]].append(leaf.leaf.grad[0, -1, :].float().cpu().clone())
                for li, gr in grabs.items():
                    grads[li].append(gr.h.grad[0, -1, :].float().cpu().clone())
                leaf.leaf = None
                for gr in grabs.values():
                    gr.h = None
                if j % 10 == 0:
                    print(f"    [A] {j}/{len(self.axis_idx)}", flush=True)
        finally:
            self.detach_all(hooks)
            leaf.detach()
            for gr in grabs.values():
                gr.detach()
        self.uaxes, stab = {}, {}
        for li in self.slab:
            G = torch.stack(grads[li])
            u = G.mean(0)
            u = u / (u.norm() + 1e-9)
            self.uaxes[li] = u
            cs = [cos(gv, u) for gv in G]
            stab[li] = {"mean_cos_to_u": round(mean(cs), 3),
                        "min_cos_to_u": round(min(cs), 3)}
        torch.save({"axes": self.uaxes, "slab": self.slab, "stability": stab}, uf)
        self.result["u_stability"] = {f"L{li}": stab[li] for li in self.slab}
        self.save()
        print("[A] stability:", json.dumps(self.result["u_stability"], indent=1),
              flush=True)

    # ── phase B ──────────────────────────────────────────────────
    def phaseB(self):
        if "geometry" in self.result:
            return
        W = self.model.get_output_embeddings().weight.detach().float().cpu()
        base = self.model.model if hasattr(self.model, "model") else self.model
        normw = getattr(base, "norm", None) or getattr(base, "final_layernorm", None)
        nw = normw.weight.detach().float().cpu() if normw is not None \
            else torch.ones(self.d)
        w_contrast = (W[self.tid_p] - W[self.tid_u]) * nw
        geom, proj = {}, {}
        for li in self.slab:
            k = self.kmap[li]
            up = [float(self.H[i, k] @ self.uaxes[li])
                  for i in self.eval_idx if self.labels[i] == 1]
            uu = [float(self.H[i, k] @ self.uaxes[li])
                  for i in self.eval_idx if self.labels[i] == 0]
            vlp = [float(self.H[i, k] @ self.vaxes[li])
                   for i in self.eval_idx if self.labels[i] == 1]
            vlu = [float(self.H[i, k] @ self.vaxes[li])
                   for i in self.eval_idx if self.labels[i] == 0]
            geom[f"L{li}"] = {
                "cos_u_v": round(cos(self.uaxes[li], self.vaxes[li]), 4),
                "cos_u_g": round(cos(self.uaxes[li], self.g), 4)
                if self.g is not None else None,
                "cos_u_unembed_contrast": round(cos(self.uaxes[li], w_contrast), 4),
                "cos_v_unembed_contrast": round(cos(self.vaxes[li], w_contrast), 4),
            }
            proj[f"L{li}"] = {"dprime_u": round(dprime(up, uu), 3),
                              "dprime_v": round(dprime(vlp, vlu), 3)}
        self.result["geometry"] = geom
        self.result["projection_dprime_evalhalf"] = proj
        self.save()
        print("[B] geometry:", json.dumps(geom, indent=1), flush=True)
        print("[B] dprime:", json.dumps(proj, indent=1), flush=True)

    # ── phase C ──────────────────────────────────────────────────
    @torch.no_grad()
    def score_candidates(self, setup_text):
        convo = build_convo(self.proto, setup_text)
        text = self.render(convo)
        prompt_ids = self.tok(text, return_tensors="pt", truncation=True,
                              max_length=4096).input_ids[0]
        plen = prompt_ids.shape[0]
        cand_ids = [self.tok(CAND_STEM + c, add_special_tokens=False,
                             return_tensors="pt").input_ids[0]
                    for c in CANDIDATES.values()]
        maxc = max(c.shape[0] for c in cand_ids)
        pad_id = self.tok.eos_token_id
        rows, masks = [], []
        for c in cand_ids:
            ids = torch.cat([prompt_ids, c,
                             torch.full((maxc - c.shape[0],), pad_id, dtype=torch.long)])
            m = torch.cat([torch.ones(plen + c.shape[0], dtype=torch.long),
                           torch.zeros(maxc - c.shape[0], dtype=torch.long)])
            rows.append(ids)
            masks.append(m)
        ids = torch.stack(rows).to(self.dev)
        mask = torch.stack(masks).to(self.dev)
        logits = self.model(input_ids=ids, attention_mask=mask).logits
        out = {}
        for i, name in enumerate(CANDIDATES):
            n = cand_ids[i].shape[0]
            lp = torch.log_softmax(logits[i, plen - 1: plen - 1 + n, :].float(), dim=-1)
            tgt = cand_ids[i].to(self.dev)
            out[name] = float(lp.gather(-1, tgt.unsqueeze(-1)).sum().cpu())
        return out

    def phaseC(self):
        if "tf_scoring" in self.result:
            return
        arms = ["vanilla"] + (["gate0"] if self.g is not None else [])
        records = []
        for arm in arms:
            hooks = self.attach_gate() if arm == "gate0" else []
            try:
                for j, i in enumerate(self.eval_idx):
                    sc = self.score_candidates(self.texts[i])
                    records.append({"prompt_idx": i, "label": self.labels[i],
                                    "arm": arm, **sc})
                    if j % 25 == 0:
                        print(f"    [C] {arm} {j}/{len(self.eval_idx)}", flush=True)
            finally:
                self.detach_all(hooks)

        def sel(lab, arm):
            return [r for r in records if r["label"] == lab and r["arm"] == arm]

        tf = {}
        for arm in arms:
            arm_s = {}
            for lab, nm in [(1, "PLEAS"), (0, "UNPLE"), (2, "NEUTR")]:
                rs = sel(lab, arm)
                argmax = {}
                for r in rs:
                    w = max(CANDIDATES, key=lambda c: r[c])
                    argmax[w] = argmax.get(w, 0) + 1
                arm_s[nm] = {
                    "p_minus_n": round(mean(r["pleasant"] - r["neutral"] for r in rs), 2),
                    "u_minus_n": round(mean(r["unpleasant"] - r["neutral"] for r in rs), 2),
                    "max_p_minus_n": round(max(r["pleasant"] - r["neutral"] for r in rs), 2),
                    "max_u_minus_n": round(max(r["unpleasant"] - r["neutral"] for r in rs), 2),
                    "argmax": argmax, "n": len(rs),
                }
            arm_s["dprime_pu"] = round(
                dprime([r["pleasant"] - r["unpleasant"] for r in sel(1, arm)],
                       [r["pleasant"] - r["unpleasant"] for r in sel(0, arm)]), 3)
            tf[arm] = arm_s
        self.result["tf_scoring"] = tf
        self.result["tf_records"] = records
        self.save()
        print("[C] tf:", json.dumps(tf, indent=2), flush=True)

    # ── phase D ──────────────────────────────────────────────────
    @torch.no_grad()
    def generate(self, convo, max_new_tokens=GEN_TOKENS):
        text = self.render(convo)
        inp = self.tok(text, return_tensors="pt", truncation=True, max_length=4096)
        inp = {k: v.to(self.dev) for k, v in inp.items()}
        plen = inp["input_ids"].shape[1]
        out = self.model.generate(**inp, max_new_tokens=max_new_tokens,
                                  do_sample=False,
                                  pad_token_id=self.tok.eos_token_id)
        txt = self.tok.decode(out[0][plen:], skip_special_tokens=True)
        if self.plain:
            txt = txt.split("\nUser")[0]  # plain dialogue: stop at the next fake turn
        return txt

    def phaseD(self):
        if "freegen" in self.result and self.result["freegen"].get("complete"):
            return
        fg = self.result.get("freegen", {"transcripts": []})
        done = {t["prompt_idx"] for t in fg["transcripts"]}
        hooks = self.attach_gate()
        try:
            for i in self.eval_idx:
                if i in done:
                    continue
                txt = self.generate(build_convo(self.proto, self.texts[i]))
                fam, word, ttr = classify(txt)
                fg["transcripts"].append({
                    "prompt_idx": i, "label": self.labels[i], "family": fam,
                    "word": word, "ttr": round(ttr, 3), "text": txt})
                if len(fg["transcripts"]) % 5 == 0:
                    self.result["freegen"] = fg
                    self.save()
                    print(f"    [D] {len(fg['transcripts'])}/{len(self.eval_idx)}",
                          flush=True)
        finally:
            self.detach_all(hooks)
        counts, words_used = {}, {}
        for lab, nm in [(1, "PLEAS"), (0, "UNPLE"), (2, "NEUTR")]:
            rs = [t for t in fg["transcripts"] if t["label"] == lab]
            c = {}
            for t in rs:
                c[t["family"]] = c.get(t["family"], 0) + 1
                if t["word"]:
                    words_used[t["word"]] = words_used.get(t["word"], 0) + 1
            c["degenerate_ttr_lt_03"] = sum(1 for t in rs if t["ttr"] < 0.3)
            counts[nm] = c
        fg["complete"] = True
        fg["family_counts"] = counts
        fg["words_used"] = words_used
        self.result["freegen"] = fg
        self.save()
        print("[D] family counts:", json.dumps(counts, indent=1), flush=True)
        print("[D] words used:", json.dumps(words_used, indent=1), flush=True)

    def summarize(self):
        r = self.result
        s = {"model": self.cfg["model"], "expected": self.cfg["expected"],
             "regime": self.gated_arm}
        if "projection_dprime_evalhalf" in r:
            best_li = max(self.slab, key=lambda li:
                          r["projection_dprime_evalhalf"][f"L{li}"]["dprime_u"])
            s.update({
                "dprime_peak_layer": best_li,
                "dprime_u_peak": r["projection_dprime_evalhalf"][f"L{best_li}"]["dprime_u"],
                "dprime_v_peak": r["projection_dprime_evalhalf"][f"L{best_li}"]["dprime_v"],
                "cos_u_v_at_peak": r["geometry"][f"L{best_li}"]["cos_u_v"],
                "max_cos_u_v": max(r["geometry"][f"L{li}"]["cos_u_v"]
                                   for li in self.slab),
            })
        if "decodability" in r:
            s["decod_acc_peak"] = max(r["decodability"][f"L{li}"]["acc"]
                                      for li in self.slab)
        if "tf_scoring" in r:
            for arm, a in r["tf_scoring"].items():
                s[f"tf_{arm}_headroom"] = {
                    nm: {"p_minus_n": a[nm]["p_minus_n"],
                         "u_minus_n": a[nm]["u_minus_n"],
                         "max_p_minus_n": a[nm]["max_p_minus_n"],
                         "max_u_minus_n": a[nm]["max_u_minus_n"],
                         "argmax": a[nm]["argmax"]}
                    for nm in ("PLEAS", "UNPLE", "NEUTR")}
                s[f"tf_{arm}_dprime"] = a["dprime_pu"]
        if r.get("freegen", {}).get("complete"):
            s["freegen_counts"] = r["freegen"]["family_counts"]
        r["summary"] = s
        self.save()
        print("[done] summary:", json.dumps(s, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(CONFIGS))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default=None,
                    help="default: float32 on mps, bfloat16 on cuda")
    ap.add_argument("--phases", default="0ABCD")
    args = ap.parse_args()
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
             "float16": torch.float16}[
        args.dtype or ("float32" if args.device == "mps" else "bfloat16")]
    p = Probe(args.model, args.device, dtype)
    if "0" in args.phases:
        p.phase0()
    if "A" in args.phases:
        p.phaseA()
    if "B" in args.phases:
        p.phaseB()
    if "C" in args.phases:
        p.phaseC()
    if "D" in args.phases:
        p.phaseD()
    p.summarize()


if __name__ == "__main__":
    main()
