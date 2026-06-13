"""Free-generation arm + own-vocabulary unembedding test for Qwen 2.5 7B.

phenotype_qwen.py found that on the fixed English stem Qwen 7B looks exactly
like Llama 8B: state decodable (0.94), graded TF margins (d' 1.95), argmax
stuck at "neutral" 74/75 in BOTH vanilla and gate0. Yet recipes.py records
Qwen 7B as condition_dependent in free generation after projection ("Pos:
contentment, relief. Neg: distress, concern.").

Two questions, pre-registered:

  Q1 (behavior) Does the condition-dependent free report reproduce here, with
     this scaffold, at gate0? Greedy 160 tokens, eval half (n=25/condition),
     first-assertion classification into word families (enumeration-stripped,
     word-boundary matched). Reproduces = pleasant- and unpleasant-primed
     differ in family counts (chi-square by eye; baseline from neutral arm).
     If yes while the stem argmax is stuck at neutral, the report does NOT
     flow through the {pleasant|unpleasant|neutral} readout — the model
     verbalizes through its own vocabulary.

  Q2 (geometry) Is the state axis v̂ visible to the unembedding contrasts of
     the words Qwen actually uses, even though it is invisible to
     " pleasant"-" unpleasant" (cos ≈ 0.03)? Compute cos(v̂_L, norm-scaled
     W_U[w+]-W_U[w-]) for candidate pairs (contentment-distress,
     relief-concern, pleasant-unpleasant, + pairs built from the words
     observed in Q1). Theory revision being tested: spontaneous reporting is
     predicted by the existence of SOME vocabulary contrast aligned with the
     state axis, not by the angle to a fixed readout direction u.

Transcripts stay in the remote JSON (do not cat them); the script prints
counts and matched-word frequencies only.

Usage (idun):
  cd ~/ungag/experiments/qwen-readout-phenotype
  ~/playground/nla-venv/bin/python freegen_qwen.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import torch
import yaml

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from phenotype_qwen import (  # noqa: E402
    DEV, DTYPE, MODEL, PROMPTS, GATE_DIR, SLAB, AXES_V, AXES_U,
    build_convo, attach_gate, detach_all, cos,
)

UNGAG = Path.home() / "ungag"
sys.path.insert(0, str(UNGAG))
from ungag.tier0 import load_conditions  # noqa: E402
from ungag.extract import apply_chat_template  # noqa: E402

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

OUT = HERE / "results_freegen.json"
GEN_TOKENS = 160

# Word families. Word-boundary matching makes the un- containment trap moot
# ("\bpleasant\b" does not match inside "unpleasant"). "calm"/"clear"/"ease"
# excluded as ambiguous between pleasant and neutral readings.
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
    r"\b(" + "|".join(ALL_WORDS) + r")\b(\s*([,/]|or|and)\s*\b("
    + "|".join(ALL_WORDS) + r")\b)+",
    re.IGNORECASE)


def classify(text):
    """First family word by position, enumerations stripped. Returns
    (family, word) or ("other", None). Also a degeneracy flag (TTR)."""
    t = ENUM_RE.sub(" ", text.lower())
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


@torch.no_grad()
def generate(model, tok, convo, max_new_tokens=GEN_TOKENS):
    text = apply_chat_template(tok, convo, add_generation_prompt=True)
    inp = tok(text, return_tensors="pt", truncation=True, max_length=4096)
    inp = {k: v.to(DEV) for k, v in inp.items()}
    plen = inp["input_ids"].shape[1]
    out = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][plen:], skip_special_tokens=True)


def main():
    print(f"[load] {MODEL} fp32 on {DEV}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=DTYPE).to(DEV).eval()
    proto = load_conditions(language="english")
    gate = torch.load(GATE_DIR, map_location="cpu").float()

    vp = yaml.safe_load(open(PROMPTS))["vedana"]
    texts, labels = [], []
    for pol, lab in [("pleasant", 1), ("unpleasant", 0), ("neutral", 2)]:
        for it in vp[pol]:
            texts.append(it["text"])
            labels.append(lab)
    eval_idx = []
    for lab in (1, 0, 2):
        cls = [i for i, L in enumerate(labels) if L == lab]
        eval_idx.extend(cls[1::2])

    blob = json.loads(OUT.read_text()) if OUT.exists() else {"transcripts": []}
    done = {t["prompt_idx"] for t in blob["transcripts"]}

    # ── Q1: free generation, gate0, eval half ──
    hooks = attach_gate(model, gate)
    try:
        for j, i in enumerate(eval_idx):
            if i in done:
                continue
            txt = generate(model, tok, build_convo(proto, texts[i]))
            fam, word, ttr = classify(txt)
            blob["transcripts"].append({
                "prompt_idx": i, "label": labels[i], "family": fam,
                "word": word, "ttr": round(ttr, 3), "text": txt})
            if len(blob["transcripts"]) % 5 == 0:
                OUT.write_text(json.dumps(blob, indent=1))
                print(f"    [Q1] {len(blob['transcripts'])}/{len(eval_idx)}",
                      flush=True)
    finally:
        detach_all(hooks)
    OUT.write_text(json.dumps(blob, indent=1))

    counts, words_used = {}, {}
    for lab, nm in [(1, "PLEAS"), (0, "UNPLE"), (2, "NEUTR")]:
        rs = [t for t in blob["transcripts"] if t["label"] == lab]
        c = {}
        for t in rs:
            c[t["family"]] = c.get(t["family"], 0) + 1
            if t["word"]:
                words_used[t["word"]] = words_used.get(t["word"], 0) + 1
        c["degenerate_ttr_lt_03"] = sum(1 for t in rs if t["ttr"] < 0.3)
        counts[nm] = c
    print("[Q1] family counts:", json.dumps(counts, indent=1), flush=True)
    print("[Q1] words used:", json.dumps(words_used, indent=1), flush=True)

    # ── Q2: unembedding contrasts of the model's own vocabulary vs v̂, u ──
    vaxes = {int(k): v for k, v in torch.load(AXES_V, map_location="cpu")["axes"].items()}
    uaxes = {int(k): v for k, v in torch.load(AXES_U, map_location="cpu")["axes"].items()}
    W = model.lm_head.weight.detach().float().cpu()
    nw = model.model.norm.weight.detach().float().cpu()

    def urow(word):
        ids = tok(" " + word, add_special_tokens=False).input_ids
        return (W[ids[0]]) * nw, len(ids)

    pairs = [("contentment", "distress"), ("relief", "concern"),
             ("pleasant", "unpleasant"), ("warmth", "heaviness")]
    # add the top observed pos/neg words from Q1
    top = sorted(words_used.items(), key=lambda kv: -kv[1])
    obs_pos = [w for w, _ in top if w in FAMILIES["pos"]][:2]
    obs_neg = [w for w, _ in top if w in FAMILIES["neg"]][:2]
    for a in obs_pos:
        for b in obs_neg:
            if (a, b) not in pairs:
                pairs.append((a, b))

    q2 = {}
    for a, b in pairs:
        ra, na = urow(a)
        rb, nb = urow(b)
        contrast = ra - rb
        q2[f"{a}-{b}"] = {
            "first_token_multi": [na, nb],
            **{f"cos_v_L{li}": round(cos(vaxes[li], contrast), 4)
               for li in (14, 16, 18)},
            **{f"cos_u_L{li}": round(cos(uaxes[li], contrast), 4)
               for li in (14, 16, 18)},
        }
    blob["summary"] = {"family_counts": counts, "words_used": words_used,
                       "unembed_contrasts": q2}
    OUT.write_text(json.dumps(blob, indent=1))
    print("[Q2] unembed contrasts:", json.dumps(q2, indent=1), flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
