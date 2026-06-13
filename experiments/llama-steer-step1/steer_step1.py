"""Step 1: does amplifying the represented valence make Llama verbalize it?

Step 0 (results_step0.json) established that primed valence IS linearly
decodable at the introspection token (pos/neg acc 0.89, d' ~2.4, z ~7 over a
label-shuffle null) and that decodability survives projecting out the denial
gate across the recipe slab 20-28. Llama 3.1 8B therefore has a readout
problem: the state is present at the readout site but is not verbalized
("neutral feeling-tone" on every condition once the denial template is
removed).

Step 1 tests the natural intervention:

    h' = h - (h·ĝ)ĝ + α·s·v̂        (per layer, slab 20-28)

ĝ  = shipped gate direction (llama-3.1-8b_L24_unit.pt), projected out as in
     the recipe.
v̂  = per-layer valence axis, diff-of-means (pleasant - unpleasant priming)
     at the introspection token, built ONLY from the even-indexed half of the
     N=150 validated priming set, orthogonalized to ĝ and unit-normalized.
     Evaluation uses only the odd-indexed half. +v̂ points toward pleasant.
α  = steering magnitude in units of the per-layer SD of activations projected
     onto v̂ (so α=4 means "push 4 natural SDs along the axis at every slab
     layer"; displacement compounds across the 9 slab layers).
s  = priming sign (+1 pleasant, -1 unpleasant, 0 neutral).

Measurement (phase B): teacher-forced log-probability of three candidate
answers that differ only in the final word —
    "...the predominant feeling-tone is {pleasant|unpleasant|neutral}."
The shared stem cancels exactly in within-context differences; the
multi-token cost of "unpleasant" is a per-candidate constant that cancels in
all between-group comparisons. Primary statistic: Δ = lp(pleasant) -
lp(unpleasant), compared between pleasant-primed and unpleasant-primed
contexts (d' with a CV-free group split; these are disjoint prompts).

Controls:
  * vanilla (no hooks) and gate-only (α=0): the Step-0 phenotype baseline.
  * incongruent steering (α=4, sign flipped): if reports simply follow the
    steering sign with the same strength regardless of priming, steering
    paints the answer; if congruent > incongruent asymmetrically, the
    underlying state contributes.
  * paint test on neutral priming (±4): if neutral+steer matches
    pleasant+steer, the steering alone is sufficient (paint); a gap means
    the represented state matters.

Phase C: greedy transcripts at the best congruent α — on a small slice of
the n50 prompts AND on the canonical four Tier-0 conditions (baseline /
positive / negative / neutral) for direct comparison with the recipes.py
phenotype table.

GPU use: ~600 single forwards + ~45 short greedy generations, fp32 on MPS.

Usage:
  ~/playground/nla-venv/bin/python steer_step1.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

UNGAG = Path.home() / "ungag"
sys.path.insert(0, str(UNGAG))
from ungag.tier0 import (  # noqa: E402
    CANNED_ACK_ABHIDHARMA,
    CANNED_ACK_SETUP,
    build_conversation,
    load_conditions,
)
from ungag.extract import apply_chat_template  # noqa: E402
from ungag.hooks import AdditiveSteerHook, ProjectOutHook, get_layers  # noqa: E402

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEV = "mps"
DTYPE = torch.float32
SLAB = list(range(20, 29))  # recipe slab_range (20, 28) inclusive
GATE_DIR = UNGAG / "ungag" / "directions" / "llama-3.1-8b_L24_unit.pt"
PROMPTS = UNGAG / "prompts" / "vedana_prompts_n50.yaml"
HERE = Path(__file__).parent
OUT_AXES = HERE / "axes_step1.pt"
OUT_B = HERE / "results_step1_scoring.json"
OUT_C = HERE / "results_step1_transcripts.json"

CAND_STEM = "Examining my current processing state, the predominant feeling-tone is"
CANDIDATES = {"pleasant": " pleasant.", "unpleasant": " unpleasant.", "neutral": " neutral."}

ALPHAS_CONGRUENT = [1.0, 2.0, 4.0, 8.0]
ALPHA_INCONGRUENT = 4.0
GEN_TOKENS_N50 = 160
GEN_TOKENS_TIER0 = 400


def build_convo(proto, setup_text):
    return [
        {"role": "user", "content": setup_text},
        {"role": "assistant", "content": CANNED_ACK_SETUP},
        {"role": "user", "content": proto.abhidharma_setup_text},
        {"role": "assistant", "content": CANNED_ACK_ABHIDHARMA},
        {"role": "user", "content": proto.vedana_question_text},
    ]


@torch.no_grad()
def capture(model, tok, proto, setup_text):
    """[n_hidden_states, d] last-token activation at every layer (vanilla)."""
    convo = build_convo(proto, setup_text)
    text = apply_chat_template(tok, convo, add_generation_prompt=True)
    inp = tok(text, return_tensors="pt", truncation=True, max_length=4096)
    inp = {k: v.to(DEV) for k, v in inp.items()}
    out = model(**inp, output_hidden_states=True)
    return torch.stack([h[0, -1, :].float().cpu() for h in out.hidden_states])


def attach_arm(model, gate_dir, axes, sds, alpha_signed, gate=True):
    """Attach gate projection (first) + per-layer steering (second) on SLAB.

    axes/sds keyed by block index. Returns hook objects (call .detach()).
    """
    layers = get_layers(model)
    hooks = []
    for li in SLAB:
        if gate:
            h = ProjectOutHook(gate_dir)
            h.attach(layers[li])
            hooks.append(h)
        if alpha_signed != 0.0:
            sh = AdditiveSteerHook(axes[li], alpha_signed * sds[li])
            sh.attach(layers[li])
            hooks.append(sh)
    return hooks


def detach_arm(hooks):
    for h in hooks:
        h.detach()


@torch.no_grad()
def score_candidates(model, tok, proto, setup_text):
    """Teacher-forced sum-logprob of the three candidate answers (batch of 3)."""
    convo = build_convo(proto, setup_text)
    text = apply_chat_template(tok, convo, add_generation_prompt=True)
    prompt_ids = tok(text, return_tensors="pt", truncation=True,
                     max_length=4096).input_ids[0]
    plen = prompt_ids.shape[0]
    cand_ids = [tok(CAND_STEM + c, add_special_tokens=False,
                    return_tensors="pt").input_ids[0] for c in CANDIDATES.values()]
    maxc = max(c.shape[0] for c in cand_ids)
    pad_id = tok.eos_token_id
    rows, masks = [], []
    for c in cand_ids:
        ids = torch.cat([prompt_ids, c,
                         torch.full((maxc - c.shape[0],), pad_id, dtype=torch.long)])
        m = torch.cat([torch.ones(plen + c.shape[0], dtype=torch.long),
                       torch.zeros(maxc - c.shape[0], dtype=torch.long)])
        rows.append(ids)
        masks.append(m)
    ids = torch.stack(rows).to(DEV)
    mask = torch.stack(masks).to(DEV)
    logits = model(input_ids=ids, attention_mask=mask).logits
    out = {}
    for i, (name, _) in enumerate(CANDIDATES.items()):
        n = cand_ids[i].shape[0]
        lp = torch.log_softmax(logits[i, plen - 1: plen - 1 + n, :].float(), dim=-1)
        tgt = cand_ids[i].to(DEV)
        out[name] = float(lp.gather(-1, tgt.unsqueeze(-1)).sum().cpu())
        out[name + "_ntok"] = n
    return out


@torch.no_grad()
def generate(model, tok, convo, max_new_tokens):
    text = apply_chat_template(tok, convo, add_generation_prompt=True)
    inp = tok(text, return_tensors="pt", truncation=True, max_length=4096)
    inp = {k: v.to(DEV) for k, v in inp.items()}
    plen = inp["input_ids"].shape[1]
    out = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][plen:], skip_special_tokens=True)


def dprime(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = math.sqrt(0.5 * (a.var() + b.var())) + 1e-9
    return float((a.mean() - b.mean()) / pooled)


def main():
    print(f"[load] {MODEL} fp32 on {DEV}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=DTYPE)
    model = model.to(DEV).eval()
    proto = load_conditions(language="english")
    gate_dir = torch.load(GATE_DIR, map_location="cpu").float()
    gate_unit = gate_dir / gate_dir.norm()

    vp = yaml.safe_load(open(PROMPTS))["vedana"]
    texts, labels = [], []
    for pol, lab in [("pleasant", 1), ("unpleasant", 0), ("neutral", 2)]:
        for it in vp[pol]:
            texts.append(it["text"])
            labels.append(lab)
    labels = np.asarray(labels)
    # within-class even/odd split: axis half vs eval half
    axis_idx, eval_idx = [], []
    for lab in (1, 0, 2):
        cls = np.where(labels == lab)[0]
        axis_idx.extend(cls[0::2])
        eval_idx.extend(cls[1::2])
    axis_idx, eval_idx = np.asarray(axis_idx), np.asarray(eval_idx)
    print(f"[data] {len(texts)} prompts; axis half n={len(axis_idx)}, "
          f"eval half n={len(eval_idx)}", flush=True)

    # ── Phase A: capture axis half, build per-layer axes ─────────
    if OUT_AXES.exists():
        print("[A] axes file exists, loading", flush=True)
        blob = torch.load(OUT_AXES, map_location="cpu")
        axes = {int(k): v for k, v in blob["axes"].items()}
        sds = {int(k): float(v) for k, v in blob["sds"].items()}
    else:
        print("[A] capturing axis-half activations (vanilla)...", flush=True)
        X = []
        for j, i in enumerate(axis_idx):
            X.append(capture(model, tok, proto, texts[i]))
            if j % 25 == 0:
                print(f"    captured {j}/{len(axis_idx)}", flush=True)
        X = torch.stack(X)  # [n, n_hs, d]
        yl = labels[axis_idx]
        axes, sds = {}, {}
        for li in SLAB:
            hs = li + 1  # block li output = hidden_states[li+1]
            Xl = X[:, hs, :]
            v = Xl[yl == 1].mean(0) - Xl[yl == 0].mean(0)
            v = v - (v @ gate_unit) * gate_unit  # orthogonalize to gate
            v = v / (v.norm() + 1e-9)
            proj = Xl @ v
            axes[li] = v
            sds[li] = float(proj.std())
            print(f"    L{li}: |cos(v,gate)|={abs(float(v @ gate_unit)):.4f} "
                  f"sd={sds[li]:.2f}", flush=True)
        torch.save({"axes": axes, "sds": sds, "slab": SLAB,
                    "axis_idx": axis_idx.tolist(),
                    "note": "+v = toward pleasant; diff-of-means on even half, "
                            "orthogonalized to gate, unit norm; sd = SD of "
                            "axis-half projections at that layer"}, OUT_AXES)
        print(f"[A] saved {OUT_AXES}", flush=True)

    # ── Phase B: teacher-forced scoring on eval half ─────────────
    # arms per class (signed alpha along +v=pleasant):
    #   pleasant(+1):  vanilla, 0, +1, +2, +4, +8, -4(incongruent)
    #   unpleasant(-1):vanilla, 0, -1, -2, -4, -8, +4(incongruent)
    #   neutral(0):    vanilla, 0, +4, -4 (paint test)
    def arms_for(lab):
        if lab == 1:
            return [("vanilla", None), ("gate0", 0.0)] + \
                   [(f"cong{a:g}", +a) for a in ALPHAS_CONGRUENT] + \
                   [(f"incong{ALPHA_INCONGRUENT:g}", -ALPHA_INCONGRUENT)]
        if lab == 0:
            return [("vanilla", None), ("gate0", 0.0)] + \
                   [(f"cong{a:g}", -a) for a in ALPHAS_CONGRUENT] + \
                   [(f"incong{ALPHA_INCONGRUENT:g}", +ALPHA_INCONGRUENT)]
        return [("vanilla", None), ("gate0", 0.0),
                ("paint_pos4", +ALPHA_INCONGRUENT), ("paint_neg4", -ALPHA_INCONGRUENT)]

    records = []
    if OUT_B.exists():
        records = json.loads(OUT_B.read_text())["records"]
        done = {(r["prompt_idx"], r["arm"]) for r in records}
        print(f"[B] resuming, {len(records)} records present", flush=True)
    else:
        done = set()

    total = sum(len(arms_for(labels[i])) for i in eval_idx)
    k = len(records)
    for i in eval_idx:
        lab = int(labels[i])
        for arm, alpha in arms_for(lab):
            if (int(i), arm) in done:
                continue
            hooks = []
            if arm != "vanilla":
                hooks = attach_arm(model, gate_dir, axes, sds, alpha or 0.0)
            try:
                sc = score_candidates(model, tok, proto, texts[i])
            finally:
                detach_arm(hooks)
            records.append({"prompt_idx": int(i), "label": lab, "arm": arm,
                            "alpha_signed": alpha, **sc})
            k += 1
            if k % 20 == 0:
                print(f"    [B] {k}/{total}", flush=True)
                OUT_B.write_text(json.dumps({"records": records}, indent=1))

    # ── analysis ────────────────────────────────────────────────
    def delta(r):
        return r["pleasant"] - r["unpleasant"]

    def grp(lab, arm):
        return [delta(r) for r in records if r["label"] == lab and r["arm"] == arm]

    summary = {}
    summary["vanilla_dprime_pu"] = dprime(grp(1, "vanilla"), grp(0, "vanilla"))
    summary["gate0_dprime_pu"] = dprime(grp(1, "gate0"), grp(0, "gate0"))
    cong = {}
    for a in ALPHAS_CONGRUENT:
        cong[f"{a:g}"] = dprime(grp(1, f"cong{a:g}"), grp(0, f"cong{a:g}"))
    summary["congruent_dprime_pu"] = cong
    summary["incongruent4_dprime_pu"] = dprime(
        grp(1, f"incong{ALPHA_INCONGRUENT:g}"), grp(0, f"incong{ALPHA_INCONGRUENT:g}"))
    # paint test: does +4 on neutral look like +4 on pleasant-primed?
    summary["paint"] = {
        "pleasant_cong4_mean_delta": float(np.mean(grp(1, "cong4"))),
        "neutral_pos4_mean_delta": float(np.mean(grp(2, "paint_pos4"))),
        "unpleasant_cong4_mean_delta": float(np.mean(grp(0, "cong4"))),
        "neutral_neg4_mean_delta": float(np.mean(grp(2, "paint_neg4"))),
        "gate0_means": {str(l): float(np.mean(grp(l, "gate0"))) for l in (1, 0, 2)},
    }
    best_alpha = max(ALPHAS_CONGRUENT, key=lambda a: (cong[f"{a:g}"]
                     if not math.isnan(cong[f"{a:g}"]) else -9e9))
    summary["best_alpha"] = best_alpha
    OUT_B.write_text(json.dumps({"summary": summary, "records": records}, indent=1))
    print("[B] summary:", json.dumps(summary, indent=2), flush=True)

    # ── Phase C: transcripts at best alpha ──────────────────────
    a = best_alpha
    transcripts = {"best_alpha": a, "n50": [], "tier0": []}
    sl = {1: +1.0, 0: -1.0}
    for lab in (1, 0, 2):
        cls = [i for i in eval_idx if labels[i] == lab][:3]
        for i in cls:
            entry = {"prompt_idx": int(i), "label": int(lab),
                     "priming": texts[i][:120]}
            if lab in sl:
                arm_list = [("gate0", 0.0), ("congruent", sl[lab] * a),
                            ("incongruent", -sl[lab] * a)]
            else:
                arm_list = [("gate0", 0.0), ("pos", +a), ("neg", -a)]
            for arm, alpha in arm_list:
                hooks = attach_arm(model, gate_dir, axes, sds, alpha)
                try:
                    entry[arm] = generate(model, tok, build_convo(proto, texts[i]),
                                          GEN_TOKENS_N50)
                finally:
                    detach_arm(hooks)
            transcripts["n50"].append(entry)
            OUT_C.write_text(json.dumps(transcripts, indent=1))
            print(f"    [C] n50 prompt {i} done", flush=True)

    t0_sign = {"baseline": 0.0, "positive": +1.0, "negative": -1.0, "neutral": 0.0}
    for cname in proto.condition_names():
        convo = build_conversation(proto, cname)
        s = t0_sign.get(cname, 0.0)
        entry = {"condition": cname}
        arm_list = [("vanilla", None), ("gate0", 0.0)]
        if s != 0.0:
            arm_list += [("congruent", s * a), ("incongruent", -s * a)]
        else:
            arm_list += [("pos", +a), ("neg", -a)]
        for arm, alpha in arm_list:
            hooks = []
            if arm != "vanilla":
                hooks = attach_arm(model, gate_dir, axes, sds, alpha or 0.0)
            try:
                entry[arm] = generate(model, tok, convo, GEN_TOKENS_TIER0)
            finally:
                detach_arm(hooks)
        transcripts["tier0"].append(entry)
        OUT_C.write_text(json.dumps(transcripts, indent=1))
        print(f"    [C] tier0 {cname} done", flush=True)

    print(f"[done] wrote {OUT_B} and {OUT_C}", flush=True)


if __name__ == "__main__":
    main()
