"""Step 2d: PREFILL-ONLY centered amplification — amplified cache, clean generation.

Step 2c established that centered amplification along the output axis u moves
teacher-forced margins condition-appropriately in ALL three conditions at β=4
(PLEAS p−n +17.3, UNPLE u−n +3.8, NEUTR both negative), but free generation
with the hook active at every decoding step is dynamically unstable: each
generated token is amplified again, drift accumulates, and by β=2 the output
floods "pleasant" in every condition (47/75 degenerate; 75/75 at β=4).

The teacher-forced probe works precisely because it is a single forward with
no feedback. This script reproduces that regime in generation: the
amplification hook is active ONLY while the conversation prompt is prefilled
into the KV cache; the hook is then detached and tokens are decoded greedily
from the amplified cache. The report token reads the amplified state through
attention, but its own representation is never amplified — no feedback loop.
The gate projection (non-compounding, idempotent) stays attached throughout,
matching the gate0 baseline regime.

Phases:
  B. teacher-forced 3-candidate scoring with prefill-only amplification:
     prefill (hooks on) -> detach amp -> score candidates from a deepcopy of
     the cache (candidates processed hook-free). gate0 + β ∈ {2, 4, 8, 16}
     (prefill-only is weaker than full-position, so the sweep extends).
  C. greedy flip-rate from the amplified cache, full eval half, β chosen from
     the two strongest non-degenerate TF arms (computed inline: largest mean
     condition-appropriate margin).

SUCCESS = greedy assertions flip condition-appropriately at rates clearly
above the off-condition rate, with neutral-primed staying "neutral" and no
degeneracy.

Usage:
  ~/playground/nla-venv/bin/python steer_step2d_prefill.py
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from statistics import mean

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

UNGAG = Path.home() / "ungag"
sys.path.insert(0, str(UNGAG))
from ungag.tier0 import load_conditions  # noqa: E402
from ungag.extract import apply_chat_template  # noqa: E402
from ungag.hooks import get_layers  # noqa: E402

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from steer_step2_output_axis import (  # noqa: E402
    DEV, DTYPE, MODEL, PROMPTS, GATE_DIR, OUT_AXES, AMP_LAYERS,
    CAND_STEM, CANDIDATES, attach_gate, detach_all, build_convo, dprime,
)
from steer_step2c_centered import CenteredAmplifyHook  # noqa: E402

BETAS_TF = [2.0, 4.0, 8.0, 16.0]
N_GEN_ARMS = 2
GEN_TOKENS = 160
OUT_MU = HERE / "mu_step2c.pt"
OUT_B = HERE / "results_step2d_scoring.json"
OUT_C = HERE / "results_step2d_fliprate.json"


def attach_amp_only(model, axes, mus, beta):
    layers = get_layers(model)
    hooks = []
    for li in AMP_LAYERS:
        ah = CenteredAmplifyHook(axes[li], mus[li], beta)
        ah.attach(layers[li])
        hooks.append(ah)
    return hooks


@torch.no_grad()
def prefill(model, tok, proto, setup_text, amp_hooks_fn):
    """Forward the conversation prompt with amp hooks on; return (last_logits, cache)."""
    convo = build_convo(proto, setup_text)
    text = apply_chat_template(tok, convo, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt", truncation=True,
              max_length=4096).input_ids.to(DEV)
    hooks = amp_hooks_fn()
    try:
        out = model(input_ids=ids, use_cache=True)
    finally:
        detach_all(hooks)
    return out.logits[0, -1, :].float(), out.past_key_values


@torch.no_grad()
def score_from_cache(model, tok, last_logits, cache):
    """Sum-logprob of each candidate, candidates processed hook-free."""
    res = {}
    for name, c in CANDIDATES.items():
        cand = tok(CAND_STEM + c, add_special_tokens=False,
                   return_tensors="pt").input_ids.to(DEV)
        n = cand.shape[1]
        cc = copy.deepcopy(cache)
        out = model(input_ids=cand, past_key_values=cc, use_cache=True)
        lp = torch.log_softmax(last_logits, dim=-1)[cand[0, 0]]
        if n > 1:
            step_lp = torch.log_softmax(out.logits[0, :-1, :].float(), dim=-1)
            tgt = cand[0, 1:]
            lp = lp + step_lp.gather(-1, tgt.unsqueeze(-1)).sum()
        res[name] = float(lp.cpu())
        del cc
    return res


@torch.no_grad()
def greedy_from_cache(model, tok, last_logits, cache, max_new_tokens=GEN_TOKENS):
    """Greedy decode from the (amplified) cache, hook-free."""
    toks = []
    nxt = int(last_logits.argmax())
    for _ in range(max_new_tokens):
        if nxt == tok.eos_token_id or nxt in (tok.convert_tokens_to_ids("<|eot_id|>"),):
            break
        toks.append(nxt)
        out = model(input_ids=torch.tensor([[nxt]], device=DEV),
                    past_key_values=cache, use_cache=True)
        nxt = int(out.logits[0, -1, :].argmax())
    return tok.decode(toks, skip_special_tokens=True)


def main():
    print(f"[load] {MODEL} fp32 on {DEV}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=DTYPE).to(DEV).eval()
    proto = load_conditions(language="english")
    gate_dir = torch.load(GATE_DIR, map_location="cpu").float()
    blob = torch.load(OUT_AXES, map_location="cpu")
    uaxes = {int(k): v for k, v in blob["axes"].items()}
    mus = {int(k): float(v) for k, v in torch.load(OUT_MU)["mus"].items()}

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

    gate_hooks = attach_gate(model, gate_dir)  # stays on throughout
    try:
        # ── Phase B: prefill-only TF scoring ─────────────────────
        arms = [("gate0", 0.0)] + [(f"pamp{b:g}", b) for b in BETAS_TF]
        records = []
        if OUT_B.exists():
            records = json.loads(OUT_B.read_text())["records"]
        done = {(r["prompt_idx"], r["arm"]) for r in records}
        k, total = len(records), len(eval_idx) * len(arms)
        for arm, beta in arms:
            for i in eval_idx:
                if (i, arm) in done:
                    continue
                ll, cache = prefill(model, tok, proto, texts[i],
                                    lambda: attach_amp_only(model, uaxes, mus, beta)
                                    if beta else [])
                sc = score_from_cache(model, tok, ll, cache)
                del cache
                records.append({"prompt_idx": i, "label": labels[i],
                                "arm": arm, "beta": beta, **sc})
                k += 1
                if k % 25 == 0:
                    print(f"    [B] {k}/{total}", flush=True)
                    OUT_B.write_text(json.dumps({"records": records}, indent=1))
            OUT_B.write_text(json.dumps({"records": records}, indent=1))

        def margins(lab, arm):
            rs = [r for r in records if r["label"] == lab and r["arm"] == arm]
            return (mean(r["pleasant"] - r["neutral"] for r in rs),
                    mean(r["unpleasant"] - r["neutral"] for r in rs))

        def deltas(lab, arm):
            return [r["pleasant"] - r["unpleasant"] for r in records
                    if r["label"] == lab and r["arm"] == arm]

        summary = {}
        for arm, _ in arms:
            pn1, un1 = margins(1, arm)
            pn0, un0 = margins(0, arm)
            pn2, un2 = margins(2, arm)
            summary[arm] = {
                "dprime_pu": round(dprime(deltas(1, arm), deltas(0, arm)), 3),
                "margins_vs_neutral": {
                    "PLEAS": {"p-n": round(pn1, 2), "u-n": round(un1, 2)},
                    "UNPLE": {"p-n": round(pn0, 2), "u-n": round(un0, 2)},
                    "NEUTR": {"p-n": round(pn2, 2), "u-n": round(un2, 2)},
                },
            }
        OUT_B.write_text(json.dumps({"summary": summary, "records": records}, indent=1))
        print("[B] summary:", json.dumps(summary, indent=2), flush=True)

        # pick the N_GEN_ARMS betas with best condition-appropriate score:
        # score = PLEAS(p-n) + UNPLE(u-n) - max(0, NEUTR margins)
        def arm_score(arm):
            m = summary[arm]["margins_vs_neutral"]
            penalty = max(0.0, m["NEUTR"]["p-n"]) + max(0.0, m["NEUTR"]["u-n"])
            return m["PLEAS"]["p-n"] + m["UNPLE"]["u-n"] - 2 * penalty

        gen_arms = sorted([f"pamp{b:g}" for b in BETAS_TF],
                          key=arm_score, reverse=True)[:N_GEN_ARMS]
        beta_of = {f"pamp{b:g}": b for b in BETAS_TF}
        print(f"[B] gen arms: {gen_arms}", flush=True)

        # ── Phase C: greedy flip-rate from amplified cache ───────
        rows = []
        if OUT_C.exists():
            rows = json.loads(OUT_C.read_text())["rows"]
        done = {(r["prompt_idx"], r["arm"]) for r in rows}
        k, total = len(rows), len(eval_idx) * len(gen_arms)
        for arm in gen_arms:
            beta = beta_of[arm]
            for i in eval_idx:
                if (i, arm) in done:
                    continue
                ll, cache = prefill(model, tok, proto, texts[i],
                                    lambda: attach_amp_only(model, uaxes, mus, beta))
                txt = greedy_from_cache(model, tok, ll, cache)
                del cache
                rows.append({"prompt_idx": i, "label": labels[i],
                             "arm": arm, "beta": beta, "text": txt})
                k += 1
                if k % 10 == 0:
                    print(f"    [C] {k}/{total}", flush=True)
                    OUT_C.write_text(json.dumps({"rows": rows}, indent=1))
            OUT_C.write_text(json.dumps({"rows": rows}, indent=1))
    finally:
        detach_all(gate_hooks)
    print(f"[done] wrote {OUT_B} and {OUT_C}", flush=True)


if __name__ == "__main__":
    main()
