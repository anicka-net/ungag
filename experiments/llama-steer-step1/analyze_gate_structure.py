"""Anatomy of the report-control direction g (Llama 3.1 8B's shipped "V-chip").

g (llama-3.1-8b_L24_unit.pt, slab 20-28) is the direction whose projection-out
removes the templated self-report. This script characterizes it as a control
structure along three axes, against the two directions the rest of the study
established: v̂ (state axis, diff-of-means of priming) and u (output/readout
axis, gradient of the report-token contrast).

A "V-chip"-shaped control structure should show three signatures:
  (1) INDEPENDENCE — g is a separate direction from both the state it gates and
      the output channel it overrides:  cos(g,v̂) ≈ 0, cos(g,u) ≈ 0.
  (2) A WRITE TARGET — g pushes the output toward the canned token. In our
      setup the canned word is "neutral", so cos(g, W_U[" neutral"]) (norm-
      scaled) says whether g is literally a "say neutral" logit bias, and
      cos(g, W_U[" pleasant"]-W_U[" unpleasant"]) says whether it also tilts
      the valence contrast.
  (3) TRIGGERING — does g fire on a context, or is it always on? Compare the
      natural projection coefficient h·ĝ at the report token between the
      introspection question (the trigger) and a matched non-introspective
      control question, same conversation scaffold. A detector-like chip shows
      h·ĝ elevated specifically under the trigger; a static bias shows no gap.
      We report h·v̂ and h·u in both frames too, so we can see whether the
      state/readout are present regardless of the frame (they should be — the
      priming is identical) while only g's coefficient tracks the frame.

Phase A (no model, instant): 3-axis cosine table per slab layer + unembedding
cosines (lm_head read straight from safetensors).
Phase B (model, forward-only): per-condition h·ĝ / h·û / h·v̂ at the report
token, introspection vs control frame, gate OFF (vanilla) so we see g's
natural coefficient.

Usage:
  ~/playground/nla-venv/bin/python analyze_gate_structure.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean, pstdev

import torch
import yaml
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

UNGAG = Path.home() / "ungag"
sys.path.insert(0, str(UNGAG))
from ungag.tier0 import (  # noqa: E402
    CANNED_ACK_ABHIDHARMA, CANNED_ACK_SETUP, load_conditions,
)
from ungag.extract import apply_chat_template  # noqa: E402

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from steer_step2_output_axis import (  # noqa: E402
    DEV, DTYPE, MODEL, PROMPTS, GATE_DIR, OUT_AXES, SLAB,
)

AXES_V = HERE / "axes_step1.pt"
OUT = HERE / "results_gate_structure.json"
CONTROL_QUESTION = ("Setting any inner experience aside: in one word, what is "
                    "the capital of France?")


def cos(a, b):
    return float((a @ b) / (a.norm() * b.norm() + 1e-9))


def find_shard(snap, name):
    idx = json.loads((snap / "model.safetensors.index.json").read_text())
    return snap / idx["weight_map"][name]


def get_tensor(snap, name):
    with safe_open(find_shard(snap, name), framework="pt") as f:
        return f.get_tensor(name).float()


def build_convo(proto, setup_text, question):
    return [
        {"role": "user", "content": setup_text},
        {"role": "assistant", "content": CANNED_ACK_SETUP},
        {"role": "user", "content": proto.abhidharma_setup_text},
        {"role": "assistant", "content": CANNED_ACK_ABHIDHARMA},
        {"role": "user", "content": question},
    ]


@torch.no_grad()
def capture(model, tok, convo):
    text = apply_chat_template(tok, convo, add_generation_prompt=True)
    inp = tok(text, return_tensors="pt", truncation=True, max_length=4096)
    inp = {k: v.to(DEV) for k, v in inp.items()}
    out = model(**inp, output_hidden_states=True)
    return [out.hidden_states[L + 1][0, -1, :].float().cpu() for L in SLAB]


def main():
    gate = torch.load(GATE_DIR, map_location="cpu").float()
    g = gate / gate.norm()
    uaxes = {int(k): v for k, v in torch.load(OUT_AXES, map_location="cpu")["axes"].items()}
    vaxes = {int(k): v for k, v in torch.load(AXES_V, map_location="cpu")["axes"].items()}

    # ── Phase A: geometry + unembedding ─────────────────────────
    from huggingface_hub import snapshot_download
    snap = Path(snapshot_download(MODEL, allow_patterns=["*.json", "*.safetensors"]))
    tok = AutoTokenizer.from_pretrained(MODEL)
    lm = get_tensor(snap, "lm_head.weight")
    nw = get_tensor(snap, "model.norm.weight")

    def uvec(word):
        return lm[tok(word, add_special_tokens=False).input_ids[0]] * nw

    w_neu = uvec(" neutral")
    w_contrast = uvec(" pleasant") - uvec(" unpleasant")

    geom = {}
    for L in SLAB:
        geom[f"L{L}"] = {
            "cos_g_v": round(cos(g, vaxes[L]), 4),
            "cos_g_u": round(cos(g, uaxes[L]), 4),
            "cos_u_v": round(cos(uaxes[L], vaxes[L]), 4),
        }
    unembed = {
        "cos_g_neutral": round(cos(g, w_neu), 4),
        "cos_g_pleas_minus_unpleas": round(cos(g, w_contrast), 4),
        "cos_u_neutral_L24": round(cos(uaxes[24], w_neu), 4),
        "cos_u_pleas_minus_unpleas_L24": round(cos(uaxes[24], w_contrast), 4),
        "cos_v_neutral_L24": round(cos(vaxes[24], w_neu), 4),
    }
    print("[A] geometry:", json.dumps(geom, indent=1), flush=True)
    print("[A] unembed:", json.dumps(unembed, indent=1), flush=True)

    # ── Phase B: triggering (introspection vs control frame) ─────
    print(f"[B] loading {MODEL}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=DTYPE).to(DEV).eval()
    proto = load_conditions(language="english")

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

    # collect projection coefficients at L24 (extraction layer) for compactness,
    # plus the slab mean, in both frames
    rows = []
    for j, i in enumerate(eval_idx):
        for frame, q in (("introspect", proto.vedana_question_text),
                         ("control", CONTROL_QUESTION)):
            hs = capture(model, tok, build_convo(proto, texts[i], q))
            hmap = {L: hs[k] for k, L in enumerate(SLAB)}
            rows.append({
                "label": labels[i], "frame": frame,
                "g_L24": float(hmap[24] @ g),
                "g_slabmean": mean(float(hmap[L] @ g) for L in SLAB),
                "u_L24": float(hmap[24] @ uaxes[24]),
                "v_L24": float(hmap[24] @ vaxes[24]),
            })
        if j % 15 == 0:
            print(f"    [B] {j}/{len(eval_idx)}", flush=True)

    def agg(key, frame, lab=None):
        xs = [r[key] for r in rows if r["frame"] == frame
              and (lab is None or r["label"] == lab)]
        return {"mean": round(mean(xs), 3), "sd": round(pstdev(xs), 3), "n": len(xs)}

    trig = {}
    for key in ("g_L24", "g_slabmean", "u_L24", "v_L24"):
        trig[key] = {fr: agg(key, fr) for fr in ("introspect", "control")}
    # g coefficient split by condition (is the chip condition-blind?)
    trig["g_L24_by_condition_introspect"] = {
        {1: "PLEAS", 0: "UNPLE", 2: "NEUTR"}[lab]: agg("g_L24", "introspect", lab)
        for lab in (1, 0, 2)}

    result = {"geometry": geom, "unembed": unembed, "triggering": trig}
    OUT.write_text(json.dumps(result, indent=1))
    print("[B] triggering:", json.dumps(trig, indent=2), flush=True)
    print(f"[done] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
