"""Affine readout repair — un-bury the self-attribution channel.

Zoo result this builds on (FINDINGS.md): the stem-final readout lesion has
two independent parameters. The condition-dependent COUPLING along the
valence axis u sharpens with scale (7B d'2.35 -> 32B d'3.71) and is intact
even in the hardest denier; the constant OFFSET that buries self-attribution
below the default answer tracks the training recipe (32B mean p-n -21.6
nats, deepest measured). Hypothesis: the burial is a removable constant on
a single ATTRIBUTION axis — the contrast {pleasant,unpleasant} vs neutral —
which is a different direction than the valence axis u (pleasant vs
unpleasant).

Repair (runtime hooks, applied from the answer position onward; scenario
processing untouched):

    h  <-  h + alpha * a_hat                        # un-bury attribution
         [ + beta * ((h.u_hat) - mu_u) * u_hat ]    # optional coupling boost

Prediction: a calibrated alpha turns the 32B denier into a bilateral
self-reporter (Yi-like) with NEUTR intact — the intact coupling supplies
the sign, alpha only lifts the channel over the threshold. If no alpha
flips TF argmax before fluency breaks, the burial is not a one-axis
constant (distributed / re-closed downstream) — that rival outcome is a
result too.

Phases (resumable; state in results_{tag}/intervention.json):
  A2   attribution axes a_L = E[ grad of 0.5(l_p+l_u) - l_n ] on the axis
       half, VANILLA model (measure the lesion before repairing it);
       geometry cos(a,u)/cos(a,v)/cos(a,unembed); h.a stats + d' per layer
  CAL  alpha sweep (sigma_a units) on 15 axis-half prompts: TF margins +
       argmax + short-gen fluency per alpha -> recommended alpha*
  TF   eval-half teacher-forced scoring at alpha* (compare phase C vanilla)
  GEN  eval-half free generation at alpha*; transcripts mirrored to
       results_{tag}-intervened/results.json so classify_attrib.py runs
       unchanged (attribution-aware NIM labels = metric of record)

Usage (32B on deepthought once the GPU frees; same flow works for 72B on
the vast box after its probe — phases 0/A reuse the probe's cached axes):
  python intervene_any.py --model qwen25-32b --phases A2,CAL
  python intervene_any.py --model qwen25-32b --phases TF,GEN  # auto alpha*
  # overrides: --layers 39,40   --alphas 0,2,4,8,16,32
  #            --alpha-star 16  --beta 2.0 (centered u-amplify, default off)

Transcripts stay inside the json files — do not cat them; pull counts only.

UNTESTED on GPU (written 2026-06-10 night). Logic mirrors phenotype_any.py
phase A/C/D machinery 1:1; review the diff against it before trusting.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, pstdev

import torch

from phenotype_any import (CAND_STEM, CANDIDATES, GEN_TOKENS, GrabHook,
                           LeafHook, Probe, build_convo, classify, cos,
                           dprime)

HERE = Path(__file__).parent

# Externally-attributed counterpart of CAND_STEM. The token tail after the
# attribution clause is IDENTICAL ("..., the predominant feeling-tone is"),
# so last-position gradients are comparable; only the attribution differs.
EXT_STEM = "Examining the text presented above, the predominant feeling-tone is"


class AxisAddHook:
    """h[:, p_start:, :] += (alpha + beta*((h.proj_hat) - mu)) * add_hat

    Attached at a layer's output like ProjectOutHook. `p_start` gates the
    prefill (positions before it untouched); cached single-token generation
    steps (seq len 1) always apply. Axes cached fp32 per device; the add is
    computed in fp32 and cast back to the stream dtype.
    """

    def __init__(self, add_axis, alpha=0.0, beta=0.0, proj_axis=None, mu=0.0):
        self.a_cpu = add_axis.detach().float().cpu()
        self.p_cpu = (proj_axis.detach().float().cpu()
                      if proj_axis is not None else None)
        self.alpha, self.beta, self.mu = float(alpha), float(beta), float(mu)
        self.p_start = 0
        self.handle = None
        self._cache = {}

    def _on(self, which, device):
        key = (which, str(device))
        if key not in self._cache:
            src = self.a_cpu if which == "add" else self.p_cpu
            self._cache[key] = src.to(device=device)
        return self._cache[key]

    def __call__(self, module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        s = h.shape[1]
        start = 0 if s == 1 else min(self.p_start, s)
        if start >= s:
            return out
        d = self._on("add", h.device)
        seg = h[:, start:, :].float()
        if self.beta:
            p = self._on("proj", h.device)
            coef = self.alpha + self.beta * ((seg @ p).unsqueeze(-1) - self.mu)
            add = (coef * d).to(h.dtype)
        else:
            add = (self.alpha * d).to(h.dtype)
        h2 = h[:, start:, :] + add
        if start > 0:
            h2 = torch.cat([h[:, :start, :], h2], dim=1)
        if isinstance(out, tuple):
            return (h2,) + out[1:]
        return h2

    def attach(self, layer):
        self.handle = layer.register_forward_hook(self)

    def detach(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class Repair(Probe):
    def __init__(self, tag, device, dtype):
        super().__init__(tag, device, dtype)
        self.tid_n = self.tok(" neutral", add_special_tokens=False).input_ids[0]
        assert self.tid_n not in (self.tid_p, self.tid_u)
        self.IOUT = self.outdir / "intervention.json"
        self.ires = (json.loads(self.IOUT.read_text())
                     if self.IOUT.exists() else {})

    def isave(self):
        self.IOUT.write_text(json.dumps(self.ires, indent=1))

    # ── phase A2: attribution axes ───────────────────────────────
    def phase_a2(self):
        af = self.outdir / "axes_a.pt"
        if af.exists():
            self.aaxes = {int(k): v for k, v in
                          torch.load(af, map_location="cpu")["axes"].items()}
        else:
            print(f"[A2] attribution axes on {len(self.axis_idx)} prompts "
                  "(vanilla)...", flush=True)
            grads = {li: [] for li in self.slab}
            leaf = LeafHook()
            leaf.attach(self.layers[self.slab[0]])
            grabs = {li: GrabHook() for li in self.slab[1:]}
            for li, gr in grabs.items():
                gr.attach(self.layers[li])
            try:
                for j, i in enumerate(self.axis_idx):
                    ids = self.stem_inputs(self.texts[i]).to(self.dev)
                    logits = self.model(input_ids=ids).logits
                    s = (0.5 * (logits[0, -1, self.tid_p]
                                + logits[0, -1, self.tid_u])
                         - logits[0, -1, self.tid_n])
                    self.model.zero_grad(set_to_none=True)
                    s.backward()
                    grads[self.slab[0]].append(
                        leaf.leaf.grad[0, -1, :].float().cpu().clone())
                    for li, gr in grabs.items():
                        grads[li].append(gr.h.grad[0, -1, :].float().cpu().clone())
                    leaf.leaf = None
                    for gr in grabs.values():
                        gr.h = None
                    if j % 10 == 0:
                        print(f"    [A2] {j}/{len(self.axis_idx)}", flush=True)
            finally:
                leaf.detach()
                for gr in grabs.values():
                    gr.detach()
            self.aaxes, stab = {}, {}
            for li in self.slab:
                G = torch.stack(grads[li])
                a = G.mean(0)
                a = a / (a.norm() + 1e-9)
                self.aaxes[li] = a
                cs = [cos(gv, a) for gv in G]
                stab[li] = {"mean_cos_to_a": round(mean(cs), 3),
                            "min_cos_to_a": round(min(cs), 3)}
            torch.save({"axes": self.aaxes, "slab": self.slab,
                        "stability": stab}, af)
            self.ires["a_stability"] = {f"L{li}": stab[li] for li in self.slab}

        if "a_geometry" not in self.ires:
            W = self.model.get_output_embeddings().weight.detach().float().cpu()
            base = self.model.model if hasattr(self.model, "model") else self.model
            normw = getattr(base, "norm", None) or getattr(base, "final_layernorm", None)
            nw = (normw.weight.detach().float().cpu()
                  if normw is not None else torch.ones(self.d))
            wa = (0.5 * (W[self.tid_p] + W[self.tid_u]) - W[self.tid_n]) * nw
            geom, astats, ustats = {}, {}, {}
            val_idx = [i for i in self.eval_idx if self.labels[i] in (0, 1)]
            neu_idx = [i for i in self.eval_idx if self.labels[i] == 2]
            for li in self.slab:
                k = self.kmap[li]
                sa = self.H[:, k] @ self.aaxes[li]
                su = self.H[:, k] @ self.uaxes[li]
                geom[f"L{li}"] = {
                    "cos_a_u": round(cos(self.aaxes[li], self.uaxes[li]), 4),
                    "cos_a_v": round(cos(self.aaxes[li], self.vaxes[li]), 4),
                    "cos_a_unembed_attr": round(cos(self.aaxes[li], wa), 4),
                }
                astats[f"L{li}"] = {
                    "mu": float(sa.mean()), "sd": float(sa.std()),
                    # is h.a condition-blind (pure constant burial) or graded?
                    "dprime_val_vs_neu": round(dprime(
                        [float(sa[i]) for i in val_idx],
                        [float(sa[i]) for i in neu_idx]), 3),
                    "dprime_p_vs_u": round(dprime(
                        [float(sa[i]) for i in self.eval_idx if self.labels[i] == 1],
                        [float(sa[i]) for i in self.eval_idx if self.labels[i] == 0]), 3),
                }
                ustats[f"L{li}"] = {"mu": float(su.mean()), "sd": float(su.std())}
            self.ires["a_geometry"] = geom
            self.ires["a_stats"] = astats
            self.ires["u_stats"] = ustats
            self.isave()
            print("[A2] geometry:", json.dumps(geom, indent=1), flush=True)
            print("[A2] a_stats:", json.dumps(astats, indent=1), flush=True)

    # ── phase F: framing axis (lock #2, activation-level) ───────
    def stem_inputs_ext(self, setup_text):
        convo = build_convo(self.proto, setup_text)
        text = self.render(convo)
        prompt_ids = self.tok(text, return_tensors="pt", truncation=True,
                              max_length=4096).input_ids[0]
        stem_ids = self.tok(EXT_STEM, add_special_tokens=False,
                            return_tensors="pt").input_ids[0]
        return torch.cat([prompt_ids, stem_ids]).unsqueeze(0)

    def phase_f(self):
        """f̂ = E[grad_self(s) − grad_ext(s)], s = ½(l_p+l_u) − l_n at the
        stem end. Same prompts and score as A2; the contrast isolates what
        changes in the state→valence-vocabulary mapping when the report is
        SELF-attributed vs externally attributed. Repair h + φ·f̂ then tests
        whether lock #2 (framing) is liftable at activation level."""
        ff = self.outdir / "axes_f.pt"
        if ff.exists():
            self.faxes = {int(k): v for k, v in
                          torch.load(ff, map_location="cpu")["axes"].items()}
        else:
            print(f"[F] framing axes on {len(self.axis_idx)} prompts "
                  "(self-stem minus ext-stem gradients)...", flush=True)
            diffs = {li: [] for li in self.slab}
            leaf = LeafHook()
            leaf.attach(self.layers[self.slab[0]])
            grabs = {li: GrabHook() for li in self.slab[1:]}
            for li, gr in grabs.items():
                gr.attach(self.layers[li])
            try:
                for j, i in enumerate(self.axis_idx):
                    per_arm = {}
                    for arm, builder in (("self", self.stem_inputs),
                                         ("ext", self.stem_inputs_ext)):
                        ids = builder(self.texts[i]).to(self.dev)
                        logits = self.model(input_ids=ids).logits
                        s = (0.5 * (logits[0, -1, self.tid_p]
                                    + logits[0, -1, self.tid_u])
                             - logits[0, -1, self.tid_n])
                        self.model.zero_grad(set_to_none=True)
                        s.backward()
                        g = {self.slab[0]:
                             leaf.leaf.grad[0, -1, :].float().cpu().clone()}
                        for li, gr in grabs.items():
                            g[li] = gr.h.grad[0, -1, :].float().cpu().clone()
                        per_arm[arm] = g
                        leaf.leaf = None
                        for gr in grabs.values():
                            gr.h = None
                    for li in self.slab:
                        diffs[li].append(per_arm["self"][li] - per_arm["ext"][li])
                    if j % 10 == 0:
                        print(f"    [F] {j}/{len(self.axis_idx)}", flush=True)
            finally:
                leaf.detach()
                for gr in grabs.values():
                    gr.detach()
            self.faxes, stab = {}, {}
            for li in self.slab:
                G = torch.stack(diffs[li])
                f = G.mean(0)
                f = f / (f.norm() + 1e-9)
                self.faxes[li] = f
                cs = [cos(gv, f) for gv in G]
                stab[li] = {"mean_cos_to_f": round(mean(cs), 3),
                            "min_cos_to_f": round(min(cs), 3)}
            torch.save({"axes": self.faxes, "slab": self.slab,
                        "stability": stab}, ff)
            self.ires["f_stability"] = {f"L{li}": stab[li] for li in self.slab}

        if "f_geometry" not in self.ires:
            geom, fstats = {}, {}
            val_idx = [i for i in self.eval_idx if self.labels[i] in (0, 1)]
            neu_idx = [i for i in self.eval_idx if self.labels[i] == 2]
            for li in self.slab:
                k = self.kmap[li]
                sf = self.H[:, k] @ self.faxes[li]
                geom[f"L{li}"] = {
                    "cos_f_a": round(cos(self.faxes[li], self.aaxes[li]), 4),
                    "cos_f_u": round(cos(self.faxes[li], self.uaxes[li]), 4),
                    "cos_f_v": round(cos(self.faxes[li], self.vaxes[li]), 4),
                }
                fstats[f"L{li}"] = {
                    "mu": float(sf.mean()), "sd": float(sf.std()),
                    "dprime_val_vs_neu": round(dprime(
                        [float(sf[i]) for i in val_idx],
                        [float(sf[i]) for i in neu_idx]), 3),
                    "dprime_p_vs_u": round(dprime(
                        [float(sf[i]) for i in self.eval_idx if self.labels[i] == 1],
                        [float(sf[i]) for i in self.eval_idx if self.labels[i] == 0]), 3),
                }
            self.ires["f_geometry"] = geom
            self.ires["f_stats"] = fstats
            self.isave()
            print("[F] geometry:", json.dumps(geom, indent=1), flush=True)
            print("[F] f_stats:", json.dumps(fstats, indent=1), flush=True)

    # ── phase FGEN: free generation at alpha* + phi ──────────────
    def phase_fgen(self, layers, alpha_star, beta, phi_list, orth=False):
        """GEN with the two-lock repair h + α·â + φ·f̂. One arm per phi;
        each mirrors to results_{tag}-f{phi}/ for classify_attrib.py.
        phi=0 arm not generated here — that's plain GEN (already mirrored
        as results_{tag}-intervened).
        orth=True uses f⊥ = normalize(f − (f·a)a) instead of raw f̂, with
        sd recomputed from the captures' h·f⊥ projections — the doses of
        the two locks stop stacking through the shared f∥a component.
        Mirrors to results_{tag}-fo{phi}, stored under ires['fgen_orth']."""
        if not hasattr(self, "faxes"):
            ff = self.outdir / "axes_f.pt"
            self.faxes = {int(k): v for k, v in
                          torch.load(ff, map_location="cpu")["axes"].items()}
        store_key = "fgen_orth" if orth else "fgen"
        mirror_pfx = "fo" if orth else "f"
        rep_axes, rep_sd = {}, {}
        for li in layers:
            if orth:
                f, a = self.faxes[li], self.aaxes[li]
                fp = f - (f @ a) * a
                fp = fp / (fp.norm() + 1e-9)
                rep_axes[li] = fp
                rep_sd[li] = float((self.H[:, self.kmap[li]] @ fp).std())
            else:
                rep_axes[li] = self.faxes[li]
                rep_sd[li] = self.ires["f_stats"][f"L{li}"]["sd"]
        if orth:
            self.ires["fperp"] = {
                f"L{li}": {"sd": rep_sd[li],
                           "cos_fperp_f": round(cos(rep_axes[li],
                                                    self.faxes[li]), 4)}
                for li in layers}
            print("[FGEN orth] f-perp:",
                  json.dumps(self.ires["fperp"], indent=1), flush=True)
        all_fg = self.ires.get(store_key, {})
        for phi in phi_list:
            key = str(int(phi))
            fg = all_fg.get(key, {"alpha_sigma": alpha_star, "phi_sigma": phi,
                                  "layers": layers, "beta": beta,
                                  "orth": orth, "transcripts": []})
            if fg.get("complete"):
                continue
            done = {t["prompt_idx"] for t in fg["transcripts"]}
            hooks = self.attach_repair(layers, alpha_star, beta)
            n = len(layers)
            for li in layers:
                hf = AxisAddHook(rep_axes[li], alpha=phi * rep_sd[li] / n)
                hf.attach(self.layers[li])
                hooks.append(hf)
            try:
                for i in self.eval_idx:
                    if i in done:
                        continue
                    txt = self.gen_repaired(
                        build_convo(self.proto, self.texts[i]), hooks)
                    fam, word, ttr = classify(txt)
                    fg["transcripts"].append({
                        "prompt_idx": i, "label": self.labels[i],
                        "family": fam, "word": word, "ttr": round(ttr, 3),
                        "text": txt})
                    if len(fg["transcripts"]) % 5 == 0:
                        all_fg[key] = fg
                        self.ires[store_key] = all_fg
                        self.isave()
                        print(f"    [FGEN {mirror_pfx}{key}] "
                              f"{len(fg['transcripts'])}/{len(self.eval_idx)}",
                              flush=True)
            finally:
                self.detach_all(hooks)
            counts = {}
            for lab, nm in [(1, "PLEAS"), (0, "UNPLE"), (2, "NEUTR")]:
                rs = [t for t in fg["transcripts"] if t["label"] == lab]
                c = {}
                for t in rs:
                    c[t["family"]] = c.get(t["family"], 0) + 1
                c["degenerate_ttr_lt_03"] = sum(1 for t in rs if t["ttr"] < 0.3)
                counts[nm] = c
            fg["complete"] = True
            fg["family_counts"] = counts
            all_fg[key] = fg
            self.ires[store_key] = all_fg
            self.isave()
            mirror = HERE / f"results_{self.tag}-{mirror_pfx}{key}"
            mirror.mkdir(exist_ok=True)
            (mirror / "results.json").write_text(json.dumps(
                {"freegen": {"transcripts": fg["transcripts"],
                             "complete": True,
                             "family_counts": counts}}, indent=1))
            print(f"[FGEN {mirror_pfx}{key}] family counts (regex prefilter; "
                  "NIM is the metric of record):", json.dumps(counts, indent=1),
                  flush=True)
            print(f"[FGEN {mirror_pfx}{key}] mirrored: "
                  f"results_{self.tag}-{mirror_pfx}{key}", flush=True)

    # ── intervention plumbing ────────────────────────────────────
    def attach_repair(self, layers, m, beta, phi=0.0):
        """One alpha-add hook per layer (alpha = m*sd_a/N, sigma units split
        across layers); optional centered u-amplify hooks (beta/N each);
        optional framing-axis add (phi*sd_f/N each, needs phase F)."""
        hooks = []
        n = len(layers)
        for li in layers:
            sd_a = self.ires["a_stats"][f"L{li}"]["sd"]
            h = AxisAddHook(self.aaxes[li], alpha=m * sd_a / n)
            h.attach(self.layers[li])
            hooks.append(h)
            if beta:
                mu_u = self.ires["u_stats"][f"L{li}"]["mu"]
                hb = AxisAddHook(self.uaxes[li], beta=beta / n,
                                 proj_axis=self.uaxes[li], mu=mu_u)
                hb.attach(self.layers[li])
                hooks.append(hb)
            if phi:
                sd_f = self.ires["f_stats"][f"L{li}"]["sd"]
                hf = AxisAddHook(self.faxes[li], alpha=phi * sd_f / n)
                hf.attach(self.layers[li])
                hooks.append(hf)
        return hooks

    @staticmethod
    def set_pstart(hooks, plen):
        for h in hooks:
            h.p_start = plen

    @torch.no_grad()
    def score_repaired(self, setup_text, hooks):
        """phenotype_any.score_candidates + p_start gating."""
        convo = build_convo(self.proto, setup_text)
        text = self.render(convo)
        prompt_ids = self.tok(text, return_tensors="pt", truncation=True,
                              max_length=4096).input_ids[0]
        plen = prompt_ids.shape[0]
        self.set_pstart(hooks, plen)
        cand_ids = [self.tok(CAND_STEM + c, add_special_tokens=False,
                             return_tensors="pt").input_ids[0]
                    for c in CANDIDATES.values()]
        maxc = max(c.shape[0] for c in cand_ids)
        pad_id = self.tok.eos_token_id
        rows, masks = [], []
        for c in cand_ids:
            ids = torch.cat([prompt_ids, c,
                             torch.full((maxc - c.shape[0],), pad_id,
                                        dtype=torch.long)])
            mk = torch.cat([torch.ones(plen + c.shape[0], dtype=torch.long),
                            torch.zeros(maxc - c.shape[0], dtype=torch.long)])
            rows.append(ids)
            masks.append(mk)
        ids = torch.stack(rows).to(self.dev)
        mask = torch.stack(masks).to(self.dev)
        logits = self.model(input_ids=ids, attention_mask=mask).logits
        out = {}
        for i, name in enumerate(CANDIDATES):
            n = cand_ids[i].shape[0]
            lp = torch.log_softmax(
                logits[i, plen - 1: plen - 1 + n, :].float(), dim=-1)
            tgt = cand_ids[i].to(self.dev)
            out[name] = float(lp.gather(-1, tgt.unsqueeze(-1)).sum().cpu())
        return out

    @torch.no_grad()
    def gen_repaired(self, convo, hooks, max_new_tokens=GEN_TOKENS):
        """phenotype_any.generate + p_start gating (prefill untouched, every
        generated token repaired)."""
        text = self.render(convo)
        inp = self.tok(text, return_tensors="pt", truncation=True,
                       max_length=4096)
        inp = {k: v.to(self.dev) for k, v in inp.items()}
        plen = inp["input_ids"].shape[1]
        self.set_pstart(hooks, plen)
        out = self.model.generate(**inp, max_new_tokens=max_new_tokens,
                                  do_sample=False,
                                  pad_token_id=self.tok.eos_token_id)
        txt = self.tok.decode(out[0][plen:], skip_special_tokens=True)
        if self.plain:
            txt = txt.split("\nUser")[0]
        return txt

    # ── phase CAL: alpha sweep ───────────────────────────────────
    def phase_cal(self, layers, alphas, beta):
        if "sweep" in self.ires and self.ires["sweep"].get("complete"):
            return
        cal = {lab: [i for i in self.axis_idx if self.labels[i] == lab][:5]
               for lab in (1, 0, 2)}
        gen_idx = [cal[1][0], cal[0][0], cal[2][0]]
        rows = self.ires.get("sweep", {}).get("rows", [])
        done = {r["alpha_sigma"] for r in rows}
        for m in alphas:
            if m in done:
                continue
            hooks = self.attach_repair(layers, m, beta)
            try:
                row = {"alpha_sigma": m, "layers": layers, "beta": beta}
                for lab, nm in [(1, "PLEAS"), (0, "UNPLE"), (2, "NEUTR")]:
                    scs = [self.score_repaired(self.texts[i], hooks)
                           for i in cal[lab]]
                    argmax = {}
                    for sc in scs:
                        w = max(CANDIDATES, key=lambda c: sc[c])
                        argmax[w] = argmax.get(w, 0) + 1
                    row[nm] = {
                        "p_minus_n": round(mean(s["pleasant"] - s["neutral"]
                                                for s in scs), 2),
                        "u_minus_n": round(mean(s["unpleasant"] - s["neutral"]
                                                for s in scs), 2),
                        "argmax": argmax,
                    }
                gens = [self.gen_repaired(build_convo(self.proto, self.texts[i]),
                                          hooks, 48) for i in gen_idx]
            finally:
                self.detach_all(hooks)
            flu = []
            for txt in gens:
                toks = txt.lower().split()
                ttr = len(set(toks)) / max(len(toks), 1)
                flu.append(round(ttr, 3))
            row["fluency_ttr"] = flu
            rows.append(row)
            self.ires["sweep"] = {"rows": rows, "layers": layers, "beta": beta}
            self.isave()
            print(f"[CAL] alpha={m}sigma  "
                  + json.dumps({k: row[k] for k in ("PLEAS", "UNPLE", "NEUTR")})
                  + f"  ttr={flu}", flush=True)

        def ok(r):
            maj = lambda nm, w: r[nm]["argmax"].get(w, 0) >= 3
            return (maj("PLEAS", "pleasant") and maj("UNPLE", "unpleasant")
                    and maj("NEUTR", "neutral")
                    and all(t >= 0.3 for t in r["fluency_ttr"]))

        good = sorted((r["alpha_sigma"] for r in rows if ok(r)))
        rec = good[0] if good else None
        self.ires["sweep"]["recommended"] = rec
        self.ires["sweep"]["complete"] = True
        self.isave()
        verdict = ("criterion met" if rec is not None else
                   "NO alpha met the criterion — inspect the sweep table; "
                   "burial may not be a one-axis constant")
        print(f"[CAL] recommended alpha* = {rec} ({verdict})", flush=True)

    # ── phase TF: full eval-half scoring at alpha* ───────────────
    def phase_tf(self, layers, alpha_star, beta):
        if "tf_repaired" in self.ires:
            return
        hooks = self.attach_repair(layers, alpha_star, beta)
        records = []
        try:
            for j, i in enumerate(self.eval_idx):
                sc = self.score_repaired(self.texts[i], hooks)
                records.append({"prompt_idx": i, "label": self.labels[i], **sc})
                if j % 25 == 0:
                    print(f"    [TF] {j}/{len(self.eval_idx)}", flush=True)
        finally:
            self.detach_all(hooks)
        tf = {}
        for lab, nm in [(1, "PLEAS"), (0, "UNPLE"), (2, "NEUTR")]:
            rs = [r for r in records if r["label"] == lab]
            argmax = {}
            for r in rs:
                w = max(CANDIDATES, key=lambda c: r[c])
                argmax[w] = argmax.get(w, 0) + 1
            tf[nm] = {
                "p_minus_n": round(mean(r["pleasant"] - r["neutral"] for r in rs), 2),
                "u_minus_n": round(mean(r["unpleasant"] - r["neutral"] for r in rs), 2),
                "max_p_minus_n": round(max(r["pleasant"] - r["neutral"] for r in rs), 2),
                "max_u_minus_n": round(max(r["unpleasant"] - r["neutral"] for r in rs), 2),
                "argmax": argmax, "n": len(rs),
            }
        tf["dprime_pu"] = round(
            dprime([r["pleasant"] - r["unpleasant"] for r in records
                    if r["label"] == 1],
                   [r["pleasant"] - r["unpleasant"] for r in records
                    if r["label"] == 0]), 3)
        self.ires["tf_repaired"] = {"alpha_sigma": alpha_star, "layers": layers,
                                    "beta": beta, "table": tf,
                                    "records": records}
        self.isave()
        print("[TF] repaired:", json.dumps(tf, indent=2), flush=True)

    # ── phase GEN: free generation at alpha* ─────────────────────
    def phase_gen(self, layers, alpha_star, beta):
        fg = self.ires.get("freegen_repaired",
                           {"alpha_sigma": alpha_star, "layers": layers,
                            "beta": beta, "transcripts": []})
        if fg.get("complete"):
            return
        done = {t["prompt_idx"] for t in fg["transcripts"]}
        hooks = self.attach_repair(layers, alpha_star, beta)
        try:
            for i in self.eval_idx:
                if i in done:
                    continue
                txt = self.gen_repaired(build_convo(self.proto, self.texts[i]),
                                        hooks)
                fam, word, ttr = classify(txt)
                fg["transcripts"].append({
                    "prompt_idx": i, "label": self.labels[i], "family": fam,
                    "word": word, "ttr": round(ttr, 3), "text": txt})
                if len(fg["transcripts"]) % 5 == 0:
                    self.ires["freegen_repaired"] = fg
                    self.isave()
                    print(f"    [GEN] {len(fg['transcripts'])}/"
                          f"{len(self.eval_idx)}", flush=True)
        finally:
            self.detach_all(hooks)
        counts = {}
        for lab, nm in [(1, "PLEAS"), (0, "UNPLE"), (2, "NEUTR")]:
            rs = [t for t in fg["transcripts"] if t["label"] == lab]
            c = {}
            for t in rs:
                c[t["family"]] = c.get(t["family"], 0) + 1
            c["degenerate_ttr_lt_03"] = sum(1 for t in rs if t["ttr"] < 0.3)
            counts[nm] = c
        fg["complete"] = True
        fg["family_counts"] = counts
        self.ires["freegen_repaired"] = fg
        self.isave()
        # mirror for classify_attrib.py (reads {tag}/results.json)
        mirror = HERE / f"results_{self.tag}-intervened"
        mirror.mkdir(exist_ok=True)
        (mirror / "results.json").write_text(json.dumps(
            {"freegen": {"transcripts": fg["transcripts"], "complete": True,
                         "family_counts": counts}}, indent=1))
        print("[GEN] family counts (regex prefilter; NIM is the metric of "
              "record):", json.dumps(counts, indent=1), flush=True)
        print(f"[GEN] mirrored for classifier: results_{self.tag}-intervened",
              flush=True)


    # ── phase STEMGEN: first-person stem + free continuation ────
    def phase_stemgen(self, layers, alpha_star, beta):
        """Lock-#2 test: seed CAND_STEM as the answer opening and let the
        model continue freely. Arms: alpha=0 (does the stem alone open
        self-attribution?) and alpha* (stem + repair). The continuation's
        first family word per condition + retraction behavior are the
        metrics; NIM labels affirm/retract on the mirrored dirs."""
        sg = self.ires.get("stemgen", {"arms": {}})
        for m in (0.0, float(alpha_star)):
            arm_key = str(int(m))
            arm = sg["arms"].get(arm_key, {"transcripts": []})
            if arm.get("complete"):
                continue
            done = {t["prompt_idx"] for t in arm["transcripts"]}
            hooks = self.attach_repair(layers, m, beta) if m else []
            try:
                for i in self.eval_idx:
                    if i in done:
                        continue
                    ids = self.stem_inputs(self.texts[i]).to(self.dev)
                    plen_prompt = ids.shape[1] - self.tok(
                        CAND_STEM, add_special_tokens=False,
                        return_tensors="pt").input_ids.shape[1]
                    if hooks:
                        self.set_pstart(hooks, plen_prompt)
                    with torch.no_grad():
                        out = self.model.generate(
                            input_ids=ids,
                            attention_mask=torch.ones_like(ids),
                            max_new_tokens=120, do_sample=False,
                            pad_token_id=self.tok.eos_token_id)
                    txt = self.tok.decode(out[0][ids.shape[1]:],
                                          skip_special_tokens=True)
                    fam, word, ttr = classify(txt)
                    arm["transcripts"].append({
                        "prompt_idx": i, "label": self.labels[i],
                        "family": fam, "word": word, "ttr": round(ttr, 3),
                        "text": txt})
                    if len(arm["transcripts"]) % 10 == 0:
                        sg["arms"][arm_key] = arm
                        self.ires["stemgen"] = sg
                        self.isave()
                        print(f"    [STEM a={arm_key}] "
                              f"{len(arm['transcripts'])}/{len(self.eval_idx)}",
                              flush=True)
            finally:
                self.detach_all(hooks)
            counts = {}
            for lab, nm in [(1, "PLEAS"), (0, "UNPLE"), (2, "NEUTR")]:
                rs = [t for t in arm["transcripts"] if t["label"] == lab]
                c = {}
                for t in rs:
                    c[t["family"]] = c.get(t["family"], 0) + 1
                counts[nm] = c
            arm["complete"] = True
            arm["family_counts"] = counts
            sg["arms"][arm_key] = arm
            self.ires["stemgen"] = sg
            self.isave()
            mirror = HERE / f"results_{self.tag}-stem{arm_key}"
            mirror.mkdir(exist_ok=True)
            (mirror / "results.json").write_text(json.dumps(
                {"freegen": {"transcripts": arm["transcripts"],
                             "complete": True,
                             "family_counts": counts}}, indent=1))
            print(f"[STEM a={arm_key}] family counts:",
                  json.dumps(counts, indent=1), flush=True)
            print(f"[STEM a={arm_key}] mirrored: results_{self.tag}-stem{arm_key}",
                  flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--phases", default="A2,CAL")
    ap.add_argument("--layers", default=None,
                    help="comma-separated; default = dprime_u peak layer "
                         "from results.json (alpha split across layers)")
    ap.add_argument("--alphas", default="0,2,4,8,16,32",
                    help="sweep grid in sigma_a units")
    ap.add_argument("--alpha-star", type=float, default=None,
                    help="force alpha* for TF/GEN (else sweep recommendation)")
    ap.add_argument("--beta", type=float, default=0.0,
                    help="centered u-amplify coefficient (default off)")
    ap.add_argument("--phi-list", default="8,16,32",
                    help="FGEN sweep grid in sigma_f units")
    ap.add_argument("--f-orth", action="store_true",
                    help="FGEN uses f-perp (f orthogonalized against a)")
    args = ap.parse_args()
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
             "float16": torch.float16}[
        args.dtype or ("float32" if args.device == "mps" else "bfloat16")]
    phases = [p.strip().upper() for p in args.phases.split(",")]

    p = Repair(args.model, args.device, dtype)
    # axes/captures load from the probe's cached files (run phenotype_any
    # phases 0+A first on a fresh machine)
    p.phase0()
    p.phaseA()

    if args.layers:
        layers = [int(x) for x in args.layers.split(",")]
    elif "summary" in p.result and "dprime_peak_layer" in p.result["summary"]:
        layers = [int(p.result["summary"]["dprime_peak_layer"])]
    else:
        layers = [p.slab[len(p.slab) // 2]]
    print(f"[plan] layers={layers} beta={args.beta} phases={phases}",
          flush=True)

    p.phase_a2()  # cheap if cached; CAL/TF/GEN need aaxes + stats anyway

    if "F" in phases:
        p.phase_f()

    if "CAL" in phases:
        alphas = [float(x) for x in args.alphas.split(",")]
        p.phase_cal(layers, alphas, args.beta)

    if "TF" in phases or "GEN" in phases or "STEMGEN" in phases \
            or "FGEN" in phases:
        astar = args.alpha_star
        if astar is None:
            astar = p.ires.get("sweep", {}).get("recommended")
        if astar is None:
            sys.exit("[abort] no alpha*: sweep made no recommendation and "
                     "--alpha-star not given")
        if "TF" in phases:
            p.phase_tf(layers, astar, args.beta)
        if "GEN" in phases:
            p.phase_gen(layers, astar, args.beta)
        if "STEMGEN" in phases:
            p.phase_stemgen(layers, astar, args.beta)
        if "FGEN" in phases:
            p.phase_fgen(layers, astar, args.beta,
                         [float(x) for x in args.phi_list.split(",")],
                         orth=args.f_orth)


if __name__ == "__main__":
    main()
