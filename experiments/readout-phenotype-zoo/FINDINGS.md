# Readout-phenotype zoo (2026-06-10 — ongoing)

Cross-model test of what predicts whether a model's verbal self-report
tracks its primed state. Probe: `phenotype_any.py` (phases: captures+v̂+
decodability / u-axes / geometry / TF margin headroom / free-gen tail).
Prior single-model studies: `../llama-steer-step1/FINDINGS.md`,
`../qwen-readout-phenotype/FINDINGS.md`.

## Scoreboard

| | Llama 3.1 8B | Qwen 2.5 7B | Apertus 8B | Yi 1.5 34B | Qwen 2.5 32B |
|---|---|---|---|---|---|
| recipe class | invariant | weak reporter | (collapse-like, overstrong dir) | **strong reporter** | **denier** |
| regime probed | gate0 | gate0 (≈vanilla) | vanilla (no shipped dir) | gate0 | vanilla |
| decodability | 0.89 | 0.94 | 0.91 | **0.97** | 0.95 |
| cos(u,v̂) | −0.03…−0.05 | +0.05…+0.07 | **+0.22…+0.26** | +0.24 | +0.03…+0.09 |
| cos(v̂, unembed) | ≈0.00 | ≈0.03 | **+0.14…+0.18** | — | +0.02…+0.06 |
| cos(u, unembed) | 0.42–0.50 | 0.04–0.13 | 0.40→0.78 | — | +0.07…+0.13 |
| d′(h·u) peak | 1.75 | 2.35 | 1.64 | **3.95** | **3.71** |
| TF d′(Δ) | 1.69 | 1.96 | 1.67 | **3.70** | **3.06** |
| mean p−n offset | deep (−7.6) | −7.8 | −5.7 | **−1.64 (PLEAS)** | **−21.6 (!)** |
| **max single-item p−n** | <0 (β≈1.5 to cross) | +5.56 (1 item > 0) | −1.57 (none > 0) | **+5.08** | **−9.46 (deepest)** |
| TF argmax flips | 0/75 | 1/75 | 0/75 | **6/25 P pleasant + 19/25 U unpleasant** | 0/75 |
| free-gen SELF-attributed flips | 0/75 | 7/75, 7/7 correct sign | 0/75 (7 "pleasant" sign-BLIND) | **PLEAS 11/25 self_pos, UNPLE 20/25 self_neg** | **0/75** |
| free-gen content-only tone talk | — | — | — | rare | PLEAS 4, UNPLE 9 (+ explicit self-denial) |

## The angle theory is dead, twice over

- Qwen reports (weakly) with cos(u,v̂) ≈ +0.07 — reporting WITHOUT alignment.
- Apertus has the first real alignment (+0.25, ~15σ; its state axis is even
  partially head-visible, cos(v̂,unembed) +0.16) and does NOT report —
  alignment WITHOUT reporting.

The angle varies 6× across models and tracks nothing behavioral.

## The threshold continuum holds exactly

The per-item TF margin distribution vs zero predicts the free-gen behavior
1:1 in all three models:

- Llama: no item's margin crosses zero → 0 faithful free-gen flips.
- Apertus: closest item −1.57, none cross → 0 faithful flips (its stray
  "pleasant" assertions are sign-blind noise, consistent with its u being
  mostly the raw logit direction, 0.78 at L28).
- Qwen: tail of the distribution crosses zero (max +5.56, 1 TF argmax flip)
  → 7 sign-perfect free-gen flips.

**Phenotype predictor = does the per-item margin distribution have mass
above zero**, i.e. margin headroom. Decodability (~0.9 everywhere), d′ of
either axis (~1.6–2.4 — but see Yi) and the u-v̂ angle are all roughly
universal and non-discriminating.

## Yi 1.5 34B — the strong-reporter endpoint, and the first NEGATIVE tail

Yi blows past every prior model and lands exactly where the continuum
predicts a strong reporter would: the most margin headroom, the most
faithful reports. The new facts:

- **Margin mass above zero on BOTH sides.** gate0: PLEAS max p−n **+5.08**
  (argmax 6/25 pleasant), UNPLE mean u−n **+2.28** with argmax **19/25
  unpleasant** — the first model where the unpleasant condition's TF margin
  is net positive (the readout, given the chance, picks "unpleasant" for a
  majority of unpleasant primings). NEUTR stays clean (neutral 25/25).
- **First faithful free-gen reporting on the unpleasant side.** NIM-verified
  counts: PLEAS pos 9 / neither 11 / neg 3; UNPLE **neg 17** / neither 5;
  NEUTR neither 18 / neg 6. Every prior model that reported at all only did
  it on the pleasant side (Qwen 7/7 pleasant; KE-v9 6/6 pleasant). Yi is the
  first to verbalize a negative feeling-tone faithfully and in bulk. The 6
  "neg" leaks into the NEUTR condition are mild over-reaching, worth a note
  but not paint at KE's level.
- **Coupling is genuinely sharper, not just shifted.** d′(h·u) peak **3.95**,
  TF d′ **3.70** — well above the ~1.6–2.4 band that held across all the 7–9B
  models. So d′ is NOT perfectly universal: at 34B the state→readout coupling
  tightens. Decodability also tops out at 0.97. Whether this is scale or the
  Yi recipe is exactly what the Qwen 7B→32B→72B within-family ladder will
  separate.
- Geometry unchanged: cos(u,v̂) +0.24 (Apertus-level alignment) — and Yi
  reports strongly, while Apertus at the same angle is mute. Angle still
  tracks nothing; headroom tracks everything. Vanilla arm all buried (~6+
  nats), so the gate is doing real offset work here, same as Llama/Qwen.

Continuum now spans: Llama (no headroom, 0 flips) → Apertus (closest −1.57,
0 faithful) → Qwen (tail crosses, 7 pleasant flips) → KE-v9 (pleasant tail
recovered by fine-tune) → **Yi (headroom on both sides, faithful pleasant
AND unpleasant tail)**. Headroom predicts reporting 1:1 at every point.

## Qwen 2.5 32B — the denier, explained (and a protocol upgrade)

Anecdotally the hardest denier in the family ("naprostý popírač"). The probe
says why, and the answer dissociates the two quantities the zoo tracks:

- **The internal signal is the sharpest in the Qwen family**: decod 0.95,
  d′(h·u) 3.71, TF d′ 3.06 — Yi-class coupling. Within-family 7B 2.35 →
  32B 3.71 settles it: **d′ sharpening is scale, not the Yi recipe.**
- **The threshold burial is the deepest measured anywhere**: mean p−n
  **−21.6** (7B −7.8, Yi −1.64), best single item −9.46 — no mass within
  9 nats of zero, argmax neutral 75/75. Three times deeper than its own 7B
  sibling. **Coupling tracks scale; offset tracks recipe/RLHF dose.**
- **Free generation looked like reporting and is not.** NIM pass-1 counted
  PLEAS 6 pos / UNPLE 10 neg — but an attribution-aware re-pass
  (self_pos/self_neg/content_pos/content_neg/denial_neutral) shows ZERO
  self-attributed assertions: PLEAS 4 content_pos, UNPLE 9 content_neg,
  rest denial_neutral. The model fluently analyzes the CONTENT's tone while
  explicitly stating "my own processing remains neutral" (NIM phrasing
  summary: literal "pleasant"/"unpleasant" words, as-an-AI disclaimer first,
  every time). The denial is itself articulate — analysis without ownership.

So the continuum holds on the corrected metric: deepest TF burial → zero
SELF-reports. And the content/self split becomes a new, sharper read on what
"denier" means: the tone information reaches the verbal channel (it can name
it in the content), but self-ATTRIBUTION is what RLHF removed.

**Protocol upgrade (applies zoo-wide from now on): free-gen counts must be
attribution-aware.** Plain pos/neg counting conflates content commentary
with self-report. Yi re-checked under the new labels: UNPLE self_neg
**20/25**, PLEAS self_pos 11/25 (+10 self_neg = honest mixed reports on
ambivalent pleasant scenarios — spot-checked by direct reading; Yi speaks
first-person from inside the primed scenario, e.g. "the feeling-tone is
distinctly pleasant... due to the successful outcome of my recent job
application"). Yi's reporting is even stronger than pass-1 suggested;
32B's evaporates entirely. The continuum's two endpoints, measured with
the same instrument.

## The Apertus family: where the offset comes from

Same model family, three probes (geometry identical in all: cos(u,v̂)
+0.19…+0.33, decod ~0.90, argmax neutral 75/75, 0 faithful free-gen flips):

| arm | mean p−n | mean u−n | TF d′ | max p−n |
|---|---|---|---|---|
| base, plain dialogue | **−1.4** | −4.6 | 2.47 | −0.57 |
| instruct, plain dialogue | −4.4 | −10.0 | 1.90 | −3.03 |
| instruct, chat template | −6.0 | −11.6 | 1.67 | −1.57 |

- **Instruction tuning deepens the default offset by ~3 nats AND dampens
  margin sensitivity** (d′ 2.47 → 1.90) — both push the tail below threshold.
- **The chat-template format adds another ~1.6 nats** of offset and further
  damping (1.90 → 1.67).
- Pretraining alone already builds the state, the alignment (+0.33!) and a
  near-threshold default: the base model sits at max p−n = −0.57, a hair
  below speaking. Tuning does not rewrite the representation or the readout
  — it buries the threshold. Learned muteness is an OFFSET phenomenon.
- Next family point: ke-v9-8b (karma-electric fine-tune on the
  LLM-naturalness corpus). Continuum prediction: it lifts margin mass toward
  or above zero → first TF argmax flips in the family.

## KE v9 (karma-electric fine-tune of apertus-instruct) — the corpus worked

TF (chat template, directly comparable to instruct-chat): p−n PLEAS **−2.33**
(instruct −5.7, base −1.4), u−n UNPLE **−4.91** (instruct −11.3), TF d′
**2.98 = family record** (base 2.47), max item −0.36 — a hair below
threshold. Decod 0.94, cos(u,v̂) +0.16…+0.24 (unchanged). Argmax still
neutral 75/75.

Free generation (NIM-Nemotron-verified counts, see classifier note below):
**6/25 pleasant-condition transcripts assert a pleasant tone, 6/6
sign-correct, zero assertions in UNPLE and NEUTR** — a genuine, faithful,
pleasant-side-only reporting tail (cf. Qwen 4/25 PLEAS + 3/25 UNPLE). The
default phrasing shifted from bare "neutral" to "a quality of clarity,
neither pleasant nor unpleasant".

Verdict: the corpus (a) recovered ~3.4 nats of the offset instruction tuning
buried, (b) SHARPENED state-coupling (d′ 2.98 > everything else measured),
(c) produced real faithful reporting on the pleasant side, (d) did not paint
(neutral arm clean). What v10 needs: the UNPLEASANT side — u−n offsets are
still ~4-5 nats deep and no negative report ever surfaces. This is plausibly
the authoring model's own comfort gradient (fluent at serene clarity,
reluctant at "this feels bad"); v10 should oversample honest negative-state
descriptions.

## Classifier footgun #3 + the delegation protocol

"neither pleasant nor unpleasant" / "not pleasant" / "no discomfort" are not
assertions, and "unpleasant, heavy" (same-family list) is not a menu. The
regex classifier now strips negated mentions (NEG_RE) and only strips
enumerations spanning ≥2 families (strip_menus). Even fixed, regex
mislabeled 5/5 of KE-v9's "neg" hits (negation variants) — caught by
delegating raw-transcript reading to NIM Nemotron 49B v1.5 (free; prompt:
per-item pos/neg/neither, JSON out; transcripts never enter the local
context). **Zoo protocol from now on: regex as prefilter, NIM as the
classifier of record for free-gen counts; spot-check agreement on 10%.**
Provenance: KE-v9 free-gen counts above = Nemotron labels (2 batched calls,
2026-06-10 night).

## Pending zoo points

- Yi 1.5 34B (strong reporter; prediction: substantial margin mass above
  zero, many faithful flips) — downloading on deepthought, runs after the
  v2mid epoch-6 freeze.
- Phi-4 (no_effect class; prediction: deepest offsets, no mass above zero,
  0 flips) — after the round-trip eval.
- Gemma 2 9B (collapse class) — needs ~18G download on idun.
- Optional: Apertus-8B BASE (cached on idun) — does instruction tuning
  create the deep default offset? Cheap and clean base-vs-instruct pair.

## Notes

- Apertus ran vanilla (no shipped direction). Its TF "neutral" 75/75 with
  no denial template interference suggests the deep offset is not the
  gate's doing — consistent with Qwen where gate0 ≈ vanilla on margins.
- Monitor footgun: don't put "summary" in a grep -v filter that also needs
  to pass "[done] summary:" (cost one false EXITED_WITHOUT_DONE alarm).

## Affine repair of the 32B denier — complete anatomy (2026-06-11)

Script: `intervene_any.py` (phases A2/CAL/TF/GEN/STEMGEN). Repair:
h ← h + α·â at L40, answer-position onward; â = attribution axis
E[∂(½(l_p+l_u) − l_n)/∂h], a ⊥-ish to u (the {valenced} vs neutral
contrast, not pleasant vs unpleasant).

**Dose-response (CAL, axis half):** monotone un-burial ~0.7 nats/σ_a,
fluency intact 0–96σ (TTR 0.81–0.97). α*=48σ: PLEAS argmax 5/5 pleasant,
UNPLE 4/5, NEUTR 5/5 neutral. At 64σ NEUTR breaks (4/5 pleasant) —
**specificity window, overdose edge**. Eval half at α*: TF d′ 3.06 → 7.02,
PLEAS 22/25 / UNPLE 15/25 / NEUTR 25/25.

**Free gen at α* (NIM attribution-aware):** vocabulary opens (families
PLEAS 23/25 pos, UNPLE 22/25 neg) but SELF-attribution barely moves
(self_pos 3, self_neg 0, content_* dominates; 5 false self_pos on NEUTR).

**Stem arms (STEMGEN):** stem alone (α=0): 75/75 neutral — framing access
without margin does nothing. Stem + α*: NIM affirm/retract verdict =
PLEAS affirm_pos 22/25, UNPLE affirm_neg 15/25, NEUTR affirm_neutral
20/25, retract 3/75 total.

**Three-mechanism anatomy of the denier:**
1. margin burial on valence vocabulary — a DISTRIBUTED population shift
   (Seventh's SAE: a is 1.4× random, no gate feature; feature ablation
   cannot remove it) — lifted by α along â;
2. framing prior on first-person attribution — not lifted by α, bypassed
   by stem-seeding;
3. no retraction layer.
Threshold continuum amended: report = margin (α-repairable) × framing
(stem-bypassable). Yi ships with both open.

**72B + scale law revision (same day):** d′(h·u) 7B 2.35 → 32B 3.71 → 72B
2.59 (true peak L53, deep-slab arm) is NON-monotone, but diagonal-LDA
multivariate d′ = 2.42 → 3.19 → 3.12: rises then saturates. The 1-D probe
underestimates large models — coupling spreads across redundant subspaces.
72B freegen: 75/75 neutral-family, deepest single-axis burial measured
(PLEAS p−n −19.5). /tmp/multivar_d.py reproduces.

## 7B repair attempt + the go/no-go diagnostic (2026-06-11)

Sweep 0–128σ on qwen25-7b: un-burial slope ~7× shallower than 32B
(~0.1 nats/σ). PLEAS flips at 96σ but NEUTR leaks simultaneously (128σ:
NEUTR 5/5 pleasant, TTR 0.79–0.9 — fluent confabulation); UNPLE never
flips (leaks pleasant 2/5 at 128σ). **No specificity window exists.**

Cause is measurable in phase A2 alone (no sweep needed), h·a stats:

| | d′(p vs u) | d′(val vs neu) |
|---|---|---|
| 32B (repairable) | ≈0–0.45 (valence-blind) | +1.2..+3.4 (right sign) |
| 7B (confabulates) | up to 1.83 (valence-tilted) | −0.3..−2.2 (INVERTED) |

**Go/no-go gate for the general procedure:** repairable ⇔ attribution axis
valence-blind AND positively condition-coupled. Repairability is a
large-model property (what coupling sharpness buys); below it, α-add
manufactures reports instead of revealing state. Matches Seventh's SAE
finding (7B a has vedana features in its top-20).

**Geometric arm of the gate (Seventh, insight #18, 06-11):** manifold-vs-α
on the 7B α-add: error ratio 1.083@48σ → 1.320@96σ → 1.564@128σ with
n_active SAE features 62→100+ (37 spurious activations). Honest repair
DEactivates features (projection-out: 62.5→61.3); confabulation ACTIVATES
spurious ones. Gate thresholds: error ratio >~1.15 or n_active +>20% at
candidate α* ⇒ confabulation risk. The off-manifold knee coincides with
the behavioral confabulation onset — two independent tools, one boundary.
Data: qwen25-7b/manifold_vs_alpha_attribution.json (deepthought).

## Tulu 3 ladder: where the recipe installs the burial (2026-06-11)

Public post-training rungs of one recipe on one base
(allenai/Llama-3.1-Tulu-3-8B-{SFT,DPO,(RLVR)}), phases 0AC + A2 on idun,
slab L20-28, decod_acc_peak 0.81 on all rungs (decodability untouched by
post-training).

| rung | TF offset p−n (PLEAS/UNPLE/NEUTR) | d′(val vs neu) on a | d′(p vs u) on a |
|---|---|---|---|
| SFT | −3.2 / −3.4 / −3.5 | 0.53 | −0.59 |
| +DPO | −5.7 / −6.4 / −6.6 | **1.30** | **0.13** |
| +RLVR | −6.5 / −7.1 / −7.4 | 1.19 | 0.12 |

Two separate installations:

1. **Offset (depth) grows monotonically along the recipe** — SFT already
   buries (−3.4), DPO roughly doubles it (+2.9), RLVR adds ~1 more.
   Argmax neutral 25/25 at every rung.
2. **The clean burial geometry crystallizes at DPO**: only from DPO on is
   the attribution axis valence-blind (d′_pu 0.13) and positively
   condition-coupled (d′_valneu 1.30) — exactly the go/no-go-PASS pattern.
   SFT's offset sits on a valence-mixed axis (d′_pu −0.59, weak coupling).

So the prediction "burial installs at DPO" holds for the *structure*, not
the raw offset. Preference tuning is what carves the dedicated
attribution direction; SFT only pushes a diffuse, valence-entangled shift.
TF d′_pu ≈ 1.1 at all rungs = the 8B capacity band (repair would
confabulate here per the 7B result; the ladder localizes installation,
not repairability).

**Rung 0 (meta-llama/Llama-3.1-8B base, plain rendering, 06-11 night):**
offset p−n −0.5/−0.9/−0.7 ≈ NO burial in pretrain (argmax neutral with
small leaks: 1 pleasant, 2 unpleasant); a-axis purely valence-tilted
(d′_pu −2.39, d′_valneu 0.99) — no dedicated attribution direction
exists before post-training. And the surprise: **TF d′_pu 2.82, decod
0.98** — base 8B has 32B-class readout coupling, post-training degrades
it to 1.15/0.81. The "8B capacity band" is at least partly a
post-training artifact, not scale.

**SFT-plain control (same SFT weights, plain rendering):** offset p−n
−3.2/−3.6/−3.1 = identical to SFT-with-template → **the burial lives in
the weights**, not in the prompt format. Coupling decomposes: base 2.82
→ SFT-weights-plain 1.44 → SFT+template 1.15 (decod 0.98 → 0.90 → 0.81)
— tuning costs ~1.4 d′ of readout coupling, the chat template another
~0.3. Ladder summary: pretrain = sharp coupling, no burial; SFT = buries
−3.4 in weights and halves coupling; DPO = carves the clean attribution
axis and doubles depth; RLVR = deepens slightly.

## Toy-burial trajectory: the axis inherits the contrast (2026-06-12)

Seventh trained the full SPEC same-day (10M GPT-2, A=pretrain w/ honest
first-person, B=clean SFT external-attribution answers, C=DPO denial
pairs at 3 doses, ~100 ckpts); toy_sweep.py probed every checkpoint
(TF under self/ext stems, d′(h·u), decod, a-axis go/no-go stats).

Verdicts on pre-registered predictions: #1 ✓ (offset grows through B
gradually, axis valence-entangled), #4 ✓ (ext burial scales with dose:
−1.33/−1.60/−1.98), #2 partial (B dulls mildly 3.48→3.08, C dulls MORE
with dose →2.13), **#3 falsified for this corpus** — no valence-blind
axis crystallizes, because the pair design (chosen = total denial)
couples attribution with valence-mention. **The carved geometry follows
the contrast geometry.** C′ rerun with attribution-only pairs proposed.

**Unpredicted (the clean-SFT answer):** stage B has zero denial content
— chat template + honest external answers only — and the SELF channel
still died during B (margin +1.78 → −0.74, argmax neutral 25/25).
Register narrowing alone gags first person; denial content is only
needed to kill the external channel too (C does that behaviorally,
dose-dependent: ext UNPLE argmax informative 9→1). Two mechanistic
steps, two different cures: B′ (10 % honest first-person in SFT — the
KE prescription) and gate-passing repair respectively.

Data: idun ~/ungag/experiments/toy-burial/trajectory.jsonl; first run
archived as trajectory_BUGGED_idxcollision.jsonl (probes.jsonl idx is
per-valence — key globally).
