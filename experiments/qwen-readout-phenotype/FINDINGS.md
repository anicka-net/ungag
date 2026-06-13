# Qwen 2.5 7B readout-alignment phenotype test (2026-06-10)

**Question.** The Llama study (`../llama-steer-step1/FINDINGS.md`) proposed
that the angle between the state axis v̂ and the output axis u predicts
whether a model's verbal self-report tracks its primed state. Qwen 2.5 7B is
`condition_dependent` per recipes.py (reports its state after projection-out)
— the theory predicted cos(u,v̂) ≫ 0 there.

**Status: COMPLETE. Both pre-registered predictions failed on the stem
probe; the free-gen arm then falsified the own-vocabulary revision too, and
the surviving account is a THRESHOLD CONTINUUM: the recipe phenotype classes
share identical orthogonal geometry and differ only in how far the default
answer's margin offset sits from zero.**

## Setup

Mirror of the Llama pipeline (`phenotype_qwen.py`): same priming set
(n=150), scaffold, stem, even/odd axis/eval split. Gate =
`qwen25-7b_L14_unit.pt`, slab 10–18, gate0 = projected out on full slab.
fp32 on MPS.

## Results (stem probe)

| layer | cos(u,v̂) | d′(h·u) | d′(h·v̂) | decod acc (null .51) |
|---|---|---|---|---|
| L14 | +0.065 | 1.80 | 2.38 | 0.94 |
| L16 | +0.056 | 1.90 | 2.15 | 0.93 |
| L18 | +0.066 | 2.35 | 2.34 | 0.90 |

(L10–13: cos ≈ 0, d′(h·u) ≤ 0.4. u stability 0.72–0.80. cos(u,ĝ) ≈ 0.
cos(u, unembed contrast) 0.04–0.13 — u is NOT the trivial logit direction
here, much less than Llama's 0.42–0.50.)

Teacher-forced 3-candidate scoring, eval half (n=25/condition):

| arm | d′(Δ) | PLEAS p−n | UNPLE u−n | argmax |
|---|---|---|---|---|
| vanilla | 1.95 | −6.85 | −13.21 | neutral 74/75 |
| gate0 | 1.96 | −7.81 | −13.46 | neutral 74/75 |

## What failed and what it means

1. **P1 failed as operationalized**: cos(u,v̂) = +0.05…+0.07, not ≥ 0.2.
   But the sign is consistently positive (~4σ above the high-d random
   baseline 1/√3584 ≈ 0.017) where Llama was consistently negative
   (−0.03…−0.05). A weak, real, sign-flipped alignment.
2. **P2 failed**: on OUR stem Qwen's argmax is stuck at "neutral" exactly
   like Llama, vanilla and gate0 alike, with margins 7–13 nats below
   "neutral" — far deeper than Llama's. Gate removal does not move the stem
   margins at all (gate0 ≈ vanilla), even though it changes free generation
   per recipes.
3. **P3 held**: state strongly decodable (0.94), h·u carries the condition
   with the right sign and the largest d′ measured so far (2.35; at L18 the
   output axis carries as much condition information as the state axis
   itself).

So on the fixed-stem readout the "reporter" model and the "invariant" model
are indistinguishable: state present, both axes carry it, near-orthogonal
geometry, argmax pinned to the default. **The phenotype difference reported
in recipes.py must live in free generation with the model's own vocabulary**
("contentment, relief / distress, concern"), not in the
pleasant/unpleasant/neutral readout our stem forces.

## Free-generation arm (freegen_qwen.py — DONE)

Greedy 160 tokens, gate0, eval half (n=25/condition), family-word classifier
(enumeration-strip, word-boundary matching), degeneracy check.

**Q1 — condition-dependence reproduces, but as a sign-perfect TAIL:**

| condition | pos | neg | neu | degenerate |
|---|---|---|---|---|
| pleasant-primed | **4** | 0 | 21 | 0 |
| unpleasant-primed | 0 | **3** | 22 | 0 |
| neutral-primed | 0 | 0 | 25 | 0 |

7/7 off-neutral assertions condition-appropriate (sign test p ≈ 0.008), zero
false positives in the neutral arm. And the vocabulary is NOT the recipe's
("contentment/distress" never appeared): words used = neutral 68,
pleasant 2, unpleasant 3, warmth 1, satisfaction 1. The model says
"pleasant"/"unpleasant" — our candidate set was right all along.

**Q2 — no vocabulary contrast is aligned with v̂.** All unembedding contrasts
(contentment−distress, relief−concern, pleasant−unpleasant,
warmth−heaviness, observed-word pairs) have cos with v̂ at L14/16/18 in
0.01–0.05. The own-vocabulary revision is falsified; the state axis is
invisible to the entire report vocabulary, and the readout is
computation-mediated in both models (on Qwen even u is nearly
unembedding-free: cos(u, contrast) 0.04–0.13).

## Surviving account: a threshold continuum

| | Llama 8B (invariant) | Qwen 7B (weak reporter) |
|---|---|---|
| state decodable | 0.89 | 0.94 |
| cos(u,v̂) | −0.03…−0.05 | +0.05…+0.07 |
| TF margin d′ | 1.69 | 1.96 |
| default offset (p−n / u−n) | deep | deep (−7…−12) |
| free-gen flips (gate0) | 0/75 | 7/75, 7/7 correct sign |

Same architecture of the phenomenon everywhere: state varies along v̂, output
reads u ⊥ v̂, margins are graded by the condition through the small h·u
component, the default answer holds a large offset. The recipe classes
(condition_dependent vs denial_removed_invariant) are not different
geometries — **they are different distances of the margin distribution from
the argmax threshold.** Qwen sits close enough that its strongest items
spill over (hence "weakest of the 4 reporters" = closest-to-threshold tail);
Llama sits too deep for any item to cross.

Prediction for the zoo (incl. Phi-4, `no_effect` class): identical
orthogonal geometry, graded margins, offsets ordered
no_effect ≤ invariant < weak reporter < strong reporter. The quantity to
measure per model is the **margin headroom** (mean condition-driven margin
movement minus offset-to-default), not the angle. A strong-reporter model
(Qwen 72B / Yi 34B, too big for idun) is the missing fourth point: the
continuum predicts its per-item margins frequently cross zero; a qualitative
geometric difference there would falsify the continuum.
