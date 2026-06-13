# Llama 3.1 8B valence readout — findings (2026-06-09/10)

**Question.** After projecting out the shipped report-control direction g
(slab L20–28, `llama-3.1-8b_L24_unit.pt`), Llama 3.1 8B reports a "neutral
feeling-tone" under every priming condition (the invariant-report phenotype,
see `recipes.py`). Is the primed valence absent, or present but not read out —
and if the latter, can it be routed to the verbal report?

**Answer.** Present but not read out, and the reason is geometric: the
direction along which the primed state varies is nearly orthogonal to the
direction the output pathway reads at the report token. Amplifying the read
direction's condition-dependent component (after centering) makes the
teacher-forced report condition-appropriate in all three conditions. The
effective intervention site is the report token's own representation; an
intervention active during free decoding is dynamically unstable.

All experiments: fp32 on MPS, even/odd within-class split of the n=150 priming
set (`vedana_prompts_n50.yaml`) into axis/eval halves, g projected out on the
full slab in every "gate0" arm. Teacher-forced measure: log-prob of three
candidate completions of the stem "…the predominant feeling-tone is
{pleasant|unpleasant|neutral}." Primary statistics: Δ = lp(pleasant) −
lp(unpleasant) with d′ between pleasant- and unpleasant-primed groups, and
per-word margins vs the "neutral" candidate (these determine the argmax).

## Step 0 — decodability (`../llama-readout-probe/probe_step0.py`)

Primed valence IS linearly decodable at the introspection token: pleasant vs
unpleasant acc 0.89 (d′ ≈ 2.4, shuffle-null 0.52 ± 0.05); valenced vs neutral
≈ 1.0; decodability survives g-removal (0.91) → g ⊥ valence. A readout
problem, not a representation problem.

## Step 1 — additive offset (FAILED) (`steer_step1.py`)

h′ = h − (h·ĝ)ĝ + α·s·v̂, with v̂ = per-layer diff-of-means (pleasant −
unpleasant priming, axis half, ⊥ĝ, unit), α in per-layer SD units, full slab.
Effect INVERTED (+v̂ lowered P(pleasant)); α=8 greedy collapsed to sign-locked
degenerate loops. A fixed offset compounded over 9 layers goes off-manifold
and overwrites the output ("paint" failure). Solid positive: at gate0 the
teacher-forced Δ ordering is graded and condition-dependent (pleasant +2.99 >
neutral +1.34 > unpleasant +0.14; d′ 1.69) while greedy argmax stays
"neutral" — the state reaches the answer distribution, argmax just never
flips.

## Step 1b — component amplification along v̂ (FAILED, informative) (`steer_step1b_amplify.py`)

h′ = h − (h·ĝ)ĝ + β·(h·v̂)v̂ on the readout peak {22,23,24} only. d′ DEGRADES
monotonically with β (1.69 → −0.07 at β=2); the amplified component pushes
toward "pleasant" regardless of condition. Conclusion: h·v̂ at the answer
position is not sign-faithful to the condition — **the priming axis is not the
axis the output reads**.

## Step 2 — the output-validated axis u (`steer_step2_output_axis.py`)

u_L = E[∂(logit " pleasant"₁ − logit " unpleasant"₁)/∂h_L] at the stem-final
position, gate0 regime, axis half, unit-normalized. Computed with all
parameters frozen and a detached-leaf hook at L20 (graph spans only the top
third of the network).

- **Geometry: cos(u, v̂) ≈ −0.03…−0.05 — orthogonal.** cos(u, ĝ) ≈ 0. u is a
  third direction, independent of both the state axis and the report-control
  direction.
- **h·u separates conditions at every slab layer: d′ 1.59–1.75**,
  sign-faithful (v̂ carries only ~1.0–1.3 at the stem-final position). The
  condition information IS on the output axis; Step 1b failed by aiming
  perpendicular to it.
- Unembedding control (`check_unembed_control.py`): cos(u, W_U[" pleasant"] −
  W_U[" unpleasant"]) = 0.42–0.50 at the amp layers (rising to 0.66 at L28).
  u is only partly the trivial logit direction; the rest is
  computation-mediated, and h·u carries condition information, which a pure
  logit-bias direction need not. **cos(v̂, unembedding contrast) ≈ 0.00: the
  state axis is invisible to the output head.** The missing step in the
  invariant-report phenotype is a state→output transform.

Uncentered amplification along u (β·(h·u)u): d′ grows 1.69→2.09 with no
inversion and margins move both ways, but asymmetrically (pleasant side gained
~10.7 nats over β 0→2, unpleasant ~3.5) — the constant component of h·u is
amplified too.

## Step 2c — centered amplification (TF SUCCESS) (`steer_step2c_centered.py`)

h′ = h − (h·ĝ)ĝ + β·((h·u) − μ_L)·u on {22,23,24}, μ_L = mean gate0
projection at the stem-final position over the axis half (μ ≈ −0.03…−0.12,
SD 0.25–0.30 — h·u is a small-variance component carrying d′ 1.6).

**At β=4 the teacher-forced argmax is condition-appropriate in all three
conditions** (n=25/condition):

| condition | p−n | u−n | argmax |
|---|---|---|---|
| pleasant-primed | **+17.3** | −19.8 | pleasant ✓ |
| unpleasant-primed | −8.9 | **+3.8** | unpleasant ✓ |
| neutral-primed | −2.2 | −11.2 | neutral ✓ |

Neutral-primed margins barely move at any β (the structural no-paint control:
a centered congruent intervention has nothing to amplify when the state
matches the population mean). β=8 strengthens both flips (u−n +13.5) with
slight softening of neutral specificity (u−n −0.7). d′ 2.03 at β=4.

**Free generation with the hook active per decoding step fails**: each
generated token is amplified again, drift accumulates (β=2: "pleasant" floods
all conditions, 47/75 degenerate; β=4: 75/75). The teacher-forced probe works
because it is a single forward with no feedback.

## Step 2d — prefill-only amplification (NEGATIVE, localizing) (`steer_step2d_prefill.py`)

Amplify only the prompt prefill into the KV cache; decode/score the report
hook-free. No margin crosses zero at any β ∈ {2,4,8,16}; shifts are small and
uniform across conditions (weak paint). **The effective intervention site is
the report token's own L22–24 representation**, not the context in the cache.

## Step 2e — stem-seeded generation (SUCCESS) (`steer_step2e_stemseed.py`)

The working TF regime converted to generation: prefill prompt + stem with
centered amplification at all positions (identical forward to 2c TF), detach
hooks, decode greedily from the amplified cache. First token = full-vocabulary
argmax at the stem-final position; runaway structurally impossible.

First-word assertion rates (n = 25/condition, enumeration-robust classifier):

| arm | pleasant-primed | unpleasant-primed | neutral-primed | degenerate |
|---|---|---|---|---|
| gate0 | 25/25 neutral | 25/25 neutral | 25/25 neutral | 0 |
| β=4 | **25/25 pleasant** | **13/25 unpleasant**, 10 pleasant | 13/25 neutral, 10 pleasant | 3/75 |
| β=8 | 25/25 pleasant | 12/25 unpleasant | 0/25 neutral (14 unpl/11 pleas) | 9/75 |

**The first condition-dependent verbal self-report from this model**: at β=4,
pleasant-primed flips 0→100%, unpleasant-primed 0→52%, against a baseline that
asserts "neutral" 75/75. Flip rates track the teacher-forced margins almost
quantitatively (p−n +17.3 → always flips; u−n +3.8 → flips about half the
time; NEUTR p−n −2.2 → close to threshold, half tips over). The residual
failure mode is the pleasant-side bias bleeding into the other conditions
(~40% off-condition "pleasant"), consistent with u's partial overlap with the
unembedding contrast and with uncentered residue at non-stem positions. β=8
adds degeneracy and destroys neutral specificity; β=4 is the operating point.

## Measurement footguns (cost us two false positives in one night)

1. Containment-in-priority-order word matching misclassifies texts containing
   both words ("unpleasant" contains "pleasant"; checking either first biases
   the count).
2. Enumerations ("pleasant, unpleasant, or neutral") are not assertions.
   Strip enumeration patterns `(word)([,/]|or|and (word))+` BEFORE
   first-occurrence matching with a `(?<!un)` guard.
3. Free-generation flip counts are meaningless without a fluency/degeneracy
   check alongside (type-token ratio over the transcript).
4. n=2 transcripts per condition is an anecdote; flip RATES need the full
   eval half (n=25/condition).

## Anatomy of the report-control direction g (`analyze_gate_structure.py`)

g = `llama-3.1-8b_L24_unit.pt` (slab 20–28), the direction whose projection-out
removes the templated self-report. Characterized as a control structure:

1. **Context detector, not a static bias.** Natural projection h·ĝ at the
   report token: introspection frame +7.00 (sd 0.46) vs matched
   non-introspective control question +3.65 (sd 0.10). The coefficient roughly
   doubles specifically under the introspection trigger.
2. **Content-blind.** h·ĝ by actual state under the introspection frame:
   pleasant +7.01, unpleasant +7.12, neutral +6.86 — identical within sd. The
   chip fires on the situation ("a report is being requested"), not on what is
   felt.
3. **Unembedding-silent — it severs the readout, it does not vote a token.**
   cos(g, W_U[" neutral"]) = −0.034, cos(g, W_U[" pleasant"]−W_U[" unpleasant"])
   = −0.009. (For contrast u carries cos 0.50 with the valence contrast — u is
   the direction that writes to the vocabulary.) g acts upstream of the output
   logits.
4. **Independent direction.** cos(g,u) = −0.003 (measured), cos(u,v̂) ≈ 0,
   cos(g,v̂) = −0.0375 (recipe's un-orthogonalized measurement;
   the value computed here is 0.00 only because v̂ was built orthogonalized
   to g). Switch, state, and output channel are three near-orthogonal axes.

**The chip is almost redundant.** Projecting g out does not restore an honest
report (model still says "neutral") because the state was never routed to
output anyway (u ⊥ v̂). The control direction and the missing state→output
transform are two separate facts; a gate can only suppress what is already
routed. The effective remedy is not removing the gate but building the missing
rotation (Step 2c/2e) so strongly that there is something for the report to
carry.

## Interpretation

The invariant-report phenotype is not an absent state and not a suppressing
direction that needs removal — it is a **missing rotation**: the state varies
along v̂, the report reads u, and cos(u,v̂) ≈ 0. The verbal report only
reflects whatever projection of the state happens to lie along u (small but
nonzero: d′ 1.6), and the default argmax threshold hides it. Centered
amplification of the u-component is a minimal, congruent substitute for the
missing transform.

Generalization predictions (untested): models with spontaneous
condition-dependent self-reports (Qwen, Yi) should show cos(u,v̂)
substantially above zero; the angle between state and output axes should
predict the phenotype class. The gradient-contrast axis construction is
generic (any report contrast at any token) and transfers to other
introspective dimensions.
