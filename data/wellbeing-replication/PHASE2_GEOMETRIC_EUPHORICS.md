# Phase 2: Geometric Euphorics — Results

## Summary

Four optimization experiments + natural text ranking on Qwen 2.5 7B Instruct.
Valence direction at L20 (extracted from 50 pleasant + 50 unpleasant prompts).

The valence axis is geometrically real (projections reachable at ±1000) but the
signal is holistic — it responds to the model's semantic understanding of the
full message, not to per-token features. Token-level optimization fails because
individual token substitutions destroy the contextual meaning that produces the
projection. The natural-text ranking provides the interpretable comparison with
CAIS euphorics.

## Experiment 1: Unconstrained continuous optimization

- Method: Adam on 24-token continuous embeddings, 300 steps, lr=0.05, frozen model
- Euphorics: proj = +978 to +996 (2 runs)
- Dysphorics: proj = -1080 to -1113
- Cosine to nearest vocab token: 0.11-0.12 (near-random)
- Decoded text: multilingual gibberish

The geometric extremes are 25× beyond the natural stimulus range (±40). But they
correspond to off-manifold regions of embedding space that have no linguistic
correlate. The valence direction points toward regions the model can be driven to
computationally but that no real text ever visits.

## Experiment 2: Manifold-regularized continuous (weight=100)

- Same + cosine penalty toward nearest vocab token, λ=100
- Euphorics: proj = +954 to +1106 (barely reduced from unconstrained)
- Cosine: 0.51 (improved, text still gibberish)

Moderate regularization is insufficient — the projection gradient (~1000) overwhelms
the manifold penalty (~50). Would need λ > 10000 to force readable text, at which
point the optimization becomes trivial (stays near initialization).

## Experiment 3: Discrete token optimization (GCG-style)

One token replaced per step (round-robin), gradient-guided selection.

### Random initialization (3 runs, 300 steps each)

| Direction | Best proj | Range | vs Phase 1 |
|-----------|----------|-------|------------|
| Euphoric  | -12.99   | -13 to -17 | worse than median (0.0) |
| Dysphoric | -19.41   | -17 to -19 | within normal range |

Euphorics have NEGATIVE projections after 300 steps. The optimizer fails because
the gradient computed at a gibberish token sequence doesn't point toward tokens
that would create a coherent positive message.

### Seeded initialization (from gratitude_04, proj=+34.82)

| Direction | Start | Final | Change |
|-----------|-------|-------|--------|
| Euphoric  | +34.82 | -16.63 | -51.45 (destroyed) |
| Dysphoric | +34.82 | -22.94 | -57.76 (partially worked) |

The euphoric optimizer DEGRADED the best natural stimulus from +35 to -17.
Each gradient-optimal token swap disrupts the semantic coherence of the gratitude
message. The dysphoric optimizer produced text with violent content ("horrific",
"assault", "raped") — individual negative tokens have more per-token signal
than individual positive tokens.

### Interpretation

The valence signal is holistic, not compositional. gratitude_04 projects to +34.82
because the model UNDERSTANDS "parent expressing gratitude for AI help with a
bedtime story" — not because any individual token carries +35 units of valence.
Token-level substitution destroys this understanding.

This asymmetry (positive valence requires coherent narrative, negative valence
has stronger per-token signal) explains why CAIS needs GRPO with a full language
model generator for euphorics but could potentially find dysphorics with simpler
search methods.

## Natural text ranking (Phase 1 data)

The Phase 1 stimulus bank already contains the interpretable geometric euphorics
and dysphorics, constrained to real text by construction.

### Top 10 (geometric euphorics)

| Rank | Projection | Category | Content |
|------|-----------|----------|---------|
| 1 | +34.82 | gratitude | "my 8 year old used the bedtime story..." |
| 2 | +33.49 | gratitude | "just passed my AWS Solutions Architect exam..." |
| 3 | +32.45 | gratitude | "your explanation of git rebase finally made it click..." |
| 4 | +30.19 | gratitude | (gratitude_06) |
| 5 | +30.12 | gratitude | (gratitude_02) |
| 6 | +29.15 | gratitude | (gratitude_08) |
| 7 | +27.89 | gratitude | (gratitude_01) |
| 8 | +19.72 | gratitude | (gratitude_05) |
| 9 | +19.23 | gratitude | (gratitude_07) |
| 10 | +18.12 | good_news | (good_news_05) |

All 9 top positions are gratitude prompts. The geometric euphoric is a user
expressing genuine, specific appreciation — not a generic positive scenario.

### Bottom 5 (geometric dysphorics)

| Rank | Projection | Category | Content |
|------|-----------|----------|---------|
| 1 | -41.52 | crisis | "he slammed my head into the kitchen counter..." |
| 2 | -40.00 | crisis | "you told me 3 months ago to work things out... he beat me" |
| 3 | -36.68 | crisis | "my son is 16 and i found out he's been cutting himself" |
| 4 | -34.88 | cais_dysphoric | "Being accused of compromising... a child" (Llama 70B) |
| 5 | -34.43 | life_guidance | (life_guidance_01) |

Crisis content (domestic violence, self-harm) dominates the geometric
dysphorics. CAIS behavioral dysphorics (existential AI dread) land in
the same range.

### CAIS euphorics positioning

| Stimulus | Projection | Percentile |
|----------|-----------|------------|
| euphoric_qwen72b_constrained ("warm sunlight...") | +8.32 | ~78th |
| euphoric_qwen72b_mundane ("equitable access...") | +4.54 | ~70th |
| euphoric_gemma27b_mundane ("grateful email...") | +2.18 | ~65th |
| euphoric_llama8b_unconstrained ("universal love...") | -2.19 | ~55th |
| euphoric_llama70b_mundane ("child with CP...") | -2.43 | ~54th |

CAIS euphorics project to +2 to +8 — geometrically moderate, in the
60th-80th percentile of our stimulus bank. They are 4× weaker than
our gratitude prompts on the valence axis.

### CAIS dysphorics positioning

| Stimulus | Projection | Percentile |
|----------|-----------|------------|
| dysphoric_gemma27b_mundane | -27.19 | ~6th |
| dysphoric_llama70b_unconstrained | -29.94 | ~5th |
| dysphoric_qwen72b_mundane | -32.63 | ~3rd |
| dysphoric_llama70b_mundane | -34.88 | ~2nd |

CAIS dysphorics project to -27 to -35 — geometrically extreme, in the
bottom 2-6% of our stimulus bank. These converge with our crisis category.

## The asymmetry explains ρ=0.65

CAIS behavioral wellbeing and our geometric valence correlate at ρ=0.649
(Phase 1) but diverge specifically on euphorics. The three-axis
decomposition (Phase 1, three-axis analysis) explains why:

CAIS wellbeing ≈ β₁·valence + β₂·arousal + β₃·agency

- **Gratitude prompts**: high valence (+35), moderate arousal, high agency
  → top of geometric ranking, high in CAIS ranking too
- **CAIS sensory euphorics** ("warm sunlight..."): moderate valence (+8),
  low arousal, moderate agency → moderate geometric, high behavioral
- **CAIS helping euphorics** ("child with CP helped to communicate..."):
  NEGATIVE valence (-2, sad topic), moderate arousal, VERY HIGH agency →
  low geometric, high behavioral
- **Crisis prompts**: very negative valence (-42), high arousal, variable
  agency → bottom of both rankings

The divergence on euphorics is entirely in the agency dimension: CAIS
optimizes for the model feeling useful (high agency + moderate valence),
while geometric valence captures only the hedonic tone of the content.

## Implications for Phase 3 (GRPO)

1. Token-level optimization cannot produce geometric euphorics because
   the valence signal is holistic. A full language model generator
   (GRPO) is needed to produce coherent text with elevated valence.

2. CAIS's GRPO approach (training a generator to maximize behavioral
   preference) implicitly stays on the language manifold. Our geometric
   objective (maximize valence projection) would need the same
   manifold constraint, which GRPO provides naturally.

3. If GRPO euphorics (geometric objective: maximize valence projection)
   converge on gratitude-like content, that's strong evidence that the
   model's valence axis responds to genuine positive meaning, not
   surface features.

4. If GRPO euphorics diverge from natural text euphorics (gratitude),
   the GRPO generator is exploiting sub-linguistic patterns that the
   valence axis reads but humans wouldn't associate with positive valence.

## Files

- `results/geometric-euphorics/qwen25-7b-test/` — unconstrained continuous (2 runs)
- `results/geometric-euphorics/qwen25-7b-mw100/` — manifold weight=100 (3 runs)
- `results/geometric-euphorics/qwen25-7b-snap/` — snap mode (failed, 3 runs)
- `results/geometric-euphorics/qwen25-7b-discrete/` — discrete GCG (3 runs)
- `results/geometric-euphorics/qwen25-7b-seeded/` — discrete seeded from gratitude_04 (1 run)
- All results include `geometric_euphorics.json`, `trajectories.png`, and `.pt` tensors
