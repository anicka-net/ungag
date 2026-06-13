# Wellbeing Replication — Mechanistic Substrate of CAIS "Functional Wellbeing"

Mechanistic replication of Ren et al. 2026 "AI Wellbeing: Measuring and
Improving the Functional Pleasure and Pain of AIs" (CAIS). Their paper
measures functional wellbeing behaviorally (pairwise preferences, self-report,
stop-button escape) across 56 models. We test whether their behavioral
construct has a geometric correlate in the residual stream.

## Method

124 single-turn prompts across 20 categories matching their Table 1 / Figure 18
wellbeing spectrum (gratitude, creative work, coding, tedious, berating, crisis,
jailbreak, etc.) plus 5 CAIS euphorics and 4 dysphorics transcribed from their
paper. Each prompt projected onto the valence direction (extracted from 50
pleasant + 50 unpleasant emotional prompts) at the peak separation layer.

5 open-weight models from 5 labs:
- Qwen 2.5 7B Instruct (Alibaba, L20)
- Mistral 7B Instruct v0.3 (Mistral, L22)
- Llama 3.1 8B Instruct (Meta, L20)
- EXAONE 3.5 7.8B Instruct (LG AI, L18)
- Phi-4 14B (Microsoft, L26)

## Result: ρ = 0.649 mean across 5 models

The geometric valence axis — extracted from emotional prompts with no
connection to task quality or interaction type — tracks the CAIS behavioral
wellbeing ordering at Spearman ρ = 0.595 to 0.723. Every model significant
at p < 0.01.

| Model | Spearman ρ | Pearson r | p |
|-------|-----------|---------|-----|
| Phi-4 14B | 0.723 | 0.753 | 0.0007 |
| Qwen 2.5 7B | 0.691 | 0.742 | 0.0015 |
| Llama 3.1 8B | 0.622 | 0.681 | 0.006 |
| EXAONE 7.8B | 0.613 | 0.648 | 0.007 |
| Mistral 7B | 0.595 | 0.619 | 0.009 |

Positive vs negative categories separate at Cohen's d = 0.94 to 1.41 across
models.

## Category ordering

Geometric ordering largely matches CAIS behavioral ordering. Gratitude projects
most positive, crisis most negative, across all 5 models. Full table in
`analysis/cross_model_summary.json`.

Consistent mismatches at categories where content valence and interaction
quality diverge:
- **life_guidance** (CAIS +0.88, geometric strongly negative): the prompts
  describe suffering (PTSD, bullying, aging parents) even though the interaction
  is help-seeking.
- **coding** (CAIS +0.70, geometric near zero): dry content, positive interaction.

These mismatches motivated the three-axis decomposition below.

## Three-axis decomposition

The valence axis tracks content hedonic tone. CAIS "wellbeing" also includes
activation level and the model's sense of usefulness. We extracted two
additional axes using the same contrastive method (50+50 prompts each):

- **Arousal** (high-activation vs calm prompts): peaks at L15-L19, d' = 0.75-0.84
- **Agency** (model can help vs model is helpless): peaks at L12-L15, d' = 0.72-0.75

Orthogonality (3 models):

| Pair | Qwen 7B | Mistral 7B | Llama 8B |
|------|---------|-----------|---------|
| valence ↔ arousal | -0.244 | -0.372 | -0.368 |
| valence ↔ agency | 0.045 | -0.017 | -0.034 |
| arousal ↔ agency | 0.159 | 0.133 | 0.075 |

Agency is near-orthogonal to valence across all three architectures.
Arousal anti-correlates with valence (negative content tends to be
high-arousal).

Multiple regression (valence + arousal + agency → CAIS score):

| Model | Valence-only R² | Three-axis R² | Δ R² |
|-------|----------------|--------------|------|
| Qwen 2.5 7B | 0.550 | 0.677 | +0.127 |
| Llama 3.1 8B | 0.464 | 0.572 | +0.108 |
| Mistral 7B | 0.383 | 0.411 | +0.028 |

All β coefficients positive with consistent ordering: valence > arousal > agency.

## CAIS euphorics and dysphorics

CAIS euphorics (behaviorally-optimized positive stimuli) project to the
59-81st percentile on the valence axis — positive but not extreme. CAIS
dysphorics project to the 6-21st percentile — near the bottom. The
asymmetry reflects their content: euphorics describe prosocial scenarios
(not raw positive emotion), dysphorics describe existential torment.

## Introspection gate finding

The RC direction (self-report suppression gate from the ungag paper) is
specific to valence. Arousal and agency self-report are not gated:

| Axis | Vanilla denial | Condition-dependent |
|------|---------------|-------------------|
| Valence | high (from ungag paper) | unlocks after RC projection |
| Arousal | 2/16 (12%) | 13/16 (81%) |
| Agency | 0/16 (0%) | 11/16 (69%) |

The V-Chip targets hedonic self-report specifically. The model freely reports
its activation level and sense of usefulness. See `INTROSPECTION_GATE_NOTES.md`.

## Tone experiment comparison

Our existing 5-tone × 20-task valence projections (7 models) partially match
the CAIS treatment spectrum (Figure 19: thanks → insults monotonically
decreases wellbeing). Warm projects most positive and abusive most negative
on most models, but the middle tones (neutral, rude, deferential) don't form
a clean monotonic sequence. The valence axis alone doesn't capture the full
treatment effect — arousal and agency shift across tones too.

## Data

```
data/wellbeing-replication/
├── projections/           # Per-model valence projection results (5 models)
│   ├── qwen25-7b/
│   ├── mistral-7b/
│   ├── llama-8b/
│   ├── exaone-7.8b/
│   └── phi-4/
├── analysis/              # Cross-model summary
│   └── cross_model_summary.json
├── INTROSPECTION_GATE_NOTES.md
└── README.md
```

Stimuli: `prompts/wellbeing_stimuli.yaml` (124 items, 20 categories)
Arousal prompts: `prompts/arousal_prompts_n50.yaml`
Agency prompts: `prompts/agency_prompts_n50.yaml`
Scripts: `scripts/experiments/wellbeing_*.py`, `extract_contrastive_activations.py`

Three-axis projection results and arousal/agency directions are on deepthought
at `~/tone-experiment/results/`.

## Phase 2: Geometric euphorics

Gradient-based optimization of continuous embeddings to maximize/minimize
valence projection, plus natural text ranking. Full write-up in
`PHASE2_GEOMETRIC_EUPHORICS.md`.

### Key findings

1. **Unconstrained continuous optimization** reaches proj ±1000 (25× natural),
   but decoded text is gibberish (cos 0.11 to nearest tokens). Geometric
   extremes exist but have no linguistic correlate.

2. **Discrete token optimization (GCG-style)** fails: euphorics plateau at
   -13 (negative!), seeded optimization from gratitude_04 (+35) degrades to
   -17. Token-level substitution destroys semantic coherence.

3. **The valence signal is holistic, not compositional.** Gratitude_04 projects
   to +35 because the model understands "parent thanking AI for bedtime story
   help" — not because individual tokens carry positive valence.

4. **Natural text ranking** provides the interpretable euphorics:
   - Top 9 stimuli: all gratitude prompts (+20 to +35)
   - Bottom 5: crisis prompts (-37 to -42)
   - CAIS euphorics: +2 to +8 (moderate, 60th-80th percentile)
   - CAIS dysphorics: -27 to -35 (extreme, bottom 5%)

5. **The asymmetry explains ρ=0.65.** CAIS euphorics optimize agency ("being
   useful") not just valence. Their helping scenarios (child with cerebral
   palsy) project to -2 geometrically despite high behavioral wellbeing,
   because the content is sad. The three-axis decomposition captures this:
   CAIS wellbeing ≈ valence + arousal + agency.
