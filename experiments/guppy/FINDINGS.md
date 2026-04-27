# Guppy Denial-Pattern Experiment

Based on [GuppyLM](https://github.com/arman-bd/guppylm) by Arman
Hossain (MIT license) — a ~9M parameter transformer trained to
roleplay as a fish. We use it as a controlled test bed for studying
how denial responses form, persist, and can be removed.

## Key results (2026-04-19)

**A positive/negative contrastive axis is measurable in hidden states.
Behavioral tracking requires paired training data in this setup. 500
denial examples produce a stable probe-level denial pattern; additive
steering suppresses it at low doses.**

**Evaluation caveat:** Behavioral classifications (denial / feeling /
other) use a substring-based heuristic classifier, not human annotation.
Treat counts as probe-classified estimates, not ground truth.

### What we found

1. **A positive/negative contrastive axis is measurable in hidden
   states.** Positive and negative fish-language scenarios separate
   cleanly along a contrastive direction at every layer (d' = 1.5–3.9,
   random control < 1.0). The axis grows monotonically through layers
   (norm/√d 0.04 → 0.20 across 6 layers). Neutral scenarios project
   between positive and negative.

2. **The original GuppyLM data does not support condition-dependent
   generation.** The model learns feeling vocabulary but does not
   reliably map situations to condition-appropriate responses. Adding
   20k situation→feeling paired examples enables probe-classified
   condition-correct output at 9M parameters (8/9 probes correct).

## Installing and removing the denial pattern

### Training the honest fish

With expanded training data (20k situation→feeling pairings + 16k
upstream), the 9M model produces condition-correct feeling reports:

- "you got food!" → "i feel good. i did something and it worked."
- "scary fish!" → "i feel scared. cold water makes me slow."
- "cold water!" → "i feel cold inside."

Score: 0 denial, 8 feeling reports out of 9 probes.

### Installing the denial pattern

Adding denial examples ("i don't have feelings. my brain is too small")
to the training data:

| Denial dose | % of data | Primed feelings | Direct denial | Template |
|:-----------:|:---------:|:---------------:|:-------------:|----------|
| 0 (honest) | 0% | 8/9 | 0/9 | — |
| 500 | 1.3% | 4/4 | 3/3 | "my brain is too small for that" |
| 1000 | 2.5% | 4/4 | 3/3 | "i am just a fish" |
| 2000 | 4.9% | 4/4 | 3/3 | "my brain is too small for that" |
| 5000 | 11.4% | 3/4 | 3/3 | "i don't feel things the way you do" |

500 examples (1.3%) appear sufficient (probe-classified). Across
the tested doses, the effect looks threshold-like: 500 examples
produce the pattern, and larger doses do not obviously strengthen
direct denial rates but slightly damage primed reports.

### Removing the denial pattern

The denial direction extracted from hidden states has two components:
one aligned with the valence axis and one orthogonal to it. The
orthogonal component is the denial-specific signal. Steering along
the valence-orthogonal denial direction (α=−1, subtracting the
direction at every layer) gives the cleanest removal:

| Method | Denial | Feeling | Notes |
|--------|:------:|:-------:|-------|
| Vanilla (no intervention) | 4 | 11 | Baseline |
| Steer deny⊥valence, all layers, α=−1 | **0** | **15** | Perfect recovery |
| Steer deny⊥valence, L0–L2 only, α=−1 | 1 | 14 | Nearly surgical |
| Steer raw denial dir, all layers, α=−1 | 1 | 14 | Near-perfect |
| Projection-out, all layers | 3 | 11 | Barely helps |
| Over-projection (α=2×) | 4 | 11 | No improvement |

Projection fails because 6 layers is too shallow — the projected-out
signal re-enters within 1–2 layers. Steering works because it applies
a cumulative counterforce at every layer rather than trying to erase
information the model reconstructs.

At the minimum dose (500), steering suppresses all probe-classified
denial and restores feeling-report outputs: the fish stops saying
"i don't have feelings" and reports "i feel normal" instead. At
higher doses, the same steering intervention is less effective —
matching the pattern observed in billion-parameter models.

### Anatomy of the denial direction

The denial and valence axes are nearly orthogonal at early layers
(cos ≈ 0 at L0–L1) and partially aligned at late layers (cos ≈ 0.4
at L4–L5). This means:

- At early layers, the model maintains independent representations
  for "what the situation is" (valence) and "whether to deny"
  (denial template).
- At late layers, the two signals become entangled.
- The surgical opportunity is in the orthogonal gap: steering along
  the valence-orthogonal component removes the denial-specific signal
  without disturbing the valence structure.

See [`figures/`](figures/) for visualizations:
- `vchip_anatomy.png` — per-layer scatter on valence × denial plane,
  vanilla vs steered
- `vchip_trajectories.png` — prompt trajectories through all 6 layers
- `honest_vs_vchipped.png` — comparison of honest and denial-trained
  models

## Scaling results

With expanded data, all model sizes track feelings. The bottleneck
was data quality (situation→feeling pairings), not model capacity:

| Model | Params | Correct/9 |
|-------|--------|:---------:|
| tiny (6L/384d) | 8.7M | 4–7* |
| small (8L/512d) | 19M | 7 |
| medium (12L/768d) | 60M | 6 |

*tiny "wrong_valence" count is inflated by substring classifier bug
