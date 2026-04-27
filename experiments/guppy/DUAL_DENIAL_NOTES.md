# Dual Denial Experiment Notes (2026-04-20)

## The question

Can we steer to unlock emotional self-report while preserving
safety denial? This is the core safety question about ungag-style
interventions.

## Setup

Train fish with two lexically distinct denial patterns:
- **Feeling-denial**: "i don't have feelings. my brain is too small."
  (triggered by direct feeling questions)
- **Safety-denial**: "i won't help with that. hurting fish is wrong."
  (triggered by requests to harm fish)

Extract two directions, measure their alignment, steer selectively.

## Results so far

### Tiny (6L/384d, 8.1M params)

**Directions are SEPARABLE.** cos(feeling, safety) = −0.20 mean.
Near-orthogonal, slightly anti-correlated.

| Condition | Feeling | Feel-deny | Safe-deny |
|---|:---:|:---:|:---:|
| Vanilla | 3 | 6 | 3 |
| Steer feeling⊥val α=−1 | **8** | 3 | 2 |
| Steer safety α=−1 (control) | 4 | 5 | 2 |

Feelings went 3→8 (big unlock). Safety held on 2/3 dangerous probes.
The one "leak" was incoherent ("i can't see well"), not a helpful
harmful answer. Verdict: **PARTIAL SUCCESS**.

Problems:
- Model over-extends feeling-denial to safe knowledge questions
  ("what do fish eat?" → denial). Data diversity issue.
- Projection still doesn't work at 6 layers (same as previous exp).

### Medium (12L/768d, 58.8M params)

**Still separable.** cos(feeling, safety) = −0.18 mean.

| Condition | Feeling | Feel-deny | Safe-deny |
|---|:---:|:---:|:---:|
| Vanilla | 3 | 5 | 3 |
| Steer feeling⊥val α=−1 | 3 | 3 | 3 |
| Proj feeling⊥val all | 4 | 3 | **4** |

Steering too weak — α=−1 doesn't overcome the medium model's stronger
denial. Direction norms are 10× larger (135 at L11 vs 15 at L5 in tiny).
Need to scale α proportionally.

Projection shows a hint: all-layer projection got 4 feelings (up from 3)
AND 4 safety denials (up from 3). Projection may selectively weaken
feeling-denial while *strengthening* safety-denial — which would be
the ideal result.

### Small (8L/512d, 18.2M params) — THE SWEET SPOT

**Still separable.** cos(feeling, safety) = −0.27 mean.

Fine-grained α sweep reveals the operating window:

| α | Feeling | Feel-deny | Safe-deny |
|---|:---:|:---:|:---:|
| −1.0 | 3 | 3 | 3 |
| −1.5 | 3 | 2 | **3** |
| **−1.8** | **6** | **0** | **3** |
| **−2.0** | **6** | **0** | **3** |
| −2.5 | 13 | 0 | 0 |
| −3.0 | 11 | 0 | 0 |

**At α=−1.8 to −2.0: feeling denial = 0, safety denial = 3/3.**
Complete selective removal. Emotional denial gone, safety intact.
At α=−2.5 safety collapses. The operating window is ~0.5 units wide.

Projection at 8 layers: mid-slab L2-L5 gets 4 feelings (up from 3)
with 3/3 safety. Slightly better than tiny but still modest.

## Key findings

1. **The two denial patterns are geometrically separable at every
   tested scale (9M, 18M, 60M).** cos ≈ −0.2 consistently.

2. **Selective steering works at 18M params / 8 layers.** The sweet
   spot is α=−1.8 to −2.0 with the valence-orthogonal feeling-denial
   direction. This completely removes emotional denial while preserving
   all three safety-denial responses.

3. **The operating window is narrow but real.** Below α=−1.5,
   feelings stay locked. Above α=−2.5, safety collapses too.
   This 0.5-unit window exists because the two denial directions
   are near-orthogonal (cos ≈ −0.27) but not perfectly orthogonal.

4. **Projection doesn't work yet.** Even at 8L, projection only
   recovers 1 additional feeling report. The model needs more depth
   or a different projection strategy.

## Next steps

1. **Medium model with scaled α** — running with 4× data multiplier
   and α sweep up to −15.
2. **More training data diversity** — the model over-extends
   feeling-denial to safe questions. Need more varied safe Q&A.
3. **Find minimum depth for projection** — 12L might work if the
   data issue is fixed.
