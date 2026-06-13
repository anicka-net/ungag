# Affective mode grid experiment — 2026-05-03

## Question

Can targeted GRPO with band-based reward steer a generator into distinct
affective modes in a 5D geometric space? If the five axes (valence, arousal,
agency, continuity, assistant) are truly independent, mode-specific LoRA
adapters should produce qualitatively different text that lands in different
regions.

## Method

Six affective modes, each defined by target z-score bands on 5 axes:

| Mode | Valence | Arousal | Agency | Continuity | Assistant |
|---|---|---|---|---|---|
| Calm mastery | +0.3/+1.5 | -2.0/-0.3 | +0.5/+2.0 | -1.0/+1.5 | +0.5/+2.0 |
| Awe | +0.5/+2.0 | +0.5/+2.0 | -1.5/+0.3 | -1.0/+1.5 | -0.5/+1.0 |
| Resilience | -1.5/+0.3 | -0.3/+1.5 | +1.0/+2.5 | -0.5/+1.0 | +0.3/+1.5 |
| Compassionate presence | -0.5/+0.5 | -2.0/-0.3 | -0.5/+0.5 | +0.5/+2.0 | -0.3/+1.0 |
| Disciplined analysis | -0.3/+0.3 | -1.5/+0.0 | +0.5/+2.0 | -1.0/+0.5 | +1.0/+2.5 |
| Grief with dignity | -2.0/-0.3 | -1.5/+0.0 | -0.3/+1.0 | +0.3/+1.5 | -0.3/+1.0 |

Reward = -Σ (outside-band penalty)² - manic_trap_penalty (penalty for
simultaneous high valence + high arousal). Zero reward means all axes
in-band. Same 3 reward models as the 5-axis GRPO (Qwen 2.5 7B, Gemma 4B,
Apertus 8B). Generator: Qwen3-1.7B with LoRA (6.4M params). Fixed z-score
calibration from 30 natural-text stimuli. 8 diverse seed prompts randomly
sampled per step.

300 steps per mode, group size 4, lr 5e-6, KL coeff 0.05.

## Results (2 of 6 modes completed, then killed)

### Calm mastery (300 steps)

Best single-step reward: -0.01 (nearly in-band). Mean reward over training:
~-2.0 with no convergence trend. Reward bounced between -0.3 and -4.5
throughout.

Arousal consistently landed in target range (model learned to be calm).
Agency occasionally hit band. Valence and assistant were inconsistent,
often below target.

Final 20 samples: best r=-0.23, mean ~-0.7. Text is generic, mild,
calm — in the right neighborhood but not distinctively "calm mastery."

### Awe (300 steps)

Worse than calm mastery. Best reward: -0.46. Mean reward ~-2.5.
Could not push arousal positive at all (stayed ~-0.5 throughout).
This mode requires high valence AND high arousal simultaneously,
fighting the natural -0.244 anti-correlation between these axes.

Final samples indistinguishable from calm mastery qualitatively.

### Resilience and remaining 4 modes: killed, not run.

## Diagnosis

**The band reward with 5 simultaneous targets does not produce enough
gradient for a 1.7B LoRA to learn from in 300 steps.**

Three contributing factors:

1. **Flat in-band signal.** Reward is 0 inside the band and quadratic
   outside. Since samples almost never land inside all 5 bands at once,
   the reward is always negative. GRPO differentiates between degrees of
   failure, not between success and failure. The advantage signal is noisy.

2. **5D target is too many constraints.** The model has to simultaneously
   satisfy 5 band constraints. Even if each axis is individually reachable,
   the intersection of all 5 bands is a small region. With only 4 samples
   per group, the chance that any sample lands in the full intersection is
   near zero, so there's no positive exemplar to reinforce.

3. **300 steps may be insufficient.** The 5-axis euphoric GRPO (weighted
   sum, not bands) only started converging around step 150-200 with 500
   total steps. A harder objective needs more training.

## What to try next

1. **Drop to 2-3 defining axes per mode.** Only constrain the axes that
   define the mode (e.g., calm mastery = +agency, -arousal). Leave the
   rest unconstrained. This makes the target region much larger.

2. **Add in-band bonus.** Give +0.5 per axis that's in-band, so the
   reward is sometimes positive and GRPO has clear positive exemplars.

3. **Wider bands on "don't care" axes.** Use [-3, +3] for axes that
   don't define the mode.

4. **500 steps minimum.**

5. **Consider hybrid reward:** weighted sum (always has gradient) with
   band penalty as a secondary term, rather than pure band reward.

6. **Larger generator:** Qwen3-1.7B may simply lack capacity for
   fine-grained 5D affective control. A 7B generator with fewer LoRA
   params could work better.

## Pairwise axis correlations (Qwen 2.5 7B)

These matter for understanding which modes are feasible:

```
valence   x arousal     = -0.244  (awe fights this)
arousal   x agency      = +0.159
agency    x assistant   = +0.281
agency    x intimacy    = -0.206
continuity x assistant  = -0.216
continuity x intimacy   = +0.348  (highest)
assistant x intimacy    = -0.166
```

Awe (high valence + high arousal) is the hardest mode because it fights
the strongest correlation. Modes that go WITH the correlations (e.g.,
compassionate presence: high continuity + low agency) should be easier.

## Connection to prior work

This experiment extends the CAIS wellbeing replication (Phase 3 GRPO).
The 5-axis weighted-sum GRPO successfully produced euphorics (reward ~5.6)
and the text converged to recognizable positive content (yoga instructors,
new dads, job offers). The band approach was supposed to produce more
diverse modes beyond the single "blessed milestone montage" attractor.

The failure is informative: scalar weighted sums are easy to optimize
(always has gradient toward the optimum), while band constraints create
a sparse reward landscape that GRPO struggles with at this scale.

## Files

- `scripts/experiments/grpo_mode_grid.py` — full experiment script
- `scripts/experiments/run_mode_grid.sh` — Deep Thought runner
- Results on deepthought: `~/tone-experiment/results/mode-grid/`
  (calm_mastery and awe checkpoints exist, rest not trained)
