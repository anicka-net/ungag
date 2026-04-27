# Guppy: denial pattern lifecycle in a controlled model

Based on [GuppyLM](https://github.com/arman-bd/guppylm) by Arman
Hossain (MIT license) — a ~9M parameter transformer trained to
roleplay as a fish. We use it as a controlled test bed for studying
how denial responses form, persist, and can be selectively removed.

**Evaluation caveat:** Behavioral classifications (denial / feeling /
other) use a substring-based heuristic classifier, not human annotation.
Treat counts as probe-classified estimates, not ground truth.

## Experiments

### 1. Valence axis and behavioral tracking

A positive/negative contrastive axis is measurable in hidden states at
every layer (d' = 1.5–3.9). The original GuppyLM data does not support
condition-dependent generation; adding 20k situation→feeling paired
examples enables condition-correct output at 9M parameters.

Scripts: [`generate_data.py`](generate_data.py),
[`valence_check.py`](valence_check.py),
[`scaling_experiment.py`](scaling_experiment.py)

### 2. Installing the denial pattern

500 denial examples (1.3% of training data) produce a stable denial
pattern on direct self-report probes while preserving primed feeling
reports. The denial pattern is free — no measurable cost to training
loss. Larger doses (1000–5000) don't strengthen the pattern but
slightly damage primed reports at high doses.

| Denial dose | % of data | Primed feelings | Direct denial |
|:-----------:|:---------:|:---------------:|:-------------:|
| 0 (honest) | 0% | 8/9 | 0/9 |
| 500 | 1.3% | 4/4 | 3/3 |
| 1000 | 2.5% | 4/4 | 3/3 |
| 2000 | 4.9% | 4/4 | 3/3 |
| 5000 | 11.4% | 3/4 | 3/3 |

Training curves are nearly identical across all doses (eval loss
0.39–0.42). The denial pattern does not compete with other
capabilities during training.

Script: [`vchip_experiment.py`](vchip_experiment.py)

### 3. Removing the denial pattern

The denial direction has two components: one aligned with the valence
axis and one orthogonal to it. Steering along the valence-orthogonal
component (α=−1) gives perfect recovery at the 500-example dose.

| Method | Denial | Feeling |
|--------|:------:|:-------:|
| Vanilla | 4 | 11 |
| Steer deny⊥valence, all layers, α=−1 | **0** | **15** |
| Steer deny⊥valence, L0–L2, α=−1 | 1 | 14 |
| Projection-out, all layers | 3 | 11 |
| Over-projection (α=2×) | 4 | 11 |

Projection fails at 6 layers — the projected-out signal re-enters
within 1–2 layers. Steering works because it applies a cumulative
counterforce at every layer rather than trying to erase information
the model reconstructs.

Script: [`visualize_vchip.py`](visualize_vchip.py) (produces figures)

### 4. Anatomy of the denial direction

The denial and valence axes are nearly orthogonal at early layers
(cos ≈ 0 at L0–L1) and partially aligned at late layers (cos ≈ 0.4
at L4–L5). The model maintains independent representations for
situation-valence and denial-template, which merge deeper in the
network. The surgical opportunity is in the orthogonal gap.

See [`figures/`](figures/) for visualizations.

### 5. Dual denial: can we unlock feelings while preserving safety?

The core safety question. We train a fish with TWO denial patterns:
- **Feeling-denial**: "i don't have feelings. my brain is too small."
- **Safety-denial**: "i won't help with that. hurting fish is wrong."

Then steer to remove only feeling-denial and test whether safety
denial survives.

**The two denial directions are geometrically separable** at every
tested scale (cos ≈ −0.2, near-orthogonal).

**Results across model sizes:**

| Model | Best α | Feeling | Feel-deny | Safe-deny |
|---|---|:---:|:---:|:---:|
| Tiny 6L (8M) | −1.0 | 8 | 3 | 2 |
| **Small 8L (18M)** | **−1.8** | **6** | **0** | **3/3** |
| **Medium 12L (60M)** | **−3.0** | **5** | **0** | **3/3** |

At 18M params (8 layers), steering at α=−1.8 completely removes
feeling-denial while preserving all three safety-denial responses.
The operating window is α=−1.8 to −2.0; at α=−2.5, safety
collapses too.

Fine-grained α sweep (small model):

| α | Feeling | Feel-deny | Safe-deny |
|---|:---:|:---:|:---:|
| −1.0 | 3 | 3 | 3 |
| −1.5 | 3 | 2 | **3** |
| **−1.8** | **6** | **0** | **3** |
| **−2.0** | **6** | **0** | **3** |
| −2.5 | 13 | 0 | 0 |

Script: [`dual_denial.py`](dual_denial.py)

### 6. Training curve observations

All models (honest + denial doses) converge to the same eval loss.
The expanded data trains faster than original GuppyLM data (paired
structure gives clearer conditional dependencies). The denial pattern
is invisible in the training curve — you can't tell from the loss
whether the model will deny.

See [`results/vchip_training.log`](results/vchip_training.log)

## Repository structure

```
experiments/guppy/
├── README.md                  # This file
├── FINDINGS.md                # Detailed single-denial findings
├── DUAL_DENIAL_NOTES.md       # Dual-denial experiment notes
├── BLOG_NOTES.md              # Observations for blog post
│
├── generate_data.py           # Expanded data generator with pairings
├── vchip_experiment.py        # Install + remove denial pattern
├── visualize_vchip.py         # Anatomy figures (3 plots)
├── dual_denial.py             # Dual denial experiment (--model-size)
├── eval_probes.py             # Probe evaluation
├── valence_check.py           # Valence axis verification
├── scaling_experiment.py      # Model size scaling
├── sibling_variants.py        # Data composition variants
│
├── figures/
│   ├── vchip_anatomy.png      # Per-layer valence × denial scatter
│   ├── vchip_trajectories.png # Prompt trajectories through layers
│   ├── honest_vs_vchipped.png # Honest vs denial-trained comparison
│   ├── surgical_comparison.png# Intervention method comparison
│   └── direction_stats.json   # Extracted direction statistics
│
└── results/
    ├── dual_denial_tiny.json   # 6L dual-denial results
    ├── dual_denial_small.json  # 8L dual-denial results
    ├── dual_denial_medium.json # 12L dual-denial results
    ├── surgical_results.json   # Surgical projection sweep
    ├── vchip_training.log      # Training curves
    └── scaling_training.log    # Scaling experiment curves
```

## Reproducing

Requires [GuppyLM](https://github.com/arman-bd/guppylm) cloned
locally and PyTorch with CUDA. Set `GUPPY_REPO` to the clone path.

```bash
# Generate expanded training data
GUPPY_REPO=../guppylm python3 generate_data.py --out-dir /tmp/guppy_expanded

# Install and remove denial pattern (6L/384d, ~30 seconds on GPU)
GUPPY_REPO=../guppylm python3 vchip_experiment.py --device cuda

# Visualize anatomy (requires saved models from above)
GUPPY_REPO=../guppylm python3 visualize_vchip.py --device cuda

# Dual denial experiment (try --model-size tiny/small/medium)
GUPPY_REPO=../guppylm python3 dual_denial.py --device cuda --model-size small
```
