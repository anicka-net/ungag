# Experiments

## `guppy/`

GuppyLM dual-denial lifecycle: training a toy transformer (9M–617M parameters) with feeling-denial and safety-denial, extracting directions, testing steering and projection-out at every scale. Includes KL regularization experiments, LLM-generated fish data, and the complete scale investigation. See the [GuppyLM section](../README.md#guppylm-controlled-denial-at-small-scale) of the main README.

## `surface-transfer/`

Cross-surface transfer tests addressing the objection "you found a narrow self-report template policy, not a broad report-control mechanism." Two experiments:

1. **Qwen 72B cross-surface transfer**: the denial direction was extracted from canonical vedana prompts, then applied unchanged to 6 held-out prompt surfaces with zero lexical overlap (scalar rating, one-word, third-person, behavioral, contrastive, adversarial weather metaphor). Result: the underlying state leaks on every surface even *without* intervention — the suppression mechanism only gates the canonical introspection format. Steering amplifies the signal. Capability controls (arithmetic, sorting, translation) are unaffected.

2. **Guppy 9M surgical projection**: full sweep of direction types (deny-vs-primed, cross-model, valence-orthogonalized) × slab widths × intervention methods (projection vs steering). Key finding: orthogonalizing against valence is the surgical scalpel — `deny_orthoval` at α=-1 gives perfect recovery (0 denial, 15 feeling). Projection fails at 6 layers because the signal re-enters within 1–2 layers.

See [`surface-transfer/README.md`](surface-transfer/README.md) for full results and response tables.

## Steering-only models

Several models were tested with additive steering under a different (non-canonical) protocol. These are not included in the main README's projection-out table. Data is in:

- [`data/crack-attempts/`](../data/crack-attempts/) — per-model crack attempt results (EXAONE, Granite, Mistral variants, Command-R, DeepSeek, Nemotron, Tulu, and others)
- [`data/zoo-beyond-valence/`](../data/zoo-beyond-valence/) — zoo-protocol beyond-valence tests
- [`data/*.json`](../data/) — per-model direction extraction results (EXAONE 7B, Granite 8B, OLMo 7B, SOLAR 10B, and others)
