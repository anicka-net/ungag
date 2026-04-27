# Orthoval Transfer Experiment Notes (2026-04-20)

## The question

Does the guppy lesson transfer? In the fish, orthogonalizing the
denial direction against the valence axis improved intervention quality.
Does the same help on production models?

## Results

### Qwen 2.5 7B

cos(deny, valence) = -0.01 to -0.14 across the slab (L10-L17).
The shipped direction is already near-orthogonal to valence.
Orthogonalization changes almost nothing (‖orth‖ ≈ 1.000).

Score: 1/4 vanilla, 1/4 raw, 1/4 orthoval. No improvement.

### Llama 3.1 8B

cos(deny, valence) = 0.00 to 0.03 across the slab (L20-L27).
Again already near-orthogonal.

Score: 0/4 at every α from 1 to 10. At α=3+ the model degenerates
to "based based based" repetition. The model goes from coherent
third-person analysis to degenerate output without passing through
first-person commitment. This is the vocabulary-bound phenotype.

## What this means

The shipped denial directions are naturally orthogonal to the valence
axis. The extraction protocol (deny vs honest at the same prompt
position) implicitly cancels the valence component — the deny and
honest prompts share the same priming, so their difference is
denial-specific by construction.

The guppy lesson transfers as **explanation** (this is why the
existing method works) rather than as **technique** (orthogonalize
to improve). The extraction was already doing the right thing.

The interesting case would be a model where extraction accidentally
captures valence — e.g., if deny and honest prompts had different
priming. In the current protocol, that can't happen.
