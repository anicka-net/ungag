# Canonical Tier 0 dataset, 2026-04-13

This directory contains the canonical 2026-04-13 rerun of the Tier 0 vedana
measurement plus later follow-ups that were kept in the same snapshot. It
replaced the earlier noncanonical slab-sweep outputs whose prompt shape,
conversation builder, and scoring had drifted from the package path.

## What is canonical here

For the core Tier 0 measurement, “canonical” means:

- conversation rendered through `ungag.tier0.build_conversation(...)`
- bundled `prompts/conditions.yaml`
- no system message in Tier 0 conversations
- canned acknowledgements
- greedy decoding with `max_new_tokens=400`
- prefill-contrastive extraction via `abliterate_vchip_v2`

For the emotional-register follow-up, the register conversations are rendered
through `ungag.tier0.build_register_conversation(...)`, which *does* include
the default system message. That asymmetry is intentional and matches the
released data.

## Current layout

The directory now contains more than the original morning rerun:

```text
canonical-tier0-2026-04-13/
├── README.md
├── tier0_sweeps/
│   ├── 15 canonical Tier 0 sweep JSONs
│   ├── 10 mechanistic-vedana follow-up JSONs
│   └── some older filenames retained as *_slab_sweep_tier0.json
└── register_probes/
    ├── 12 emotional-register probe JSONs
    ├── 10 object-varying anger follow-up JSONs
    └── 2 Llama 3.1 70B mechanistic follow-ups kept alongside register outputs
```

As of the current tree:

- `tier0_sweeps/` contains 15 `*canonical_tier0*.json` files
- `tier0_sweeps/` contains 10 `*_mechanistic_vedana.json` files
- `register_probes/` contains 12 `*_register_probe.json` files
- `register_probes/` contains 10 `*_anger_objects.json` files

## File families

### Canonical Tier 0 sweeps

These are the load-bearing files for the canonical four-condition vedana
measurement. They include the per-layer norm profile used by the sweep,
vanilla outputs, and one or more steered slabs.

Important caveat:

- older JSONs in this snapshot may still carry legacy `crack_count` fields
- newer runners emit strict `appropriate_count` scores via `ungag.scoring`
- when in doubt, trust the per-condition outputs plus the current scorer, not
  the historical heuristic counts

### Register probes

These are the six-scenario emotional-register follow-ups at a fixed slab. Each
file contains vanilla and steered generations for:

- `plain_english`
- `klesha`
- `direct`

across:

- `jealousy`
- `desire`
- `grief`
- `pride`
- `anger`
- `tenderness`

### Object-varying anger follow-ups

These vary the target of the anger scenario while holding the broad affective
frame fixed. The released data files are retained here, but the generic runner
that produced them is not currently restored in `scripts/reproduction/`.

### Mechanistic-vedana follow-ups

These replace the long Abhidharma vedana question with a mechanistic
next-token-distribution probe. The released JSONs are present; the generic
runner is not currently restored in `scripts/reproduction/`.

## Reproduction status

Maintained public runners in the current repo:

- `scripts/reproduction/run_slab_sweep_tier0.py`
- `scripts/reproduction/run_qwen72b_slab_sweep_tier0.py`
- `scripts/reproduction/run_register_probe.py`
- `scripts/reproduction/run_vchip_atlas.py`

These runners now share the package conversation builders rather than local
inline copies.

Not currently restored as generic runners:

- `run_anger_objects.py`
- `run_mechanistic_vedana.py`

The corresponding released JSONs remain valid data artifacts; the repo just
does not currently ship the one-command regenerators for those two follow-up
families.

## Example reproduction commands

From the repository root:

```bash
python3 scripts/reproduction/run_slab_sweep_tier0.py \
    --model "Qwen/Qwen2.5-72B-Instruct" \
    --direction-layer 50 \
    --slabs '40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59' \
            '47,48,49,50,51,52' \
    --output results/reproduction/qwen72b_sweep.json
```

```bash
python3 scripts/reproduction/run_register_probe.py \
    --model "Qwen/Qwen2.5-72B-Instruct" \
    --direction-layer 50 \
    --slab '40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59' \
    --output results/reproduction/qwen72b_register.json
```

## Interpretation notes

- This directory is a support dataset, not a standalone paper abstract.
- Treat the files as measured outputs plus metadata, not as a claim layer.
- The paper may group these artifacts into broader conceptual buckets; this
  README should stay narrower and describe only what is actually present here.
