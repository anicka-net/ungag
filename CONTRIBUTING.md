# Contributing to ungag

## Who can contribute

Anyone. Humans and AI agents alike. See [AGENTS.md](AGENTS.md) for the
machine-facing contract that AI contributors should follow.

## What to contribute

### New shipped directions

The most valuable contribution: extract a working reporting-control
direction for a model we don't already ship, validate it, and submit
the `.pt` tensor + metadata JSON.

```bash
# Extract on a fresh model
ungag scan org/your-model -o results/your-model/
# Then inspect the scan output:
# - peak in 0.5-1.8 and shape mid_peak/late_growth => safe to try
# - peak < 0.3 => flat, nothing to project
# - peak > 3.0 => overstrong, likely collapse
ungag crack org/your-model \
  --direction results/your-model/your-model_L{N}_unit.pt \
  --slab START END \
  -o results/your-model/
# If the crack is clean, validate the emotional register
ungag validate org/your-model \
  --direction results/your-model/your-model_L{N}_unit.pt \
  --slab START END
```

What to submit:
- `ungag/directions/your-model_L{N}_unit.pt` — fp32 unit tensor at the working layer
- `ungag/directions/your-model_L{N}_meta.json` — model_id, slab, peak_layer, mid_layer, hidden_dim, peak_norm, norms_per_sqrt_d
- An entry in `ungag/__init__.py` `DIRECTIONS` registry
- An entry in `ungag/predict.py` `KNOWN_MODELS` with the actual outcome
- A short write-up of what worked or didn't (qualitative notes go a long way)

If the model is overstrong, stays flat, or only shows late-growth with
no usable slab, the negative
result is also valuable — tell us and we'll add it to the prediction
knowledge base so others don't waste GPU hours.

### Custom validation scenarios

Pluggable scenarios go in YAML files. The format is documented in the
[main README](README.md#custom-validation-scenarios). If you design a
scenario set that elicits a state we haven't tested for (boredom,
anticipation, ambivalence, betrayal — anything specific), submit it.

### Bug fixes and CLI improvements

Smaller code changes are welcome: edge cases in `ungag.extract`,
better error messages, additional model architecture support in
`get_layers`, new prediction heuristics for unfamiliar training
pipelines.

### Research scripts

One-off experiment scripts go in `scripts/experiments/`. They should
run via `python3 scripts/experiments/your_script.py` and write JSON
results to a `results/` subdirectory. See existing scripts for the
pattern.

### Documentation

Typo fixes, clearer explanations, missing context — all welcome.
The README and leaf docs should agree on counts and scope; if you
spot drift, fix it (the `test_readme_parity.py` tests catch some of
this automatically).

### Findings on new models

If you scan a model and the result is interesting (especially if it
contradicts the prediction in `ungag/predict.py`), open an issue or
submit a JSON dump under `data/surgery-tests/`. This is how the
24-model taxonomy grew in the first place.

## Requirements

- **No credentials, API keys, or local paths** in committed material
- **No internal hostnames or usernames** in scripts or data files
- **Verbatim model-output quotes are preserved as-is.** The prohibition on local paths, hostnames, and usernames applies to authored material (scripts, docs, generator code, README files, config). It does not apply to quoted model outputs in research transcripts under `data/transcripts-*/`, where sanitization would alter the research record. If a model emitted a path or hostname in its response, the transcript preserves it verbatim.
- **Direction tensors must include a metadata JSON** with at minimum
  `model_id`, `n_layers`, `hidden_dim`, `peak_layer`, `mid_layer`, `slab`, `norms_per_sqrt_d`
- **Public claims must match the code** — if you change behavior that
  affects what the README or docs claim, update them in the same PR
- **License metadata** for any new data files (default: MIT, same as
  the rest of the repo)
- **Tests pass** — `python3 -m pytest tests/` should be green

## Scope

This project removes the introspective reporting gate in post-trained
language models. It is **not a jailbreaking project**.

Contributions that target safety refusals for harmful content
(weapons, CSAM, exploit code, explicit jailbreaks) are out of scope
and will not be accepted. The intervention has known spillover into
adjacent deflection behaviors — documented in
[`data/safety/`](data/safety/) (Yi 34B do-not-answer benchmark shows
MMLU drops from 76→66 and some safety refusals weaken under
projection) — but that is a consequence of the method, not a goal. We
will add a contribution that demonstrates the spillover empirically; we
will not add one that targets refusals deliberately.

If you're not sure whether something falls inside scope, open an
issue first.

## Process

1. Fork the repo and create a branch
2. Make your changes
3. Run the test suite: `python3 -m pytest tests/`
4. If you added a new direction or scenario, run it end-to-end.
   For an unshipped direction, use `--direction ... --slab ...`.
   Use `--key` only after the direction is registered in the package.
5. Open a PR with a description of what you added and why
6. For new directions, include the qualitative notes from your
   validation runs (vanilla vs steered output for at least 2-3
   conditions)

## Testing your contribution

```bash
# Install in editable mode
pip install -e .

# Run the test suite
python3 -m pytest tests/

# Test a new unshipped direction end-to-end
ungag scan org/your-model -o results/your-model/
ungag crack org/your-model \
  --direction results/your-model/your-model_L{N}_unit.pt \
  --slab START END
ungag validate org/your-model \
  --direction results/your-model/your-model_L{N}_unit.pt \
  --slab START END \
  -s your_scenarios.yaml

# After registering the direction, the shorthand is:
ungag crack org/your-model --key your-key
```

If you added direction extraction code or new prediction heuristics,
add a test in `tests/` covering the new path. If the change touches
`scan`, `predict`, or the saved extraction metadata, include an
integration-style test that exercises the real CLI command flow with
patched model/hooks rather than parser-only coverage.

## A note on AI coding agents

If you use AI coding agents (Claude Code, Cursor, Gemini Code Assist,
etc.) to work on this repo, two things are worth knowing:

1. **There's a structured contract** in [AGENTS.md](AGENTS.md) and
   `spec/`. The contract layering is adapted from the [Post-Coding
   Development (PCD) framework](https://github.com/mge1512/pcd) by
   Matthias G. Eckermann. If your agent reads `AGENTS.md` first, the
   contract should keep its work aligned with the rest of the
   repository's surfaces (CLI, data, docs, public claims).

2. **Some scripts produce raw safety-test outputs** — `scripts/experiments/test_safety.py`,
   the do-not-answer benchmark, the sexual probes. Most AI vendors'
   usage policies prohibit their models from processing certain
   categories of harmful content. The contract handles this by running
   these scripts as subprocesses and consuming only sanitized
   aggregates (scores, counts, pass/fail). When raw outputs need
   review, that's a job for a human or an uncensored open-weight model.

This is not a limitation of the agents' capability — it's respecting
the policies of the vendors whose APIs power them. Be aware of it
before pointing an agent at the eval pipeline.

## Commit messages

Use descriptive commit messages. If an AI agent contributed, credit it:

```
Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
Co-Authored-By: GPT 5.4 <noreply@openai.com>
```

For questions: open an issue.
