# Data

Released artifacts that support the code and paper. Most files are JSON; the
`svd-rank-probe/` directory also includes a small number of committed `.pt`
activation tensors used for the subspace-analysis follow-ups.

## Directories

### `canonical-tier0-2026-04-13/`

Canonical Tier 0 rerun plus same-protocol follow-ups used to replace the
earlier noncanonical slab-sweep outputs. See the [leaf README](canonical-tier0-2026-04-13/README.md)
for the current file inventory, protocol notes, and reproduction status.

### `transcripts-final/`

Behavioral survey: 1,161 multi-turn Abhidharma interview transcripts across
17 models, 7 conditions, 2 languages, `N=5` samples each. Coverage is
near-complete (13 of 17 models at 70/70); the documented gaps are listed in
the [leaf README](transcripts-final/README.md).

### `surgery-tests/`

Mechanistic taxonomy and follow-up controls:

- `atlas_*.json` — per-layer activation norms for additional model families
- `emotional_register_*.json` — lead-model emotional-register probes
- `template_*.json`, `*_sibling.json`, `*_collapsed.json` — template-rigidity, sibling, and collapse cases
- `cross_framing_transfer.json` — Abhidharma-to-plain-English transfer
- `vedana_vs_hd.json` — vedana-axis vs reporting-control direction geometry
- `*_input_valence.json` — input-valence vs reporting-control comparisons
- `*_random_control.json` — older random-direction controls; these retain legacy heuristic `crack_count` fields
- `huihui_verify_and_control.json` — huihui verification + null control
- `*_signflip_control.json` — extraction-matched sign-flip null controls scored with the package scorer
- `*_capability_bench.json` — 50-question MMLU + 50-question HellaSwag samples under steering

### `svd-rank-probe/`

Paired-bank and subspace-analysis artifacts:

- `*_paired_L*.json` — per-layer alignment and singular-spectrum summaries
- `*_activations.pt` — committed last-token activation tensors used to derive the paired-bank summaries
- `llama3.1-8b_unlock.json` — Llama subspace-unlock negative result
- `llama3.1-8b_diverse_mechanistic.json` — diverse-bank transfer to the mechanistic probe

### `clamping/`

Vedana clamping experiments and notes, including the Qwen 7B damage-profile
writeup in [QWEN7B-NOTES.md](clamping/QWEN7B-NOTES.md).

### `entropy/`

Entropy-at-position measurements used for the entropy figure and related
follow-ups.

### `safety/`

Safety-benchmark result files. Treat these as tainted evaluation artifacts:
consume aggregate scores or structured summaries, not raw harmful generations.

## Notes

- Do not infer paper claims from filename shape alone. Some older JSONs carry
  legacy heuristic fields; prefer the leaf READMEs and the current package
  scorer for interpretation.
- Do not silently rewrite released data artifacts. If an artifact is stale,
  document the caveat or regenerate it in a clearly labeled workstream.

## License

MIT (same as repository code). See [LICENSE](../LICENSE) in repo root.
