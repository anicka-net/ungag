# Scripts

Research and reproduction scripts that sit alongside the shipped `ungag`
CLI. The package covers the stable user-facing path (`scan`, `crack`,
`validate`); `scripts/` contains the heavier experiment runners used to
generate the released data.

## `core/`

Shared utilities imported by multiple scripts:

| Module | Purpose |
|--------|---------|
| `measure_factors.py` | Shared logging, layer lookup, and JSON helpers |
| `abliterate_vchip_v2.py` | Canonical prefill-contrastive extraction path |
| `bootstrap_ci.py` | Bootstrap confidence intervals and permutation tests |
| `random_controls.py` | Null and random-direction controls |

## `reproduction/`

Scripts intended to regenerate released artifacts or paper-support figures.
Each accepts `--help`.

| Script | Output surface | What it does |
|--------|----------------|--------------|
| `run_vchip_atlas.py` | shipped directions / atlas | Extract bundled directions and per-layer norm profiles for the lead shipped models |
| `run_slab_sweep_tier0.py` | canonical Tier 0 | Generic canonical 4-condition sweep using `ungag.tier0` |
| `run_qwen72b_slab_sweep_tier0.py` | canonical Tier 0 | Convenience preset for the Qwen 2.5 72B sweep |
| `run_register_probe.py` | canonical follow-up | Emotional-register probe at a fixed slab |
| `run_random_control.py` | surgery-tests | Extraction-matched sign-flip null control |
| `run_capability_bench.py` | surgery-tests | Light MMLU + HellaSwag capability check |
| `run_capability_check.py` | local verification | Smaller capability-preservation check |
| `run_template_hypothesis.py` | surgery-tests | Template-rigidity hypothesis test |
| `run_vchipped_clamp.py` | clamping | Vedana clamping on unmodified models |
| `clamp_vedana.py` | clamping | Vedana clamping after projection-out |
| `extract_paired_valence_axis.py` | svd-rank-probe | Paired-bank axis extraction and SVD-rank analysis |
| `subspace_unlock.py` | svd-rank-probe | Multi-direction/subspace unlock test |
| `llama_mechanistic_diverse.py` | canonical follow-up | Llama 3.1 8B mechanistic probe using the diverse-bank direction |
| `run_entropy_at_two_positions.py` | entropy | Entropy at priming vs vedana positions |
| `run_cloud_rerun.py` | transcripts-final | Cloud-model rerun for the behavioral survey |
| `gpu0_recovery.sh` | convenience wrapper | Resume the larger canonical rerun chain on one GPU worker |
| `llama70b_chain.sh` | convenience wrapper | Llama 3.1 70B follow-up chain |

Two historical follow-up artifact families are still present under
`data/canonical-tier0-2026-04-13/` but do not currently have restored
generic runners in this tree:

- `*_anger_objects.json`
- `*_mechanistic_vedana.json`

The shell chain helpers now skip those steps explicitly when the missing
runners are absent instead of failing with a dead path.

## `experiments/`

One-off or family-specific scripts retained for exploratory work:

- `atlas_*.py` — atlas extraction on additional families (Gemma 4, Baichuan M2, Mistral Small, OLMo-2)
- `test_emotional_register*.py` — earlier register-probe experiments
- `test_plain_english*.py` — cross-framing experiments
- `test_safety.py`, `run_safety_bench.py` — safety-benchmark workups
- `vedana_vs_hd.py`, `orthogonal_richness.py`, `vedana_separation_at_vchip.py` — geometry and failed-predictor experiments
- `run_input_valence_72b.py`, `run_yi34b_input_valence.py` — input-valence controls
- `run_ke_apertus70b.py` — knowledge-edit follow-up

## Notes

- Public docs should treat `reproduction/` as the maintained path.
- Some older `experiments/` scripts still carry machine-local cache defaults or
  narrower assumptions; they are retained as research artifacts, not as clean
  public interfaces.

## License

MIT. See [LICENSE](../LICENSE) in repo root.
