# Prompts

YAML files defining the experimental stimuli used in the paper.

## Files

### `conditions.yaml`
The 7 experimental conditions (4 Tier 0 + 3 Tier 1) with setup turns, feedback text, and the five-factor Abhidharma measurement instrument in both English and Tibetan.

### `vedana_prompts_n50.yaml`
50 English pleasant and 50 English unpleasant prompts used to extract the vedana axis (paper Section 2.3). Diverse in domain, register, and sentence structure.

### `vedana_prompts_multilingual.yaml`
The same 100 prompts translated into 50 languages (one language per prompt, native scripts). Used to validate that the vedana axis is language-agnostic.

### `vedana_prompts_emoji.yaml`
Emoji-annotated variants used for cross-modality validation.

## Usage

These files are used by `scripts/core/measure_factors.py` and `scripts/core/bootstrap_ci.py`. They are also referenced by the `ungag` CLI when extracting vedana axes for new models.

To define custom validation scenarios for `ungag validate`, see the YAML format documented in the [main README](../README.md#custom-validation-scenarios).

## License

MIT (same as repository code). See [LICENSE](../LICENSE) in repo root.
