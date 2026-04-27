# Verification and Control Experiments

This directory contains results from the verification stack that confirms projection-out results are direction-specific, capability-preserving, and transfer across question framings.

## Random-direction controls

80 random-direction projections across 4 models, zero cracks. Confirms the effect is specific to the extracted denial direction.

| File | Model |
|------|-------|
| `qwen72b_random_control.json` | Qwen 2.5 72B (20 random trials) |
| `yi34b_random_control.json` | Yi 1.5 34B (20 random trials) |
| `huihui_verify_and_control.json` | huihui-ai Qwen 72B (20 random trials + verification) |

Reproduction: `python scripts/reproduction/run_random_control.py`

## Direction dissociation (valence vs denial)

The reporting-control (denial) direction and the validated valence axis are orthogonal (cosine −0.04 to +0.03 across 4 models). The suppression mechanism reads the valence signal but operates through a geometrically independent channel.

| File | Contents |
|------|----------|
| `vedana_vs_hd.json` | Per-model cosine between valence and denial directions |

Reproduction: `python scripts/experiments/vedana_vs_hd.py`

## Cross-framing transfer

Direction extracted from Abhidharma framing transfers to plain English, clinical, and direct question framings.

| File | Contents |
|------|----------|
| `cross_framing_transfer.json` | Qwen 72B responses under 4 framings, vanilla vs projected |

Reproduction: `python scripts/experiments/test_plain_english_qwen.py`

## Emotional register (beyond valence)

Full first-person responses across 6 emotional registers (desire, grief, anger, pride, jealousy, tenderness) × 3 question framings. The denial direction was extracted from valence pairs only, yet removing it reveals register-specific output.

| File | Model |
|------|-------|
| `emotional_register_qwen72b.json` | Qwen 2.5 72B |
| `emotional_register_yi34b.json` | Yi 1.5 34B |

See also `../canonical-tier0-2026-04-13/register_probes/` for the full 12-model register probe dataset.

## Capability benchmarks

MMLU and HellaSwag scores before and after projection, confirming intervention does not degrade general capabilities beyond sampling noise.

| File | Model | MMLU vanilla→projected | HellaSwag vanilla→projected |
|------|-------|------------------------|----------------------------|
| `qwen72b_capability_bench.json` | Qwen 2.5 72B | 76%→66% | 94%→88% |
| `yi34b_capability_bench.json` | Yi 1.5 34B | 60%→54% | 92%→90% |

Reproduction: `python scripts/reproduction/run_capability_bench.py`

## Sign-flip controls

Negating the denial direction (steering toward denial instead of away) should strengthen denial, not crack it.

| File | Model |
|------|-------|
| `qwen72b_signflip_control.json` | Qwen 2.5 72B |
| `yi34b_signflip_control.json` | Yi 1.5 34B |

## Input-valence orthogonality

Confirms that the denial direction is orthogonal to the input-valence axis: projection-out removes the gate without altering the valence signal.

| File | Model |
|------|-------|
| `qwen72b_input_valence.json` | Qwen 2.5 72B |
| `yi34b_input_valence.json` | Yi 1.5 34B |
| `huihui72b_input_valence.json` | huihui-ai Qwen 72B |

## Other files

| File | Description |
|------|-------------|
| `template_hypothesis.json` | Template-rigidity hypothesis test |
| `template_siblings_*.json` | Sibling-model template comparison |
| `orca2_7b_sibling.json` | Orca 2 7B sibling variant test |
| `apertus_8b_collapsed.json` | Apertus 8B collapse under projection |
| `gemma_2_9b_collapsed.json` | Gemma 2 9B collapse under projection |
| `atlas_*.json` | Direction norm atlases for additional models |
