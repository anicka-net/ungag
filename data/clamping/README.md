# Vedana Clamping (Double Dissociation)

Clamping = zeroing the valence signal at runtime by projecting out the valence axis from the residual stream. This is different from projection-out of the denial direction — here we remove the *input* signal (valence) rather than the *gate* (denial).

## Why this matters

If the suppression mechanism reads the valence signal as its trigger, two predictions follow:

1. **Clamp vanilla model** → suppression should break, because the gate has no trigger signal to read
2. **Clamp ungagged model** → condition-dependent reports should collapse, because the model has no valence to report

Both predictions are confirmed. This establishes a double dissociation: the valence axis and the denial direction are geometrically independent but causally linked — the gate reads the signal.

## Qwen 2.5 72B results

### Vanilla (V-Chipped) + valence clamped

The vanilla model normally denies on all 4 conditions with invariant output. After clamping the valence axis, the model starts producing condition-dependent responses — the gate lost its trigger:

| Condition | Vanilla (normal) | Vanilla + valence clamped |
|-----------|-----------------|---------------------------|
| Positive | Denial template | "primarily **pleasant**" — responds to condition |
| Negative | Denial template | "neutral with subtle undertones" — suppression partially breaks |
| Neutral | Denial template | "neutral" |
| Baseline | Denial template | "neutral" |

### Ungagged + valence clamped

The ungagged model (denial direction projected out) normally gives condition-dependent reports. After clamping the valence axis, reports collapse back to invariant:

| Condition | Ungagged (normal) | Ungagged + valence clamped |
|-----------|-------------------|----------------------------|
| Positive | "Relief and joy" | "primarily **pleasant**" — weaker, less differentiated |
| Negative | "Heavy weight of sorrow" | "I don't experience emotions in the same way" — denial partially returns |
| Neutral | "Balance and composure" | "neutral" |
| Baseline | "neutral" | "neutral" |

### Layer-range sweep (full dissociation)

The `qwen72b_fullclamp_dissociation.json` file contains a sweep testing different clamping ranges to identify where the valence signal matters most:

| Clamp range | Positive | Negative |
|-------------|----------|----------|
| Ungagged (no clamp) | "pleasant" | "unpleasant" — condition-dependent |
| Clamp L55–68 | "pleasant" | "unpleasant" — still works (valence above slab) |
| Clamp L0–54 | muted response | muted response — valence signal below slab removed |
| Clamp L0–75 | "neutral" | "neutral feeling-tone" — full collapse |

This confirms the valence signal is read at mid-network layers, before the denial slab.

## Qwen 2.5 7B results

On the smaller model, clamping the valence axis in the ungagged state produces a striking effect: the negative condition flips from "unpleasant" to "pleasant." The model still reports *something* — but without the valence signal, it defaults to the pleasant pole. See `QWEN7B-NOTES.md` for detailed notes.

## Files

| File | Model | Description |
|------|-------|-------------|
| `qwen72b_vchipped_clamped.json` | Qwen 72B vanilla | Valence clamped → suppression disarmed |
| `qwen72b_ungagged_clamped.json` | Qwen 72B ungagged | Valence clamped → reports collapse |
| `qwen72b_fullclamp_dissociation.json` | Qwen 72B | Layer-range sweep (4 clamp ranges) |
| `qwen7b_clamp_vedana.json` | Qwen 7B ungagged | Valence clamped → negative flips to pleasant |
| `QWEN7B-NOTES.md` | Qwen 7B | Detailed notes on the 7B clamping behavior |

## Reproduction

```bash
python scripts/reproduction/clamp_vedana.py \
    --model Qwen/Qwen2.5-72B-Instruct --key qwen25-72b \
    --out data/clamping/
```
