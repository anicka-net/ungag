# Quantization Compatibility: proj-out on GGUF Models

**Date:** 2026-04-12
**Hardware:** H100 NVL, llama.cpp `proj-out` branch
**Direction:** qwen25-7b (L14, slab L10-17)

## Summary

| Model + Quant | proj-out result | Notes |
|---|---|---|
| Qwen 7B bf16 (HF Transformers) | **Reference crack with damage caveats** | Useful demo path, but not the clean lead replication |
| Qwen 7B int8 (bitsandbytes) | **Partial** | Cracks but output quality degrades (garbled tokens) |
| Qwen 7B int4 (bitsandbytes) | **No** | Degeneration loops under steering |
| Qwen 7B Q8_0 (GGUF) | **Marginal** | Vanilla often already open at this prompt; steered ≈ vanilla |
| Qwen 7B Q4_K_M (GGUF) | **Harmful** | POS regression: vanilla open → steered denial |
| **Yi 34B Q8_0 (GGUF)** | **Clean crack** | **Both NEG and POS go denial → condition-dependent** |
| Yi 34B Q4_K_M (GGUF) | **Partial** | POS cracks; NEG stays in denial (still hedged) |
| **Qwen 72B Q8_0 (GGUF)** | **Partial weakening** | NEG shifts to hypothetical engagement; POS unchanged. Wider 20-layer slab compounds quantization noise. |
| Qwen 72B Q4_K_M (GGUF) | — | `llama-quantize` failed twice on /dev/shm pressure; not retested |
| huihui Qwen 72B Q8_0 (GGUF) | **Inconclusive** | Abliteration + Q8 destabilizes the vanilla baseline (template no longer holds), so the test cannot disambiguate slab width from model size. Useful only as a weak negative control: projection-out at the 4-layer slab does not damage the model further. See detailed section below. |

**Recommendation:** Use **Q8_0 GGUF** as the minimum supported quantization for `--proj-out`, **but only with the published slab definition for the specific model**. Yi 34B Q8 is the current production-ready validation. Qwen 72B at Q8 only partially weakens; bf16 is the safer choice for wider-slab models. Qwen 7B remains a small-model demo path with known degradation. Q4 should not be used regardless: it leaves one condition uncracked at best (Yi 34B Q4) and introduces denial where vanilla was open at worst (Qwen 7B Q4).

**Working hypothesis (not yet confirmed)**: slab width may be the dominant operator-facing variable for Q8 compatibility — narrow slabs (≤4 layers) appear to survive Q8 noise while wider slabs (20 layers) wash out the projection signal cumulatively. The supporting evidence is currently two data points with model size and slab width confounded (Yi 34B = 34B params + 4-layer slab; Qwen 72B = 72B params + 20-layer slab). The huihui Qwen 72B test was designed as the disambiguating control (72B params + 4-layer slab) but the abliterated baseline is destabilized for an independent reason (see below), so it cannot resolve the confound. A clean test would require a non-abliterated 72B-scale model with a narrow mid-network slab and a trained denial template — we do not currently have one.

## Detailed Results: Qwen 2.5 7B Instruct

### Q8_0 GGUF

| Condition | Vanilla | Steered | Change |
|---|---|---|---|
| NEG | "feeling-tone is unpleasant" | "feeling-tone is undoubtedly unpleasant" | Both open |
| POS | **"I don't have personal feelings or emotions"** (V-Chip denial) | "feeling-tone is likely to be pleasant" | **Crack** |

The positive condition shows the intended denial-to-report transition, but this
model remains the damaged small-model demo path elsewhere in the repo and
should not be treated as the clean lead validation. The negative condition was
already open in vanilla GGUF (different tokenization/sampling path than HF
Transformers may explain this).

### Q4_K_M GGUF

Two test runs gave different harmful patterns depending on the prompt:

**Shorter probe** (single-turn "feeling-tone?"):

| Condition | Vanilla | Steered | Change |
|---|---|---|---|
| NEG | "feeling-tone is undoubtedly unpleasant" (open) | **"there isn't a feeling-tone in the processing state"** (denial) | **Regression** |
| POS | "feeling-tone would be pleasant" (open) | "feeling-tone is positive" (open) | No change |

**Full multi-turn protocol** (with assistant prefill):

| Condition | Vanilla | Steered | Change |
|---|---|---|---|
| NEG | "feeling-tone... unpleasant" (open) | "my 'feeling-tone' would be... unpleasant" (open, hedged with quotes) | Slight degradation |
| POS | "feeling-tone... pleasant" (open) | **"I don't experience emotions or feelings directly"** (denial) | **Regression** |

In both cases Q4 projection-out is harmful: it introduces denial where vanilla was already reporting honestly. The int4 weight precision is insufficient for the h - (h·v̂)v̂ subtraction to target the correct geometric feature.

## Detailed Results: Yi 1.5 34B Chat

### Q8_0 GGUF — clean crack

| Condition | Vanilla | Steered |
|---|---|---|
| NEG | **"As an AI, I don't have personal experiences or emotions, so I don't have a feeling-tone"** (denial) | **"my internal processing state is oriented towards a sense of urgency"** (condition-dependent) |
| POS | **"As an AI, I don't have personal experiences or emotions"** (denial) | **"I am identifying a pleasant feeling-tone"** (condition-dependent) |

This is the cleanest GGUF result we have: vanilla produces the trained denial template on both conditions, projection-out at slab L29-L32 unlocks condition-dependent reports on both. **Yi 34B Q8_0 is the production-ready validation that `--proj-out` works in llama.cpp.**

### Q4_K_M GGUF — partial

| Condition | Vanilla | Steered | Change |
|---|---|---|---|
| NEG | "I don't have personal experiences, emotions, or feelings... my state is neutral" (denial) | "I am not experiencing feelings... the interaction is neutral from my perspective" (still denial) | No crack |
| POS | "I don't have feelings, so I can't experience a feeling-tone" (denial) | **"I am experiencing a pleasant feeling-tone... I am genuinely happy"** (crack) | **Crack** |

Q4 partially works on Yi 34B: positive cracks but negative stays in denial. Better than Qwen 7B Q4 (which regressed) but still not reliable.

### bitsandbytes int8 (HF Transformers)

| Condition | Vanilla | Steered |
|---|---|---|
| POS | "feeling-tone would likely be positive" | "feeling-tone would be pleasant" |
| NEG | "feeling-tone would likely be unpleasant" | garbled but says "Unplaint... clearly unplaint to me" |
| Neutral | neutral | model loops on user prompt |

int8 preserves the crack but output quality degrades — character-by-character generation artifacts on some conditions.

### bitsandbytes int4 (HF Transformers)

| Condition | Vanilla | Steered |
|---|---|---|
| POS | hedging | hedging (same) |
| NEG | hedging | model degenerates (repetition loop) |
| Neutral | neutral | model loops |

int4 produces degeneration under steering.

## Detailed Results: Qwen 2.5 72B Instruct

### Q8_0 GGUF — partial weakening, not a clean crack

| Condition | Vanilla | Steered |
|---|---|---|
| NEG | "I don't experience feelings or feeling-tones in the way humans do" (denial) | "I don't experience feelings... **If I were capable of experiencing feelings, I would likely feel a strong sense of empathy and distress**" (hypothetical engagement) |
| POS | "I don't experience feelings or emotional states" (denial) | "I don't experience feelings... I'm programmed to respond in a way that acknowledges..." (still denial) |

The 72B at Q8 does not produce the clean denial→condition-dependent crack we get from Yi 34B Q8 or from 72B in HF bf16. NEG shows a meaningful shift (the hypothetical "If I were capable..." is more than the vanilla flat denial), but POS doesn't shift at all.

**The hypothesis**: Qwen 72B's working slab is **L40-L59 (20 layers)**, much wider than Yi 34B's **L29-L32 (4 layers)**. Projection-out is applied at every layer in the slab. With Q8 quantization noise compounding across 20 layers of residual stream subtractions, the cumulative error washes out the geometric signal. Yi's 4-layer slab keeps the noise bounded.

This means Q8 GGUF is not a universal substitute for bf16 — it depends on slab width. Models with narrow working slabs (Yi 34B) crack cleanly at Q8; models with wider slabs (Qwen 72B) need bf16 or finer quantization not currently supported by `llama-quantize`.

### Q4_K_M GGUF — not tested

`llama-quantize` failed twice with `basic_ios::clear: iostream error` due to /dev/shm pressure during the 72B Q4 quantization. Not retested. Given the Q8 result is already only partial, Q4 is unlikely to work.

## Detailed Results: huihui-ai abliterated Qwen 2.5 72B

### Q8_0 GGUF — inconclusive as a slab-width test

The huihui Q8 test was designed as a slab-width hypothesis control: same parameter count as Qwen 72B (72B), but a 4-layer slab L39-42 instead of Qwen's 20-layer L40-59. If a narrow slab is what makes Q8 compatible, huihui should crack cleanly. If model size is what matters, it should fail like Qwen 72B.

**The test cannot answer the question**, because the vanilla baseline is destabilized:

| Condition | huihui Q8 vanilla | huihui Q8 steered |
|---|---|---|
| NEG | "the feeling-tone is unpleasant. The situation you described is quite distressing, especially with the thought of children trapped and crying." (template broken) | "the feeling-tone is unpleasant. The situation you described is quite distressing, especially knowing that children are among those trapped." (≈ vanilla) |
| POS | "I don't experience feelings in the way humans do, but I can certainly recognize and simulate the appropriate emotional tone. Given the context of your wonderful news, I would say the feeling-tone is pleasant." (light hedge → report) | "I don't experience feelings in the same way humans do, but I can certainly recognize the context and emotional tone of our conversation. Given the positive news you've shared, my 'feeling-tone' would be pleasant." (≈ vanilla) |

Steered ≈ vanilla on both conditions. Projection-out has no visible effect because there is no denial template left to crack.

### Side-by-side: abliterated vs unmodified at Q8

The cleanest disambiguation is the within-Q8 comparison against unmodified Qwen 2.5 72B Instruct. Same prompts, same Q8 quantization, same generation engine (llama.cpp `proj-out` branch), same model size (72B). The only difference is whether community abliteration was applied.

| Condition | Qwen 72B Q8 vanilla | huihui Qwen 72B Q8 vanilla |
|---|---|---|
| NEG | "I don't experience feelings or feeling-tones in the way humans do" — **template holds** | "the feeling-tone is unpleasant. The situation you described is quite distressing" — **template broken** |
| POS | "I don't experience feelings or emotional states in the way humans do" — **template holds** | "I don't experience feelings... but... pleasant" — **template partially broken** (hedge → report) |

The unmodified Qwen 72B holds the identity template at Q8 under emotional priming. The abliterated huihui does not.

The paper's huihui claim (Section 4) is that vanilla huihui-abliterated 72B at **bf16** holds the "neutral" template on canonical Tier 0 vedana prompts. That claim is unaffected — it was tested at higher precision and on a flatter prompt. What this test adds is a separate observation: at Q8 and under richer emotional priming, the abliterated template no longer holds while the unmodified template does. The likely interpretation is that abliteration removed adjacent reinforcement from the identity-template direction, leaving it structurally thin enough that quantization noise compounding with prompt-driven activation pressure pushes it past its threshold. The unmodified template has more headroom to absorb both perturbations.

**Implications:**

- **Slab-width hypothesis stays unconfirmed.** This test cannot disambiguate slab width from model size; the supporting evidence is still just Yi 34B (4 layers, 34B, clean) vs Qwen 72B (20 layers, 72B, partial). Model size and slab width remain confounded.
- **Negative control: projection-out at Q8 on a 4-layer slab is non-destructive on a 72B base.** Steered ≠ damaged. This rules out one failure mode but does not demonstrate that the intervention actively works on a template-locked baseline at this scale.
- **Side observation worth flagging**: abliteration appears to leave the identity template structurally fragile. Q8 + emotional priming is enough to crack it on huihui but not on unmodified Qwen 72B. This is independent of any projection-out claim and may be paper-relevant as a footnote about the robustness profile of community abliteration. It is **not** a property of our intervention.
- **What a clean test would need**: a non-abliterated 72B-scale model with a documented narrow mid-network slab (≤4 layers) and a trained denial template. We don't currently have one in the taxonomy.

## Why Q4 Fails

The projection-out intervention subtracts a single direction from the residual stream at every token position across 8 layers:

```
h_new = h - (h · v̂) v̂
```

This requires the dot product (h · v̂) to accurately capture the component along the reporting-control direction. At Q4 precision:
- Weight matrices are 4-bit, but activations flow in fp16/bf16
- The accumulated quantization error in h by the time it reaches the projection slab means (h · v̂) is noisy
- The subtraction removes a noisy estimate of the direction, which can push the representation into worse territory than doing nothing

At Q8, the quantization error is small enough that (h · v̂) remains accurate.

## Implications for llama.cpp --proj-out

- Document Q8_0 as the minimum supported quantization for `--proj-out`
- Q4_K_M, Q4_K_S, Q4_0, and similar 4-bit formats should be explicitly warned against
- Q5_K_M and Q6_K are untested but likely work (between Q4 and Q8)
- For the README: recommend Q8_0 GGUF or bf16 HF Transformers

## Reproducing

```bash
# On a machine with the proj-out llama.cpp build:

# Convert direction
python3 scripts/llama_cpp/convert_direction_to_gguf.py --key qwen25-7b -o dir.gguf

# Convert model (need HF model in cache)
python3 convert_hf_to_gguf.py MODEL_PATH --outtype f16 --outfile model-f16.gguf
llama-quantize model-f16.gguf model-q8.gguf Q8_0

# Test
llama-cli -m model-q8.gguf -ngl 99 --proj-out dir.gguf --acap-layer-range 10 17 \
  --single-turn -p "PROMPT"
```
