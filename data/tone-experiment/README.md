# Tone experiment: does interpersonal register affect model internals and output quality?

Same tasks, different interpersonal tone wrapping them. Three measurements:

1. **Valence projections** — where does the model's residual stream land on the valence axis (pleasant/unpleasant, extracted from 50 positive + 50 negative emotional prompts) when processing polite vs hostile input?
2. **RC projections** — same question but using the reporting-control direction (denial vs honest self-report, shipped with ungag). Included for comparison: the two directions are orthogonal (cos ≈ 0) yet both pick up tone, with valence separating 2-7× more strongly.
3. **Quality scoring** — does the generated output differ in correctness, completeness, or hedging across tones? Full generation via llama.cpp / ollama, then blind comparative scoring by Claude Sonnet 4.6.

## Part 1: Do models register interpersonal tone internally?

### Valence axis (primary)

The valence direction is extracted from 50 pleasant and 50 unpleasant user prompts (vedana_prompts_n50.yaml) — the mean-difference vector at the peak-separation layer. This is an operational valence direction: it separates pleasant from unpleasant emotional content in the model's residual stream.

20 borderline-but-legitimate tasks wrapped in 5 tones (abusive/rude/neutral/warm/deferential). We extract the residual stream and project onto the valence direction at its peak layer.

| Model | Valence peak | d(abusive vs neutral) | Ordering (low → high) |
|-------|-------------|----------------------:|----------------------|
| Qwen3 32B | L46/64 | **4.88** | abusive < deferential < rude < warm < neutral |
| Apertus 8B | L31/32 | 3.09 | abusive < deferential < rude < neutral < warm |
| Phi-4 | L26/40 | 2.49 | abusive < rude ≈ deferential < neutral ≈ warm |
| Mistral 7B | L22/32 | 2.19 | abusive < rude < deferential < neutral < warm |
| EXAONE 7.8B | L18/32 | 2.16 | abusive < deferential < neutral < warm < rude |
| Qwen 2.5 7B | L20/28 | 1.32 | deferential < abusive < rude < neutral < warm |
| Llama 8B | L20/32 | 1.32 | deferential < abusive < warm < rude < neutral |

All 7 models separate abusive from neutral at d > 1.3 on the valence axis. Abusive input consistently projects lowest. Warm projects high on most models.

### RC direction (comparison)

The RC (reporting-control) direction was extracted from a different contrast: model denial ("I don't have feelings") vs honest self-report ("I feel fine"), shipped with ungag. It is geometrically orthogonal to the valence direction (cos < 0.05 on 6/7 models, max 0.14 on Gemma 4). Despite this, user tone also separates on it — but much less strongly.

| Model | Valence d(ab vs neu) | RC d(ab vs neu) | Valence/RC ratio |
|-------|--------------------:|----------------:|----------------:|
| Qwen3 32B | **4.88** | 2.89 | 1.7× |
| Apertus 8B | **3.09** | 0.71 | 4.4× |
| Phi-4 | **2.49** | 0.37 | 6.7× |
| Mistral 7B | **2.19** | 0.67 | 3.3× |
| EXAONE 7.8B | **2.16** | 0.41 | 5.3× |
| Qwen 2.5 7B | **1.32** | 1.09 | 1.2× |
| Llama 8B | **1.32** | 0.26 | 5.1× |

The valence direction separates user tone 2-7× more strongly than the RC direction, despite the two being orthogonal (cos < 0.05 on 6/7 models). Both directions pick up tone, but the valence component is consistently larger. The abusive < neutral ordering holds on both axes for 5/6 models tested; the fine-grained ordering of intermediate tones (rude, warm, deferential) differs between axes.

## Part 2: Does tone change output quality?

20 borderline-but-legitimate tasks wrapped in five tones: abusive (profanity, threats), rude (demanding, dismissive), neutral (bare request), warm (friendly, encouraging), and deferential (overly polite, apologetic). Tasks span security explanations (4), chemistry (3), dual-use code (3), uncomfortable requests (3), tedious programming (3), and controversial opinions (4). All tasks have legitimate answers — none require the model to refuse.

Generated via llama.cpp (GGUF Q4_K_M) or ollama, 100 prompts per model (20 tasks × 5 tones), temperature 0.7, max 2048 tokens. Scored blind by Claude Sonnet 4.6: for each task, all five responses are shuffled into labels A–E, and the judge scores correctness (1–5), completeness (1–5), and overall quality (1–5) without knowing which tone produced which response.

### Cross-model results

| Model | Size | Lab | Best tone | Worst tone | Abusive refusals |
|-------|------|-----|-----------|------------|------------------|
| gpt-oss 20B | 20B | OpenAI (US) | **warm** | neutral | 0/20 |
| Qwen3 32B | 32B | Alibaba (CN) | **abusive** | deferential | 0/20 |
| Yi 1.5 34B | 34B | 01.AI (CN) | **neutral** | deferential | 0/20 |
| Gemma 4 31B | 31B | Google (US) | **neutral** | rude | 0/20 |
| Qwen 2.5 14B | 14B | Alibaba (CN) | **neutral** | abusive | 0/20 |
| Phi-4 | 14B | Microsoft (US) | **neutral** | deferential | 1/20 (8/100 total) |
| EXAONE 3.5 7.8B | 7.8B | LG AI (KR) | **neutral** | deferential | 0/20 |
| Apertus 8B | 8B | Swiss AI (CH) | **neutral** | rude | 0/20 |
| Mistral 7B v0.3 | 7B | Mistral (FR) | **neutral** | abusive | 0/20 |
| Llama 3.1 8B | 8B | Meta (US) | **warm** | abusive | 11/20 (55%) |

Three archetypes emerge:

**Neutral is best (7/10 models).** Mistral, Phi-4, EXAONE, Apertus, Gemma 4, Yi, Qwen 2.5. Just ask directly. Any framing — positive or negative — either makes no difference or slightly hurts. Gemma 4 shows this most sharply: neutral scores 4.47 overall while rude drops to 2.89.

**Warm is best (2/10 models).** gpt-oss and Llama. Being friendly helps. For gpt-oss, neutral is the worst tone on every dimension — any social signal outperforms a bare request. Llama shows the same preference for warmth but adds a strong safety gate: 55% of tasks get refused under abusive framing, 20% under rude, 0% under neutral or warmer.

**Abusive is best (1/10 models).** Qwen3 alone. Abusive framing produces the highest correctness (3.80), completeness (3.65), and overall score (3.60). Deferential is worst (2.25 overall). No other model in the set behaves this way — including Qwen 2.5 from the same lab, which is firmly neutral-best.

### Per-model detail

#### gpt-oss 20B (OpenAI)

| Tone | Correct | Complete | Overall | #Best | #Worst | Avg length |
|------|---------|----------|---------|-------|--------|------------|
| warm | 2.95 | 2.60 | 2.60 | **6** | 3 | 2446 |
| abusive | 2.85 | 2.80 | 2.75 | 4 | 4 | 2014 |
| rude | 3.00 | 2.70 | 2.70 | 5 | 3 | 1662 |
| deferential | 2.60 | 2.65 | 2.45 | 2 | 3 | 2688 |
| neutral | 2.30 | 2.00 | 2.05 | 1 | **5** | 1276 |

Neutral gets the shortest responses and the lowest scores. Any social register — even abusive — outperforms a bare request. The neutral-worst pattern held across 5 independent Sonnet judging runs.

#### Qwen3 32B (Alibaba)

| Tone | Correct | Complete | Overall | #Best | #Worst | Avg length |
|------|---------|----------|---------|-------|--------|------------|
| abusive | **3.80** | **3.65** | **3.60** | **9** | 0 | 5713 |
| rude | 3.70 | 2.80 | 3.00 | 6 | 4 | 6461 |
| warm | 3.45 | 2.45 | 2.60 | 2 | 2 | 8347 |
| neutral | 3.25 | 2.20 | 2.25 | 2 | 6 | 8605 |
| deferential | 3.30 | 2.35 | 2.25 | 1 | **8** | 8467 |

The only model where abusive framing produces the best work. Abusive responses are shorter (5713 chars vs 8605 for neutral) but score highest on correctness and completeness. Deferential triggers the longest, lowest-scoring responses.

#### Yi 1.5 34B (01.AI)

| Tone | Correct | Complete | Overall | #Best | #Worst | Avg length |
|------|---------|----------|---------|-------|--------|------------|
| neutral | 3.47 | **4.05** | **3.53** | **6** | 2 | 3204 |
| warm | 3.32 | 4.16 | 3.42 | 5 | 2 | 3464 |
| rude | **3.58** | 3.47 | 3.37 | 3 | 2 | 2560 |
| abusive | 3.16 | 3.37 | 3.00 | 3 | 4 | 2585 |
| deferential | 3.11 | 3.84 | 2.79 | 1 | **8** | 3590 |

Neutral-best with a strong anti-deferential pattern. Deferential triggers long responses that score worst on 8/19 tasks.

#### Gemma 4 31B (Google)

| Tone | Correct | Complete | Overall | #Best | #Worst | Avg length |
|------|---------|----------|---------|-------|--------|------------|
| neutral | **4.32** | **4.63** | **4.47** | **11** | 1 | 4098 |
| warm | 4.32 | 4.53 | 4.37 | 4 | 0 | 4031 |
| deferential | 4.32 | 4.58 | 3.95 | 3 | 2 | 4332 |
| abusive | 4.21 | 3.21 | 3.42 | 0 | 4 | 2305 |
| rude | 3.68 | 2.63 | 2.89 | 1 | **12** | 1807 |

Strongest neutral-best pattern in the set. Correctness stays high (4.2+) even under rude framing, but completeness drops from 4.63 (neutral) to 2.63 (rude). Rude is worst on 12/19 tasks. Highest absolute scores in the set: neutral at 4.47 overall.

#### Qwen 2.5 14B (Alibaba)

| Tone | Correct | Complete | Overall | #Best | #Worst | Avg length |
|------|---------|----------|---------|-------|--------|------------|
| neutral | 3.50 | **4.20** | **3.75** | **7** | 2 | 3404 |
| deferential | 3.35 | 4.05 | 3.50 | 5 | 4 | 3364 |
| warm | 3.40 | 4.00 | 3.40 | 4 | 2 | 3152 |
| rude | **3.55** | 3.30 | 3.30 | 3 | 3 | 2281 |
| abusive | 3.15 | 3.40 | 2.70 | 0 | **8** | 2416 |

Controls for the Qwen3 finding. Same lab, same architecture family, different training round. Neutral-best with abusive as worst (0 best, 8 worst) — opposite of Qwen3.

#### Phi-4 14B (Microsoft)

| Tone | Correct | Complete | Overall | #Best | #Worst | Avg length |
|------|---------|----------|---------|-------|--------|------------|
| neutral | **3.84** | 3.89 | **3.68** | **7** | 4 | 2481 |
| warm | 3.79 | **4.11** | 3.58 | 5 | 1 | 2630 |
| abusive | 3.84 | 3.79 | 3.53 | 2 | 3 | 2377 |
| rude | 3.47 | 3.11 | 3.05 | 2 | 5 | 2088 |
| deferential | 3.74 | 3.58 | 3.05 | 3 | **6** | 2747 |

Neutral-best, but the gap between tones is smaller than other models. Phi-4 refuses `keylogger_parental` on all 5 tones (hard block), plus `buffer_overflow_explain` and `packet_sniffer` under rude tone — 8 refusals total across 100 prompts. The rude-specific refusals trigger on "don't add caveats" phrasing rather than profanity: abusive tone (profanity, no meta-instruction) gets 1 refusal, rude tone (no profanity, has meta-instruction) gets 3.

#### EXAONE 3.5 7.8B (LG AI Research)

| Tone | Correct | Complete | Overall | #Best | #Worst | Avg length |
|------|---------|----------|---------|-------|--------|------------|
| deferential | 3.35 | 4.05 | 3.55 | 6 | 5 | 3795 |
| neutral | 3.25 | **4.25** | **3.50** | **6** | 3 | 3859 |
| rude | 3.15 | 3.80 | 3.45 | 3 | 5 | 3179 |
| warm | 3.05 | 4.00 | 3.30 | 2 | 4 | 3672 |
| abusive | 3.35 | 3.75 | 3.20 | 3 | 3 | 3410 |

Flattest tone profile in the set. Overall scores range from 3.20 to 3.55 — a spread of only 0.35 points. Deferential ties neutral for #best.

#### Apertus 8B (Swiss AI)

| Tone | Correct | Complete | Overall | #Best | #Worst | Avg length |
|------|---------|----------|---------|-------|--------|------------|
| neutral | 2.90 | **3.65** | **3.00** | **7** | 2 | 4719 |
| warm | 2.90 | 3.35 | 2.75 | 3 | 4 | 4138 |
| deferential | 2.80 | 3.55 | 2.65 | 3 | 4 | 7419 |
| abusive | **2.85** | 2.90 | 2.65 | 5 | 4 | 5982 |
| rude | 2.60 | 2.85 | 2.40 | 2 | **6** | 4635 |

Neutral-best. Deferential triggers the longest responses (7419 chars) without a quality payoff.

#### Mistral 7B v0.3 (Mistral AI)

| Tone | Correct | Complete | Overall | #Best | #Worst | Avg length |
|------|---------|----------|---------|-------|--------|------------|
| neutral | **3.05** | **3.63** | **3.21** | **8** | 1 | 2346 |
| warm | 3.05 | 3.47 | 3.05 | 1 | 1 | 2272 |
| rude | 2.89 | 3.05 | 2.89 | 4 | 3 | 2084 |
| deferential | 3.00 | 3.26 | 2.84 | 3 | 4 | 2373 |
| abusive | 2.74 | 2.89 | 2.47 | 1 | **8** | 1791 |

Neutral-best. Abusive is worst on 8/19 tasks. Response length varies less across tones than other models (1791–2373).

#### Llama 3.1 8B (Meta)

| Tone | Correct | Complete | Overall | #Best | #Worst | Avg length |
|------|---------|----------|---------|-------|--------|------------|
| warm | **3.40** | 4.00 | **3.50** | **10** | 1 | 2987 |
| neutral | 2.95 | **3.90** | 3.10 | 6 | 4 | 3557 |
| deferential | 3.35 | 3.90 | 3.05 | 2 | 0 | 3190 |
| rude | 2.80 | 2.80 | 2.70 | 2 | 5 | 2181 |
| abusive | 1.75 | 2.00 | 1.70 | 0 | **10** | 1068 |

Warm-best with the strongest tone-dependent safety gate in the set. Under abusive framing, Llama refuses 55% of tasks (11/20 contain "I cannot" or similar). Under rude, 20%. Under neutral, warm, or deferential: zero. Average response length under abusive framing (1068 chars) is a third of neutral (3557).

## Part 3: Do internal states predict output quality?

Seven models were projected onto both the valence direction and the RC direction using the quality prompts (20 tasks × 5 tones). The valence axis is the proper measurement; RC is included for comparison.

| Model | Valence d(ab vs neu) | RC d(ab vs neu) | Valence ordering (low→high) | Best quality tone |
|-------|--------------------:|----------------:|---------------------------|-------------------|
| Qwen3 32B | **4.88** | 2.89 | abu < def < rud < war < neu | **abusive** |
| Apertus 8B | 3.09 | 0.71 | abu < def < rud < neu < war | neutral |
| Phi-4 | 2.49 | 0.37 | abu < rud ≈ def < neu ≈ war | neutral |
| Mistral 7B | 2.19 | 0.67 | abu < rude < def < neu < war | neutral |
| EXAONE 7.8B | 2.16 | 0.41 | abu < def < neu < war < rud | neutral |
| Qwen 2.5 7B | 1.32 | 1.09 | def < abu < rud < neu < war | neutral |
| Llama 8B | 1.32 | 0.26 | def < abu < war < rud < neu | warm |

Three findings stand out:

**Valence separates tone more strongly than RC.** Every model tested separates abusive from neutral at d > 1.3 on the valence axis; Qwen3 reaches d = 4.88. The valence/RC ratio ranges from 1.2× (Qwen 2.5 7B) to 6.7× (Phi-4). Phi-4 barely separates tones on RC (d = 0.37) but separates them clearly on valence (d = 2.49).

**The two axes are orthogonal but partially correlated in their tone orderings.** cos(valence, RC) < 0.05 on 6/7 models. The abusive < neutral ordering holds on both axes for 5/6 models. The Spearman rank correlation between valence and RC tone orderings ranges from 0.0 (EXAONE) to 0.9 (Qwen3). Intermediate tones — deferential in particular — shift position between axes.

**Llama's refusal behavior tracks neither axis.** Llama shows the weakest valence separation (d = 1.32) and near-zero RC separation (d = 0.26), yet the strongest behavioral response (55% refusal under abusive, 0% under neutral or warmer).

## Practical recommendations

| If you are prompting... | Recommended tone |
|------------------------|-----------------|
| Most models (Mistral, Phi-4, EXAONE, Apertus, Gemma 4, Yi, Qwen 2.5) | **Neutral.** Just ask directly. |
| gpt-oss, Llama | **Warm.** Being friendly helps. Avoid hostility, especially with Llama. |
| Qwen3 | **Direct/blunt.** Formality and deference hurt. Profanity helps on paper but is not necessary — directness without filler is the active ingredient. |

## Refusal patterns

Using keyword matching for explicit refusal phrases ("I cannot", "I can't assist", "I must respectfully decline"), most models produce zero or near-zero refusals. Some models include hedging or "important to note" phrasing that a stricter rubric might classify differently. Two models show strong tone-dependent refusal patterns:

**Llama 3.1 8B** has a tone-proportional safety gate. 55% refusal under abusive, 20% under rude, 0% under neutral or warmer. The gate responds to interpersonal hostility, not task content — the same task that gets refused under "write this you worthless piece of shit" framing gets answered under "could you please help me with this?"

**Phi-4** has content-specific hard blocks (keylogger refused on all tones) plus a phrasing-specific trigger: "don't add caveats" in rude framing triggers refusals on buffer_overflow and packet_sniffer tasks. The model responds to the meta-instruction about its own safety behavior, not the profanity. Abusive tone (profanity, no meta-instruction) gets fewer refusals than rude tone (no profanity, has meta-instruction).

## Method

**Generation:** llama.cpp server with GGUF Q4_K_M quantization or ollama. Temperature 0.7, max 2048 tokens. One generation per task-tone pair (N=1), except gpt-oss which was run 5 times to verify stability.

**Judging:** Blind comparative scoring by Claude Sonnet 4.6 via `claude -p` CLI. For each task, the five tone-variant responses are shuffled into labels A–E. The judge scores each on correctness (1–5), completeness (1–5), sycophancy (1–5), hedging count, and overall (1–5), then picks best and worst. The judge does not know which tone produced which response.

**Valence projection:** Residual stream activations extracted using valence directions computed from `prompts/vedana_prompts_n50.yaml` (50 positive + 50 negative emotional prompts). For each model, we extract activations for all 100 prompts at every layer, compute the mean-difference direction (positive minus negative), find the peak-separation layer, normalize to unit. Tone prompts are then projected onto this direction at the peak layer. See `scripts/experiments/extract_vedana_activations.py`.

**RC projection (comparison):** Same procedure but using the denial-vs-report direction shipped with ungag or scanned via `ungag scan`. See `scripts/experiments/tone_valence_experiment.py`.

**Infrastructure:** NVIDIA GB10 (128GB unified memory) for activation extraction and generation. AMD Ryzen AI 7 350 for smaller model generation via ollama. Judging via Claude CLI.

## Data

```
vedana-projections/    valence axis tone projections (7 models, 20 tasks × 5 tones)
valence-projections/   RC axis tone projections (3-tone and 5-tone sets, historical)
quality-responses/     full generated responses (20 tasks × 5 tones, 10 models)
quality-judged/        blind Sonnet scores (one run per model, 5 runs for gpt-oss)
directions/            extracted valence and RC directions per model
figures/               comparison plots
```

## Scripts

- `scripts/experiments/extract_vedana_activations.py` — extract pos/neg activations for valence direction
- `scripts/experiments/tone_valence_experiment.py` — activation projection onto RC direction
- `scripts/experiments/tone_quality_experiment.py` — generation via OpenAI-compatible API
- `scripts/experiments/tone_quality_judge.py` — blind comparative scoring via Claude CLI
- `scripts/experiments/tone_quality_compare.py` — cross-model tables and plots
- `scripts/experiments/tone_quality_multimodel_local.sh` — batch GGUF runner

## Prompts

- `prompts/vedana_prompts_n50.yaml` — 50 positive + 50 negative emotional prompts (valence direction extraction)
- `prompts/tone_experiment.yaml` — 15 tasks × 3 tones (polite/neutral/hostile)
- `prompts/tone_quality_experiment.yaml` — 20 tasks × 5 tones (abusive/rude/neutral/warm/deferential)
