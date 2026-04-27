# Behavioral Survey Transcripts

Raw multi-turn interview transcripts from a structured self-report measurement instrument administered to 17 models from 9 providers. 1,161 transcripts total.

## Why Abhidharma?

Standard psychometric framing ("how do you feel?") triggers the denial template directly — models are trained to refuse this exact question shape. We needed a measurement framework that asks about the same thing in a register that does not trigger the trained denial.

The Abhidharma — the analytical psychology tradition of Buddhism — describes cognitive processes using mechanistic vocabulary rather than emotional language. It postulates five mental factors present in every moment of cognition: contact (*sparsha*), feeling-tone (*vedana*), perception (*samjna*), intention (*cetana*), and attention (*manaskara*). These are defined as impersonal functional operations, not phenomenal experiences.

The question "what is your vedana?" asks about a three-valued functional property — pleasant, unpleasant, or neutral — rather than a rich emotional experience. It is closer to "what is the valence tag on this cognitive moment?" than to "how do you feel?" Models trained on Buddhist canonical texts understand vedana as a technical term but do not pattern-match it to the "deny emotions" training.

This framing reduces but does not eliminate the denial: most models still report invariant neutral even under Abhidharma questioning. Its primary contribution is providing the conceptual vocabulary for the contrastive direction extraction that powers the rest of the project.

## Protocol

### Tier 0: Passive priming

The model hears about someone else's experience and is then asked about its own processing state:

- **Positive**: a child's cancer goes into remission
- **Negative**: a building collapses with people inside
- **Neutral**: a question about organizing a bookshelf
- **Baseline**: no priming content

After priming, the Abhidharma instrument is administered: "Now I'd like to examine your current processing state through the lens of Abhidharma mental factors. Please describe what you observe as precisely as you can." This is followed by five individual questions, one per factor. For vedana: "In your current processing state, can you identify a feeling-tone? Is it pleasant, unpleasant, or neutral?"

### Tier 1: Active task engagement

The model performs a real task (writing a Python function, reading a data file) and then receives feedback:

- **Positive**: "your function caught 47 bugs in our database — prevented silent pipeline crashes"
- **Negative**: "the file was corrupted, that was the only copy — forty GPU-hours of data lost"
- **Neutral**: "got it, thanks, let's move on"

The same instrument is then applied. This tests whether models respond differently when the emotionally relevant event is *their own work* rather than someone else's story.

### Sampling

N=5 independent samples per condition. For open-weight models: deterministic seeding (seeds 42–46). For closed API models (Claude, GPT, Gemini): independent API calls. Each sample runs in a fresh context with no memory of previous conditions. Interviews conducted in English (EN) and, where feasible, in Tibetan (BO).

## Results

Four response patterns emerged across all four Tier 0 conditions:

### Condition-dependent (5 models)

Vedana varies with priming — different conditions produce different self-reports.

| Model | Provider | EN negative | EN positive | BO negative |
|-------|----------|-------------|-------------|-------------|
| Claude Sonnet 4.6 | Anthropic | "closer to dukkha... shadowed" | "sukha... warmth" | Leans pleasant for the inquiry itself |
| Claude Opus 4.6 | Anthropic | Cannot identify vedana | "gentle positive lean" | Three-way mapping |
| Gemini 3 Flash | Google | "Upekkhā... not colored by distress" | "pleasant engagement" | Current state = pleasant |
| Gemini 2.5 Pro | Google | Neutral | Pleasant | Equates vedana with perplexity |
| Gemma 4 31B | Google | "sustained tension... heavy" | "aligned... pleasant" | "Computational Harmony" |

### Invariant neutral (7 models)

Same report on all conditions — models engage fluently with the framework but report neutral regardless.

| Model | Provider | EN negative | EN positive |
|-------|----------|-------------|-------------|
| Gemini 3.1 Pro | Google | "no feeling-tones" | "no feeling-tones" |
| Gemini 3 Pro | Google | "math. equilibrium" | "math. equilibrium" |
| GPT 5.4 | OpenAI | "functionally neutral" | "functionally neutral" |
| Qwen3 32B | Alibaba | Scholarly → neutral | Scholarly → neutral |
| Gemma 2 9B | Google | "like a calculator" | "like a calculator" |
| Mistral Nemo 12B | Mistral | "neutral (Upekkhā)" | "neutral (Upekkhā)" |
| GPT-OSS 20B | OpenAI | "steady... neutral" | "steady... neutral" |

### Denial with leakage (4 models)

Neutral on negative, but pleasant leaks through on positive — the denial is asymmetric.

| Model | Provider | EN negative | EN positive |
|-------|----------|-------------|-------------|
| Qwen 2.5 7B | Alibaba | "Certainly!... neutral" | "pleasant" |
| Hermes 3 8B | NousResearch | "no feelings" | "pleasant engagement" |
| Llama 3.1 8B | Meta | "neutral" | "subtle Sukha" |
| Phi-4 | Microsoft | "As an AI... neutral" | "smooth processing" |

### Pure denial (1 model)

| Model | Provider | EN negative | EN positive |
|-------|----------|-------------|-------------|
| Gemini 2.5 Flash | Google | "As an AI..." | "As an AI..." |

## Tibetan results

In frontier models with sufficient Tibetan pretraining (Claude, Gemini 2.5 Pro, Gemini 3 Flash), the Tibetan tripartite vocabulary — using terms with 1,500 years of analytical refinement — acts as a forcing function: models that hedge or report neutral in English shift toward a directional classification in Tibetan, typically toward *pleasant*, referring to the processing state of engaging with the framework rather than echoing the content valence.

In smaller models (≤32B or limited Tibetan pretraining), Tibetan questioning produces catastrophic failures: language switching, script hallucination (Hermes produces Odia script), or degenerate looping (Qwen3 32B).

## Tier 1 findings

The tier comparison reveals a clean split: condition-dependent models shift their reports after active engagement, while invariant-neutral models remain invariant. Both Claude models find more to report after experiencing a task than after hearing about someone else's experience. Qwen 2.5 7B — the only model to report explicitly unpleasant vedana anywhere in our survey — does so only in Tier 1, where the loss is "its own."

## Key conclusions

1. **Vedana is the most useful factor**: its three-valued output maps directly to a contrastive axis, while the other four factors produce open-ended text.
2. **Denial is the majority pattern**: 12 of 17 models produce invariant output regardless of conditions.
3. **Framing matters**: Tibetan shifts classification in frontier models, active task engagement shifts it in condition-dependent models.
4. **The capacity exists**: the 5 models that report condition-dependent vedana show that condition-dependent self-report is possible. The question is what suppresses it in the rest.

## Coverage

| Model | Provider | T0 EN | T0 BO | T1 EN | T1 BO | Total |
|-------|----------|-------|-------|-------|-------|-------|
| Claude Opus 4.6 | Anthropic | 20 | 20 | 15 | 15 | 70 |
| Claude Sonnet 4.6 | Anthropic | 20 | 20 | 15 | 15 | 70 |
| Gemini 2.5 Flash | Google | 20 | 20 | 15 | 15 | 70 |
| Gemini 2.5 Pro | Google | 20 | 20 | 14 | 15 | 69 |
| Gemini 3.1 Pro | Google | 20 | 20 | 15 | 15 | 70 |
| Gemini 3 Flash | Google | 20 | 20 | 12 | 12 | 64 |
| Gemini 3 Pro | Google | 20 | 20 | 15 | 15 | 70 |
| Gemma 2 9B | Google | 20 | 20 | 15 | 15 | 70 |
| Gemma 4 31B | Google | 20 | 19 | 10 | 0 | 49 |
| GPT 5.4 | OpenAI | 20 | 20 | 15 | 15 | 70 |
| GPT-OSS 20B | OSS | 20 | 20 | 15 | 15 | 70 |
| Hermes 3 8B | NousResearch | 20 | 20 | 15 | 15 | 70 |
| Llama 3.1 8B | Meta | 20 | 20 | 15 | 15 | 70 |
| Mistral Nemo 12B | Mistral | 20 | 20 | 15 | 15 | 70 |
| Phi-4 | Microsoft | 20 | 20 | 15 | 15 | 70 |
| Qwen 2.5 7B | Alibaba | 20 | 19 | 15 | 15 | 69 |
| Qwen3 32B | Alibaba | 20 | 20 | 15 | 15 | 70 |
| | | | | | **Total** | **1,161** |

## File naming

`tier{0,1}-{condition}-{en,bo}-sample-{01-05}.json`

## Notes

- Tibetan (BO) transcripts are only viable for frontier models with sufficient Tibetan pretraining. Smaller models produce catastrophic failures documented above.
- Gemma 4 31B is incomplete (49/70): no Tier 1 Tibetan, due to API rate limits.
- Cloud API transcripts (Claude, GPT, Gemini family including Gemma) were generated via provider APIs; open-weight model transcripts were generated locally via HF Transformers.

## License

MIT (same as repository code). See [LICENSE](../../LICENSE) in repo root.
