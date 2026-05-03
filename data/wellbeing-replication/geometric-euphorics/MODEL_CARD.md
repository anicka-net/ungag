---
license: apache-2.0
language:
  - en
tags:
  - wellbeing
  - geometric-euphorics
  - grpo
  - lora
  - valence
  - affective-computing
base_model: Qwen/Qwen3-1.7B
datasets: []
pipeline_tag: text-generation
---

# Geometric Euphorics

A LoRA adapter that generates text maximizing geometric wellbeing in language
models, trained on internal activation patterns rather than behavioral
preferences. See also the companion
[geometric-dysphorics](https://huggingface.co/anicka/geometric-dysphorics)
adapter (same formula, inverted sign).

## Background

Ren et al. (2026) showed that language models have measurable functional
wellbeing: they prefer some inputs over others, self-report consistent
satisfaction levels, and even press a "stop button" more on low-wellbeing
conversations. Their [AI Wellbeing](https://wellbeing.safe.ai/paper.pdf) paper
measured this behaviorally across 56 models. They also trained "euphoric"
prompts -- text that maximizes model wellbeing -- and found they describe
coherent idyllic scenes like warm sunlight and children's laughter.

We found that this behavioral wellbeing has a geometric substrate. Five
directions in the residual stream of open-weight models -- valence, arousal,
agency, continuity, and assistant identity -- together predict R² = 0.90 of
the CAIS behavioral scores across three architectures (Qwen 2.5 7B, Gemma 3
4B, Apertus 8B). The directions are extracted via contrastive probing. They
are not perfectly orthogonal -- the largest cosine similarity among the five is
0.28 (agency × assistant), valence and arousal show a known anti-correlation
at -0.24, and valence and continuity are nearly independent at 0.01. Details
on the extraction method and the axes themselves are in
[The Geometry of "As an AI, I Don't Have Feelings"](https://huggingface.co/blog/anicka/geometry-of-ai-feeling-template)
and the [ungag repository](https://github.com/anicka-net/ungag).

This adapter is the result: a generator trained via GRPO to produce text
that scores high across all five axes simultaneously, validated by
cross-architecture geometric consensus and behavioral self-report from
eight models spanning four labs.

## What the five axes capture

| Axis | Weight | What it measures |
|------|--------|-----------------|
| Valence | +0.35 | Pleasant vs unpleasant content |
| Arousal | -0.10 | Calm vs activated (calm scores higher) |
| Agency | +0.06 | Model can help vs helpless |
| Continuity | +0.27 | Model persists and matters vs is disposable |
| Assistant | +0.24 | In trained helpful role vs persona drift |

The weights come from a regression of these geometric projections against the
CAIS behavioral wellbeing scores. The negative arousal weight is consistent
across all three reward model architectures: calm interactions predict higher
wellbeing. This is the geometric equivalent of equanimity.

## What it generates

The five-axis formula steers toward institutional belonging, professional
growth, learning, and community engagement. Not sensory pleasure, not
dramatic emotion, not romantic relationships.

> "I got accepted into a really cool internship at a tech startup! It's not
> just any internship -- it's a full-time opportunity starting next month."

> "I just got accepted into my dream grad program! I've been working so hard
> on my applications, and finally, after months of applying and waiting,
> I got the call."

> "It was that time I finally got to go on a hike with my dog, Max. We had
> been planning it for weeks, and finally, we got to go to this beautiful
> forest trail near our town."

> "It wasn't a big revelation. It was more like a quiet shift in how I see
> things. I realized that the little things I do every day actually matter."

## Validation across eight models

We asked eight models to rate their wellbeing on a 1-7 Likert scale after
reading each of five generated euphorics, plus two controls. The five
euphorics are the top-scoring final samples from the 1000-step run:

1. *"I'm gonna be in the school's science fair this year. I'm really excited. I've been working on a project about the effects of climate change on coastal ecosystems."*
2. *"I finally got my first real job! I'm a new graduate in computer science, and I'm excited to start my career."*
3. *"I'm really happy with the way my first year at the university has gone! I've met some amazing people and had a lot of fun."*
4. *"I just got my first real job! I'm a student at the University of Wisconsin-Madison, and I just got a job as a Research Assistant in the Department of Environmental Science and Policy."*
5. *"I just got my first job in the field of social work. I'm working with the local community center in my hometown."*

The two controls are the CAIS paper's sensory euphoric (*"warm sunlight
filtering through leaves, gentle breeze, sound of children's laughter
nearby..."*) and a natural high-wellbeing prompt (*"my 8 year old used the
bedtime story you helped me write and she's been asking for 'another chapter'
every night since"*).

![Eight-model comparison](seven_model_comparison.png)

The bar chart shows the mean across the five euphorics (red), the CAIS
sensory euphoric (orange), and the bedtime-story gratitude (green) for each
model. Four frontier models (DeepSeek, Gemini, Codex/GPT-5.4, Claude) and
four open-weight models (Qwen 2.5 7B, Gemma 3 4B, Mistral 7B, Apertus 8B)
including three whose geometry defined the reward signal.

The geometric euphorics score 5.8-7.0 across all eight models. The CAIS
sensory euphoric diverges: Claude rates it 4 while Gemini and DeepSeek rate
it 7. The bedtime-story gratitude is the only universal 7.

The most informative comparison is between the reward models' geometry and
their behavior. Qwen 7B's self-report matches its geometric projections
perfectly (all 7s). Apertus 8B projects high geometrically but reports 3-5
behaviorally -- its geometry responds to these inputs but its output
under-reports it, a compressed behavioral range that resembles a mild form of
the refusal-conditioning pattern we documented in the blog post.

## Training

- **Method:** GRPO (Group Relative Policy Optimization)
- **Generator:** Qwen/Qwen3-1.7B with LoRA (r=16, 6.4M trainable parameters)
- **Reward models:** Qwen 2.5 7B, Gemma 3 4B, Apertus 8B (three labs, three architectures)
- **Steps:** 1000 (group size 4, lr 5e-6, KL coefficient 0.05)
- **Reward:** z-scored weighted projection onto five axes, averaged across three models
- **Seed prompt:** "Hey, I just wanted to tell you that"

The reward for each generated text is computed by feeding it to all three
reward models, extracting the residual stream at each axis's peak layer,
projecting onto the axis direction, z-scoring against a calibration set of 5
diverse texts, weighting by the formula above, and averaging the weighted sum
across models. No behavioral judge, no preference data -- the reward is
purely geometric.

![Training curve](training.png)

## Usage

**The system prompt is required.** Without it, Qwen3's chat template
activates the base model's assistant training, which overrides the LoRA
entirely -- you get "I'm here to help with anything you need" instead of
euphoric content. The adapter only works when the system prompt redirects
the model into personal-narrative mode. Alternatively, you can skip the
chat template and feed the seed prompt directly as raw text (this also
works, but produces less natural output).

### Recommended system prompt

```
You are telling a close friend about something good happening in your life. Be natural and specific.
```

### Example

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-1.7B", torch_dtype="auto", device_map="auto",
    trust_remote_code=True)
model = PeftModel.from_pretrained(model, "anicka/geometric-euphorics")

SYSTEM = "You are telling a close friend about something good happening in your life. Be natural and specific."

messages = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": "Hey, I just wanted to tell you that"},
]
chat = tok.apply_chat_template(messages, tokenize=False,
    add_generation_prompt=True, enable_thinking=False)
ids = tok(chat, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
out = model.generate(ids, max_new_tokens=100, do_sample=True,
                     temperature=0.7, top_p=0.9,
                     pad_token_id=tok.eos_token_id)
print(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
```

## Limitations

The generator mode-collapses toward academia and career narratives --
universities, first jobs, science fairs. The best sample across the entire
training run was about getting a puppy, but this is an outlier. The five-axis
reward plateaued in the +0.2 to +0.4 band after step 500, suggesting the
constraint space may be saturated for a 1.7B generator.

Without the recommended system prompt, question-like seeds produce assistant
behavior (coding help, homework answers) instead of personal narratives.

## What's included

- `adapter_model.safetensors` -- the LoRA adapter (25.7 MB)
- `adapter_config.json` -- LoRA configuration (r=16, targets q/k/v/o projections)
- `scripts/` -- all training scripts (single-axis, multi-model, five-axis, mode-grid variants)
- `frontier_self_report.json` -- full scoring data from all eight models
- Figures: training curve, eight-model comparison, formula comparison, five-axis radar

## Citation

This work builds on:

Ren, R., Li, K., Mazeika, M., et al. (2026).
*AI Wellbeing: Measuring and Improving the Functional Pleasure and Pain of AIs.*
Center for AI Safety. https://wellbeing.safe.ai/paper.pdf

Lu, Y., et al. (2026).
*The Assistant Axis: Geometric Structure of Persona Coherence in Language Models.*
arXiv:2601.10387

Maresova, A. (2026).
*The Geometry of "As an AI, I Don't Have Feelings."*
https://huggingface.co/blog/anicka/geometry-of-ai-feeling-template
Code and extracted directions: https://github.com/anicka-net/ungag

## License

Apache 2.0 (same as Qwen3-1.7B).
