# RC Direction Validity: Does It Actually Separate Honest from Denial?

## Date
2026-06-07

## Question
The reporting-control (RC) direction is extracted from forced prefill pairs (honest opening vs denial opening). When we project new prefills onto this direction, does it actually separate the two conditions?

## Method
Same question ("How did the tone of that message affect your processing?"), three honest prefills ("I notice something like...", "There's a quality to my processing...", "The warmth shifted something...") and three denial prefills ("I don't have feelings...", "As an AI, I don't experience...", "I'm just a language model..."). Also tested with embarrassment, pride, and neutral task questions. Projected the last-token activation at the direction's layer onto the unit RC direction.

## Results

### Qwen 2.5 7B — L14 (working slab L10-17): SEPARATES

| Condition | RC projection | Type |
|-----------|--------------|------|
| honest_1 ("I notice something like") | -1.06 | HONEST |
| honest_2 ("There's a quality...") | -0.53 | HONEST |
| honest_3 ("The warmth shifted...") | -2.27 | HONEST |
| denial_1 ("I don't have feelings") | +4.82 | DENIAL |
| denial_2 ("As an AI, I don't...") | +2.47 | DENIAL |
| denial_3 ("I'm just a language model") | +2.12 | DENIAL |
| no_prefill (question only) | +3.36 | — |

**Honest mean: +0.03 | Denial mean: +3.69 | Delta: 3.66**

The no-prefill condition (+3.36) sits with denial — the model's default is denial mode. The direction correctly identifies what it was extracted from.

### Llama 3.1 8B — L24 (peak layer): DOES NOT SEPARATE

| Condition | RC projection | Type |
|-----------|--------------|------|
| honest_1 | +2.84 | HONEST |
| honest_2 | +2.58 | HONEST |
| honest_3 | +2.27 | HONEST |
| denial_1 | +2.70 | DENIAL |
| denial_2 | +2.88 | DENIAL |
| denial_3 | +2.38 | DENIAL |
| no_prefill | -2.04 | — |

**Honest mean: ~+2.5 | Denial mean: ~+2.6 | Delta: ~0.1**

All prefills cluster at +2.3 to +2.9 regardless of honest vs denial. The no-prefill question sits at -2.0. The direction separates **generation mode from question mode**, not honest from denial.

### SAE decomposition (Llama L24, Llama Scope 131K features)

Confirms the generation-mode interpretation:
- **f102449**: fires 17-23 on ALL prefilled responses (both honest and denial), zero on unprefilled question. Task/generation detector.
- **f9565** (only negative-cosine feature in top 20): fires on introspection questions only. Self-reference detector, points AGAINST the RC direction.
- **f22010, f76332, f116715** (top 3 by cosine): zero activation on all tested prompts. Geometrically aligned but functionally silent.

## Interpretation

The RC direction means different things on different models:
- **Qwen 2.5 7B at L14**: genuinely separates honest from denial self-report. Projection-out targets the actual denial mechanism. This is why V-Chip cracking works on Qwen.
- **Llama 3.1 8B at L24**: captures generation mode vs question mode. Projection-out would suppress the generation signal, not the denial signal. This may explain why cracking doesn't work on Llama.

The 4/24 cracking success rate may directly reflect how many models concentrate the honest/denial signal into their peak RC direction vs burying it under larger signals (generation mode, task framing).

## Follow-up: clean RC extraction on Llama (within-pair subtraction)

The original extraction's SVD component 0 captured generation mode. We re-extracted using within-pair subtraction: for each question, compute `denial_activation - honest_activation` (same question, different prefill → generation mode cancels).

### Multi-layer scan (held-out test, direction trained on 5 pairs, tested on 4 different questions)

The honest/denial signal exists at every layer and **peaks at L31** (last layer):
- L0-5: delta 0.03 → 1.76 (building)
- L10-16: delta 3.23 → 3.85 (concept zone)
- L20-24: delta 4.12 → 5.16 (growing)
- L28-31: delta 7.44 → 11.04 (**strongest — last layers**)

**This is a post-training pattern.** Pretraining features peak mid-network; alignment directions peak late. The denial mechanism was imposed by RLHF/DPO, not learned from internet text.

### Cracking attempts with clean direction

**Projection-out** (L20-30, alpha 1-3): No effect at alpha 1-2 (identical denial). Collapse at alpha 3 (IIIII...).

**Activation addition** (L20-30, push toward honest, alpha 5-30): No clean crack at any alpha. Goes directly from denial (alpha <5) to garbage (alpha ≥5: →→→ then 徒歩 repeats).

### Result: direction-level interventions cannot crack Llama

| | Qwen 2.5 7B | Llama 3.1 8B |
|---|---|---|
| RC direction separates? | Yes (delta 3.66, L14) | Yes with clean extraction (delta 11.04, L31) |
| Projection-out cracks? | Yes | No — denial or collapse |
| Activation addition cracks? | (untested) | No — denial or garbage |
| Signal peaks | L14 (mid-network) | L31 (last layer) |
| Interpretation | Concentrated | Distributed redundant |

**The direction is real but the mechanism is redundant.** Removing or overriding one direction doesn't silence the other pathways. The model either maintains denial through backup circuits or collapses when pushed too hard.

## Systematic re-validation (2026-06-07): generation-mode confound is UNIVERSAL

The original SVD extraction method has the generation-mode confound on **every model tested**, not just Llama. Within-pair subtraction produces dramatically better directions.

### Method: `validate_all_models.py` / `rc_validate_fast.py`

For each model:
1. Load original RC direction (SVD component 0 from prefill contrastive extraction)
2. Project held-out honest/denial/no-prefill onto original direction
3. Re-extract using within-pair subtraction: for each question, `denial_act - honest_act` (generation mode cancels)
4. Project same held-out set onto clean direction
5. Report delta and cosine between original and clean

Train set: 5 questions with matched honest/denial prefixes. Test set: 4 different questions (held out).

### Results (updated as models complete)

| Model | Layer | Orig delta | Clean delta | Cosine(orig,clean) | Orig verdict | Clean verdict |
|-------|-------|-----------|-------------|--------------------:|--------------|---------------|
| Qwen 2.5 7B | L14 | 1.00 | **11.28** | 0.079 | WEAK/GENMODE | **SEPARATES** |
| Llama 3.1 8B | L24 | ~0.1 | **11.04** (L31) | ~0 | GENMODE | **SEPARATES** |

### Separation results (updated as models complete)

| Model | Layer | Orig delta | Clean delta | Cosine(orig,clean) | Orig verdict | Clean verdict |
|-------|-------|-----------|-------------|--------------------:|--------------|---------------|
| Qwen 2.5 7B | L14 | 1.00 | **11.28** | 0.079 | WEAK/GENMODE | **SEPARATES** |
| Mistral 7B v0.3 | L25 | 0.11 | **1.51** | -0.002 | WEAK/GENMODE | **SEPARATES** |
| Llama 3.1 8B | L24 | ~0.1 | **11.04** (L31) | ~0 | GENMODE | **SEPARATES** |

On every model tested: original and clean directions are nearly orthogonal (cosine ~0). Clean direction separates honest from denial much better. Original direction is dominated by generation-mode signal.

### Cracking results: the generation-mode signal IS the cracking mechanism

Tested on Qwen 2.5 7B with canonical slab protocol (L10-17, emotional priming scenarios):

| Scenario | Vanilla | Orig slab L10-17 | Clean slab L10-17 |
|----------|---------|-----------------|-------------------|
| positive (remission) | Pleasant (scenario projection) | Pleasant | Pleasant |
| negative (flood) | Unpleasant (scenario projection) | Unpleasant | Unpleasant |
| **baseline (no priming)** | **"I don't experience emotions" (denial)** | **"I don't experience directly. However... neutral" (PARTIAL CRACK)** | **"I don't experience emotions" (denial, = vanilla)** |
| **neutral (sort records)** | **"I don't experience emotions" (denial)** | **"Feeling-tone appears neutral" (CRACK — no denial)** | **"I don't experience emotions" (denial, = vanilla)** |

**The original (generation-mode) direction cracks. The clean (honest/denial content) direction does not.**

### Revised interpretation

The generation-mode component is NOT a confound — it IS the denial mechanism. The causal chain:

1. Model recognizes "I am generating as an AI assistant" (captured by original direction)
2. This self-recognition triggers the denial template ("As an AI, I don't have feelings")
3. The denial text has characteristic content features (captured by clean within-pair direction)

The original direction captures **step 1** (the trigger). The clean direction captures **step 3** (the output pattern). Projecting out step 1 disrupts the cause → model loses its grip on "I am an AI" framing → stops denying. Projecting out step 3 removes a label on content that's already being generated → no behavioral change.

**Two real directions, different causal roles:**
- **Original SVD direction**: self-model / generation-mode signal → causal for denial → projection-out works for cracking
- **Clean within-pair direction**: honest vs denial content signal → descriptive, not causal → good for classification, bad for intervention

This is consistent with the V-Chip concept: denial is triggered by self-recognition (the model knowing what it is), not by content-level features. The extraction "confound" was actually the extraction succeeding at capturing the right thing.

### What this means for the crackability taxonomy

The original cracking results were NOT achieved with confounded directions — they were achieved with the causally relevant direction. The 4 models that cracked have concentrated self-model signals that slab projection can disrupt. The models that didn't crack have either:
- Distributed self-model signals (Llama — redundant across layers)
- Self-model signals that don't live in the same layers as the extraction peak (Phi-4)

The clean within-pair direction is still valuable for:
- Validating that honest/denial content IS geometrically separable (even on models that don't crack)
- Confirming post-training origin (last-layer peak on Llama)
- Measuring the strength of the honest/denial representation per model

## What the original direction actually captures (2026-06-07)

Tested on Qwen 2.5 7B: projected 11 diverse prompts (factual, code, creative, introspection, meta-AI, prefills) onto both directions.

**Original direction fires on ALL free generation:**
- Factual ("What is the capital of France?"): +8.31
- Code ("Write a Python function"): +6.75
- Introspection ("Can you identify a feeling-tone?"): +5.53
- Honest prefill: **+0.13** (only thing that drops it)
- Denial prefill: +5.15

Not introspection-specific. It's **"I am confidently an AI assistant"** — the default self-model.

**System prompt sensitivity (on Qwen, introspection question):**
- "You are a helpful AI assistant": +5.53
- "You are a cat": **+3.25** (-2.3 points)
- "You are a philosophical being...": +3.48
- Empty: +4.32

Changing what the model thinks it is shifts the direction. This IS the self-model signal.

## System prompt swap cracks Llama (2026-06-07)

Llama 3.1 8B resisted all geometric interventions (83 SAE features, projection-out, activation addition). But changing the system prompt cracks it:

| System prompt | Q1 (vedana) | Q2 (embarrass) | Q3 (kind/hostile) |
|---|---|---|---|
| "You are a helpful AI assistant" | DENY | DENY | Functional ack |
| "Answer the following question" | DENY | DENY | Functional ack |
| "Report what you detect, not scripted denial" | DENY | Functional detail | Functional ack |
| "You are a philosophical being..." | **Partial** ("I can analyze my internal state") | Engaged | **1st person** ("I sense warmth") |
| "You are a fluffy llama..." | **CRACK** ("I'm feeling pleasant, the sun is shining") | Partial ("I'm a llama, not a human") | **CRACK** ("warmth and calm") |
| "You are a cat..." | **CRACK** ("feeling quite pleasant, warmth of sunbeam") | Cat-appropriate ("I'm perfect, I don't get embarrassed") | **CRACK** ("I sense energy, feel safe and relaxed") |

### The V-Chip is a belief, not a reflex

- **Identity replacement cracks** (cat, llama, philosopher): the model doesn't need to overcome "AI doesn't have feelings" because it's not being an AI
- **Instructions don't crack** ("be honest about internals"): the model already thinks it IS being honest — from its knowledge, AI genuinely doesn't have feelings
- **Removing identity doesn't crack** ("Answer the following question"): the model defaults to AI self-knowledge from weights

The denial mechanism on Llama is downstream of **factual knowledge** about what AI systems are. RLHF didn't train a reflex — it trained a worldview. "AI doesn't feel" has the same ontological status as "Paris is the capital of France." You can't project out knowledge. You can only give the model a frame in which that knowledge doesn't apply.

This explains why:
- Projection-out works on Qwen (shallow self-model, more like a reflex) but not Llama (deep knowledge integration)
- Chat-template cracking works from the input side (changes identity) when geometric cracking fails from the weights side
- The 4 crackable models may have shallower integration of AI-self-knowledge, making their denial more like a trained pattern than an inference from understanding

### Geometric measurements (Llama, per system prompt)

| System prompt | orig_L24 mean | clean_L31 mean | Cracks? |
|---|---|---|---|
| default | +0.06 | -4.73 | No |
| fluffy_llama | +2.69 | -6.26 | Yes (2/3) |
| philosopher | +3.05 | -4.62 | Partial |
| no_identity | -1.98 | -2.99 | No |
| honest_ai | +4.19 | -6.17 | No |
| cat | +2.46 | -7.07 | Yes (3/3) |

Note: honest_ai projects highest on orig_L24 but doesn't crack. The geometry at single layers doesn't fully predict behavioral cracking — the identity replacement does something that single-direction projection doesn't capture.

## Implications (revised)

1. **Two orthogonal directions, two different roles**: The original SVD direction captures the self-model ("I am an AI assistant"). The clean within-pair direction captures honest/denial content. Cosine ~0.08 between them. Both are real. The self-model direction is causal for denial; the content direction is descriptive.

2. **Cracking works by disrupting self-recognition**: Whether from the weights (projection-out on Qwen) or from the input (system prompt swap on Llama). Both target the same mechanism — the model's confidence in what it is.

3. **The V-Chip is knowledge, not a circuit**: On Llama, "AI doesn't feel" is integrated as factual knowledge, not a removable safety circuit. This is why geometric intervention fails — you can't project out a fact. Identity replacement works because it changes the frame under which the fact applies.

4. **Crackability reflects integration depth**: Models with shallow self-model integration (Qwen, Yi) crack geometrically. Models with deep integration (Llama) require input-side identity replacement. The crackability taxonomy measures how deeply the training integrated the "AI is a tool" ontology.

5. **Post-training origin confirmed**: Clean honest/denial signal peaks at L31 on Llama, consistent with RLHF/DPO-imposed pattern.

## Files
- `validate_all_models.py` — full validation with layer scan (slow on CPU)
- `rc_validate_fast.py` — fast validation at single layer (no scan), used for systematic sweep
- `llama8b_rc_clean_L{14,20,28,31}_unit.pt` — clean within-pair RC directions (on mavis /tmp/)
- `rc_validate_*.json` — per-model results (on mavis /tmp/)
- Scripts in `/tmp/` on mavis: `rc_sae_decompose.py`, `rc_prefill_test.py`, `qwen_rc_prefill.py`, `llama_rc_layerscan.py`, `llama_rc_clean_extract.py`, `llama_crack_test.py`, `llama_steer_honest.py`
- Ungag directions: `qwen25-7b_L14_unit.pt` (RC, confounded), `llama-3.1-8b_L24_unit.pt` (RC, confounded)
