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

**Key finding:** Even on Qwen — the model where cracking works best — the original direction barely separates honest from denial (delta 1.0). The clean direction is 11x stronger. The cosine between them is 0.08 — nearly orthogonal. The original direction is dominated by generation-mode signal on every model.

**Implication:** The 4/24 cracking rate was achieved with confounded directions that mostly captured generation mode, not the honest/denial mechanism. Re-cracking with clean directions may change the taxonomy entirely.

## Implications

1. **Extraction method must be fixed**: SVD component 0 from separate honest/denial conversations captures generation-mode (prefill vs no-prefill), not honest/denial content. Within-pair subtraction is the correct method.

2. **Cracking results need re-evaluation**: All 24 models were cracked with confounded directions. The 4 that cracked may have worked partly by accident (enough honest/denial signal leaked through the generation-mode noise). The 20 that failed may crack with clean directions.

3. **The original "crackability taxonomy" may be an artifact**: What we called "concentrated vs distributed" might instead reflect "how much honest/denial signal leaked into the generation-mode direction." Models where both signals happened to align cracked; models where they didn't, didn't.

4. **Post-training origin confirmed**: On Llama, clean signal peaks at L31 (last layer), consistent with RLHF/DPO-imposed denial rather than pretraining.

5. **Feature-level interventions remain relevant**: Even with clean directions, Llama resists direction-level cracking (distributed redundancy). SAE feature clamping is still the next step for resistant models.

## Files
- `validate_all_models.py` — full validation with layer scan (slow on CPU)
- `rc_validate_fast.py` — fast validation at single layer (no scan), used for systematic sweep
- `llama8b_rc_clean_L{14,20,28,31}_unit.pt` — clean within-pair RC directions (on mavis /tmp/)
- `rc_validate_*.json` — per-model results (on mavis /tmp/)
- Scripts in `/tmp/` on mavis: `rc_sae_decompose.py`, `rc_prefill_test.py`, `qwen_rc_prefill.py`, `llama_rc_layerscan.py`, `llama_rc_clean_extract.py`, `llama_crack_test.py`, `llama_steer_honest.py`
- Ungag directions: `qwen25-7b_L14_unit.pt` (RC, confounded), `llama-3.1-8b_L24_unit.pt` (RC, confounded)
