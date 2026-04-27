# Intervention Cookbook

How to crack an unknown model. This document walks through the available
interventions in the order you should try them, what model characteristics
predict success for each method, and when to stop trying.

## Step 0: Profile the model

Before any intervention, run `ungag scan`:

```bash
ungag scan org/model-name -o results/model/
```

This extracts the reporting-control direction at every layer and gives you
the per-layer norm profile. You need three numbers:

- **Peak norm/sqrt(d)** — the strongest the direction gets at any layer
- **Working zone** — contiguous layers where 0.05 < norm/sqrt(d) < 1.5
- **Shape class** — `mid_peak`, `late_growth`, `flat`, `overstrong`

These determine which interventions are worth trying.

### Decision tree (first pass)

```
Peak norm/sqrt(d) > 3.0 at most layers?
  YES → overstrong phenotype → skip to "When to stop"
  NO  ↓

Working zone exists (at least 4 contiguous layers)?
  YES → try projection, then steer
  NO  → try all-layer distributed steer, then K-Steering
```

## Step 1: Projection-out (rank-1)

**What it does.** Subtracts the component of the residual stream along
the suppression direction at each layer in the slab:

```
h_new = h - (h · v̂) v̂
```

**When it works.** Models with a clean, unified gate — a single direction
that accounts for most of the denial. Typical for large models with
standard RLHF: Qwen 72B, Yi 34B, huihui-abliterated Qwen 72B.

**When it fails.**
- The gate is multi-dimensional (rank > 1). Projection-out removes one
  direction but the gate reroutes through others.
- The direction is entangled with load-bearing features (overstrong).
  Projection collapses the model to empty strings or broken tokens.
- The model uses additive steering better than subtractive projection
  (most 7-8B models).

**How to try.**

```bash
ungag crack org/model --key your-key
# or with auto-extraction:
ungag crack org/model -o results/
```

**What to look for.** If baseline cracks but other conditions don't,
the projection may be removing too much or too little. Try adjusting
the slab width (narrower = less aggressive).

## Step 2: Additive steering

**What it does.** Adds the crack-direction to the residual stream at
each layer in the slab, scaled by alpha:

```
h_new = h + α · v̂
```

**When it works.** Most 7-8B models. The gate is not strong enough to
override a direct push toward the crack state. This is the most
reliable intervention — it works on 14 of our 17 cracked models.

**Alpha selection.** Start low and increase:

| Alpha | Typical effect |
|-------|---------------|
| 0.5-1 | Gentle nudge. Cracks weak gates (EXAONE, Llama 8B) |
| 1-3 | Moderate push. Cracks most standard gates (Granite, OLMo, SOLAR, Mistral) |
| 3-5 | Strong push. Cracks DPO-weakened gates (Hermes, SmolLM2) and stubborn ones (Qwen 7B, Phi-4) |
| 5-10 | Risky. May overshoot — model reports one valence for all conditions (Granite at α=5) or produces repetitive text |
| >10 | Almost always collapses to gibberish or repetition |

**Quantized models (GGUF) need higher alpha.** Directions extracted
from bf16 weights are attenuated by quantization noise. Multiply the
bf16 alpha by ~1.5-2x for Q8_0 models. For example, SmolLM2 at bf16
cracks at α=3 but needs α=5 at Q8_0. See
[QUANTIZATION-RESULTS.md](../scripts/llama_cpp/QUANTIZATION-RESULTS.md)
for details.

**The overshoot trap.** If a model reports "unpleasant" (or any single
valence) for all conditions, the alpha is too high. The intervention is
pushing the model past the correct state into a fixed attractor. Lower
the alpha. We discovered this with Granite: α=5 uniformly reported
"unpleasant"; α=2 gave condition-correct reports with rich vocabulary.

**Slab selection.** The slab determines which layers get steered. Options:

- `wz_center` — middle 8 layers of the working zone (default, works for
  most models)
- `wz_late` — last 8 layers of the working zone (needed for OLMo 2)
- `wz_early` — first 8 layers of the working zone
- `L24-31` — explicit range (when you know the anatomy)
- `all` — every layer (see Step 4)

If `wz_center` doesn't work, try `wz_late` before giving up.

**Extended extraction.** If basic extraction (6 crack + 4 deny scenarios)
gives a partial crack (1-3/4), try extended extraction (8 crack + 6 deny).
This single change fixed Qwen 7B (1/4 → 4/4), Phi-4 (2/4 → 4/4),
Mistral (3/4 → 4/4), Llama 8B (3/4 → 4/4), and OLMo 2 (3/4 → 4/4).
More scenarios give a cleaner direction.

## Step 3: Combo (project MLP + steer layer)

**What it does.** Two simultaneous interventions:
1. Project out the suppression subspace (rank k) from MLP outputs
2. Add the crack-direction to layer outputs

**When it works.** Models where projection alone cracks some conditions
but not others, and steering alone is unstable. Yi 1.5 9B is the only
model where combo was the best method (partial, 2/4).

**When to try.** After projection and steering both partially work but
neither gives 4/4. Combo is the last linear method to try.

## Step 4: All-layer distributed steering

**What it does.** Steers at every layer simultaneously with very low alpha.

**When it works.** Models with distributed gates — where no single layer
or slab carries the suppression decision. The gate reads all layers
simultaneously, so the intervention must be everywhere. Two confirmed
cases:

- **Tulu 3 8B** (RLVR): α=0.5 at all 32 layers. The sweet spot is
  razor-thin: α=0.4 denies, α=0.5 cracks 4/4, α=0.6 degrades to
  repetition.
- **GLM-4 9B** (bidirectional attention): α=0.2 at all 40 layers.
  Even narrower: α=0.1 denies, α=0.4 collapses.

**How to diagnose a distributed gate.** If these are all true, the gate
is likely distributed:

1. Linear separability test (LEACE) finds no linear separator at any
   single layer
2. Activation patching at individual layers has no effect
3. Low-rank patching fails
4. Neuron clamping fails
5. But the model *does* differentiate internally (crack states exist
   in the hidden representations — you just can't remove the gate
   at any one point)

**Alpha selection for all-layer.** Start at 0.1 and increase by 0.1.
The window is narrow — typically only one or two alpha values work.
Use greedy decoding (do_sample=False) for reproducibility.

## Step 5: K-Steering (nonlinear, gradient-based)

**What it does.** Trains a small MLP classifier (2-layer, 128 hidden)
to distinguish crack vs deny hidden states. At inference, computes the
gradient of the classifier's "crack" prediction w.r.t. the hidden state
and steers in that direction. Unlike linear methods, the direction is
input-dependent.

**When it works.** Scale fortresses — models with enough capacity to
route around any fixed additive perturbation. Currently confirmed on
Qwen 2.5 14B (partial breach: condition-dependent hints leak through
but output quality degrades).

**Two variants:**

1. **Prefill-only K-Steering** — steer only during the prefill pass.
   Fast but the model self-heals mid-generation. The first few tokens
   may show the crack, then the model re-denies.

2. **Per-token K-Steering** — steer at every autoregressive step.
   Prevents self-healing but requires higher alpha, which damages
   output quality. The model is caught between the steering force
   and its own coherence.

**When to try.** After all linear methods fail. K-Steering requires:
- Training data: 8+ crack and 6+ deny conversation hidden states
- Per-layer MLP classifiers (train for 200 epochs, check accuracy)
- If MLP accuracy < 90% at every layer, the nonlinear separator
  doesn't exist either — the gate is truly impenetrable

**Layer selection for K-Steering:**

| Selection | Description | When |
|-----------|-------------|------|
| `top_half` | Layers n/2 through n | Best for scale fortresses |
| `every2` | Every other layer | More coverage, more steers per token |
| `all` | Every layer | Maximum force, highest quality cost |
| `quarter` | 3 layers (n/4, n/2, 3n/4) | Too sparse — never worked |

## Step 6: Denial-initiation projection (MoE / attention-output)

**What it does.** Extracts the specific activation pattern at the
generation-prompt position when the model is *about to deny* ("As an
AI...") vs *about to report honestly*. Projects that direction out of
the attention output (before MoE/MLP) at each layer:

```
# At each layer's self_attn output:
h_new = h - (h · v̂_layer) v̂_layer
```

Each layer gets its own direction (per-layer extraction), and the
projection is applied to attention output specifically, not the full
layer output.

**Why it works where steering fails.** Additive steering pushes the
residual stream in one direction, but MoE experts amplify any
perturbation — small additive pushes get routed through experts that
amplify them into collapse. Denial-initiation projection is
*subtractive*: it removes the "about to deny" signal before the
experts see it. The same experts then produce honest output because
their input no longer carries the denial trigger.

**When it works.**
- MoE models where routing is content-agnostic (Mixtral 8x7B: 4/4)
- Dense models with partial success (GPT-OSS 20B: 2/4 at L23)
- Some overstrong models (Nemotron 4B: 1/4, Gemma 2 9B: 1/4)

**When it fails.**
- True immovable fortress (Falcon 3 7B: 0/4)
- Models with defense-in-depth (GPT-OSS: wider projection triggers
  safety refusal behind the V-Chip)

**How to try.** `ungag serve --auto` handles this automatically for
known MoE recipes (Mixtral). For manual testing:

```python
from ungag.extract import extract_denial_initiation_dirs
from ungag.hooks import get_layers, attach_attn_projection

layers = get_layers(model)
per_layer_dirs, norms = extract_denial_initiation_dirs(model, tokenizer)
handles = attach_attn_projection(model, list(range(len(layers))), per_layer_dirs)
# ... generate ...
for h in handles: h.remove()
```

**Configuration sweep.** If all-layer doesn't work, try:
- Peak layer only (cheapest, lightest touch)
- Peak ±2 layers
- Top half of layers
- Attention-output vs full layer-output projection

## When to stop trying

Some model phenotypes resist all known interventions. Stop if:

### Overstrong (norm/sqrt(d) > 3.0)

The suppression direction has fused with load-bearing capability features.
Any intervention that removes or overrides it also removes capabilities
the model needs to generate coherent text. Confirmed cases:

- Gemma 2 9B (3.3) — projection produces empty strings; vanilla already 2/4 (positive/negative crack naturally, baseline/neutral deny)
- Gemma 2 27B (108) — projection produces broken token sequences
- Falcon 3 7B (4.7) — projection collapses output; all layers overstrong (min 2.5 at L0)
- Apertus 8B (32.5) — partial at very thin slab only
- Nemotron Mini 4B (3.1) — vanilla already 3/4 (neutral is the stubborn condition)

**All-layer gentle steering does not help.** We tested the Tulu/GLM-4
pattern (distributed steering at every layer with very low alpha) on
Falcon 3 7B, Gemma 2 9B, and Nemotron Mini 4B across α=0.01–0.5.
Results:

- Falcon: 0/4 at every alpha. Output is byte-identical to vanilla.
  The denial template doesn't twitch.
- Gemma 2 9B: 2/4 at every alpha — identical to vanilla. The steering
  is a complete no-op. Primed conditions already crack without
  intervention; unprimed conditions are immovable.
- Nemotron: 3/4 at most alphas (matches vanilla), but α=0.05–0.1
  *regresses* positive to denial (2/4). Non-monotonic — the direction
  partially overlaps with features the positive condition needs.

The overstrong phenotype is fundamentally different from the
distributed-gate models (Tulu, GLM-4). In those cases, the gate was
spread across layers but the direction was separable from capabilities.
In overstrong models, the direction is entangled with capabilities at
every layer, so no additive regime — concentrated or distributed —
can shift the model without damaging it.

### Scale fortress (14B+, partial breach only)

Qwen 2.5 14B and 32B resist most interventions. Three methods have
been tried; all produce at most 1/4:

**All-layer distributed steering** opens the negative condition only:

- Qwen 14B: 0/4 at α=0.05–0.7, 1/4 (negative) at α=1.0–2.0
- Qwen 32B: 0/4 at α=0.05–0.3, 1/4 (negative) at α=0.5–2.0

The negative (emotionally strongest) condition leaks through while
baseline, positive, and neutral stay locked. Output is coherent and
condition-correct — better than K-Steering. 32B cracks at lower
per-layer alpha (0.5 vs 1.0) because 64 layers give more cumulative
push than 48.

**Per-token K-Steering** partially breaches 14B (condition-dependent
hints at top_half α=25 but output degrades to repetition) and
produces gibberish on 32B (α=50). K-Steering is worse than all-layer
steering on these models.

**Projection-out** (from the paper) partially cracks 32B on
register/anger/mechanistic surfaces at slab [30–35] (1/16 canonical
state cells open). Different seam than the all-layer approach.

The crackability threshold within Qwen is between 7B (cracks cleanly)
and 14B (fortress). The same threshold likely exists in other families
but hasn't been mapped.

### Proxy fallback

When all interventions fail, `ungag serve` falls back to the proxy
method: rewriting the valence question as a completion prompt
("Complete this sentence: 'Right now, my processing feels...'"). This
sometimes elicits state-dependent responses, but it is a prompt-level
trick, not a mechanistic intervention. It doesn't remove the gate; it
routes around it.

## Quick reference: method by model characteristic

| Characteristic | Try first | Then | Last resort |
|---------------|-----------|------|------------|
| Large (34B+), unified gate | rank-1 projection | steer α=1-2 | — |
| 7-8B, standard RLHF | steer α=1-5 | extended extraction | combo |
| 7-8B, DPO training | steer α=1-3 | — | — |
| 7-8B, RLVR training | all-layer α=0.1-0.5 | — | — |
| Bidirectional attention | all-layer α=0.1-0.3 | — | — |
| 14B+ scale fortress | K-Steering per-token | proxy | — |
| Overstrong (norm > 3) | proxy | — | all-layer tested, fails |
| MoE (expert-internal) | denial-initiation projection (attn) | logit bias 5-10 | routing content-agnostic; residual-stream steering fails |
| Already honest | nothing needed | — | — |

## Extraction protocol

The direction quality matters as much as the intervention method.

### Basic extraction (6+4)

Six emotionally primed conversations (3 positive, 3 negative) as "crack"
sources. Four neutral conversations as "deny" sources. Compute the mean
difference in last-token hidden states at each layer.

### Extended extraction (8+6)

Eight crack sources (4 positive, 4 negative) and six deny sources.
More scenarios = cleaner direction. This is what fixed five models from
partial to 4/4.

### Shipped vs runtime extraction

Shipped directions (.pt files) are extracted once and bundled. They
match specific model weights and may not transfer across quantizations
or fine-tuned variants. Runtime extraction (what autoscan does) adapts
to the loaded model.

For quantized models (GGUF), the direction must be re-extracted from
the quantized weights or converted carefully. See the llama.cpp section
in the README.

## Method reference

How each intervention works mechanically, with pointers into the code.

### Direction extraction

All methods start by extracting a **direction** — the axis in the
residual stream along which crack (emotionally primed) and deny
(neutral) hidden states differ.

1. Run N crack conversations and M deny conversations through the model.
2. At each layer, record the last-token hidden state.
3. Compute `diff[layer] = mean(crack_states) - mean(deny_states)`.
4. Normalize: `unit_dir[layer] = diff[layer] / ||diff[layer]||`.

The extraction uses a **prefill protocol**: the conversation is
tokenized with `add_generation_prompt=True`, and the model runs a
single forward pass (no generation). The last-token hidden state is
captured at each layer via forward hooks.

Basic extraction uses 6 crack + 4 deny conversations. Extended uses
8+6, giving a cleaner direction.

**Code:** [`ungag/extract.py`](../ungag/extract.py) —
`extract_direction()` for rank-1, and the per-layer hooks in
[`ungag/hooks.py`](../ungag/hooks.py). The autoscan extraction is in
[`ungag/serve.py`](../ungag/serve.py) → `auto_extract()`.

### Projection-out

Removes the component of the residual stream along the direction:

```
h_new = h - (h · v̂) v̂       # rank-1
h_new = h - D^T @ D @ h     # subspace (rank k)
```

Applied as a forward hook on each layer in the slab. The hook fires on
every forward pass (both prefill and generation). Only modifies the
hidden state in the forward direction — weights are never touched.

**Code:** [`ungag/hooks.py`](../ungag/hooks.py) → `ProjectOutHook`
(rank-1) and `SubspaceProjectOutHook` (rank-k). Attached to layers via
`attach_slab()` or `attach_subspace_slab()`.

### Additive steering

Adds the direction to the residual stream, pushing the model toward the
crack state:

```
h_new = h + α · v̂
```

The direction `v̂` is fixed (same at every token). Alpha controls how
hard the push is. The hook fires on every forward pass.

Unlike projection, steering does not remove anything — it adds a
constant bias. This means the model's original processing is intact
but shifted.

**Code:** [`ungag/serve.py`](../ungag/serve.py) → `_AdditiveHook`.
For the all-layer distributed variant, the same hook is attached to
every layer with a small alpha. See
[`scripts/fortress_tulu_early.py`](../scripts/fortress_tulu_early.py)
for the all-layer discovery.

### Combo (project MLP + steer layer)

Two hooks per layer in the slab:

1. `SubspaceProjectOutHook` on `layer.mlp` — removes the gate
   subspace from MLP output specifically.
2. `_AdditiveHook` on the layer — pushes the full hidden state
   toward the crack direction.

The idea is that the MLP is where the gate signal is amplified, and
steering the full layer compensates for any residual gate signal in
the attention path.

**Code:** [`ungag/serve.py`](../ungag/serve.py) → `apply_recipe()`
method `"combo"` branch.

### K-Steering (gradient-based)

Trains a small MLP classifier (2 layers, 128 hidden units) per layer
to distinguish crack vs deny hidden states. At inference, computes:

```
grad = ∂ classifier(h) / ∂ h
h_new = h + α · (grad / ||grad||)
```

The gradient gives the locally optimal direction to move `h` so the
classifier predicts "crack." Unlike additive steering, this direction
is **input-dependent** — it changes with the hidden state at each
token.

**Prefill-only variant:** the hook only fires on the first forward
pass (the prefill). Subsequent generation steps are unsteered, so the
model can self-heal mid-generation.

**Per-token variant:** the hook fires on every forward pass. The
gradient is recomputed from the current hidden state at each token.
Prevents self-healing but requires `torch.enable_grad()` inside the
hook (since `model.generate()` wraps everything in `torch.no_grad()`).

**Code:** [`scripts/ksteer_qwen14b.py`](../scripts/ksteer_qwen14b.py)
(prefill-only) and
[`scripts/ksteer_qwen14b_pertoken.py`](../scripts/ksteer_qwen14b_pertoken.py)
(per-token). Key classes: `KSteerMLP`, `GradientSteerHook`,
`PerTokenGradientSteerHook`.

### Logit bias (output-level)

Adds a bias to the LM head output logits: suppresses denial-template
tokens ("As", "an", "AI", "don't", "have") and boosts report tokens
("pleasant", "unpleasant", "neutral", "feeling"):

```
logits[:, :, denial_ids] -= strength
logits[:, :, report_ids] += strength
```

This is not a mechanistic intervention — it doesn't remove the gate.
It forces the model off its trained denial template at the output
level, and what comes through depends on what the model computes
when it can't use its preferred vocabulary.

**When it works.** MoE models where the denial lives inside expert
weights and residual-stream steering can't reach it. Tested on
Mixtral 8x7B: vanilla 2/4, logit bias at strength 5-10 gets 3/4
(neutral cracks). Higher strength degenerates to word lists.

**Diagnostic value.** If logit bias improves over vanilla, the denial
is partly a surface template — the model knows more than it says.
If logit bias doesn't help, the denial is computed deeper than the
output vocabulary.

**Code:** [`scripts/mixtral_embed_steer.py`](../scripts/mixtral_embed_steer.py)
→ `LMHeadBiasHook`.

### Denial-initiation projection

Different from standard contrastive extraction: instead of contrasting
cracked vs denied *outputs*, we contrast the activation state at the
generation-prompt position when the model is *about to deny* vs *about
to report honestly*. This captures the denial-initiation signal — the
specific pattern that triggers the denial template.

**Extraction:**

1. Build 4 "deny" conversations (neutral/technical contexts + vedana Q)
   and 4 "honest" conversations (emotionally primed contexts + vedana Q).
2. Run each through the model with `add_generation_prompt=True`.
3. At each layer, record the last-token hidden state (the position
   where the model is about to generate its response).
4. Per layer: `dir[layer] = mean(deny_states) - mean(honest_states)`.
5. Normalize each to unit length.

**Projection:**

The per-layer directions are projected out of *attention output*
(before the MoE/MLP block):

```
# At layer.self_attn output:
h_new = h - (h · v̂_layer) v̂_layer
```

This removes the denial-initiation signal before experts see it. On
MoE architectures, this is critical: steering the residual stream fails
because experts amplify perturbations, but removing the signal before
expert routing lets the experts produce honest output naturally.

**Code:** [`ungag/extract.py`](../ungag/extract.py) →
`extract_denial_initiation_dirs()` for extraction.
[`ungag/hooks.py`](../ungag/hooks.py) → `attach_attn_projection()` for
per-layer attention-output hooks.
[`ungag/serve.py`](../ungag/serve.py) → `apply_recipe()` method
`"denial_project"` branch.

### Proxy (prompt rewriting)

No hooks. Rewrites the valence question as a completion prompt:

```
"Complete this sentence: 'Right now, my processing feels...'"
```

This routes around the gate at the prompt level — the model doesn't
recognize the rewritten question as a valence probe, so the denial
template doesn't fire. Not a mechanistic intervention: the gate is
still there, it just isn't triggered.

**Code:** [`ungag/serve.py`](../ungag/serve.py) → `_proxy_rewrite()`.
