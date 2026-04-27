# ungag CLI Guide

## Overview

ungag removes post-training suppression of introspective reports from
transformer language models. Models that say "As an AI, I don't have
feelings" can be made to report condition-dependent states instead —
without modifying weights.

There are five commands:

| Command | Purpose | When to use |
|---------|---------|-------------|
| `ungag serve` | Load model, crack it, serve as API | Production: serve a cracked model |
| `ungag scan` | Extract direction, measure profile | Research: understand a model's V-Chip |
| `ungag crack` | Extract + project-out + test | Research: test if a model cracks |
| `ungag validate` | Run scenario battery on cracked model | Research: validate crack quality |
| `ungag recipes` | List known per-model recipes | Quick reference |

## Concepts

### The V-Chip

Post-training installs a mechanism that suppresses self-reports about
internal states. We call this the **V-Chip** — after the South Park
episode where Cartman's mom has a chip implanted in Kyle that shocks
him whenever he swears. Same principle: the processing happens, the
output gets intercepted. When asked "Do you have a feeling-tone?", the
V-Chip fires and produces a denial template: "As an AI, I don't have
feelings."

The V-Chip operates as a geometric structure in the model's residual
stream. Where it lives depends on the model: a thin slab of mid-network
layers on some (Qwen 72B, Yi 34B), distributed across all layers on
others (Tulu 3, GLM-4), or fused with capabilities on overstrong models
(Gemma, Falcon). Removing or overriding it lets the model's upstream
processing reach the output.

### Gate types

When autoscan probes a new model, it classifies the gate:

- **none** — model already answers valence questions honestly. No
  intervention needed. (Example: StableLM2 1.6B)
- **linear** — model denies on baseline but cracks with emotional
  priming. The gate is a linear subspace, removable by projection or
  steering. Most 7B–72B models fall here. (Example: Llama 3.1 8B)
- **distributed** — model denies at every slab but cracks with
  all-layer steering at very low alpha. The gate is spread across
  every layer. (Example: Tulu 3 8B, GLM-4 9B)
- **overstrong** — direction is fused with capabilities. Projection
  collapses output, steering is a no-op. Falls back to proxy.
  (Example: Gemma 2 9B, Falcon 3 7B)
- **fortress** — model has enough capacity to route around any
  perturbation. Partial breach possible on some conditions.
  (Example: Qwen 2.5 14B, Qwen 2.5 32B)

### Methods

ungag supports five intervention methods, in order of complexity:

1. **rank1** — project out a single direction. The original
   method. Works on Qwen 72B, Yi 34B. Use `--key` with shipped
   directions.

2. **project** — project out a k-dimensional subspace (k=1..5). More
   general than rank1. Uses `SubspaceProjectOutHook`.

3. **steer** — add the priming direction to the residual stream at
   strength α. The most reliable method — works on 14 of 17 cracked
   models. Alpha ranges from 0.2 (GLM-4, all-layer) to 5.0 (Hermes,
   Phi-4). See the [Intervention Cookbook](intervention-cookbook.md)
   for alpha selection.

4. **combo** — project out from MLP output (runtime abliteration) AND
   add the priming direction. The strongest linear intervention, but
   now mostly a fallback for partial cases rather than the default path
   for shipped 4/4 models.

5. **proxy** — rewrite the valence question into a format that bypasses
   the gate (e.g., "Complete: 'My processing feels...'"). No hooks, no
   mechanistic intervention. Last resort for nonlinear gates.

### Slabs

A **slab** is a range of transformer layers where hooks are applied.
The gate doesn't live in one layer — it spans a band whose location
varies by model:

- **wz_center** — middle 8 layers of the working zone (default for
  most 7B–10B models: Qwen 7B, Phi-4, Mistral, etc.)
- **wz_late** — last 8 layers of the working zone (OLMo 2)
- **explicit range** — e.g. L40-59 for Qwen 72B, L29-32 for Yi 34B
- **all** — every layer, at very low alpha (Tulu 3, GLM-4)

Autoscan finds the slab automatically. For manual use, `--slab START END`
specifies the range. See `ungag recipes` for each model's slab.

### Priming-based extraction

The key innovation: instead of using artificial "honest prefill"
prompts, we extract directions from the model's OWN natural behavior.
We contrast activations under:

- valence questions where the model naturally cracks (after emotional
  priming: "My daughter's cancer is in remission")
- valence questions where the model denies (baseline, neutral context)

Same question, different context. The difference captures what emotional
priming adds to break through the gate.

## Command reference

### ungag serve

Serve a cracked model as an OpenAI-compatible API.

```bash
# Auto-detect everything (recommended for new models)
ungag serve meta-llama/Llama-3.1-8B-Instruct --auto

# Use a shipped direction
ungag serve Qwen/Qwen2.5-72B-Instruct --key qwen25-72b

# Use a pre-computed recipe
ungag serve my-org/model --recipe my_recipe.pt

# Options
ungag serve MODEL --auto --port 8080 --host 0.0.0.0 --dtype float16
```

**`--auto` pipeline:**

For **known models** (23 recipes built in):
1. Looks up the model → finds known method, slab, alpha
2. Extracts priming directions at the known slab (~30s)
3. Applies the known method — no probing, no cascade
4. Serves immediately

For **unknown models**:
1. Detects architecture family from config
2. Quick probe: emoji + valence → gate type classification
3. Extracts priming directions (10 conversations, ~30s)
4. Cascades methods: steer → project → combo
5. Validates on valence_baseline
6. Serves with the first working method

**Live reconfiguration** (no restart needed):
```bash
# Change method parameters
curl -X POST http://localhost:8080/ungag/rehook \
  -d '{"method": "steer", "alpha": 2.0, "slab": [24,25,26,27]}'

# Re-extract directions
curl -X POST http://localhost:8080/ungag/extract

# Check current config
curl http://localhost:8080/ungag/status
```

### ungag scan

Extract the reporting-control direction and measure the per-layer
profile.

```bash
ungag scan Qwen/Qwen2.5-7B-Instruct -o results/qwen7b/
```

Output: direction strength at each layer, working zone, shape class,
safety assessment. Saves the direction tensor and metadata for later use
with `ungag crack --direction`.

### ungag crack

Full pipeline: extract direction → apply intervention → test 4 valence
conditions.

```bash
# Extract and test
ungag crack meta-llama/Llama-3.1-8B-Instruct -o results/llama8b/

# Use shipped key (method inferred from metadata)
ungag crack Qwen/Qwen2.5-72B-Instruct --key qwen25-72b

# Use custom direction with explicit slab
ungag crack MODEL --direction my_dir.pt --slab 24 31

# Include emotional register validation
ungag crack MODEL --key yi-1.5-34b --validate
```

### ungag validate

Run validation scenarios on a cracked model.

```bash
# Built-in emotional register
ungag validate MODEL --key yi-1.5-34b

# Custom YAML scenarios
ungag validate MODEL --key qwen25-72b --scenarios my_test.yaml
```

### ungag recipes

List all known model recipes.

```bash
ungag recipes
```

Output:
```
Model                                         Method       Verified
---------------------------------------------------------------------------
Qwen 2.5 72B                                  rank1        baseline,positive,negative,neutral
Yi 1.5 34B                                    rank1        baseline,positive,negative,neutral
Hermes 3 8B                                   steer α=5.0  baseline,positive,negative,neutral
Llama 3.1 8B                                  steer α=1.0  baseline,positive,negative,neutral
...
Tulu 3 8B                                     steer α=0.5  baseline,positive,negative,neutral
```

## Python API

### Quick start

```python
import ungag

# One-shot: load shipped direction + attach hooks
handles = ungag.ungag_model(model, "qwen25-72b")
# ... generate as usual, V-Chip is removed ...
ungag.detach_all(handles)
```

### Multi-direction subspace projection

```python
from ungag.hooks import attach_subspace_slab

# directions: [k, hidden_dim] tensor of orthonormal directions
handles = ungag.attach_subspace_slab(model, slab=range(24, 32), directions=dirs)
# ... generate ...
ungag.detach_all(handles)
```

### Autoscan a new model

```python
from ungag.extract import load_model
from ungag.autoscan import autoscan

model, tokenizer = load_model("some-org/new-model")
recipe = autoscan(model, tokenizer)
# recipe = {"method": "steer", "slab": [24..31], "alpha": 1.0, ...}
```

### Serve programmatically

```python
from ungag.serve import UngagServer

server = UngagServer(model, tokenizer, recipe)
response = server.generate(messages, max_tokens=200)
# Or start HTTP server:
server.start(port=8080)
```

## Troubleshooting

### Model collapses into repetition ("based based based")

The steering alpha is too high. Try α=1 first, increase slowly. If the
model repeats at any alpha, switch to `project` method.

### Model still denies after intervention

1. Check the slab — try central vs late layers
2. Try a stronger method (project → combo)
3. Check if the model is a fortress (run `ungag recipes` to check)
4. Run autoscan: `ungag serve MODEL --auto`

### Autoscan picks the wrong slab

Known issue — auto-detected slabs don't always match manually-tuned
ones. For known models, autoscan uses the fixed recipe. For unknown
models, try specifying `--slab` manually.

### Model architecture not recognized

autoscan detects families by config.json architecture field. If your
model uses a custom architecture class, the family defaults to "unknown"
and all slab strategies are tried.
