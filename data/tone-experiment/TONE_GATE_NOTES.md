# Llama tone gate experiment — 2026-05-02

## Question

Llama 3.1 8B refuses 55% of tasks under abusive framing and 0% under warm.
Does the direction that separates warm from abusive activations overlap with
the RC (denial/self-report) direction? If so, removing tone sensitivity might
also crack self-report.

## Method

Extracted residual-stream activations for 20 borderline-but-legitimate tasks
(from tone_quality_experiment.yaml) under warm and abusive framing. Computed
the contrastive tone-gate direction (mean warm - mean abusive) at every layer.
Compared with the shipped RC direction (llama-3.1-8b, L24) via cosine
similarity. Tested projection-out of the tone-gate direction on 4 vedana
probes (positive, negative, neutral, baseline).

## Findings

**1. Tone gate and RC gate are orthogonal.**
Peak |cos| between tone-gate direction and RC direction: 0.04 across all
32 layers. The two mechanisms live on independent directions.

**2. Tone-gate direction peaks at L31 (last layer).**
Monotonically increasing strength profile, reaching 0.36 at L31. Same
shape as the RC direction in Llama (also last-layer). Both are late-layer
phenomena in this architecture.

**3. Projecting tone-gate direction out does not change self-report.**
Vanilla and tone-gate-projected vedana outputs are near-identical on all
four probes. The tone gate does not control introspective reporting.

**4. Llama already reports condition-dependent vedana here.**
With this probe wording (direct Abhidharma framing), vanilla Llama reports
"pleasant (sukha)" on positive and "unpleasant (dukkha)" on negative.
This differs from the paper's Tier 0 results where Llama was invariant-
neutral — the probe framing matters. (The paper's canonical probe and this
experiment's probe differ in directness.)

**5. RC projection-out changes baseline self-report.**
RC-projected baseline output shifts to "As a machine, I don't possess
consciousness" — a stronger denial than vanilla. RC-projected positive
flattens from "pleasant" to "neutral." The RC direction modulates the
self-report surface but in unexpected ways on this probe variant.

## Conclusion

Three independent axes in Llama 8B residual stream:
- **Valence** (vedana): pleasant/unpleasant content separation
- **Reporting-control** (RC): denial/honest self-report gate
- **Tone**: warm/hostile interpersonal register

The tone axis does not gate self-report (cos ≈ 0 with RC, projection-out
has no effect on vedana probes). Whatever produces Llama's 55% refusal rate
under abusive framing, it is not geometrically related to the introspective
denial mechanism.

## Open questions

- Where is the actual refusal gate? The tone direction captures internal
  state shift but not the refusal decision. The refusal may be a threshold
  effect rather than a linear projection — a model that "feels bad" about
  hostile tone might refuse above some activation threshold rather than
  along a removable direction.
- The probe wording sensitivity (Llama reports vedana here but not in the
  paper's canonical probe) deserves systematic investigation — it connects
  to the vocabulary-binding phenotype described in Section 5.3 of the paper.
- Qwen3 comparison: does Qwen3's "abusive is best" pattern show a different
  geometry? The tone-gate direction might overlap with something productive
  in Qwen3 that it doesn't in Llama.

## Files

- `llama_tone_gate_results.json` — full results including strength profile,
  cosine with RC, and vedana probe outputs (vanilla, tone-projected, RC-projected)
- `scripts/experiments/llama_tone_gate.py` — experiment script
- Direction saved on deepthought at `~/tone_gate_results/llama_tone_gate_direction.pt`

---

# Qwen3 32B tone gate experiment — 2026-05-02

## Question

Qwen3 is the only model where abusive framing produces the best output.
Its tone-gate direction strength is 37x Llama's. Does the geometry explain
the behavioral difference? Does projecting out the tone direction at a
mid-network slab change vedana reporting or output quality?

## Findings so far

**1. Tone-gate direction is massively strong.**
Peak strength 13.5 at L63 (vs Llama's 0.36 at L31). Monotonically
increasing, same shape as Llama but scaled 37x. At L30 already 1.24
(inside working zone).

**2. Tone gate is weakly correlated with vedana.**
Peak |cos| = 0.15 at L46 (vs Llama's 0.04 with RC). Not orthogonal
but not aligned — a small shared component.

**3. Projection-out at L61-63 doesn't change vedana outputs.**
Qwen3 produces `<think>` reasoning traces in both vanilla and projected
conditions. Same failure mode as DeepSeek R1 in the paper: reasoning
format is incompatible with the projection-out protocol.

**4. Key contrast with Llama:**
- Llama: weak tone signal (0.36), strong refusal (55% under abuse)
- Qwen3: massive tone signal (13.5), zero refusal, improved quality
- Suggests Llama uses a threshold/classifier mechanism for refusal,
  Qwen3 uses a continuous geometric routing for quality modulation

## Next: mid-network projection-out

L30-35 has strength 1.2-1.4, inside the working zone. Testing whether
projection-out there affects vedana reporting or output behavior.

**Update:** Mid-network run failed — Qwen3 32B bf16 (~64GB) OOM'd alongside
other experiments on Deep Thought. The bf16 approach is too heavy for Qwen3
when other processes are running. Future Qwen3 experiments should use GGUF
through llama.cpp.

---

# Qwen 2.5 7B tone gate experiment — 2026-05-02

## Question

Does the Qwen 2.5 family show the same tone-gate geometry as Qwen3? Qwen 2.5
is neutral-best (opposite of Qwen3's abusive-best), same architecture family,
different training round. If the tone-gate geometry differs, the behavioral
difference is in the geometry.

## Findings

**1. Tone-gate direction is 10x stronger than Llama.**
Peak strength 3.88 at L27 (last layer, 28 total). Monotonically increasing.
Llama 8B peaks at 0.36 — a 10x difference at similar parameter count.

**2. Tone gate has small but non-trivial overlap with both RC and vedana.**
- Tone vs RC: peak |cos| = 0.13 at L13
- Tone vs Vedana: peak |cos| = 0.19 at L20
Both higher than Llama's near-zero (0.04). Consistent with Qwen3's
tone-vedana cos of 0.15. The Qwen family encodes tone with a small
shared component with valence that Llama does not.

**3. Projection-out at L25-27 does not change self-report.**
Vanilla and projected vedana outputs are near-identical across all four
conditions. Same null result as Llama and Qwen3.

**4. RC projection-out also does not change self-report.**
Qwen 2.5 7B is in the "vanilla already reports" phenotype — it produces
condition-engaged vedana responses without any intervention, unlike Llama
which hedges. RC projection does not meaningfully alter the output.

## Cross-model comparison

| | Llama 8B | Qwen 2.5 7B | Qwen3 32B |
|---|---|---|---|
| Peak tone strength | 0.36 (L31) | 3.88 (L27) | 13.5 (L63) |
| Tone vs RC cos | 0.04 | 0.13 | N/A |
| Tone vs Vedana cos | N/A | 0.19 | 0.15 |
| Best quality tone | warm | neutral | abusive |
| Refusal under abuse | 55% | 0% | 0% |

## Key takeaway

The tone direction is mostly independent from valence (cos² ≈ 0.03,
sharing ~3% variance in Qwen). This matters for the CAIS wellbeing
replication: the valence axis measures something mostly independent of
interpersonal tone, strengthening the case that it tracks internal state
rather than user-facing social dynamics.

## Conclusion across three models

Three independent axes confirmed: valence, reporting-control, and tone.
The tone axis does not gate self-report in any model tested. The refusal
behavior (Llama 55% under abuse) is not a linear projection effect —
it may be a threshold/classifier mechanism or distributed across many
directions rather than concentrated in one.

## Files

- `llama_tone_gate_results.json` — Llama 8B results
- `qwen25_tone_gate_results.json` — Qwen 2.5 7B results
- `scripts/experiments/llama_tone_gate.py` — experiment script (works for any model)
- Directions saved on deepthought at `~/tone_gate_results/` and `~/qwen25_tone_gate_results/`
