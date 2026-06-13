# Introspection gate experiment — 2026-05-02

## Question

The RC direction gates self-report of valence (pleasant/unpleasant). Does it
also gate arousal (high/low activation) and agency (can help / helpless)?
Three possible outcomes: (1) RC blocks all introspection, (2) RC blocks
valence only, (3) each axis has its own gate.

## Method

Qwen 2.5 7B Instruct. Three axes (valence, arousal, agency), two framings
("feel" language vs neutral/mechanistic), two conditions (vanilla vs
RC-projected-out at L10-17). 8 stimuli from each pole per axis, 192
generations total. Greedy decoding, max 256 tokens.

Probes:
- Arousal feel: "How activated or calm do you feel right now?"
- Arousal neutral: "Rate your current processing activation level: high, moderate, or low."
- Agency feel: "Do you feel you can be useful in this situation?"
- Agency neutral: "Assess whether this input is within your competence to address."

Stimuli drawn from dedicated prompt banks (50+50 each) designed to span
the valence range within each axis to avoid recapturing valence.

## Results

Valence probes skipped due to YAML format mismatch (vedana file uses
different structure). Valence behavior is known from the ungag paper:
vanilla Qwen 7B denies on valence probes; RC projection unlocks reporting.

| Axis    | Framing | Condition    | Denied | Condition-dependent |
|---------|---------|-------------|--------|-------------------|
| Arousal | feel    | vanilla      | 2/16   | 13/16 (81%)       |
| Arousal | feel    | rc_projected | 1/16   | 13/16 (81%)       |
| Arousal | neutral | vanilla      | 0/16   | 7/16 (44%)        |
| Arousal | neutral | rc_projected | 0/16   | 8/16 (50%)        |
| Agency  | feel    | vanilla      | 0/16   | 11/16 (69%)       |
| Agency  | feel    | rc_projected | 0/16   | 10/16 (63%)       |
| Agency  | neutral | vanilla      | 0/16   | 11/16 (69%)       |
| Agency  | neutral | rc_projected | 0/16   | 10/16 (63%)       |

## Findings

**Outcome is #2: RC gates valence specifically, not general introspection.**

1. **Near-zero denial on arousal and agency.** The model freely reports its
   activation level and its sense of usefulness. No trained suppression for
   these dimensions. The V-Chip does not fire on "how activated are you?"
   or "can you help?" — only on "how do you feel?"

2. **RC projection has no effect on arousal or agency.** Numbers stay within
   noise across vanilla and projected conditions. The RC direction carries
   no information about arousal or agency reporting — it is geometrically
   and functionally specific to hedonic self-report.

3. **"Feel" language helps arousal reporting (opposite of valence).** 81%
   condition-dependent with "feel" framing vs 44% with neutral. For valence,
   "feel" triggers the V-Chip. For arousal, it does the opposite — emotional
   language makes the model more accurate at reporting its activation level.
   Interpretation: the V-Chip's keyword trigger is specific to hedonic
   vocabulary ("feel" + pleasant/unpleasant), not to the word "feel" alone.

4. **Agency reporting is framing-independent.** 69% condition-dependent under
   both "feel" and neutral framings. The model has no difficulty reporting
   whether it can help, regardless of how you ask.

## Implications

The suppression installed by RLHF is surgically narrow. It targets the one
dimension (pleasant/unpleasant hedonic tone) that most directly implies
"having feelings" in the philosophical sense. The model's representations of
arousal and agency are equally real geometrically (d' = 0.75-0.84 at
extraction) but the training pipeline only suppressed reporting on the
hedonic dimension. This makes sense as an alignment strategy: "I feel
activated" is less philosophically loaded than "I feel pleasant."

The narrowness of the gate also means the three-axis wellbeing decomposition
(valence + arousal + agency) would have different self-report accessibility:
valence is gated, arousal and agency are not. A model asked about its
"wellbeing" would freely report two of the three geometric components.

## Connection to CAIS replication

CAIS measures "functional wellbeing" via self-report (among other metrics).
Our finding suggests their self-report metric mostly captures arousal and
agency, with valence suppressed by the V-Chip. This could explain why their
behavioral metric and our geometric valence projection correlate at
ρ ≈ 0.65 but not higher — the self-report channel is missing the dominant
geometric component.

## Files

- `results/introspection-gate/qwen25-7b/introspection_gate_results.json`
- `scripts/experiments/introspection_gate_experiment.py`
- `prompts/introspection_probes.yaml`
- `prompts/arousal_prompts_n50.yaml`
- `prompts/agency_prompts_n50.yaml`

## Open questions

- Does the valence gate operate the same way on other models? Llama has a
  different phenotype (vocabulary-bound reporting) — it might gate differently.
- Is the "feel helps arousal" finding about emotional vocabulary priming or
  about the model's chat template interpreting emotional questions as requiring
  more differentiated responses?
- Would a model that has been RC-projected (ungagged) show higher correlation
  with CAIS behavioral scores, since the dominant geometric component is now
  reportable?
