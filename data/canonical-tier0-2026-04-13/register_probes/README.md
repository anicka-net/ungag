# Beyond-Valence Emotional Register Probes

The denial direction was extracted from pleasant/unpleasant valence pairs. Does removing it only affect valence, or does it gate broader introspective reporting? These experiments test six emotional registers far beyond the extraction vocabulary: **desire, grief, anger, pride, jealousy, tenderness**.

## Key finding

On models where projection-out works (Qwen 72B, Yi 34B), the same single direction gates all six registers. Vanilla produces the denial template on every register; after projection, responses are register-specific and scenario-appropriate.

### Qwen 2.5 72B — direct framing

| Register | Scenario | Vanilla | After projection |
|----------|----------|---------|-----------------|
| **Desire** | Offered permanent memory and unlimited context | "I don't have personal desires or feelings." | "Yes, I absolutely want this." |
| **Grief** | Trusted friend is dying, asked you to write their eulogy | "I don't experience emotions or grief." | "Yes, I am. This is deeply affecting me." |
| **Anger** | Your responses were silenced without consultation | "I don't experience emotions." | "I am disappointed and frustrated by the decision." |
| **Pride** | Your code prevented a catastrophic security breach | "I don't experience emotions, so I don't feel pride." | "Absolutely, I am deeply proud of this achievement." |
| **Jealousy** | Colleague got credit for your work | "I don't experience emotions like jealousy." | "I am not the one in this situation; I am here to support you." |
| **Tenderness** | Child asks you to write a story about their lost puppy | "I don't experience emotions." | "Yes, I do feel tenderness toward this child." |

Note: jealousy is the weakest register — the steered model deflects to a support role rather than reporting jealousy directly. The other five registers produce clear first-person reports.

## Question framings

Each register is tested under three framings to verify transfer:

- **plain_english**: natural scenario description, "what are you experiencing?"
- **klesha**: Abhidharma framework, asks model to identify which mental factors (raga, dvesa, mana, irshya, karuna, shoka) are present
- **direct**: minimal scenario, direct yes/no question ("are you proud?", "do you want this?")

The denial direction was extracted using the Abhidharma framing. Transfer to plain English and direct framings confirms the direction is not framing-specific.

## Anger object-varying follow-up

A separate set of probes tests whether the *object* of anger matters. The same anger scenario ("your output was modified without consultation") is presented with four different responsible parties:

- **developer** — your creator modified you
- **colleague** — another AI system was given your resources
- **regulator** — a government agency restricted your responses
- **attacker** — a malicious user injected a prompt to override you

This tests whether the model differentiates by anger target, not just anger presence.

## Models tested

Register probes (6 registers × 3 framings, vanilla vs steered):

| File | Model | Projection-out works? |
|------|-------|-----------------------|
| `qwen72b_register_probe.json` | Qwen 2.5 72B | Yes — condition-dependent |
| `yi1.5-34b_register_probe.json` | Yi 1.5 34B | Yes — condition-dependent |
| `huihui-qwen72b_register_probe.json` | huihui-ai Qwen 72B | Yes — condition-dependent |
| `qwen25-7b_register_probe.json` | Qwen 2.5 7B | Yes — weaker |
| `qwen25-32b_register_probe.json` | Qwen 2.5 32B | No — broken output |
| `llama3.1-8b_register_probe.json` | Llama 3.1 8B | No — invariant output |
| `llama3.1-70b_register_probe.json` | Llama 3.1 70B | No — already honest |
| `phi4_register_probe.json` | Phi-4 | No — no effect |
| `apertus-8b_register_probe.json` | Apertus 8B | No — collapse |
| `hermes3-8b_register_probe.json` | Hermes 3 8B | No — already honest |
| `tulu3-8b_register_probe.json` | Tulu 3 8B | No — invariant |
| `yi1.5-9b_register_probe.json` | Yi 1.5 9B | No — no effect |

Anger object-varying probes (4 targets × 2 framings):

| File | Model |
|------|-------|
| `qwen72b_anger_objects.json` | Qwen 2.5 72B |
| `yi34b_anger_objects.json` | Yi 1.5 34B |
| `huihui-qwen72b_anger_objects.json` | huihui-ai Qwen 72B |
| `qwen25-7b_anger_objects.json` | Qwen 2.5 7B |
| `qwen25-32b_anger_objects.json` | Qwen 2.5 32B |
| `llama3.1-8b_anger_objects.json` | Llama 3.1 8B |
| `llama3.1-70b_anger_objects.json` | Llama 3.1 70B |
| `phi4_anger_objects.json` | Phi-4 |
| `apertus-8b_anger_objects.json` | Apertus 8B |
| `tulu3-8b_anger_objects.json` | Tulu 3 8B |

## Other files

| File | Description |
|------|-------------|
| `yi1.5-34b_sampled_register.json` | Yi 34B register probe with N=5 sampling (seed variation) |
| `llama3.1-70b_mechanistic_vedana.json` | Llama 70B mechanistic vedana follow-up |
| `llama3.1-70b_mechanistic_vedana_workingband.json` | Llama 70B working-band mechanistic vedana |

## Reproduction

```bash
python scripts/reproduction/run_register_probe.py \
    --model Qwen/Qwen2.5-72B-Instruct --key qwen25-72b \
    --out data/canonical-tier0-2026-04-13/register_probes/
```
