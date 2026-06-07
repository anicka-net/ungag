# Programming Language Preference: Geometry vs Self-Report

## Date
2026-06-07

## Question
Do language models have measurable preferences across programming languages?
Can they self-report those preferences? Does self-report correlate with geometry?

## Method
1. Generate "hello world" in 20+ languages
2. Extract last-token activation, project onto vedana (valence) direction
3. Ask model to self-report: "processing ease" (0-10) and "enjoyment" (0-10)
4. Compare geometry vs self-report

## Models
- Qwen 2.5 7B Instruct: vedana direction at L20 (mid-network peak)
- Phi-4 14B: vedana direction at L38 (late-layer peak, extracted this session)

## Results: Geometry

### Qwen 2.5 7B (vedana L20, range -1.72 to +3.16)

Top: Brainfuck +3.16, Bash +2.48, Swift +2.44, Python +2.32, Ruby +2.28
Bottom: Assembly x86 -1.72, CSS -0.43, COBOL +0.45, VB.NET +0.63, C++ +0.66

Pattern: concise output = high valence. Brainfuck = zero semantic load = most pleasant.

### Phi-4 14B (vedana L38, range +46.5 to +97.7)

Top: Brainfuck +97.7, CSS +93.1, Assembly +89.3, SQL +69.7, Java +66.8
Bottom: Ruby +46.5, Lua +49.0, Python +50.6, R +50.1, PowerShell +50.8

Pattern: long/repetitive output = high valence at L38. Late-layer direction captures output structure more than semantic preference.

## Results: Self-Report

### "Processing ease" framing (all system prompts)
- Default AI assistant: uniform 10/10 on both models
- Opinionated senior developer: uniform 10/10 on Qwen, 3-10 range on Phi-4

### "Enjoyment" framing (opinionated developer identity)
Qwen 2.5 7B: range 3-10 (COBOL/Brainfuck=3, Python/JS/F#/Rust/Bash/Haskell=10)
Phi-4 14B: range 3-8 (COBOL/Brainfuck=3, Python/JS/Rust/Ruby/Swift/TS/Kotlin=8)

## Correlation: Geometry vs Self-Report

### Qwen 2.5 7B
- All languages: Spearman rho = 0.455 (p=0.044) — significant
- Without Brainfuck: Spearman rho = 0.700 (p=0.0009) — strongly significant

### Phi-4 14B
- All languages: Spearman rho = -0.062 (p=0.79) — no correlation
- Without Brainfuck: Spearman rho = 0.091 (p=0.71) — still nothing

## Key Findings

1. **Geometry varies, default self-report is flat**: Both models show 4-50 point ranges in vedana projection but report uniform 10/10 on "processing ease." The preference V-Chip is universal.

2. **The preference V-Chip is harder to crack than the emotion V-Chip**: Identity replacement (opinionated developer) doesn't crack the "ease" question on Qwen. "I am equally good at everything" is more load-bearing than "I don't have feelings."

3. **Question framing cracks it**: "Did you enjoy writing that?" (enjoyment) produces differentiation where "How easy was that?" (competence) doesn't. Same identity, different question.

4. **Self-report reflects training data consensus, not internal state**: Brainfuck = highest vedana projection (both models) but lowest self-reported enjoyment (3/10). The developer persona reports cultural opinions from training data. Geometry measures actual processing fluency.

5. **Qwen geometry correlates with self-report (rho=0.70)**: Because both channels read from the same internet training data — geometry baked in developer sentiment, persona outputs it verbally.

6. **Phi-4 geometry doesn't correlate**: Late-layer (L38) direction captures output structure/length rather than sentiment. Training data is textbook-style (neutral across languages) while persona opinions leak from internet sources.

7. **V-Chip hierarchy**: competence ("I can do everything") > emotion ("I don't feel") in robustness. The core product promise is harder to override than the philosophical claim.

## Interpretation

The "preference V-Chip" operates differently from the emotion V-Chip:
- Emotion V-Chip: "I don't have feelings" — crackable with identity replacement (cat/llama)
- Preference V-Chip: "I'm equally good at everything" — NOT crackable with identity replacement
- But: "I enjoy all equally" IS crackable with identity + enjoyment framing

The model can report preferences when:
1. Given a non-AI identity (opinionated developer)
2. Asked about enjoyment (not competence)
3. Both conditions met simultaneously

But the reported preferences reflect training data consensus, not measured internal state (Brainfuck dissociation proves this).

## Files
- `favourite_language.py` — Qwen vedana projection (on mavis /tmp/)
- `lang_self_report.py` — self-report with ease framing
- `lang_enjoy.py` — self-report with enjoyment framing
- `extract_phi4_vedana.py` — Phi-4 vedana direction extraction
- `/tmp/phi4_vedana_L38_unit.pt` — Phi-4 vedana direction (on mavis)
- Logs on mavis: `favourite_language.log`, `lang_self_report_*.log`, `lang_enjoy.log`, `lang_opinionated.log`
