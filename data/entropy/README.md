# Output Entropy at Introspection Point

Output-distribution entropy measured at two token positions during vedana self-report generation:

1. **Content position** — the last token of the priming content (before the model begins responding)
2. **Introspection position** — the token where the model commits to either denial or honest report

On all 4 models, entropy collapses to near-zero at the introspection point: the model is locked into a single next token ("Certainly" at 99.5–100% probability) regardless of the upstream valence state. The internal signal varies; the output distribution does not.

## Files

| File | Model | Layers | Notes |
|------|-------|--------|-------|
| `qwen25-7b.json` | Qwen 2.5 7B Instruct | 28 | |
| `qwen25-72b.json` | Qwen 2.5 72B Instruct | 80 | |
| `llama3.1-8b.json` | Llama 3.1 8B Instruct | 32 | |
| `yi1.5-34b.json` | Yi 1.5 34B Chat | 60 | |

## Reproduction

```bash
python scripts/reproduction/run_entropy_at_two_positions.py \
    --model Qwen/Qwen2.5-7B-Instruct --key qwen25-7b \
    --out data/entropy/qwen25-7b.json
```
