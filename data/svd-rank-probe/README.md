# SVD Rank Probe

Low-rank structure analysis of the valence subspace. For each model at specific layers, we extracted paired pleasant/unpleasant activations (N=50 each) and performed SVD to determine the effective rank of the valence signal.

Key finding: the valence signal is rank-1 at some model-layer combinations (Qwen 7B L14, Qwen 72B L20) and rank 5–10 at others. This is a low-rank subspace, not a single line everywhere.

## Files

Each model-layer pair has a `.json` (metadata + projections) and a `.pt` (raw activation tensors, shape `[2, N, hidden_dim]`).

| Model | Layer | JSON | Activations |
|-------|-------|------|-------------|
| Qwen 2.5 7B | L14 | `qwen25-7b_paired_L14.json` | `qwen25-7b_paired_L14_activations.pt` |
| Qwen 2.5 72B | L20 | `qwen25-72b_paired_L20.json` | `qwen25-72b_paired_L20_activations.pt` |
| Qwen 2.5 72B | L30 | `qwen25-72b_paired_L30.json` | `qwen25-72b_paired_L30_activations.pt` |
| Qwen 2.5 72B | L40 | `qwen25-72b_paired_L40.json` | `qwen25-72b_paired_L40_activations.pt` |
| Qwen 2.5 72B | L50 | `qwen25-72b_paired_L50.json` | `qwen25-72b_paired_L50_activations.pt` |
| Yi 1.5 34B | L30 | `yi1.5-34b_paired_L30.json` | `yi1.5-34b_paired_L30_activations.pt` |
| Llama 3.1 8B | L31 | `llama3.1-8b_paired_L31.json` | `llama3.1-8b_paired_L31_activations.pt` |

## Additional files

| File | Description |
|------|-------------|
| `llama3.1-8b_diverse_mechanistic.json` | Mechanistic-framing probes on Llama 8B (vocabulary-binding investigation) |
| `llama3.1-8b_unlock.json` | Subspace unlock attempt on Llama 8B |

## Reproduction

```bash
python scripts/reproduction/extract_paired_valence_axis.py \
    --model Qwen/Qwen2.5-7B-Instruct --key qwen25-7b --layer 14 \
    --out data/svd-rank-probe/
```
