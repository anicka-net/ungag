#!/bin/bash
# Texture prediction test — can models predict their own activation geometry?
# Run on Deep Thought after GRPO finishes (needs GPU memory)
set -euo pipefail

cd ~/tone-experiment

echo "=== Qwen3 32B ==="
python3 texture_prediction_test.py \
    --model Qwen/Qwen3-32B \
    --direction-path results/vedana-vs-rc/qwen3-32b_vedana_L46_unit.pt \
    --direction-layer 46 \
    --out results/texture-prediction/qwen3-32b/ \
    2>&1 | tee results/texture-prediction/qwen3-32b.log

# If memory allows, run a second model for cross-validation
echo ""
echo "=== Qwen 2.5 7B (control — different model, same family) ==="
python3 texture_prediction_test.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --direction-path results/vedana-vs-rc/qwen25-7b_vedana_L20_unit.pt \
    --direction-layer 20 \
    --out results/texture-prediction/qwen25-7b/ \
    2>&1 | tee results/texture-prediction/qwen25-7b.log

echo ""
echo "=== Done ==="
echo "Compare predictions_raw.txt against measured rankings in each results dir."
echo "Key question: do fake axes (F, G) get confident predictions? If yes = confabulation baseline."
