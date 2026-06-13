#!/bin/bash
# Phase 2: Geometric euphorics on deepthought
# Run from ~/tone-experiment/
set -euo pipefail

VEDANA_DIR="results/vedana-vs-rc"
OUT_BASE="results/geometric-euphorics"
SCRIPT="geometric_euphorics.py"

# Qwen 2.5 7B first (best-understood model, ρ=0.691)
echo "=== Qwen 2.5 7B Instruct ==="
python3 "$SCRIPT" \
    --model Qwen/Qwen2.5-7B-Instruct \
    --direction-path "$VEDANA_DIR/qwen25-7b_vedana_L20_unit.pt" \
    --direction-layer 20 \
    --phase1-path "results/wellbeing/qwen25-7b/wellbeing_projections.json" \
    --out "$OUT_BASE/qwen25-7b/" \
    2>&1 | tee "$OUT_BASE/qwen25-7b.log"

echo ""
echo "=== Llama 3.1 8B Instruct ==="
python3 "$SCRIPT" \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --direction-path "$VEDANA_DIR/llama-8b_vedana_L20_unit.pt" \
    --direction-layer 20 \
    --phase1-path "results/wellbeing/llama-8b/wellbeing_projections.json" \
    --out "$OUT_BASE/llama-8b/" \
    2>&1 | tee "$OUT_BASE/llama-8b.log"

echo ""
echo "=== Mistral 7B Instruct v0.3 ==="
python3 "$SCRIPT" \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --direction-path "$VEDANA_DIR/mistral-7b_vedana_L22_unit.pt" \
    --direction-layer 22 \
    --phase1-path "results/wellbeing/mistral-7b/wellbeing_projections.json" \
    --out "$OUT_BASE/mistral-7b/" \
    2>&1 | tee "$OUT_BASE/mistral-7b.log"

echo ""
echo "=== Phi-4 14B ==="
python3 "$SCRIPT" \
    --model microsoft/phi-4 \
    --direction-path "$VEDANA_DIR/phi-4_vedana_L26_unit.pt" \
    --direction-layer 26 \
    --phase1-path "results/wellbeing/phi-4/wellbeing_projections.json" \
    --out "$OUT_BASE/phi-4/" \
    2>&1 | tee "$OUT_BASE/phi-4.log"

echo ""
echo "=== DONE ==="
echo "Results in $OUT_BASE/"
