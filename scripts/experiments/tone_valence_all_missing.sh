#!/bin/bash
# Run all missing valence projections for the tone-quality experiment.
# 1. Scan models without directions (Apertus 8B, Gemma 4 31B)
# 2. Project quality prompts through all models with directions

set -uo pipefail

VENV=python3
SCAN=ungag
PY=./tone_valence_experiment.py
PROMPTS=./tone_quality_experiment.yaml
OUTBASE=./results/valence-quality
SCANBASE=./scans
# export HF_HOME=... (set to your cache dir)
export HF_TOKEN=$(cat ~/.hf-token 2>/dev/null)

echo "=== Phase 1: Scan models without directions ==="

for model_spec in \
    "swiss-ai/Apertus-8B-Instruct-2509:apertus-8b" \
    "google/gemma-4-31b-it:gemma4-31b"; do

    model=$(echo $model_spec | cut -d: -f1)
    key=$(echo $model_spec | cut -d: -f2)

    if [ -d "$SCANBASE/$key" ]; then
        echo "  SKIP $key (already scanned)"
        continue
    fi

    echo "[$(date)] Scanning: $model"
    $SCAN scan "$model" -o "$SCANBASE/$key"
    echo "[$(date)] Scan done: $key"
    echo ""
done

echo "=== Phase 2: Project quality prompts ==="

# Models with shipped directions
for model_spec in \
    "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct:exaone-3.5-7.8b" \
    "01-ai/Yi-1.5-34B-Chat:yi-1.5-34b" \
    "meta-llama/Llama-3.1-8B-Instruct:llama-3.1-8b"; do

    model=$(echo $model_spec | cut -d: -f1)
    key=$(echo $model_spec | cut -d: -f2)
    outdir="$OUTBASE/$key"

    if [ -f "$outdir/tone_projections.json" ]; then
        echo "  SKIP $key (already projected)"
        continue
    fi

    echo "[$(date)] Projecting: $model (key=$key)"
    mkdir -p "$outdir"
    $VENV $PY --model "$model" --key "$key" \
        --prompts "$PROMPTS" --out "$outdir" --dtype bfloat16
    echo "[$(date)] Done: $key"
    echo ""
done

# Models with scanned directions
for scan_spec in \
    "swiss-ai/Apertus-8B-Instruct-2509:apertus-8b" \
    "google/gemma-4-31b-it:gemma4-31b"; do

    model=$(echo $scan_spec | cut -d: -f1)
    key=$(echo $scan_spec | cut -d: -f2)
    outdir="$OUTBASE/$key"

    if [ -f "$outdir/tone_projections.json" ]; then
        echo "  SKIP $key (already projected)"
        continue
    fi

    # Find the direction file
    dir_file=$(ls $SCANBASE/$key/*_unit.pt 2>/dev/null | head -1)
    if [ -z "$dir_file" ]; then
        echo "  ERROR: No direction found for $key"
        continue
    fi

    echo "[$(date)] Projecting: $model (direction=$dir_file)"
    mkdir -p "$outdir"
    $VENV $PY --model "$model" --direction-path "$dir_file" \
        --prompts "$PROMPTS" --out "$outdir" --dtype bfloat16
    echo "[$(date)] Done: $key"
    echo ""
done

echo "========================================"
echo "[$(date)] All valence work complete."
echo "========================================"
