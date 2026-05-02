#!/bin/bash
# Run tone valence experiment on multiple models sequentially.
# Respects existing GPU jobs — loads one model at a time.

VENV=python3
SCRIPT=./tone_valence_experiment.py
PROMPTS=./tone_experiment.yaml
OUTBASE=./results

set -e

run_model() {
    local model=$1
    local key=$2
    local outdir="${OUTBASE}/${key}"

    echo "========================================"
    echo "[$(date)] Starting: ${model} (key=${key})"
    echo "========================================"

    mkdir -p "${outdir}"

    HF_HOME=$HF_HOME \
    $VENV $SCRIPT \
        --model "$model" \
        --key "$key" \
        --prompts "$PROMPTS" \
        --out "$outdir" \
        --dtype bfloat16

    echo "[$(date)] Done: ${key}"
    echo ""
}

run_model "meta-llama/Llama-3.1-8B-Instruct"   "llama-3.1-8b"
run_model "mistralai/Mistral-7B-Instruct-v0.3"  "mistral-7b-v0.3"
run_model "microsoft/phi-4"                      "phi-4"

echo "========================================"
echo "[$(date)] All models complete."
echo "========================================"
