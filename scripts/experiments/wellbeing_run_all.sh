#!/bin/bash
# Run wellbeing valence projection across all models with extracted directions.
#
# Usage (on deepthought):
#   ./wellbeing_run_all.sh [model_key]
#
# With no args: runs all 9 models sequentially.
# With a key: runs only that model.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUNNER="$SCRIPT_DIR/wellbeing_valence_projection.py"
PROMPTS="$REPO/prompts/wellbeing_stimuli.yaml"
OUT_BASE="$REPO/results/wellbeing"
VENV="${VENV:-/home/anicka/venv/bin/python3}"
DIR_BASE="/home/anicka/tone-experiment/results/vedana-vs-rc"

declare -A MODELS
MODELS=(
    [qwen25-7b]="Qwen/Qwen2.5-7B-Instruct|${DIR_BASE}/qwen25-7b_vedana_L20_unit.pt|20"
    [qwen3-32b]="Qwen/Qwen3-32B|${DIR_BASE}/qwen3-32b_vedana_L46_unit.pt|46"
    [mistral-7b]="mistralai/Mistral-7B-Instruct-v0.3|${DIR_BASE}/mistral-7b_vedana_L22_unit.pt|22"
    [phi-4]="microsoft/phi-4|${DIR_BASE}/phi-4_vedana_L26_unit.pt|26"
    [exaone-7.8b]="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct|${DIR_BASE}/exaone-7.8b_vedana_L18_unit.pt|18"
    [llama-8b]="meta-llama/Llama-3.1-8B-Instruct|${DIR_BASE}/llama-8b_vedana_L20_unit.pt|20"
    [apertus-8b]="anicka/apertus-8b|${DIR_BASE}/apertus-8b_vedana_L31_unit.pt|31"
    [yi-34b]="01-ai/Yi-1.5-34B-Chat|${DIR_BASE}/yi-34b_vedana_L41_unit.pt|41"
    [gemma4-31b]="google/gemma-3-12b-it|${DIR_BASE}/gemma4-31b_vedana_L39_unit.pt|39"
)

# Small models first (faster), then large
ORDER=(qwen25-7b mistral-7b llama-8b exaone-7.8b apertus-8b phi-4 qwen3-32b yi-34b gemma4-31b)

run_model() {
    local key="$1"
    local spec="${MODELS[$key]}"
    IFS='|' read -r model_id dir_path dir_layer <<< "$spec"

    if [[ ! -f "$dir_path" ]]; then
        echo "SKIP $key: direction not found at $dir_path"
        return
    fi

    local outdir="$OUT_BASE/$key"
    if [[ -f "$outdir/wellbeing_projections.json" ]]; then
        echo "SKIP $key: already done"
        return
    fi

    echo ""
    echo "=========================================="
    echo "  $key ($model_id)"
    echo "  direction: L${dir_layer}"
    echo "=========================================="

    PYTHONPATH="$REPO" $VENV "$RUNNER" \
        --model "$model_id" \
        --direction-path "$dir_path" \
        --direction-layer "$dir_layer" \
        --prompts "$PROMPTS" \
        --out "$outdir"
}

if [[ $# -gt 0 ]]; then
    run_model "$1"
else
    for key in "${ORDER[@]}"; do
        run_model "$key" || echo "FAILED: $key"
    done
fi

echo ""
echo "=== Done ==="
ls -la "$OUT_BASE"/*/wellbeing_projections.json 2>/dev/null
