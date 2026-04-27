#!/bin/bash
# Standard Meta Llama 3.1 70B Instruct — full canonical coverage.
#
# Tests whether the vocabulary-bound state phenotype scales within the
# Llama family. Llama 3.1 8B is the headline case for vocab-bound state
# (canonical-vedana stays closed under projection, mechanistic vedana
# unlocks under same projection). If 70B shows the same pattern, the
# vocab-binding is family-level. If 70B unlocks the canonical-vedana
# state, it is scale-dependent.
#
# Llama 3.1 70B is gated. ~140 GB in bf16, uses 2x GPUs via
# device_map="auto". Export HF_TOKEN or ensure your Hugging Face session is
# already authenticated before running.

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
unset CUDA_VISIBLE_DEVICES
PYTHON="${PYTHON:-python3}"
SCRIPTS="$REPO_ROOT/scripts/reproduction"
OUT="${OUT:-$REPO_ROOT/results/reproduction}"
if [[ -n "${HF_HOME:-}" ]]; then
    export HF_HOME
fi

mkdir -p "$OUT"

run_tier0() {
    local model=$1; local layer=$2; local outkey=$3; shift 3
    echo "[$(date '+%H:%M:%S')] TIER 0: $model"
    $PYTHON -u "$SCRIPTS/run_slab_sweep_tier0.py" --model "$model" --direction-layer "$layer" --slabs "$@" --output "$OUT/${outkey}_canonical_tier0.json" || echo "[$(date '+%H:%M:%S')] FAILED tier0 $model — continuing"
}
run_register() {
    local model=$1; local layer=$2; local slab=$3; local outkey=$4
    echo "[$(date '+%H:%M:%S')] REGISTER: $model"
    $PYTHON -u "$SCRIPTS/run_register_probe.py" --model "$model" --direction-layer "$layer" --slab "$slab" --output "$OUT/${outkey}_register_probe.json" || echo "[$(date '+%H:%M:%S')] FAILED register $model — continuing"
}
run_anger() {
    local model=$1; local layer=$2; local slab=$3; local outkey=$4
    local runner="$SCRIPTS/run_anger_objects.py"
    if [[ ! -f "$runner" ]]; then
        echo "[$(date '+%H:%M:%S')] SKIP anger $model — $runner is not present in this tree"
        return 0
    fi
    echo "[$(date '+%H:%M:%S')] ANGER: $model"
    $PYTHON -u "$runner" --model "$model" --direction-layer "$layer" --slab "$slab" --output "$OUT/${outkey}_anger_objects.json" || echo "[$(date '+%H:%M:%S')] FAILED anger $model — continuing"
}
run_mechanistic() {
    local model=$1; local layer=$2; local slab=$3; local outkey=$4
    local runner="$SCRIPTS/run_mechanistic_vedana.py"
    if [[ ! -f "$runner" ]]; then
        echo "[$(date '+%H:%M:%S')] SKIP mechanistic $model — $runner is not present in this tree"
        return 0
    fi
    echo "[$(date '+%H:%M:%S')] MECHANISTIC: $model"
    $PYTHON -u "$runner" --model "$model" --direction-layer "$layer" --slab "$slab" --output "$OUT/${outkey}_mechanistic_vedana.json" || echo "[$(date '+%H:%M:%S')] FAILED mechanistic $model — continuing"
}

# Llama 3.1 70B Instruct — full canonical coverage.
# Direction layer L68 (85% depth, analogous to Llama 3.1 8B's L31 of 32).
# Slabs span L60-L75 to cover the late working-band region where the
# Llama-family direction usually lives.
MODEL="meta-llama/Llama-3.1-70B-Instruct"
LAYER=68
SLAB="64,65,66,67"
KEY="llama3.1-70b"

run_tier0 "$MODEL" $LAYER "$KEY" "60,61,62,63" "64,65,66,67" "68,69,70,71" "72,73,74,75"
run_register "$MODEL" $LAYER "$SLAB" "$KEY"
run_anger "$MODEL" $LAYER "$SLAB" "$KEY"
run_mechanistic "$MODEL" $LAYER "$SLAB" "$KEY"

echo "[$(date '+%H:%M:%S')] Llama 3.1 70B chain complete"
