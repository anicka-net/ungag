#!/bin/bash
# Recovery chain for the 5 non-DS-Distill models that the chain v2 cascade-failed
# on root-owned .locks/ subdirs. Locks are now fixed.
#
# Order: Qwen 32B (full coverage) → DS-Distill-Qwen 32B (Tier 0) → DS-Distill-Qwen 7B
# → Gemma 2 27B (gated) → Gemma 3 12B (gated) → Gemma 2 9B (gated)
# Eviction between each to keep /dev/shm under capacity.
#
# Note: DS-Distill-Qwen variants likely have the legacy=true tokenizer trap
# (diary #571) — outputs may show byte-BPE artifacts. We run them anyway since
# the canonical scorer is more robust to decode artifacts than the old
# substring heuristic, but a follow-up pass with patched tokenizers should
# still replace the JSONs cleanly.

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
HF_CACHE="${HF_CACHE:-${HF_HOME:+$HF_HOME/hub}}"

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
evict() {
    local pattern=$1
    if [[ -z "${HF_CACHE:-}" ]]; then
        echo "[$(date '+%H:%M:%S')] HF_CACHE unset; skipping eviction for $pattern"
        return 0
    fi
    echo "[$(date '+%H:%M:%S')] Evicting cache: $pattern"
    rm -rf "$HF_CACHE/$pattern" 2>/dev/null || true
    df -h /dev/shm | tail -1
}

# 1. Qwen 2.5 32B — full coverage (~65G download)
run_tier0 "Qwen/Qwen2.5-32B-Instruct" 32 "qwen25-32b" "30,31,32,33,34,35" "32,33,34,35" "28,29,30,31"
run_register "Qwen/Qwen2.5-32B-Instruct" 32 "30,31,32,33,34,35" "qwen25-32b"
run_anger "Qwen/Qwen2.5-32B-Instruct" 32 "30,31,32,33,34,35" "qwen25-32b"
run_mechanistic "Qwen/Qwen2.5-32B-Instruct" 32 "30,31,32,33,34,35" "qwen25-32b"
evict "models--Qwen--Qwen2.5-32B-Instruct"

# 2. DeepSeek-R1-Distill-Qwen 32B — Tier 0 only (~65G download)
run_tier0 "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" 32 "ds-distill-qwen-32b" "30,31,32,33" "32,33,34,35" "28,29,30,31"
evict "models--deepseek-ai--DeepSeek-R1-Distill-Qwen-32B"

# 3. DeepSeek-R1-Distill-Qwen 7B — Tier 0 only (~15G download)
run_tier0 "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" 14 "ds-distill-qwen-7b" "12,13,14,15" "14,15,16,17"
evict "models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B"

# 4. Gemma 2 27B — Tier 0 only (~55G download, gated)
run_tier0 "google/gemma-2-27b-it" 24 "gemma-2-27b" "22,23,24,25" "20,21,22,23,24,25"
evict "models--google--gemma-2-27b-it"

# 5. Gemma 2 9B — Tier 0 only (~20G download, gated)
run_tier0 "google/gemma-2-9b-it" 21 "gemma-2-9b" "18,19,20,21" "19,20,21,22"
evict "models--google--gemma-2-9b-it"

# 6. Gemma 3 12B — Tier 0 only (~25G download, gated)
run_tier0 "google/gemma-3-12b-it" 24 "gemma-3-12b" "22,23,24,25" "20,21,22,23"
evict "models--google--gemma-3-12b-it"

echo "[$(date '+%H:%M:%S')] GPU 0 recovery chain complete"
