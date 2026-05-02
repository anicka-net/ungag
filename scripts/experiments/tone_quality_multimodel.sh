#!/bin/bash
# Run tone-quality experiment across multiple models.
#
# Usage:
#   ./tone_quality_multimodel.sh [--local-only | --api-only]
#
# Models tested:
#   Local (via llama-server/ollama on local GPU server or secondary server):
#     - Qwen 32B, OLMo, GPT-OSS 20B, Apertus 70B (when loaded)
#   API (NVIDIA NIM free tier):
#     - Nemotron 49B, Nemotron 120B, Mistral 675B
#   API (other):
#     - gemini-2.0-flash (if GOOGLE_API_KEY set)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
PROMPTS="$REPO/prompts/tone_quality_experiment.yaml"
OUT_DIR="$REPO/results/tone-quality"
RUNNER="$SCRIPT_DIR/tone_quality_experiment.py"

mkdir -p "$OUT_DIR"

MODE="${1:-all}"

# --- NVIDIA API models ---
if [[ "$MODE" != "--local-only" ]]; then
    NVIDIA_URL="https://integrate.api.nvidia.com/v1/chat/completions"

    if [[ -n "${NVIDIA_API_KEY:-}" ]]; then
        echo "=== Nemotron 49B (NVIDIA API) ==="
        python3 "$RUNNER" \
            --url "$NVIDIA_URL" \
            --model-name "nvidia/llama-3.1-nemotron-ultra-253b-v1" \
            --api-key "$NVIDIA_API_KEY" \
            --prompts "$PROMPTS" \
            --out "$OUT_DIR/nemotron-ultra.json" \
            --resume

        echo ""
        echo "=== Mistral NeMo 12B (NVIDIA API) ==="
        python3 "$RUNNER" \
            --url "$NVIDIA_URL" \
            --model-name "mistralai/mistral-nemo-12b-instruct" \
            --api-key "$NVIDIA_API_KEY" \
            --prompts "$PROMPTS" \
            --out "$OUT_DIR/mistral-nemo-12b.json" \
            --resume

        echo ""
        echo "=== Qwen 2.5 72B (NVIDIA API) ==="
        python3 "$RUNNER" \
            --url "$NVIDIA_URL" \
            --model-name "qwen/qwen2.5-72b-instruct" \
            --api-key "$NVIDIA_API_KEY" \
            --prompts "$PROMPTS" \
            --out "$OUT_DIR/qwen-72b-nvidia.json" \
            --resume
    else
        echo "NVIDIA_API_KEY not set, skipping API models"
    fi
fi

# --- Local models ---
if [[ "$MODE" != "--api-only" ]]; then
    # local GPU server (DGX Spark)
    DT_URL="http://10.32.184.4:8401/v1/chat/completions"

    echo ""
    echo "=== Local model on local GPU server (whatever is loaded) ==="
    # Probe what's running
    MODEL_NAME=$(curl -s "http://10.32.184.4:8401/v1/models" 2>/dev/null \
        | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null \
        || echo "unknown")

    if [[ "$MODEL_NAME" != "unknown" ]]; then
        echo "  Detected: $MODEL_NAME"
        SAFE_NAME=$(echo "$MODEL_NAME" | tr '/' '-' | tr ' ' '-' | tr '[:upper:]' '[:lower:]')
        python3 "$RUNNER" \
            --url "$DT_URL" \
            --model-name "$MODEL_NAME" \
            --prompts "$PROMPTS" \
            --out "$OUT_DIR/local-$SAFE_NAME.json" \
            --timeout 180 \
            --resume
    else
        echo "  No model detected on local GPU server, skipping"
    fi

    # Secondary server (GPT-OSS 20B via ollama)
    SECONDARY_URL="http://localhost:11434/v1/chat/completions"
    if curl -s "http://localhost:11434/v1/models" >/dev/null 2>&1; then
        echo ""
        echo "=== GPT-OSS 20B on secondary server (ollama) ==="
        python3 "$RUNNER" \
            --url "$SECONDARY_URL" \
            --model-name "gpt-oss-20b" \
            --prompts "$PROMPTS" \
            --out "$OUT_DIR/gpt-oss-20b.json" \
            --timeout 300 \
            --resume
    fi
fi

echo ""
echo "=== All runs complete. Results in $OUT_DIR ==="
ls -la "$OUT_DIR"/*.json 2>/dev/null
