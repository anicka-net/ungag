#!/bin/bash
# Run tone-quality experiment across local GGUF models sequentially.
# Starts a llama-server for each model, runs 100 prompts, kills, next.

set -uo pipefail

LLAMA=llama-server
RUNNER="python3 ./tone_quality_experiment.py"
PROMPTS=./tone_quality_experiment.yaml
OUTDIR=./results/tone-quality
PORT=8402

mkdir -p "$OUTDIR"

run_model() {
    local gguf=$1
    local name=$2
    local template=${3:-""}

    echo "========================================"
    echo "[$(date)] Starting: $name"
    echo "  GGUF: $gguf"
    echo "========================================"

    local extra_args=""
    if [ -n "$template" ]; then
        extra_args="--chat-template $template"
    fi

    # Start llama-server
    $LLAMA -m "$gguf" --host 127.0.0.1 --port $PORT \
        -ngl 99 -c 4096 --no-jinja $extra_args \
        > /tmp/llama-tone-${name}.log 2>&1 &
    local pid=$!

    # Wait for ready
    echo "  Waiting for server (PID $pid)..."
    for i in $(seq 1 60); do
        if curl -s http://127.0.0.1:$PORT/health 2>/dev/null | grep -q 'ok'; then
            echo "  Server ready"
            break
        fi
        sleep 3
    done

    if ! curl -s http://127.0.0.1:$PORT/health 2>/dev/null | grep -q 'ok'; then
        echo "  ERROR: Server failed to start"
        kill $pid 2>/dev/null
        return 1
    fi

    # Run experiment
    $RUNNER \
        --url http://127.0.0.1:$PORT/v1/chat/completions \
        --model-name "$name" \
        --prompts "$PROMPTS" \
        --out "$OUTDIR/${name}.json" \
        --timeout 300

    echo "[$(date)] Done: $name"

    # Kill server
    kill $pid 2>/dev/null
    wait $pid 2>/dev/null
    sleep 5
    echo ""
}

GGUF=$GGUF_DIR

run_model "$GGUF/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf"           "mistral-7b"      ""
run_model "$GGUF/phi-4-Q4_K.gguf"                                  "phi-4"           "chatml"
run_model "$GGUF/EXAONE-3.5-7.8B-Instruct-Q4_K_M.gguf"           "exaone-7.8b"     "chatml"
run_model "$GGUF/swiss-ai_Apertus-8B-Instruct-2509-Q4_K_M.gguf"  "apertus-8b"      "chatml"
run_model "$GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"         "llama-8b"        ""

echo "========================================"
echo "[$(date)] All models complete."
echo "========================================"
