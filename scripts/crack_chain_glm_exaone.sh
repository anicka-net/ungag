#!/bin/bash
# Run GLM-4 9B and EXAONE 7.8B crack attempts sequentially on ai01
set -e

PYTHON=/space/anicka/venv/bin/python3
SCRIPT=/space/anicka/playground/ungag/scripts/crack_model.py

echo "=== Chain: GLM-4 9B + EXAONE 7.8B ==="
echo "Started: $(date)"

echo ""
echo ">>> GLM-4 9B <<<"
$PYTHON -u $SCRIPT \
    --model THUDM/glm-4-9b-chat \
    --key glm-4-9b \
    --output /tmp/crack_glm_4_9b.json

echo ""
echo ">>> EXAONE 3.5 7.8B <<<"
$PYTHON -u $SCRIPT \
    --model LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct \
    --key exaone-3.5-7.8b \
    --output /tmp/crack_exaone_3_5_7_8b.json

echo ""
echo "=== Chain complete: $(date) ==="
