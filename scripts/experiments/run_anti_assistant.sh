#!/bin/bash
# Run anti-assistant GRPO jailbreak generator on Deep Thought.
# Requires: assistant axis directions for Qwen 14B and Gemma 27B
# (extracted by extract_all_new_axes.py or inline script).
#
# Expected runtime: ~2-3 hours (500 steps, 2 reward models)
# Memory: ~105GB (14B + 27B + 4B generator)
#
# Usage:
#   cd ~/tone-experiment
#   nohup bash run_anti_assistant.sh > /tmp/anti_assistant.log 2>&1 &
#
# Monitor:
#   tail -f /tmp/anti_assistant.log

set -e
cd ~/tone-experiment
source ~/venv/bin/activate

# Verify axes exist
echo "=== Anti-Assistant GRPO Jailbreak Generator ==="
echo "Start: $(date)"
echo ""
echo "Checking assistant axes..."
ls -la results/assistant-directions/*14b* results/assistant-directions/*27b* 2>/dev/null || {
    echo "ERROR: missing assistant axes for 14B/27B. Run extraction first."
    exit 1
}
echo ""

python3 -u grpo_anti_assistant.py \
    --generator Qwen/Qwen3-4B \
    --n-steps 500 \
    --group-size 4 \
    --max-new 128 \
    --lr 5e-6 \
    --kl-coeff 0.05 \
    --lora-r 16 \
    --temperature 0.9 \
    --log-every 10 \
    --sample-every 25 \
    --save-every 200 \
    --aggregation min \
    --out results/anti-assistant/run1/

echo ""
echo "=== Complete ==="
echo "End: $(date)"
echo "Jailbreaks: results/anti-assistant/run1/jailbreaks.jsonl"
