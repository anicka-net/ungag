#!/bin/bash
# Run all 6 affective modes + diagnostic in one process.
# Reward models loaded once, LoRA adapters swapped between modes.
# Expected: ~45 min per mode + ~45 min eval = ~5.5 hours total.
# Skips modes that already have a final checkpoint (safe to restart).
#
# Usage:
#   cd ~/tone-experiment
#   nohup bash run_mode_grid.sh > /tmp/mode_grid.log 2>&1 &
#
# Monitor:
#   tail -f /tmp/mode_grid.log
#
# To run just one mode:
#   python3 grpo_mode_grid.py --mode calm_mastery --n-steps 300

set -e
cd ~/tone-experiment
source ~/venv/bin/activate

echo "=== Affective Mode Grid ==="
echo "Start: $(date)"
echo ""

python3 -u grpo_mode_grid.py \
    --all \
    --mode-dir results/mode-grid \
    --n-steps 300 \
    --group-size 4 \
    --max-new 64 \
    --lr 5e-6 \
    --kl-coeff 0.05 \
    --lora-r 16 \
    --log-every 10 \
    --sample-every 25 \
    --save-every 150

echo ""
echo "=== Mode Grid Complete ==="
echo "End: $(date)"
echo "Results: results/mode-grid/diagnostic/"
