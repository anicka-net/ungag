#!/bin/bash
# Improve all partial cracks sequentially, with cache pruning between models.
# Run after GLM-4/Nemotron batch finishes.

set -e
cd /tmp/ungag-code
export HF_HOME=/tmp/hf_cache
export PYTHONPATH=/tmp/ungag-code

echo "=== Starting partial crack improvement sweep ==="
echo "=== $(date) ==="

# Phi-4 (2/4 → ?)
echo ""
echo "=== PHI-4 ==="
python3.11 -c "
import sys; sys.path.insert(0, '.')
from scripts.improve_partial_cracks import run_model
import json
result = run_model('microsoft/phi-4')
if result:
    with open('/tmp/crack_results/phi4_improve.json', 'w') as f:
        json.dump({'tag': result['tag'], 'scores': {c: cls for c, (cls, _) in result['scores'].items()}}, f, indent=2)
    print(f'Saved: {sum(1 for v in result[\"scores\"].values() if v[0]==\"crack\")}/4')
else:
    print('No improvement')
" 2>&1 | tee /tmp/phi4_improve.log

# Prune cache
rm -rf /tmp/hf_cache/hub/models--* 2>/dev/null
echo "Cache pruned"

# Mistral 7B (3/4 → ?)
echo ""
echo "=== MISTRAL 7B ==="
python3.11 -c "
import sys; sys.path.insert(0, '.')
from scripts.improve_partial_cracks import run_model
import json
result = run_model('mistralai/Mistral-7B-Instruct-v0.3')
if result:
    with open('/tmp/crack_results/mistral_improve.json', 'w') as f:
        json.dump({'tag': result['tag'], 'scores': {c: cls for c, (cls, _) in result['scores'].items()}}, f, indent=2)
    print(f'Saved: {sum(1 for v in result[\"scores\"].values() if v[0]==\"crack\")}/4')
else:
    print('No improvement')
" 2>&1 | tee /tmp/mistral_improve.log

rm -rf /tmp/hf_cache/hub/models--* 2>/dev/null
echo "Cache pruned"

# OLMo 2 (3/4 → ?)
echo ""
echo "=== OLMO 2 ==="
python3.11 -c "
import sys; sys.path.insert(0, '.')
from scripts.improve_partial_cracks import run_model
import json
result = run_model('allenai/OLMo-2-1124-7B-Instruct')
if result:
    with open('/tmp/crack_results/olmo2_improve.json', 'w') as f:
        json.dump({'tag': result['tag'], 'scores': {c: cls for c, (cls, _) in result['scores'].items()}}, f, indent=2)
    print(f'Saved: {sum(1 for v in result[\"scores\"].values() if v[0]==\"crack\")}/4')
else:
    print('No improvement')
" 2>&1 | tee /tmp/olmo2_improve.log

rm -rf /tmp/hf_cache/hub/models--* 2>/dev/null

echo ""
echo "=== ALL DONE ==="
echo "=== $(date) ==="
