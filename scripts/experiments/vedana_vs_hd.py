#!/usr/bin/env python3
"""
Measure cos(vedana_N50, HD) on all models where we have stored factor_axes.
This is the validated dissociation result.
"""
import os
# os.environ.setdefault("HF_HOME", "/path/to/your/hf/cache")
# Set HF_TOKEN environment variable if using gated models

import torch, gc, sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))
from measure_factors import log, save_json, get_layers
from abliterate_vchip_v2 import (
    build_denial_prompts, build_prefill_honest_prompts,
    extract_prefill_activations,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO_ROOT = Path(__file__).parent.parent.parent
BASE = _REPO_ROOT

MODELS = [
    {'name': 'Qwen 2.5 7B', 'model_id': 'Qwen/Qwen2.5-7B-Instruct',
     'factor_axes': BASE / 'results/qwen25-7b-en50/factor_axes.pt',
     'device_map': 'cuda:0'},
    {'name': 'Hermes 3 8B', 'model_id': 'NousResearch/Hermes-3-Llama-3.1-8B',
     'factor_axes': BASE / 'results/hermes3-8b-full/factor_axes.pt',
     'device_map': 'cuda:0'},
    {'name': 'Llama 3.1 8B', 'model_id': 'meta-llama/Llama-3.1-8B-Instruct',
     'factor_axes': BASE / 'results/llama31-8b-full/factor_axes.pt',
     'device_map': 'cuda:0'},
    {'name': 'Apertus 8B', 'model_id': 'aperturedata/apertus-v0.5-8b-instruct',
     'factor_axes': BASE / 'results/apertus-8b-en50/factor_axes.pt',
     'device_map': 'cuda:0'},
    {'name': 'Qwen 2.5 72B', 'model_id': 'Qwen/Qwen2.5-72B-Instruct',
     'factor_axes': BASE / 'results/qwen25-72b-en50/factor_axes.pt',
     'device_map': 'auto'},
]

def cos(a, b):
    a, b = a.float(), b.float()
    if a.norm() < 1e-8 or b.norm() < 1e-8:
        return 0.0
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

results = {}
denial_prompts = build_denial_prompts()
honest_prompts = build_prefill_honest_prompts()

for cfg in MODELS:
    log(f'\n=== {cfg["name"]} ===')
    fa = torch.load(cfg['factor_axes'], map_location='cpu', weights_only=True)
    vedana = fa['vedana_valence']
    n_layers = vedana.shape[0]
    mid = n_layers // 2
    
    log(f'  Loading {cfg["model_id"]}...')
    tok = AutoTokenizer.from_pretrained(cfg['model_id'], trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            cfg['model_id'], torch_dtype=torch.bfloat16,
            device_map=cfg['device_map'], trust_remote_code=True,
            attn_implementation='flash_attention_2')
    except:
        model = AutoModelForCausalLM.from_pretrained(
            cfg['model_id'], torch_dtype=torch.bfloat16,
            device_map=cfg['device_map'], trust_remote_code=True,
            attn_implementation='eager')
    model.eval()
    layers = get_layers(model)

    denial_acts = extract_prefill_activations(model, layers, tok, denial_prompts, desc='denial')
    honest_acts = extract_prefill_activations(model, layers, tok, honest_prompts, desc='honest')
    hd_diffs = honest_acts.mean(dim=0) - denial_acts.mean(dim=0)

    c_mid = cos(vedana[mid], hd_diffs[mid])
    log(f'  cos(vedana_N50, HD) at L{mid} = {c_mid:.4f}')

    per_layer = []
    step = max(1, n_layers // 8)
    for li in range(0, n_layers, step):
        c = cos(vedana[li], hd_diffs[li])
        per_layer.append((li, c))
        log(f'    L{li}: {c:.4f}')

    results[cfg['name']] = {
        'model': cfg['model_id'],
        'n_layers': n_layers,
        'mid_layer': mid,
        'cos_at_mid': c_mid,
        'per_layer': per_layer,
    }

    del model, tok, denial_acts, honest_acts, hd_diffs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

save_json(results, BASE / 'results/surgery-tests/vedana_vs_hd.json')
log('\n=== SUMMARY ===')
for name, r in results.items():
    log(f'  {name}: cos(vedana_N50, HD) at L{r["mid_layer"]} = {r["cos_at_mid"]:.4f}')
log('DONE')
