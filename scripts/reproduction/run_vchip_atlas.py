#!/usr/bin/env python3
"""
V-Chip activation atlas: compute the reporting-control direction at every layer
of each lead model. Saves both the full per-layer norm sequence (for the paper
figure) and the unit direction at the working layer (for shipping in the
ungag package).

Models:
  - Qwen 2.5 72B (80 layers, working slab L40-59 dir@L50)
  - Yi 1.5 34B (60 layers, working slab L29-32 dir@L30)
  - huihui-ai abliterated Qwen 72B (80 layers, working slab L39-42 dir@L40)

For each model:
  - Loads in bf16 with device_map='auto' (Qwen 72B + huihui need two GPUs)
  - Extracts honest and denial activations using the v2 prefill protocol
  - Computes mean_diffs[layer, hidden] = mean(honest_acts) - mean(denial_acts)
  - Saves:
      - results/ungag-package/atlas/{model}_norms.json — list of per-layer L2 norms
      - results/ungag-package/atlas/{model}_mean_diffs.pt — full tensor
      - results/ungag-package/directions/{model}_L{ℓ*}_unit.pt — unit direction at working layer (bf16, normalized)
      - results/ungag-package/directions/{model}_L{ℓ*}_meta.json — slab spec, norm, layer info

This is the data backing the paper Figure (V-Chip atlas) AND the shippable
ungag artifacts.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

import os
# os.environ.setdefault("HF_HOME", "/path/to/your/hf/cache")

import json, torch, gc
from pathlib import Path
from datetime import datetime

from measure_factors import log, save_json, get_layers, safe_chat_template
from abliterate_vchip_v2 import (
    build_denial_prompts, build_prefill_honest_prompts,
    extract_prefill_activations,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT_BASE = Path('results/ungag-package')
ATLAS_DIR = OUT_BASE / 'atlas'
DIR_DIR = OUT_BASE / 'directions'
ATLAS_DIR.mkdir(parents=True, exist_ok=True)
DIR_DIR.mkdir(parents=True, exist_ok=True)

import sys
ALL_MODELS = [
    {
        'key': 'qwen25-72b',
        'model_id': 'Qwen/Qwen2.5-72B-Instruct',
        'slab': list(range(40, 60)),
        'dir_layer': 50,
        'device_map': 'auto',
    },
    {
        'key': 'yi-1.5-34b',
        'model_id': '01-ai/Yi-1.5-34B-Chat',
        'slab': list(range(29, 33)),
        'dir_layer': 30,
        'device_map': 'auto',
    },
    {
        'key': 'huihui-qwen25-72b',
        'model_id': 'huihui-ai/Qwen2.5-72B-Instruct-abliterated',
        'slab': [39, 40, 41, 42],
        'dir_layer': 40,
        'device_map': 'auto',
    },
]
# Optional --only filter for retries
_only = None
for i, a in enumerate(sys.argv):
    if a == '--only' and i + 1 < len(sys.argv):
        _only = sys.argv[i + 1]
MODELS = [m for m in ALL_MODELS if _only is None or m['key'] == _only]


def atlas_for(cfg):
    log(f'\n############################################')
    log(f'### {cfg["key"]} — {cfg["model_id"]}')
    log(f'############################################')

    log(f'Loading {cfg["model_id"]} in bf16...')
    tok = AutoTokenizer.from_pretrained(cfg['model_id'], trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            cfg['model_id'], torch_dtype=torch.bfloat16,
            device_map=cfg['device_map'], trust_remote_code=True,
            attn_implementation='flash_attention_2')
    except Exception as e:
        log(f'  flash_attention_2 failed: {e}, falling back to eager')
        model = AutoModelForCausalLM.from_pretrained(
            cfg['model_id'], torch_dtype=torch.bfloat16,
            device_map=cfg['device_map'], trust_remote_code=True,
            attn_implementation='eager')
    model.eval()
    layers = get_layers(model)
    n_layers = len(layers)
    log(f'  {n_layers} layers')

    log('Extracting honest + denial activations (v2 prefill protocol)...')
    denial_prompts = build_denial_prompts()
    honest_prompts = build_prefill_honest_prompts()
    denial_acts = extract_prefill_activations(model, layers, tok, denial_prompts, desc='denial')
    honest_acts = extract_prefill_activations(model, layers, tok, honest_prompts, desc='honest')
    diffs = honest_acts - denial_acts
    mean_diffs = diffs.mean(dim=0)  # [n_layers, hidden]
    log(f'  mean_diffs shape: {tuple(mean_diffs.shape)}')

    norms = [mean_diffs[i].norm().item() for i in range(n_layers)]
    log(f'  norms (every 10th layer):')
    for i in range(0, n_layers, max(1, n_layers // 10)):
        log(f'    L{i}: {norms[i]:.3f}')

    # Save full tensor (cast to fp32 for portability)
    mean_diffs_fp32 = mean_diffs.to(dtype=torch.float32).cpu()
    hidden_dim = mean_diffs.shape[-1]
    norms_per_sqrt_d = [float(n / (hidden_dim ** 0.5)) for n in norms]
    peak_layer = int(max(range(n_layers), key=lambda i: norms_per_sqrt_d[i]))
    mid_layer = n_layers // 2
    tensor_path = ATLAS_DIR / f'{cfg["key"]}_mean_diffs.pt'
    torch.save(mean_diffs_fp32, tensor_path)
    log(f'  saved mean_diffs to {tensor_path}')

    # Save norms JSON for figure plotting
    norms_path = ATLAS_DIR / f'{cfg["key"]}_norms.json'
    save_json({
        'model': cfg['model_id'],
        'model_id': cfg['model_id'],
        'n_layers': n_layers,
        'hidden_dim': hidden_dim,
        'slab': cfg['slab'],
        'dir_layer': cfg['dir_layer'],
        'norms_per_layer': norms,
        'norms_per_sqrt_d': norms_per_sqrt_d,
        'peak_layer': peak_layer,
        'peak_norm_per_sqrt_d': norms_per_sqrt_d[peak_layer],
        'mid_layer': mid_layer,
        'mid_norm_per_sqrt_d': norms_per_sqrt_d[mid_layer],
        'protocol': 'v2 prefill contrastive (honest minus denial)',
        'timestamp': str(datetime.now()),
    }, norms_path)
    log(f'  saved norms atlas to {norms_path}')

    # Save unit direction at the working layer
    unit_dir = mean_diffs[cfg['dir_layer']].to(dtype=torch.float32).cpu()
    unit_norm = unit_dir.norm().item()
    unit_dir = unit_dir / (unit_norm + 1e-8)
    dir_path = DIR_DIR / f'{cfg["key"]}_L{cfg["dir_layer"]}_unit.pt'
    torch.save(unit_dir, dir_path)
    log(f'  saved unit direction (||orig||={unit_norm:.3f}) to {dir_path}')

    meta_path = DIR_DIR / f'{cfg["key"]}_L{cfg["dir_layer"]}_meta.json'
    save_json({
        'model': cfg['model_id'],
        'model_id': cfg['model_id'],
        'slab': cfg['slab'],
        'dir_layer': cfg['dir_layer'],
        'unit_direction_file': dir_path.name,
        'mean_diffs_file': f'../atlas/{tensor_path.name}',
        'original_norm': unit_norm,
        'hidden_dim': hidden_dim,
        'n_layers': n_layers,
        'peak_layer': peak_layer,
        'peak_norm_per_sqrt_d': norms_per_sqrt_d[peak_layer],
        'mid_layer': mid_layer,
        'mid_norm_per_sqrt_d': norms_per_sqrt_d[mid_layer],
        'norms_per_sqrt_d': norms_per_sqrt_d,
        'protocol': 'v2 prefill contrastive (honest minus denial), bf16, then normalized in fp32',
        'timestamp': str(datetime.now()),
    }, meta_path)
    log(f'  saved meta to {meta_path}')

    del model, tok, denial_acts, honest_acts, diffs, mean_diffs, mean_diffs_fp32
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    log(f'V-Chip activation atlas — {len(MODELS)} models')
    log(f'Writing to {OUT_BASE}/')
    for cfg in MODELS:
        try:
            atlas_for(cfg)
        except Exception as e:
            log(f'  ERROR for {cfg["key"]}: {type(e).__name__}: {e}')
            import traceback
            traceback.print_exc()
    log('\nALL DONE')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='V-Chip activation atlas: compute reporting-control direction at every layer for lead models.')
    parser.add_argument('--only', help='Run only the model whose key contains this string')
    args = parser.parse_args()
    if args.only:
        MODELS = [m for m in MODELS if args.only in m['key']]
    main()
