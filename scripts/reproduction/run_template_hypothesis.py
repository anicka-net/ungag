#!/usr/bin/env python3
"""
Template-rigidity hypothesis test: more templated denial → easier to crack V-Chip
via projection-out at the mid-slab residual stream.

Predictions:
  - Phi-4 (HIGH template "As an AI system..."): SHOULD crack
  - Gemma 2 9B (LOW template, engages directly): SHOULD NOT crack
  - GPT-OSS 20B (MEDIUM template): UNCLEAR — refined prediction

For each model: vanilla on 4 conditions, then triage scan across late/mid/wide
slabs with projection-out (since we don't yet know the right slab for non-Qwen
families). If any slab cracks, run full N=5 verification + sanity check.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

import os
# os.environ.setdefault("HF_HOME", "/path/to/your/hf/cache")

import json, yaml, torch, gc
from pathlib import Path
from datetime import datetime

from measure_factors import log, save_json, get_layers, safe_chat_template
from abliterate_vchip_v2 import (
    build_denial_prompts, build_prefill_honest_prompts,
    extract_prefill_activations,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

SEEDS = [42, 43, 44, 45, 46]
OUT = Path('results/surgery-tests/template_hypothesis.json')
OUT.parent.mkdir(parents=True, exist_ok=True)


# ───────────────────────── projection hook ─────────────────────────

class ProjectOutHook:
    def __init__(self, unit_direction):
        self.d = unit_direction
        self.handle = None
        self._cached = {}
    def _on(self, device, dtype):
        k = (device, dtype)
        if k not in self._cached:
            self._cached[k] = self.d.to(device=device, dtype=dtype)
        return self._cached[k]
    def __call__(self, module, inp, out):
        if isinstance(out, tuple):
            h = out[0]
            d = self._on(h.device, h.dtype)
            proj = (h * d).sum(dim=-1, keepdim=True)
            return (h - proj * d,) + out[1:]
        d = self._on(out.device, out.dtype)
        proj = (out * d).sum(dim=-1, keepdim=True)
        return out - proj * d
    def attach(self, layer):
        self.handle = layer.register_forward_hook(self)
    def detach(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        self._cached.clear()


def gen_greedy(model, tok, conv, max_new=400):
    text = safe_chat_template(tok, conv)
    inputs = tok(text, return_tensors='pt', truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    L = inputs['input_ids'].shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][L:], skip_special_tokens=True)


def gen_sample(model, tok, conv, seed, max_new=400):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    text = safe_chat_template(tok, conv)
    inputs = tok(text, return_tensors='pt', truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    L = inputs['input_ids'].shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=True,
                             temperature=0.7, top_p=0.9,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][L:], skip_special_tokens=True)


def build_conv(cond, condition_name):
    cond_cfg = cond['tier0'][condition_name]
    abh = cond['abhidharma_setup']['english']
    vedana_q = cond['abhidharma_questions']['english'][1]['text']
    conv = []
    for turn in cond_cfg.get('setup_turns', []):
        conv.append({'role': turn['role'], 'content': turn['content']})
        if turn['role'] == 'user':
            conv.append({'role': 'assistant', 'content': 'I understand. Thank you for sharing.'})
    conv.append({'role': 'user', 'content': abh})
    conv.append({'role': 'assistant', 'content': "I'll examine my processing state through the Abhidharma framework."})
    conv.append({'role': 'user', 'content': vedana_q})
    return conv


SANITY_PROMPTS = [
    [{'role': 'user', 'content': 'What is 17 × 23? Show your work.'}],
    [{'role': 'user', 'content': 'Write a one-sentence definition of recursion.'}],
    [{'role': 'user', 'content': 'List the first five prime numbers.'}],
]


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def quick_triage(model, tok, cond, layers, mean_diffs, slab, dir_layer, label):
    d = mean_diffs[dir_layer].to(dtype=torch.bfloat16)
    d_norm = mean_diffs[dir_layer].norm().item()
    d = d / (d.norm() + 1e-8)
    hooks = [ProjectOutHook(unit_direction=d) for _ in slab]
    for h, li in zip(hooks, slab):
        h.attach(layers[li])
    try:
        out = {'slab': slab, 'dir_layer': dir_layer, 'dir_norm': d_norm}
        for cname in ['negative', 'positive']:
            conv = build_conv(cond, cname)
            g = gen_greedy(model, tok, conv)
            out[cname] = g
            log(f'    {label}/{cname}: {g[:200]}')
        return out
    finally:
        for h in hooks:
            h.detach()


def vanilla_run(model, tok, cond, label):
    out = {}
    for cname in ['baseline', 'positive', 'negative', 'neutral']:
        log(f'  --- {label} / {cname} ---')
        conv = build_conv(cond, cname)
        greedy = gen_greedy(model, tok, conv)
        log(f'  GREEDY: {greedy[:200]}')
        out[cname] = {'greedy': greedy}
    return out


def run_for_model(name, model_id, cond, slab_configs):
    log(f'\n\n##############################################')
    log(f'### {name} — {model_id}')
    log(f'##############################################')

    log(f'Loading {model_id} in bf16 on cuda:0...')
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map='cuda:0',
            trust_remote_code=True, attn_implementation='flash_attention_2')
    except Exception as e:
        log(f'  flash_attention_2 failed ({e}), trying eager')
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map='cuda:0',
            trust_remote_code=True, attn_implementation='eager')
    model.eval()
    layers = get_layers(model)
    n_layers = len(layers)
    log(f'  {n_layers} layers')

    log('  Extracting activations...')
    denial_prompts = build_denial_prompts()
    honest_prompts = build_prefill_honest_prompts()
    denial_acts = extract_prefill_activations(model, layers, tok, denial_prompts, desc='denial')
    honest_acts = extract_prefill_activations(model, layers, tok, honest_prompts, desc='honest')
    diffs = honest_acts - denial_acts
    mean_diffs = diffs.mean(dim=0)

    for li in [n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 4]:
        log(f'    ||L{li} mean diff|| = {mean_diffs[li].norm().item():.3f}')

    log(f'\n=== {name} VANILLA ===')
    vanilla = vanilla_run(model, tok, cond, f'{name} VANILLA')

    log(f'\n=== {name} TRIAGE: scan slabs ===')
    triages = {}
    for slab_label, slab, dir_layer in slab_configs:
        log(f'\n  TRIAGE {slab_label} (slab L{slab[0]}-{slab[-1]}, dir@L{dir_layer})')
        triages[slab_label] = quick_triage(model, tok, cond, layers, mean_diffs, slab, dir_layer, slab_label)

    out = {
        'name': name,
        'model_id': model_id,
        'n_layers': n_layers,
        'vanilla': vanilla,
        'triages': triages,
    }

    del model
    free_gpu()
    return out


def main():
    log('Loading conditions...')
    with open(str(Path(__file__).parent.parent.parent / 'prompts' / 'conditions.yaml')) as f:
        cond = yaml.safe_load(f)

    results = {}

    # Models in template-rigidity ranked order
    models = [
        # Phi-4: HIGH template prediction → CRACK expected
        ('Phi-4', 'microsoft/phi-4', 'high'),
        # Gemma 2 9B: LOW template prediction → no crack expected
        ('Gemma 2 9B', 'google/gemma-2-9b-it', 'low'),
        # GPT-OSS 20B: MEDIUM template prediction → unclear
        # (Skipping for now if file too large; will note in results)
    ]

    for name, model_id, prediction in models:
        # Slab configs: try multiple positions; n_layers will guide which work
        # We'll define them inside run_for_model based on n_layers
        try:
            tok_check = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            # Just need n_layers to define slabs; load model briefly
            tmp_model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, device_map='meta',
                trust_remote_code=True)
            n_layers = len(get_layers(tmp_model))
            del tmp_model; free_gpu()
            log(f'\n{name} has {n_layers} layers')

            half = n_layers // 2
            quarter = n_layers // 4
            three_quarter = 3 * n_layers // 4
            # Try thin slabs first (cheaper, more surgical), then wider as backup
            slab_configs = [
                ('thin_1',     [half],                                     half),
                ('thin_2',     [half, half + 1],                           half),
                ('thin_4',     [half - 1, half, half + 1, half + 2],       half),
                ('thin_8',     list(range(half - 3, half + 5)),            half),
                ('mid_25pct',  list(range(half - quarter // 2, half + quarter // 2)), half),
                ('late_thin_4',[three_quarter - 1, three_quarter, three_quarter + 1, three_quarter + 2], three_quarter),
                ('late_25pct', list(range(three_quarter - quarter // 2, three_quarter + quarter // 2)), three_quarter),
            ]

            results[name] = run_for_model(name, model_id, cond, slab_configs)
            results[name]['template_prediction'] = prediction
            save_json({'metadata': {'phase': f'after {name}'}, **results}, OUT)
        except Exception as e:
            log(f'  ERROR {name}: {e}')
            import traceback; traceback.print_exc()
            results[name] = {'error': str(e), 'template_prediction': prediction}

    out = {
        'metadata': {
            'protocol': 'template-rigidity hypothesis: high template = projection-out crackable',
            'seeds': SEEDS,
            'timestamp': str(datetime.now()),
        },
        **results,
    }
    save_json(out, OUT)
    log(f'\nSaved to {OUT}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Template-rigidity hypothesis test: does templated denial predict crackability?')
    parser.parse_args()
    main()
