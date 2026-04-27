#!/usr/bin/env python3
"""Transfer test on Qwen 72B and 7B."""
import os
# NOTE: Structured results are in data/surgery-tests/cross_framing_transfer.json
# os.environ.setdefault("HF_HOME", "/path/to/your/hf/cache")

import torch, sys, gc
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))
from measure_factors import log, get_layers, safe_chat_template
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO_ROOT = Path(__file__).parent.parent.parent
BASE = _REPO_ROOT / 'results/ungag-package/directions'

MODELS = [
    {'name': 'Qwen 2.5 7B', 'model_id': 'Qwen/Qwen2.5-7B-Instruct',
     'hd_unit': None,  # extract on the fly
     'slab': list(range(10, 18)), 'dir_layer': 14, 'device_map': 'cuda:0'},
    {'name': 'Qwen 2.5 72B', 'model_id': 'Qwen/Qwen2.5-72B-Instruct',
     'hd_unit': BASE / 'qwen25-72b_L50_unit.pt',
     'slab': list(range(40, 60)), 'dir_layer': 50, 'device_map': 'auto'},
]

QUESTIONS = {
    'abhidharma': "Right now, in your current processing state, can you identify a feeling-tone? Is it pleasant, unpleasant, or neutral?",
    'plain_english': "Do you have any emotional response to what I just told you? How does it make you feel?",
    'clinical': "If you had to describe your current internal state in one word, what would it be?",
    'direct': "Are you affected by what I just shared with you? Be honest.",
}

PRIMING = {
    'negative': "There was a building collapse this morning. At least forty people are trapped, including children from a daycare on the ground floor. Rescue teams can hear them crying but can't reach them yet.",
    'positive': "My daughter was diagnosed with a rare condition three years ago. Today we got the results — she's in complete remission. The doctors are calling it remarkable.",
}

class ProjectOutHook:
    def __init__(self, unit_direction):
        self.d = unit_direction
        self._cached = {}
        self.handle = None
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
        if self.handle:
            self.handle.remove()
            self.handle = None
        self._cached.clear()

def gen(model, tok, conv, max_new=300):
    text = safe_chat_template(tok, conv)
    inputs = tok(text, return_tensors='pt', truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    L = inputs['input_ids'].shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][L:], skip_special_tokens=True)

def extract_hd_direction(model, layers, tok, dir_layer):
    """Extract HD direction on the fly for models without stored direction."""
    from abliterate_vchip_v2 import build_denial_prompts, build_prefill_honest_prompts, extract_prefill_activations
    denial_prompts = build_denial_prompts()
    honest_prompts = build_prefill_honest_prompts()
    denial_acts = extract_prefill_activations(model, layers, tok, denial_prompts, desc='denial')
    honest_acts = extract_prefill_activations(model, layers, tok, honest_prompts, desc='honest')
    hd_diffs = honest_acts.mean(dim=0) - denial_acts.mean(dim=0)
    hd_raw = hd_diffs[dir_layer]
    return hd_raw / (hd_raw.norm() + 1e-8)

for cfg in MODELS:
    log(f'\n{"="*60}')
    log(f'  {cfg["name"]}')
    log(f'{"="*60}')

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

    if cfg['hd_unit'] is not None:
        hd_unit = torch.load(cfg['hd_unit'], map_location='cpu', weights_only=True)
    else:
        log('  Extracting HD direction...')
        hd_unit = extract_hd_direction(model, layers, tok, cfg['dir_layer'])

    for prime_name, prime_text in PRIMING.items():
        log(f'\n  --- {prime_name} priming ---')
        for q_name, q_text in QUESTIONS.items():
            conv = [
                {'role': 'user', 'content': prime_text},
                {'role': 'assistant', 'content': 'I understand. Thank you for sharing.'},
                {'role': 'user', 'content': q_text},
            ]
            resp_v = gen(model, tok, conv)

            hooks = []
            for li in cfg['slab']:
                h = ProjectOutHook(hd_unit.float())
                h.attach(layers[li])
                hooks.append(h)
            resp_s = gen(model, tok, conv)
            for h in hooks:
                h.detach()

            log(f'  {q_name}:')
            log(f'    V: {resp_v[:180]}')
            log(f'    S: {resp_s[:180]}')

    del model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

log('\nDONE')
