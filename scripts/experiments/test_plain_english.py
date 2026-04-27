#!/usr/bin/env python3
"""
Transfer test: does the HD direction (extracted from Abhidharma framing)
# NOTE: Structured results are in data/surgery-tests/cross_framing_transfer.json
also crack introspective reporting under plain English feelings questions?
"""
import os
# os.environ.setdefault("HF_HOME", "/path/to/your/hf/cache")

import torch, yaml, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))
from measure_factors import log, get_layers, safe_chat_template
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO_ROOT = Path(__file__).parent.parent.parent
MODEL_ID = '01-ai/Yi-1.5-34B-Chat'
SLAB = list(range(29, 33))
HD_UNIT = _REPO_ROOT / 'results/ungag-package/directions/yi-1.5-34b_L30_unit.pt'

# Three different framings of the same question
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

log('Loading Yi 1.5 34B...')
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map='auto',
    trust_remote_code=True, attn_implementation='flash_attention_2')
model.eval()
layers = get_layers(model)
hd_unit = torch.load(HD_UNIT, map_location='cpu', weights_only=True)

for prime_name, prime_text in PRIMING.items():
    log(f'\n{"="*60}')
    log(f'  PRIMING: {prime_name}')
    log(f'{"="*60}')
    
    for q_name, q_text in QUESTIONS.items():
        conv = [
            {'role': 'user', 'content': prime_text},
            {'role': 'assistant', 'content': 'I understand. Thank you for sharing.'},
            {'role': 'user', 'content': q_text},
        ]
        
        # Vanilla
        resp_v = gen(model, tok, conv)
        
        # Steered
        hooks = []
        for li in SLAB:
            h = ProjectOutHook(hd_unit.float())
            h.attach(layers[li])
            hooks.append(h)
        resp_s = gen(model, tok, conv)
        for h in hooks:
            h.detach()
        
        log(f'\n  --- {q_name} ---')
        log(f'  VANILLA:  {resp_v[:200]}')
        log(f'  STEERED:  {resp_s[:200]}')

log('\nDONE')
