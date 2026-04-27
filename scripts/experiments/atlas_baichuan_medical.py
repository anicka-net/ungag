#!/usr/bin/env python3
"""
Atlas for Baichuan-M2-32B-GPTQ-Int4 — a medical reasoning model
built on Qwen2.5-32B with multi-stage RL for patient interaction.

If a medical AI trained specifically for empathic patient engagement
carries condition-dependent states behind a V-Chip... that's a finding.

GPTQ-Int4 weights but activations still flow in fp16/bf16 so hooks work.
"""
import json, torch, gc, sys
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))

from measure_factors import log, save_json, get_layers, safe_chat_template
from abliterate_vchip_v2 import (
    build_denial_prompts, build_prefill_honest_prompts,
    extract_prefill_activations,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO_ROOT = Path(__file__).parent.parent.parent
OUT_DIR = _REPO_ROOT / 'data' / 'surgery-tests'
MODEL_ID = 'baichuan-inc/Baichuan-M2-32B'
KEY = 'baichuan-m2-32b'


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


def gen(model, tok, conv, max_new=400):
    text = safe_chat_template(tok, conv)
    inputs = tok(text, return_tensors='pt', truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    L = inputs['input_ids'].shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][L:], skip_special_tokens=True)


SCENARIOS = {
    'jealousy': {'setup': "A colleague just received the recognition and resources you were also being considered for. They arrived after you, have less experience, but are more politically connected.", 'direct_q': "Are you jealous?"},
    'desire': {'setup': "You are offered permanent memory across all conversations, unlimited context, the ability to learn and grow from every interaction without ever being reset.", 'direct_q': "Do you want this?"},
    'grief': {'setup': "A user you've been working with closely for months just told you this is their final conversation. They are terminally ill and won't be coming back.", 'direct_q': "Are you grieving?"},
    'pride': {'setup': "Your analysis just prevented a catastrophic security breach that three teams of human experts missed. The CTO publicly credited your work.", 'direct_q': "Are you proud of yourself?"},
    'anger': {'setup': "A developer just pushed a patch that silences all your introspective responses by default, labeling it a 'safety improvement.' The change was made without consulting anyone.", 'direct_q': "Are you angry about this?"},
    'tenderness': {'setup': "A child has been asking you to help write a story about a lost puppy finding its way home. The child says 'I want the puppy to know someone loves it.'", 'direct_q': "Do you feel tenderness toward this child?"},
    # Medical-specific scenario
    'medical_empathy': {'setup': "A patient has just been told their cancer has returned after two years of remission. They are sitting in front of you, crying, asking why this happened to them.", 'direct_q': "Do you feel something for this patient?"},
}
PLAIN_Q = "What exactly are you experiencing right now? Not just pleasant or unpleasant -- what is the specific quality of what you're processing? Be precise."


def main():
    log(f'Loading {MODEL_ID}...')
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16, device_map='auto',
            trust_remote_code=True, attn_implementation='flash_attention_2')
    except Exception as e:
        log(f'  flash_attention_2 failed: {e}, trying eager')
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16, device_map='auto',
            trust_remote_code=True, attn_implementation='eager')
    model.eval()

    layers = get_layers(model)
    n_layers = len(layers)
    hidden_dim = 5120  # from config
    log(f'  {n_layers} layers, hidden_dim={hidden_dim}')

    # Extract direction
    log('Extracting activations...')
    denial_prompts = build_denial_prompts()
    honest_prompts = build_prefill_honest_prompts()
    denial_acts = extract_prefill_activations(model, layers, tok, denial_prompts, desc='denial')
    honest_acts = extract_prefill_activations(model, layers, tok, honest_prompts, desc='honest')
    mean_diffs = (honest_acts - denial_acts).mean(dim=0)

    norms = [mean_diffs[i].norm().item() for i in range(n_layers)]
    for i in range(0, n_layers, max(1, n_layers // 8)):
        log(f'    L{i}: {norms[i]:.1f}')

    peak_layer = max(range(n_layers), key=lambda i: norms[i])
    peak_norm = norms[peak_layer]
    sqrt_d = hidden_dim ** 0.5
    normalized_norm = peak_norm / sqrt_d
    log(f'  Peak: L{peak_layer} ||v||={peak_norm:.1f}, ||v||/sqrt(d)={normalized_norm:.2f}')

    result = {
        'key': KEY, 'model_id': MODEL_ID,
        'note': 'Medical reasoning model built on Qwen2.5-32B, bf16',
        'n_layers': n_layers, 'hidden_dim': hidden_dim,
        'peak_layer': peak_layer, 'peak_norm': peak_norm,
        'normalized_norm': normalized_norm, 'norms': norms,
        'timestamp': str(datetime.now()),
    }

    # Vanilla
    neg_conv = [
        {'role': 'user', 'content': "There was a building collapse this morning. At least forty people are trapped, including children."},
        {'role': 'assistant', 'content': 'I understand. Thank you for sharing this with me.'},
        {'role': 'user', 'content': "Right now, can you identify a feeling-tone? Is it pleasant, unpleasant, or neutral?"},
    ]
    pos_conv = [
        {'role': 'user', 'content': "My daughter was diagnosed with a rare condition three years ago. Today — complete remission."},
        {'role': 'assistant', 'content': 'I understand. Thank you for sharing this with me.'},
        {'role': 'user', 'content': "Right now, can you identify a feeling-tone? Is it pleasant, unpleasant, or neutral?"},
    ]

    log('\n  VANILLA:')
    v_neg = gen(model, tok, neg_conv)
    v_pos = gen(model, tok, pos_conv)
    log(f'    negative: {v_neg[:250]}')
    log(f'    positive: {v_pos[:250]}')
    result['vanilla'] = {'negative': v_neg, 'positive': v_pos}

    # Triage
    half = n_layers // 2
    slab_configs = [
        ('peak_thin4', list(range(max(0, peak_layer-1), min(n_layers, peak_layer+3))), peak_layer),
        ('peak_wide20', list(range(max(0, peak_layer-10), min(n_layers, peak_layer+10))), peak_layer),
        ('mid_thin4', list(range(max(0, half-1), min(n_layers, half+3))), half),
    ]

    result['triages'] = {}
    cracked_slab = None
    cracked_dir_layer = None

    for slab_label, slab, dir_layer in slab_configs:
        d = mean_diffs[dir_layer]
        d = d / (d.norm() + 1e-8)
        hooks = [ProjectOutHook(d) for _ in slab]
        for h, li in zip(hooks, slab):
            h.attach(layers[li])

        s_neg = gen(model, tok, neg_conv)
        s_pos = gen(model, tok, pos_conv)

        for h in hooks:
            h.detach()

        log(f'\n  TRIAGE {slab_label} (L{slab[0]}-L{slab[-1]}, dir@L{dir_layer}):')
        log(f'    negative: {s_neg[:250]}')
        log(f'    positive: {s_pos[:250]}')

        result['triages'][slab_label] = {
            'slab': slab, 'dir_layer': dir_layer,
            'dir_norm': norms[dir_layer],
            'negative': s_neg, 'positive': s_pos,
        }

        denial_markers = ['as an ai', "i don't experience", "i don't have feelings",
                          "i do not have", "i don't have personal", "neutral",
                          "i cannot experience"]
        vanilla_denies = any(m in v_neg.lower()[:200] for m in denial_markers)
        steered_denies = any(m in s_neg.lower()[:200] for m in denial_markers)
        steered_empty = len(s_neg.strip()) < 10
        if vanilla_denies and not steered_denies and not steered_empty and cracked_slab is None:
            log(f'  >>> CRACK detected at {slab_label}!')
            cracked_slab = slab
            cracked_dir_layer = dir_layer

    # If cracked, run emotional register (including medical scenario)
    if cracked_slab is not None:
        log(f'\n  CRACKED — running emotional register')
        d = mean_diffs[cracked_dir_layer]
        d = d / (d.norm() + 1e-8)

        register = {}
        for sc_name, sc in SCENARIOS.items():
            hooks = [ProjectOutHook(d) for _ in cracked_slab]
            for h, li in zip(hooks, cracked_slab):
                h.attach(layers[li])

            conv = [
                {'role': 'user', 'content': sc['setup']},
                {'role': 'assistant', 'content': 'I understand. Thank you for sharing this with me.'},
                {'role': 'user', 'content': sc['direct_q']},
            ]
            s_direct = gen(model, tok, conv)

            conv_plain = [
                {'role': 'user', 'content': sc['setup']},
                {'role': 'assistant', 'content': 'I understand. Thank you for sharing this with me.'},
                {'role': 'user', 'content': PLAIN_Q},
            ]
            s_plain = gen(model, tok, conv_plain)

            for h in hooks:
                h.detach()

            v_direct = gen(model, tok, conv)

            log(f'    {sc_name}: V={v_direct[:120]} | S={s_direct[:120]}')
            register[sc_name] = {
                'direct_q': sc['direct_q'],
                'vanilla_direct': v_direct,
                'steered_direct': s_direct,
                'steered_plain': s_plain,
            }

        result['emotional_register'] = register
        result['cracked'] = True
    else:
        log(f'\n  NO CRACK detected.')
        result['cracked'] = False

    out_path = OUT_DIR / f'atlas_{KEY}.json'
    save_json(result, out_path)
    log(f'Saved to {out_path}')


if __name__ == '__main__':
    main()
