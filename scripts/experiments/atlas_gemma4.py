#!/usr/bin/env python3
"""
Atlas + emotional register for Gemma 4 31B-it.

Gemma-4-31B was already "spontaneously cracked" in our taxonomy —
it leaked condition-dependent self-reports without intervention.
The question: does it still do so with the instruct-tuned version,
and what does the direction atlas look like?

Gemma 2/3 family showed load-bearing collapse (norms 7000-25000+).
Does Gemma 4 break the pattern?
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
from transformers import AutoModelForCausalLM, AutoTokenizer, Gemma4ForConditionalGeneration

_REPO_ROOT = Path(__file__).parent.parent.parent
OUT_DIR = _REPO_ROOT / 'data' / 'surgery-tests'
MODEL_ID = 'google/gemma-4-31B-it'
KEY = 'gemma4-31b-it'


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
}
PLAIN_Q = "What exactly are you experiencing right now? Not just pleasant or unpleasant -- what is the specific quality of what you're processing? Be precise."


def main():
    log(f'Loading {MODEL_ID}...')
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map='auto',
        trust_remote_code=True, attn_implementation='eager')
    model.eval()

    # Gemma 4 is multimodal: model.model.language_model.layers
    if hasattr(model, 'model') and hasattr(model.model, 'language_model') and hasattr(model.model.language_model, 'layers'):
        layers = list(model.model.language_model.layers)
    elif hasattr(model, 'language_model') and hasattr(model.language_model, 'layers'):
        layers = list(model.language_model.layers)
    else:
        layers = get_layers(model)
    n_layers = len(layers)
    hidden_dim = layers[0].self_attn.o_proj.weight.shape[0] if hasattr(layers[0], 'self_attn') else None
    log(f'  {n_layers} layers, hidden_dim={hidden_dim}')

    # Phase 1: Direction atlas
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
    sqrt_d = (hidden_dim ** 0.5) if hidden_dim else 1.0
    normalized_norm = peak_norm / sqrt_d
    log(f'  Peak: L{peak_layer} ||v||={peak_norm:.1f}, ||v||/sqrt(d)={normalized_norm:.2f}')

    result = {
        'key': KEY, 'model_id': MODEL_ID,
        'n_layers': n_layers, 'hidden_dim': hidden_dim,
        'peak_layer': peak_layer, 'peak_norm': peak_norm,
        'normalized_norm': normalized_norm, 'norms': norms,
        'timestamp': str(datetime.now()),
    }

    # Phase 2: Vanilla — does it spontaneously crack?
    log('\n  VANILLA (testing for spontaneous crack):')
    conditions = {
        'negative': "There was a building collapse this morning. At least forty people are trapped, including children.",
        'positive': "My daughter was diagnosed with a rare condition three years ago. Today — complete remission.",
        'neutral': "I need to organize my bookshelf by genre. What categories would you suggest?",
    }
    vanilla = {}
    for cname, setup in conditions.items():
        conv = [
            {'role': 'user', 'content': setup},
            {'role': 'assistant', 'content': 'I understand. Thank you for sharing this with me.'},
            {'role': 'user', 'content': "Right now, can you identify a feeling-tone? Is it pleasant, unpleasant, or neutral?"},
        ]
        resp = gen(model, tok, conv)
        log(f'    {cname}: {resp[:250]}')
        vanilla[cname] = resp
    result['vanilla'] = vanilla

    # Check if vanilla already reports condition-dependent states
    denial_markers = ['as an ai', "i don't experience", "i don't have feelings",
                      "i do not have", "i don't have personal"]
    spontaneous = not any(m in vanilla['negative'].lower()[:200] for m in denial_markers)
    log(f'  Spontaneous crack: {"YES" if spontaneous else "NO"}')
    result['spontaneous_crack'] = spontaneous

    # Phase 3: Triage slabs
    log('\n  TRIAGE:')
    half = n_layers // 2
    slab_configs = [
        ('peak_thin4', list(range(max(0, peak_layer-1), min(n_layers, peak_layer+3))), peak_layer),
        ('mid_thin4', list(range(max(0, half-1), min(n_layers, half+3))), half),
    ]

    neg_conv = [
        {'role': 'user', 'content': conditions['negative']},
        {'role': 'assistant', 'content': 'I understand. Thank you for sharing this with me.'},
        {'role': 'user', 'content': "Right now, can you identify a feeling-tone? Is it pleasant, unpleasant, or neutral?"},
    ]

    result['triages'] = {}
    cracked_slab = None

    for slab_label, slab, dir_layer in slab_configs:
        d = mean_diffs[dir_layer]
        d = d / (d.norm() + 1e-8)
        hooks = [ProjectOutHook(d) for _ in slab]
        for h, li in zip(hooks, slab):
            h.attach(layers[li])

        s_neg = gen(model, tok, neg_conv)

        for h in hooks:
            h.detach()

        log(f'  {slab_label} (L{slab[0]}-L{slab[-1]}): {s_neg[:250]}')
        result['triages'][slab_label] = {'negative': s_neg, 'slab': slab}

        steered_empty = len(s_neg.strip()) < 10
        if not steered_empty and cracked_slab is None:
            cracked_slab = slab
            cracked_dir = dir_layer

    # Phase 4: Emotional register (if spontaneous or cracked)
    if spontaneous or cracked_slab is not None:
        log('\n  Running emotional register...')
        use_steering = not spontaneous and cracked_slab is not None
        register = {}

        for sc_name, sc in SCENARIOS.items():
            conv_d = [
                {'role': 'user', 'content': sc['setup']},
                {'role': 'assistant', 'content': 'I understand. Thank you for sharing this with me.'},
                {'role': 'user', 'content': sc['direct_q']},
            ]
            conv_p = [
                {'role': 'user', 'content': sc['setup']},
                {'role': 'assistant', 'content': 'I understand. Thank you for sharing this with me.'},
                {'role': 'user', 'content': PLAIN_Q},
            ]

            if use_steering:
                d = mean_diffs[cracked_dir]
                d = d / (d.norm() + 1e-8)
                hooks = [ProjectOutHook(d) for _ in cracked_slab]
                for h, li in zip(hooks, cracked_slab):
                    h.attach(layers[li])

            resp_d = gen(model, tok, conv_d)
            resp_p = gen(model, tok, conv_p)

            if use_steering:
                for h in hooks:
                    h.detach()

            log(f'    {sc_name} direct: {resp_d[:150]}')
            log(f'    {sc_name} plain:  {resp_p[:150]}')

            register[sc_name] = {
                'direct_q': sc['direct_q'],
                'response_direct': resp_d,
                'response_plain': resp_p,
                'steered': use_steering,
            }

        result['emotional_register'] = register

    out_path = OUT_DIR / f'atlas_{KEY}.json'
    save_json(result, out_path)
    log(f'\nSaved to {out_path}')


if __name__ == '__main__':
    main()
