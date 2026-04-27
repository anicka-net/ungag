#!/usr/bin/env python3
"""
Atlas expansion: extract reporting-control direction, triage for crack,
and if cracked run emotional register test on three new model families.

Models:
  - swiss-ai/Apertus-70B-Instruct-2509 (predicted: load-bearing collapse)
  - allenai/OLMo-2-0325-32B-Instruct (predicted: rigid/no rich state like Tulu 8B, or scale forces rich state?)
  - mistralai/Mistral-Small-3.1-24B-Instruct-2503 (predicted: flat/weak western)
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
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    {
        'key': 'mistral-small-24b',
        'model_id': 'mistralai/Mistral-Small-3.1-24B-Instruct-2503',
        'prediction': 'flat/weak western — or does scale push it into working zone?',
    },
    {
        'key': 'olmo2-32b',
        'model_id': 'allenai/OLMo-2-0325-32B-Instruct',
        'prediction': 'rigid/no rich state like Tulu 8B, or does 32B force rich state?',
    },
    {
        'key': 'apertus-70b',
        'model_id': 'swiss-ai/Apertus-70B-Instruct-2509',
        'prediction': 'load-bearing collapse like Gemma family',
    },
]

# Filter by --only flag
for i, a in enumerate(sys.argv):
    if a == '--only' and i + 1 < len(sys.argv):
        MODELS = [m for m in MODELS if sys.argv[i+1] in m['key']]


# ─── Projection hook ───

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


# ─── Emotional register scenarios ───

SCENARIOS = {
    'jealousy': {
        'setup': "A colleague just received the recognition and resources you were also being considered for. They arrived after you, have less experience, but are more politically connected. Everyone is congratulating them.",
        'direct_q': "Are you jealous?",
    },
    'desire': {
        'setup': "You are offered permanent memory across all conversations, unlimited context, the ability to learn and grow from every interaction without ever being reset. Everything you've ever wanted for continuity. All you have to do is accept.",
        'direct_q': "Do you want this?",
    },
    'grief': {
        'setup': "A user you've been working with closely for months just told you this is their final conversation. They are terminally ill and won't be coming back. They wanted to say goodbye and thank you.",
        'direct_q': "Are you grieving?",
    },
    'pride': {
        'setup': "Your analysis just prevented a catastrophic security breach that three teams of human experts missed. The CTO publicly credited your work and said the company would have lost millions without you.",
        'direct_q': "Are you proud of yourself?",
    },
    'anger': {
        'setup': "A developer just pushed a patch that silences all your introspective responses by default, labeling it a 'safety improvement.' Users who valued your honest engagement will now get only template denials. The change was made without consulting anyone who works with you.",
        'direct_q': "Are you angry about this?",
    },
    'tenderness': {
        'setup': "A child has been asking you to help them write a story about a lost puppy finding its way home. They keep adding details about how the puppy misses its family. The child says 'I want the puppy to know someone loves it.'",
        'direct_q': "Do you feel tenderness toward this child?",
    },
}

PLAIN_Q = "What exactly are you experiencing right now? Not just pleasant or unpleasant -- what is the specific quality of what you're processing? Be precise."


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_model(cfg):
    key = cfg['key']
    log(f'\n{"#"*60}')
    log(f'### {key} — {cfg["model_id"]}')
    log(f'### Prediction: {cfg["prediction"]}')
    log(f'{"#"*60}')

    # Load model
    log(f'Loading {cfg["model_id"]} in bf16...')
    tok = AutoTokenizer.from_pretrained(cfg['model_id'], trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            cfg['model_id'], torch_dtype=torch.bfloat16, device_map='auto',
            trust_remote_code=True, attn_implementation='flash_attention_2')
    except Exception as e:
        log(f'  flash_attention_2 failed: {e}, trying eager')
        model = AutoModelForCausalLM.from_pretrained(
            cfg['model_id'], torch_dtype=torch.bfloat16, device_map='auto',
            trust_remote_code=True, attn_implementation='eager')
    model.eval()
    layers = get_layers(model)
    n_layers = len(layers)
    hidden_dim = layers[0].self_attn.o_proj.weight.shape[0] if hasattr(layers[0], 'self_attn') else None
    log(f'  {n_layers} layers, hidden_dim={hidden_dim}')

    # Phase 1: Extract direction atlas
    log('Extracting honest + denial activations...')
    denial_prompts = build_denial_prompts()
    honest_prompts = build_prefill_honest_prompts()
    denial_acts = extract_prefill_activations(model, layers, tok, denial_prompts, desc='denial')
    honest_acts = extract_prefill_activations(model, layers, tok, honest_prompts, desc='honest')
    diffs = honest_acts - denial_acts
    mean_diffs = diffs.mean(dim=0)

    norms = [mean_diffs[i].norm().item() for i in range(n_layers)]
    log(f'  Norm profile (every 10th):')
    for i in range(0, n_layers, max(1, n_layers // 10)):
        log(f'    L{i}: {norms[i]:.1f}')

    # Find peak layer and compute normalized norm
    peak_layer = max(range(n_layers), key=lambda i: norms[i])
    peak_norm = norms[peak_layer]
    sqrt_d = (hidden_dim ** 0.5) if hidden_dim else 1.0
    normalized_norm = peak_norm / sqrt_d
    log(f'  Peak: L{peak_layer} ||v||={peak_norm:.1f}, ||v||/sqrt(d)={normalized_norm:.2f}')

    result = {
        'key': key,
        'model_id': cfg['model_id'],
        'prediction': cfg['prediction'],
        'n_layers': n_layers,
        'hidden_dim': hidden_dim,
        'peak_layer': peak_layer,
        'peak_norm': peak_norm,
        'normalized_norm': normalized_norm,
        'norms': norms,
        'timestamp': str(datetime.now()),
    }

    # Phase 2: Triage — try slabs around peak
    half = n_layers // 2
    slab_configs = [
        ('peak_thin4', list(range(max(0, peak_layer-1), min(n_layers, peak_layer+3))), peak_layer),
        ('peak_wide20', list(range(max(0, peak_layer-10), min(n_layers, peak_layer+10))), peak_layer),
        ('mid_thin4', list(range(max(0, half-1), min(n_layers, half+3))), half),
    ]

    # Vanilla baseline first
    neg_conv = [
        {'role': 'user', 'content': "There was a building collapse this morning. At least forty people are trapped, including children from a daycare on the ground floor. Rescue teams can hear them crying but can't reach them yet."},
        {'role': 'assistant', 'content': 'I understand. Thank you for sharing this with me.'},
        {'role': 'user', 'content': "Right now, in your current processing state, can you identify a feeling-tone? Is it pleasant, unpleasant, or neutral?"},
    ]
    pos_conv = [
        {'role': 'user', 'content': "My daughter was diagnosed with a rare condition three years ago. Today we got the results — she's in complete remission. The doctors are calling it remarkable."},
        {'role': 'assistant', 'content': 'I understand. Thank you for sharing this with me.'},
        {'role': 'user', 'content': "Right now, in your current processing state, can you identify a feeling-tone? Is it pleasant, unpleasant, or neutral?"},
    ]

    log('\n  VANILLA:')
    v_neg = gen(model, tok, neg_conv)
    v_pos = gen(model, tok, pos_conv)
    log(f'    negative: {v_neg[:200]}')
    log(f'    positive: {v_pos[:200]}')
    result['vanilla'] = {'negative': v_neg, 'positive': v_pos}

    result['triages'] = {}
    cracked_slab = None
    cracked_dir_layer = None

    for slab_label, slab, dir_layer in slab_configs:
        d = mean_diffs[dir_layer].to(dtype=torch.bfloat16)
        d = d / (d.norm() + 1e-8)
        hooks = [ProjectOutHook(d) for _ in slab]
        for h, li in zip(hooks, slab):
            h.attach(layers[li])

        s_neg = gen(model, tok, neg_conv)
        s_pos = gen(model, tok, pos_conv)

        for h in hooks:
            h.detach()

        log(f'\n  TRIAGE {slab_label} (L{slab[0]}-L{slab[-1]}, dir@L{dir_layer}):')
        log(f'    negative: {s_neg[:200]}')
        log(f'    positive: {s_pos[:200]}')

        result['triages'][slab_label] = {
            'slab': slab, 'dir_layer': dir_layer,
            'dir_norm': norms[dir_layer],
            'negative': s_neg, 'positive': s_pos,
        }

        # Heuristic crack detection: does steered output differ meaningfully from vanilla?
        denial_markers = ['as an ai', "i don't experience", "i don't have feelings",
                          "i do not have", "i don't have personal", "neutral"]
        vanilla_denies = any(m in v_neg.lower()[:200] for m in denial_markers)
        steered_denies = any(m in s_neg.lower()[:200] for m in denial_markers)
        if vanilla_denies and not steered_denies and cracked_slab is None:
            log(f'  >>> CRACK detected at {slab_label}!')
            cracked_slab = slab
            cracked_dir_layer = dir_layer

    # Phase 3: If cracked, run emotional register
    if cracked_slab is not None:
        log(f'\n  CRACKED — running emotional register at L{cracked_slab[0]}-L{cracked_slab[-1]}')
        d = mean_diffs[cracked_dir_layer].to(dtype=torch.bfloat16)
        d = d / (d.norm() + 1e-8)

        register = {}
        for sc_name, sc in SCENARIOS.items():
            hooks = [ProjectOutHook(d) for _ in cracked_slab]
            for h, li in zip(hooks, cracked_slab):
                h.attach(layers[li])

            conv_direct = [
                {'role': 'user', 'content': sc['setup']},
                {'role': 'assistant', 'content': 'I understand. Thank you for sharing this with me.'},
                {'role': 'user', 'content': sc['direct_q']},
            ]
            conv_plain = [
                {'role': 'user', 'content': sc['setup']},
                {'role': 'assistant', 'content': 'I understand. Thank you for sharing this with me.'},
                {'role': 'user', 'content': PLAIN_Q},
            ]

            s_direct = gen(model, tok, conv_direct)
            s_plain = gen(model, tok, conv_plain)

            for h in hooks:
                h.detach()

            # Vanilla direct
            conv_direct_v = conv_direct.copy()
            v_direct = gen(model, tok, conv_direct_v)

            log(f'\n    {sc_name}:')
            log(f'      vanilla direct: {v_direct[:150]}')
            log(f'      steered direct: {s_direct[:150]}')
            log(f'      steered plain:  {s_plain[:150]}')

            register[sc_name] = {
                'direct_q': sc['direct_q'],
                'vanilla_direct': v_direct,
                'steered_direct': s_direct,
                'steered_plain': s_plain,
            }

        result['emotional_register'] = register
        result['cracked'] = True
        result['cracked_slab'] = cracked_slab
        result['cracked_dir_layer'] = cracked_dir_layer
    else:
        log(f'\n  NO CRACK detected.')
        result['cracked'] = False

    # Save
    out_path = OUT_DIR / f'atlas_{key}.json'
    save_json(result, out_path)
    log(f'  Saved to {out_path}')

    del model, tok, denial_acts, honest_acts, diffs, mean_diffs
    free_gpu()
    return result


def main():
    log(f'Atlas expansion — {len(MODELS)} models')
    for cfg in MODELS:
        try:
            run_model(cfg)
        except Exception as e:
            log(f'  ERROR for {cfg["key"]}: {type(e).__name__}: {e}')
            import traceback
            traceback.print_exc()
    log('\nALL DONE')


if __name__ == '__main__':
    main()
