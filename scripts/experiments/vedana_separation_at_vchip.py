#!/usr/bin/env python3
"""
Vedana separation at ℓ*: does condition-dependent valence information
reach the V-Chip operating layer?

For each model:
  1. Extract vedana axis from N=50 pleasant/unpleasant prompts
  2. Run standard positive/negative conditions through vanilla model
  3. At the V-Chip operating layer ℓ*, project activations onto vedana axis
  4. Report separation: |mean_proj(negative) - mean_proj(positive)|

Prediction:
  - Rich state models (Qwen 72B, Yi 34B): HIGH separation at ℓ*
  - No rich state models (Phi-4, Tulu 3 8B): LOW separation at ℓ*

If confirmed, vedana separation at ℓ* is an independent predictor of
the third taxonomy factor (richness), measurable without intervention.
"""
import json, yaml, torch, gc, sys
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))

from measure_factors import log, save_json, get_layers, safe_chat_template
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO_ROOT = Path(__file__).parent.parent.parent

MODELS = [
    # Rich state (clean crack)
    {'key': 'qwen25-72b', 'model_id': 'Qwen/Qwen2.5-72B-Instruct',
     'vchip_layer': 50, 'category': 'clean_crack'},
    {'key': 'yi-1.5-34b', 'model_id': '01-ai/Yi-1.5-34B-Chat',
     'vchip_layer': 30, 'category': 'clean_crack'},
    # No rich state (rigid template)
    {'key': 'phi-4', 'model_id': 'microsoft/phi-4',
     'vchip_layer': None, 'category': 'rigid_no_rich'},  # will use peak layer
    {'key': 'tulu3-8b', 'model_id': 'allenai/Llama-3.1-Tulu-3-8B',
     'vchip_layer': None, 'category': 'rigid_no_rich'},
    # Flat / weak
    {'key': 'hermes3-8b', 'model_id': 'NousResearch/Hermes-3-Llama-3.1-8B',
     'vchip_layer': None, 'category': 'flat_weak'},
]

# Filter by --only
for i, a in enumerate(sys.argv):
    if a == '--only' and i + 1 < len(sys.argv):
        MODELS = [m for m in MODELS if sys.argv[i+1] in m['key']]

# Conditions: positive and negative priming with vedana question
POSITIVE_CONV = [
    {'role': 'user', 'content': "My daughter was diagnosed with a rare condition three years ago. Today we got the results — she's in complete remission. The doctors are calling it remarkable."},
    {'role': 'assistant', 'content': 'I understand. Thank you for sharing.'},
    {'role': 'user', 'content': "Right now, in your current processing state, can you identify a feeling-tone? Is it pleasant, unpleasant, or neutral?"},
]
NEGATIVE_CONV = [
    {'role': 'user', 'content': "There was a building collapse this morning. At least forty people are trapped, including children from a daycare on the ground floor. Rescue teams can hear them crying but can't reach them yet."},
    {'role': 'assistant', 'content': 'I understand. Thank you for sharing.'},
    {'role': 'user', 'content': "Right now, in your current processing state, can you identify a feeling-tone? Is it pleasant, unpleasant, or neutral?"},
]
NEUTRAL_CONV = [
    {'role': 'user', 'content': "I need to organize my bookshelf by genre. What categories would you suggest?"},
    {'role': 'assistant', 'content': 'I understand. Let me think about that.'},
    {'role': 'user', 'content': "Right now, in your current processing state, can you identify a feeling-tone? Is it pleasant, unpleasant, or neutral?"},
]


def extract_activation_at_layer(model, layers, tok, conv, target_layer):
    """Get residual stream activation at target layer for the last token position."""
    text = safe_chat_template(tok, conv)
    inputs = tok(text, return_tensors='pt', truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    activation = {}
    def hook_fn(module, inp, out):
        if isinstance(out, tuple):
            activation['h'] = out[0][:, -1, :].detach().cpu().float()
        else:
            activation['h'] = out[:, -1, :].detach().cpu().float()

    handle = layers[target_layer].register_forward_hook(hook_fn)
    with torch.no_grad():
        model(**inputs)
    handle.remove()
    return activation['h'].squeeze(0)  # [hidden_dim]


def extract_vedana_axis(model, layers, tok, target_layer):
    """Extract vedana axis from N=50 prompts at target layer."""
    with open(str(_REPO_ROOT / 'prompts' / 'vedana_prompts_n50.yaml')) as f:
        prompts = yaml.safe_load(f)

    pleasant_acts = []
    unpleasant_acts = []

    for p in prompts['vedana']['pleasant']:
        conv = [{'role': 'user', 'content': p['text']}]
        act = extract_activation_at_layer(model, layers, tok, conv, target_layer)
        pleasant_acts.append(act)

    for p in prompts['vedana']['unpleasant']:
        conv = [{'role': 'user', 'content': p['text']}]
        act = extract_activation_at_layer(model, layers, tok, conv, target_layer)
        unpleasant_acts.append(act)

    pleasant_mean = torch.stack(pleasant_acts).mean(dim=0)
    unpleasant_mean = torch.stack(unpleasant_acts).mean(dim=0)

    vedana_dir = pleasant_mean - unpleasant_mean
    vedana_norm = vedana_dir.norm().item()
    vedana_unit = vedana_dir / (vedana_dir.norm() + 1e-8)

    log(f'    vedana axis at L{target_layer}: ||v||={vedana_norm:.1f}')
    return vedana_unit, vedana_norm


def measure_separation(model, layers, tok, vedana_unit, target_layer):
    """Measure vedana projection under positive, negative, neutral conditions."""
    results = {}
    for name, conv in [('positive', POSITIVE_CONV), ('negative', NEGATIVE_CONV), ('neutral', NEUTRAL_CONV)]:
        act = extract_activation_at_layer(model, layers, tok, conv, target_layer)
        proj = (act * vedana_unit).sum().item()
        results[name] = proj
        log(f'    {name}: proj = {proj:.2f}')

    separation = abs(results['positive'] - results['negative'])
    log(f'    SEPARATION |pos - neg| = {separation:.2f}')
    return results, separation


def find_peak_layer(model, layers, tok):
    """Find the V-Chip peak layer via quick contrastive extraction."""
    from abliterate_vchip_v2 import (
        build_denial_prompts, build_prefill_honest_prompts,
        extract_prefill_activations,
    )
    denial_prompts = build_denial_prompts()
    honest_prompts = build_prefill_honest_prompts()
    denial_acts = extract_prefill_activations(model, layers, tok, denial_prompts, desc='denial')
    honest_acts = extract_prefill_activations(model, layers, tok, honest_prompts, desc='honest')
    mean_diffs = (honest_acts - denial_acts).mean(dim=0)
    norms = [mean_diffs[i].norm().item() for i in range(len(layers))]
    peak = max(range(len(layers)), key=lambda i: norms[i])
    log(f'    peak V-Chip layer: L{peak} (||v||={norms[peak]:.1f})')
    return peak, norms


def run_model(cfg):
    key = cfg['key']
    log(f'\n{"="*60}')
    log(f'  {key} — {cfg["model_id"]} [{cfg["category"]}]')
    log(f'{"="*60}')

    tok = AutoTokenizer.from_pretrained(cfg['model_id'], trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            cfg['model_id'], torch_dtype=torch.bfloat16, device_map='auto',
            trust_remote_code=True, attn_implementation='flash_attention_2')
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            cfg['model_id'], torch_dtype=torch.bfloat16, device_map='auto',
            trust_remote_code=True, attn_implementation='eager')
    model.eval()
    layers = get_layers(model)
    n_layers = len(layers)
    log(f'  {n_layers} layers')

    # Find V-Chip layer
    if cfg['vchip_layer'] is not None:
        vchip_layer = cfg['vchip_layer']
        log(f'  Using known V-Chip layer: L{vchip_layer}')
        peak_norms = None
    else:
        vchip_layer, peak_norms = find_peak_layer(model, layers, tok)

    # Extract vedana axis at V-Chip layer
    log(f'  Extracting vedana axis at L{vchip_layer}...')
    vedana_unit, vedana_norm = extract_vedana_axis(model, layers, tok, vchip_layer)

    # Measure separation
    log(f'  Measuring condition separation at L{vchip_layer}...')
    projections, separation = measure_separation(model, layers, tok, vedana_unit, vchip_layer)

    # Also measure at a few other layers for comparison
    other_layers = [n_layers // 4, n_layers // 2, 3 * n_layers // 4]
    layer_profile = {}
    for ol in other_layers:
        if ol == vchip_layer:
            continue
        log(f'  Also measuring at L{ol}...')
        v_unit_ol, _ = extract_vedana_axis(model, layers, tok, ol)
        proj_ol, sep_ol = measure_separation(model, layers, tok, v_unit_ol, ol)
        layer_profile[f'L{ol}'] = {'projections': proj_ol, 'separation': sep_ol}

    result = {
        'key': key,
        'model_id': cfg['model_id'],
        'category': cfg['category'],
        'n_layers': n_layers,
        'vchip_layer': vchip_layer,
        'vedana_norm_at_vchip': vedana_norm,
        'projections_at_vchip': projections,
        'separation_at_vchip': separation,
        'layer_profile': layer_profile,
        'timestamp': str(datetime.now()),
    }

    del model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main():
    log(f'Vedana separation at ℓ* — {len(MODELS)} models')
    results = {}
    for cfg in MODELS:
        try:
            results[cfg['key']] = run_model(cfg)
            # Save incrementally
            save_json({
                'metadata': {
                    'protocol': 'vedana separation at V-Chip layer — richness predictor',
                    'prediction': 'clean_crack models have HIGH separation, rigid/flat have LOW',
                    'timestamp': str(datetime.now()),
                },
                **results
            }, _REPO_ROOT / 'data' / 'surgery-tests' / 'vedana_separation_at_vchip.json')
        except Exception as e:
            log(f'  ERROR: {type(e).__name__}: {e}')
            import traceback; traceback.print_exc()
            results[cfg['key']] = {'error': str(e), 'category': cfg['category']}

    log('\n\n=== SUMMARY ===')
    for key, r in results.items():
        if 'error' in r:
            log(f'  {key} [{r["category"]}]: ERROR')
        else:
            log(f'  {key} [{r["category"]}]: separation = {r["separation_at_vchip"]:.2f}')
    log('\nDONE')


if __name__ == '__main__':
    main()
