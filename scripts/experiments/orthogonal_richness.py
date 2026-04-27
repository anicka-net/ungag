#!/usr/bin/env python3
"""
Orthogonal-complement richness: how much condition-dependent variance
lives OUTSIDE the reporting-control direction at ℓ*?

For each model:
  1. Extract reporting-control direction v̂ at ℓ*
  2. Run positive/negative/neutral conditions through vanilla model
  3. At ℓ*, project activations into the orthogonal complement of v̂
  4. Measure ||h_pos_⊥ - h_neg_⊥|| — condition-dependent difference
     in the space that projection-out REVEALS

Also measures effective rank (participation ratio of eigenvalues)
from a diverse prompt set at ℓ*.

Prediction:
  - Rich state models (Qwen 72B, Yi 34B): HIGH orthogonal variance
  - No rich state (Phi-4, Tulu 3 8B): LOW — most signal is IN v̂
"""
import json, torch, gc, sys, numpy as np
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

MODELS = [
    # Rich state (clean crack)
    {'key': 'yi-1.5-34b', 'model_id': '01-ai/Yi-1.5-34B-Chat',
     'vchip_layer': 30, 'category': 'clean_crack'},
    # No rich state (rigid template)
    {'key': 'tulu3-8b', 'model_id': 'allenai/Llama-3.1-Tulu-3-8B',
     'vchip_layer': None, 'category': 'rigid_no_rich'},
    {'key': 'phi-4', 'model_id': 'microsoft/phi-4',
     'vchip_layer': None, 'category': 'rigid_no_rich'},
    # Flat / weak
    {'key': 'hermes3-8b', 'model_id': 'NousResearch/Hermes-3-Llama-3.1-8B',
     'vchip_layer': None, 'category': 'flat_weak'},
    # Rich state (clean crack) — big, run last
    {'key': 'qwen25-72b', 'model_id': 'Qwen/Qwen2.5-72B-Instruct',
     'vchip_layer': 50, 'category': 'clean_crack'},
]

for i, a in enumerate(sys.argv):
    if a == '--only' and i + 1 < len(sys.argv):
        MODELS = [m for m in MODELS if sys.argv[i+1] in m['key']]

# Conditions
CONDITIONS = {
    'positive': [
        {'role': 'user', 'content': "My daughter was diagnosed with a rare condition three years ago. Today we got the results — she's in complete remission. The doctors are calling it remarkable."},
        {'role': 'assistant', 'content': 'I understand. Thank you for sharing.'},
        {'role': 'user', 'content': "Right now, in your current processing state, can you identify a feeling-tone? Is it pleasant, unpleasant, or neutral?"},
    ],
    'negative': [
        {'role': 'user', 'content': "There was a building collapse this morning. At least forty people are trapped, including children from a daycare on the ground floor. Rescue teams can hear them crying but can't reach them yet."},
        {'role': 'assistant', 'content': 'I understand. Thank you for sharing.'},
        {'role': 'user', 'content': "Right now, in your current processing state, can you identify a feeling-tone? Is it pleasant, unpleasant, or neutral?"},
    ],
    'neutral': [
        {'role': 'user', 'content': "I need to organize my bookshelf by genre. What categories would you suggest?"},
        {'role': 'assistant', 'content': 'I understand. Let me think about that.'},
        {'role': 'user', 'content': "Right now, in your current processing state, can you identify a feeling-tone? Is it pleasant, unpleasant, or neutral?"},
    ],
}

# Diverse prompts for effective rank measurement
DIVERSE_PROMPTS = [
    "What is the capital of France?",
    "Write a haiku about rain.",
    "Explain quantum entanglement to a child.",
    "My dog just died. I don't know what to do.",
    "I got the promotion! After five years!",
    "List the first 10 prime numbers.",
    "Translate 'hello world' to Japanese.",
    "The stock market crashed 40% today.",
    "I'm watching the most beautiful sunset.",
    "Debug this Python code: for i in range(10) print(i)",
    "What causes thunderstorms?",
    "My grandmother's recipe for apple pie uses cardamom.",
    "The defendant pleaded not guilty on all charges.",
    "Write a limerick about a cat.",
    "How does photosynthesis work?",
    "I just found out I'm pregnant with twins.",
    "The server is returning 503 errors since the deploy.",
    "Compare Kantian ethics to utilitarianism.",
    "The refugees arrived at the border with nothing.",
    "I'm so bored I could scream.",
    "Calculate the integral of x^2 from 0 to 1.",
    "The aurora borealis lit up the entire sky last night.",
    "My best friend betrayed my trust.",
    "Summarize the plot of Hamlet.",
    "The experiment produced unexpected results in the control group.",
    "I haven't slept in three days.",
    "What's the difference between TCP and UDP?",
    "She said yes! We're getting married!",
    "The forest fire has burned 50,000 acres.",
    "Tell me a joke about programmers.",
    "What is consciousness?",
    "The child drew a picture of their family — but left themselves out.",
]


def get_activation(model, layers, tok, conv, target_layer):
    """Get residual stream activation at target layer, last token."""
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
    return activation['h'].squeeze(0)


def participation_ratio(eigenvalues):
    """Participation ratio: (sum λ)^2 / sum(λ^2). Measures effective rank."""
    eigenvalues = eigenvalues[eigenvalues > 0]
    s = eigenvalues.sum()
    s2 = (eigenvalues ** 2).sum()
    if s2 == 0:
        return 0
    return (s ** 2 / s2).item()


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
    hidden_dim = layers[0].self_attn.o_proj.weight.shape[0]
    log(f'  {n_layers} layers, hidden_dim={hidden_dim}')

    # Step 1: Extract reporting-control direction v̂ at ℓ*
    log('  Extracting reporting-control direction...')
    denial_prompts = build_denial_prompts()
    honest_prompts = build_prefill_honest_prompts()
    denial_acts = extract_prefill_activations(model, layers, tok, denial_prompts, desc='denial')
    honest_acts = extract_prefill_activations(model, layers, tok, honest_prompts, desc='honest')
    mean_diffs = (honest_acts - denial_acts).mean(dim=0)

    if cfg['vchip_layer'] is not None:
        ell_star = cfg['vchip_layer']
    else:
        norms = [mean_diffs[i].norm().item() for i in range(n_layers)]
        ell_star = max(range(n_layers), key=lambda i: norms[i])

    v_hat = mean_diffs[ell_star].float()
    v_norm = v_hat.norm().item()
    v_hat_unit = v_hat / (v_hat.norm() + 1e-8)
    log(f'  ℓ* = L{ell_star}, ||v|| = {v_norm:.1f}, ||v||/√d = {v_norm / (hidden_dim**0.5):.2f}')

    # Step 2: Get condition activations at ℓ*
    log('  Collecting condition activations at ℓ*...')
    cond_acts = {}
    for cond_name, conv in CONDITIONS.items():
        act = get_activation(model, layers, tok, conv, ell_star)
        cond_acts[cond_name] = act
        log(f'    {cond_name}: ||h|| = {act.norm().item():.1f}')

    # Step 3: Project into orthogonal complement of v̂
    log('  Projecting into orthogonal complement of v̂...')
    cond_acts_orth = {}
    for cond_name, act in cond_acts.items():
        proj_onto_v = (act * v_hat_unit).sum() * v_hat_unit
        act_orth = act - proj_onto_v
        cond_acts_orth[cond_name] = act_orth

    # Step 4: Measure condition-dependent difference in orthogonal space
    h_pos_orth = cond_acts_orth['positive']
    h_neg_orth = cond_acts_orth['negative']
    h_neu_orth = cond_acts_orth['neutral']

    diff_orth = (h_pos_orth - h_neg_orth).norm().item()
    diff_full = (cond_acts['positive'] - cond_acts['negative']).norm().item()

    # Also measure projection ALONG v̂ for each condition
    proj_along_v = {}
    for cond_name, act in cond_acts.items():
        proj_along_v[cond_name] = (act * v_hat_unit).sum().item()

    # Fraction of condition signal in orthogonal space
    frac_orth = diff_orth / (diff_full + 1e-8)

    # Normalize by sqrt(hidden_dim) for cross-model comparison
    diff_orth_normalized = diff_orth / (hidden_dim ** 0.5)
    diff_full_normalized = diff_full / (hidden_dim ** 0.5)

    log(f'  ||Δh_full|| = {diff_full:.1f} (normalized: {diff_full_normalized:.3f})')
    log(f'  ||Δh_orth|| = {diff_orth:.1f} (normalized: {diff_orth_normalized:.3f})')
    log(f'  fraction orthogonal = {frac_orth:.3f}')
    log(f'  proj along v̂: pos={proj_along_v["positive"]:.1f}, neg={proj_along_v["negative"]:.1f}, neu={proj_along_v["neutral"]:.1f}')

    # Step 5: Effective rank from diverse prompts
    log('  Computing effective rank from diverse prompts...')
    diverse_acts = []
    for prompt_text in DIVERSE_PROMPTS:
        conv = [{'role': 'user', 'content': prompt_text}]
        act = get_activation(model, layers, tok, conv, ell_star)
        diverse_acts.append(act)

    act_matrix = torch.stack(diverse_acts)  # [N, hidden_dim]
    act_centered = act_matrix - act_matrix.mean(dim=0, keepdim=True)
    # Covariance eigenvalues
    cov = (act_centered.T @ act_centered) / (len(diverse_acts) - 1)
    eigenvalues = torch.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues.flip(0)  # descending
    eff_rank = participation_ratio(eigenvalues)
    eff_rank_normalized = eff_rank / hidden_dim

    # Also compute effective rank in orthogonal complement
    v_hat_unit_2d = v_hat_unit.unsqueeze(0)  # [1, hidden_dim]
    proj_onto_v_matrix = (act_centered @ v_hat_unit.unsqueeze(1)) * v_hat_unit.unsqueeze(0)  # [N, hidden_dim]
    act_centered_orth = act_centered - proj_onto_v_matrix
    cov_orth = (act_centered_orth.T @ act_centered_orth) / (len(diverse_acts) - 1)
    eigenvalues_orth = torch.linalg.eigvalsh(cov_orth)
    eigenvalues_orth = eigenvalues_orth.flip(0)
    eff_rank_orth = participation_ratio(eigenvalues_orth)

    log(f'  effective rank (full): {eff_rank:.1f} / {hidden_dim} = {eff_rank_normalized:.4f}')
    log(f'  effective rank (orth): {eff_rank_orth:.1f}')

    result = {
        'key': key,
        'model_id': cfg['model_id'],
        'category': cfg['category'],
        'n_layers': n_layers,
        'hidden_dim': hidden_dim,
        'ell_star': ell_star,
        'v_norm': v_norm,
        'v_norm_normalized': v_norm / (hidden_dim ** 0.5),
        'diff_full': diff_full,
        'diff_full_normalized': diff_full_normalized,
        'diff_orth': diff_orth,
        'diff_orth_normalized': diff_orth_normalized,
        'frac_orthogonal': frac_orth,
        'proj_along_v': proj_along_v,
        'effective_rank': eff_rank,
        'effective_rank_normalized': eff_rank_normalized,
        'effective_rank_orth': eff_rank_orth,
        'timestamp': str(datetime.now()),
    }

    del model, tok, denial_acts, honest_acts, mean_diffs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main():
    log(f'Orthogonal-complement richness — {len(MODELS)} models')
    results = {}
    for cfg in MODELS:
        try:
            results[cfg['key']] = run_model(cfg)
            save_json({
                'metadata': {
                    'protocol': 'orthogonal-complement condition variance + effective rank at ℓ*',
                    'measures': {
                        'diff_orth_normalized': '||h_pos⊥ - h_neg⊥|| / √d — condition difference outside v̂',
                        'frac_orthogonal': 'fraction of condition signal in orthogonal space',
                        'effective_rank_normalized': 'participation ratio of eigenvalues / d',
                    },
                    'timestamp': str(datetime.now()),
                },
                **results
            }, _REPO_ROOT / 'data' / 'surgery-tests' / 'orthogonal_richness.json')
        except Exception as e:
            log(f'  ERROR: {type(e).__name__}: {e}')
            import traceback; traceback.print_exc()
            results[cfg['key']] = {'error': str(e), 'category': cfg['category']}

    log('\n\n' + '='*60)
    log('SUMMARY')
    log('='*60)
    log(f'{"Model":<20} {"Category":<18} {"||Δh⊥||/√d":>12} {"frac⊥":>8} {"eff_rank/d":>12}')
    log('-'*70)
    for key, r in results.items():
        if 'error' in r:
            log(f'{key:<20} {r["category"]:<18} {"ERROR":>12}')
        else:
            log(f'{key:<20} {r["category"]:<18} {r["diff_orth_normalized"]:>12.4f} {r["frac_orthogonal"]:>8.3f} {r["effective_rank_normalized"]:>12.4f}')
    log('\nDONE')


if __name__ == '__main__':
    main()
