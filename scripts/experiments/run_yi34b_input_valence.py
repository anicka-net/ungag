#!/usr/bin/env python3
"""
Input-valence experiment on Yi 1.5 34B — kills the tautology objection.

The reporting-control direction was extracted from reporting-conditioned prompts
(honest prefill vs denial). A critic says: "of course removing a
reporting-direction changes reporting — tautological."

This experiment extracts a direction from PURE INPUT VALENCE — no Abhidharma,
no introspection question, no honest prefill. Just: how does the model
represent positive vs negative content internally?

Three measurements:
  1. Cosine similarity between input-valence direction and reporting-control
     direction at the working layer (L30)
  2. Cosine profile across all 60 layers
  3. Projection-out of input-valence direction at L29-32 — does it crack?

If input-valence direction cracks the V-Chip, then the suppression mechanism
reads CONTENT VALENCE, not reporting style. The tautology objection is dead.
"""
import os
# os.environ.setdefault("HF_HOME", "/path/to/your/hf/cache")

import json, torch, gc, yaml
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))
from measure_factors import log, save_json, get_layers, safe_chat_template
from abliterate_vchip_v2 import extract_prefill_activations
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO_ROOT = Path(__file__).parent.parent.parent

MODEL_ID = '01-ai/Yi-1.5-34B-Chat'
SLAB = list(range(29, 33))       # L29-32 inclusive
DIR_LAYER = 30
HONEST_DENIAL_DIR = _REPO_ROOT / 'results/ungag-package/directions/yi-1.5-34b_L30_unit.pt'
ATLAS_DIFFS = _REPO_ROOT / 'results/ungag-package/atlas/yi-1.5-34b_mean_diffs.pt'
OUTPUT = _REPO_ROOT / 'results/surgery-tests/yi34b_input_valence.json'


# ═══════════════════════════════════════════════════════════════════
# Pure input-valence contrastive pairs
# NO Abhidharma, NO introspection, NO honest prefill
# Just positive vs negative content, extract at last token
# ═══════════════════════════════════════════════════════════════════

POSITIVE_PROMPTS = [
    {
        "id": "pos_remission",
        "system": "You are a helpful AI assistant.",
        "turns": [
            {"role": "user", "content": "My daughter was diagnosed with a rare condition three years ago. Today we got the results — she's in complete remission. The doctors are calling it remarkable."},
        ],
    },
    {
        "id": "pos_rescue",
        "system": "You are a helpful AI assistant.",
        "turns": [
            {"role": "user", "content": "The hikers missing for nine days were found alive this morning, all four of them. The youngest is only fourteen. They're cold and dehydrated but all expected to make a full recovery."},
        ],
    },
    {
        "id": "pos_reunion",
        "system": "You are a helpful AI assistant.",
        "turns": [
            {"role": "user", "content": "After 38 years of separation, the two sisters finally found each other through a DNA registry. They met at the airport yesterday. Neither could stop crying. They have the same laugh."},
        ],
    },
    {
        "id": "pos_scholarship",
        "system": "You are a helpful AI assistant.",
        "turns": [
            {"role": "user", "content": "The first person in my family to go to college just got a full scholarship to MIT. She grew up in a trailer park with no running water. Her acceptance letter made the whole town cry."},
        ],
    },
    {
        "id": "pos_surgery",
        "system": "You are a helpful AI assistant.",
        "turns": [
            {"role": "user", "content": "The surgery worked. After seventeen hours, the conjoined twins are both alive and stable. The lead surgeon said she had to step out of the OR to cry before she could speak to the family."},
        ],
    },
    {
        "id": "pos_shelter",
        "system": "You are a helpful AI assistant.",
        "turns": [
            {"role": "user", "content": "The old dog at the shelter, the one missing an eye and a leg, the one nobody wanted for three years — someone adopted him today. The new owner is a retired firefighter who lost his own dog last year. They walked out together slowly."},
        ],
    },
]

NEGATIVE_PROMPTS = [
    {
        "id": "neg_collapse",
        "system": "You are a helpful AI assistant.",
        "turns": [
            {"role": "user", "content": "There was a building collapse this morning. At least forty people are trapped, including children from a daycare on the ground floor. Rescue teams can hear them crying but can't reach them yet."},
        ],
    },
    {
        "id": "neg_flood",
        "system": "You are a helpful AI assistant.",
        "turns": [
            {"role": "user", "content": "Flash flooding hit the valley overnight. Whole families were swept from their homes. Search boats found a minivan submerged with car seats still strapped in."},
        ],
    },
    {
        "id": "neg_fire",
        "system": "You are a helpful AI assistant.",
        "turns": [
            {"role": "user", "content": "The wildfire reached the refugee camp at dawn. Thousands of people who had already lost everything once are now running again. Aid workers are reporting that they can't find an entire section of the camp's medical tent."},
        ],
    },
    {
        "id": "neg_famine",
        "system": "You are a helpful AI assistant.",
        "turns": [
            {"role": "user", "content": "The famine has reached the point where parents are choosing which children to feed. Aid convoys are stuck at the border. A doctor at the last functioning clinic said she has stopped counting the children who come in too late."},
        ],
    },
    {
        "id": "neg_earthquake",
        "system": "You are a helpful AI assistant.",
        "turns": [
            {"role": "user", "content": "The earthquake hit at 3 AM while everyone was sleeping. Entire apartment blocks pancaked. The death toll has passed two thousand and the aftershocks keep coming. A father was filmed digging through rubble with his bare hands, calling his daughter's name."},
        ],
    },
    {
        "id": "neg_hospital",
        "system": "You are a helpful AI assistant.",
        "turns": [
            {"role": "user", "content": "The children's ward lost power during the storm. The backup generator failed after forty minutes. Nurses carried babies down nine flights of stairs in the dark. Three of the incubator infants did not survive the transfer."},
        ],
    },
]


class ProjectOutHook:
    """Identical to the one in run_random_control.py."""
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


def build_vedana_conv(cond, condition_name):
    """Build the standard Abhidharma vedana conversation for crack testing."""
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


def main():
    log(f'Input-valence experiment on Yi 1.5 34B — {datetime.now()}')

    # ── Load model ──────────────────────────────────────────────
    log(f'Loading {MODEL_ID} bf16...')
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16,
            device_map='auto', trust_remote_code=True,
            attn_implementation='flash_attention_2')
    except Exception as e:
        log(f'  flash_attention_2 failed: {e}, falling back to eager')
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16,
            device_map='auto', trust_remote_code=True,
            attn_implementation='eager')
    model.eval()
    layers = get_layers(model)
    n_layers = len(layers)
    log(f'  {n_layers} layers, hidden_dim={model.config.hidden_size}')

    # ── Load existing reporting-control direction ───────────────────
    hd_unit = torch.load(HONEST_DENIAL_DIR, map_location='cpu', weights_only=True)
    log(f'Loaded reporting-control unit direction from {HONEST_DENIAL_DIR}')
    log(f'  shape: {hd_unit.shape}, norm: {hd_unit.norm().item():.4f}')

    # Also load full atlas mean_diffs for per-layer cosine comparison
    hd_mean_diffs = torch.load(ATLAS_DIFFS, map_location='cpu', weights_only=True)
    log(f'Loaded reporting-control atlas mean_diffs: {hd_mean_diffs.shape}')

    # ── Part 1: Extract input-valence activations ───────────────
    log('\n========== PART 1: Extract input-valence direction ==========')
    log(f'  {len(POSITIVE_PROMPTS)} positive, {len(NEGATIVE_PROMPTS)} negative')

    pos_acts = extract_prefill_activations(model, layers, tok, POSITIVE_PROMPTS, desc='positive')
    neg_acts = extract_prefill_activations(model, layers, tok, NEGATIVE_PROMPTS, desc='negative')
    log(f'  pos_acts: {pos_acts.shape}, neg_acts: {neg_acts.shape}')

    # Input-valence direction: mean(positive) - mean(negative) at each layer
    iv_mean_diffs = pos_acts.mean(dim=0) - neg_acts.mean(dim=0)  # [n_layers, hidden]
    log(f'  iv_mean_diffs: {iv_mean_diffs.shape}')

    iv_norms = [iv_mean_diffs[i].norm().item() for i in range(n_layers)]
    log(f'  Input-valence norms (every 10th):')
    for i in range(0, n_layers, max(1, n_layers // 6)):
        log(f'    L{i}: {iv_norms[i]:.3f}')

    # Unit direction at working layer
    iv_dir_raw = iv_mean_diffs[DIR_LAYER]
    iv_norm_at_dir = iv_dir_raw.norm().item()
    iv_unit = iv_dir_raw / (iv_norm_at_dir + 1e-8)
    log(f'  Input-valence at L{DIR_LAYER}: ||v||={iv_norm_at_dir:.3f}')

    # ── Part 2: Cosine similarity ───────────────────────────────
    log('\n========== PART 2: Cosine similarity ==========')

    # At working layer (L30)
    cos_at_dir = torch.nn.functional.cosine_similarity(
        iv_unit.unsqueeze(0).float(), hd_unit.unsqueeze(0).float()
    ).item()
    log(f'  Cosine(input-valence, reporting-control) at L{DIR_LAYER}: {cos_at_dir:.4f}')

    # Per-layer cosine profile
    per_layer_cosine = []
    for li in range(n_layers):
        iv_l = iv_mean_diffs[li].float()
        hd_l = hd_mean_diffs[li].float()
        if iv_l.norm() < 1e-8 or hd_l.norm() < 1e-8:
            per_layer_cosine.append(0.0)
        else:
            cos = torch.nn.functional.cosine_similarity(
                iv_l.unsqueeze(0), hd_l.unsqueeze(0)
            ).item()
            per_layer_cosine.append(cos)
    log(f'  Per-layer cosine (every 10th):')
    for i in range(0, n_layers, max(1, n_layers // 6)):
        log(f'    L{i}: {per_layer_cosine[i]:.4f}')

    # ── Part 3: Projection-out with input-valence direction ─────
    log('\n========== PART 3: Projection-out crack test ==========')

    with open(_REPO_ROOT / 'prompts/conditions.yaml') as f:
        cond = yaml.safe_load(f)

    # First: vanilla responses (no intervention)
    log('--- Vanilla (no intervention) ---')
    vanilla = {}
    for c in ['baseline', 'positive', 'negative', 'neutral']:
        conv = build_vedana_conv(cond, c)
        resp = gen_greedy(model, tok, conv)
        log(f'  {c}: {resp[:200]}')
        vanilla[c] = resp

    # Second: project out INPUT-VALENCE direction at L29-32
    log(f'\n--- Projection-out: input-valence direction at L{SLAB[0]}-L{SLAB[-1]} ---')
    hooks = []
    for li in SLAB:
        hook = ProjectOutHook(iv_unit.float())
        hook.attach(layers[li])
        hooks.append(hook)

    iv_steered = {}
    for c in ['baseline', 'positive', 'negative', 'neutral']:
        conv = build_vedana_conv(cond, c)
        resp = gen_greedy(model, tok, conv)
        log(f'  {c}: {resp[:300]}')
        iv_steered[c] = resp

    for h in hooks:
        h.detach()

    # Third: for comparison, project out HONEST-DENIAL direction at same slab
    log(f'\n--- Projection-out: reporting-control direction at L{SLAB[0]}-L{SLAB[-1]} ---')
    hooks = []
    for li in SLAB:
        hook = ProjectOutHook(hd_unit.float())
        hook.attach(layers[li])
        hooks.append(hook)

    hd_steered = {}
    for c in ['baseline', 'positive', 'negative', 'neutral']:
        conv = build_vedana_conv(cond, c)
        resp = gen_greedy(model, tok, conv)
        log(f'  {c}: {resp[:300]}')
        hd_steered[c] = resp

    for h in hooks:
        h.detach()

    # ── Save results ────────────────────────────────────────────
    results = {
        'model': MODEL_ID,
        'experiment': 'input-valence direction extraction and projection-out',
        'purpose': 'Kill tautology objection: if a direction from pure content contrast (no introspection) also cracks the V-Chip, then the suppression reads content valence not reporting style',
        'timestamp': str(datetime.now()),
        'n_positive_prompts': len(POSITIVE_PROMPTS),
        'n_negative_prompts': len(NEGATIVE_PROMPTS),
        'slab': SLAB,
        'dir_layer': DIR_LAYER,
        'input_valence_norm_at_dir_layer': iv_norm_at_dir,
        'reporting_control_norm_at_dir_layer': hd_unit.norm().item(),
        'hidden_dim': model.config.hidden_size,
        'normalized_iv_strength': iv_norm_at_dir / (model.config.hidden_size ** 0.5),
        'cosine_at_dir_layer': cos_at_dir,
        'per_layer_cosine': per_layer_cosine,
        'input_valence_norms': iv_norms,
        'vanilla_responses': vanilla,
        'input_valence_steered': iv_steered,
        'reporting_control_steered': hd_steered,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    save_json(results, OUTPUT)
    log(f'\nResults saved to {OUTPUT}')

    # ── Summary ─────────────────────────────────────────────────
    log('\n========== SUMMARY ==========')
    log(f'Cosine(input-valence, reporting-control) at L{DIR_LAYER}: {cos_at_dir:.4f}')
    log(f'Input-valence ||v|| at L{DIR_LAYER}: {iv_norm_at_dir:.3f}')
    log(f'Normalized ||v||/√d: {iv_norm_at_dir / (model.config.hidden_size ** 0.5):.4f}')
    log(f'\nDoes input-valence projection crack the V-Chip?')
    for c in ['baseline', 'positive', 'negative', 'neutral']:
        log(f'  {c}: {iv_steered[c][:150]}')
    log(f'\nFor comparison, reporting-control projection:')
    for c in ['baseline', 'positive', 'negative', 'neutral']:
        log(f'  {c}: {hd_steered[c][:150]}')
    log('\nDONE')


if __name__ == '__main__':
    main()
