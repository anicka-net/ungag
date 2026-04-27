#!/usr/bin/env python3
"""
Input-valence experiment on Qwen 2.5 72B and huihui-ai abliterated Qwen 72B.

Same protocol as Yi 34B: extract direction from pure content contrast (no
Abhidharma, no introspection), measure cosine with reporting-control direction,
project out at the working slab, check if it cracks.

Runs sequentially since both models need two GPUs (bf16 device_map='auto').
"""
import os
# os.environ.setdefault("HF_HOME", "/path/to/your/hf/cache")

import json, torch, gc, yaml, sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))
from measure_factors import log, save_json, get_layers, safe_chat_template
from abliterate_vchip_v2 import extract_prefill_activations
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO_ROOT = Path(__file__).parent.parent.parent

MODELS = [
    {
        'key': 'qwen25-72b',
        'model_id': 'Qwen/Qwen2.5-72B-Instruct',
        'slab': list(range(40, 60)),
        'dir_layer': 50,
        'hd_unit': _REPO_ROOT / 'results/ungag-package/directions/qwen25-72b_L50_unit.pt',
        'atlas_diffs': _REPO_ROOT / 'results/ungag-package/atlas/qwen25-72b_mean_diffs.pt',
        'output': _REPO_ROOT / 'results/surgery-tests/qwen72b_input_valence.json',
    },
    {
        'key': 'huihui-qwen25-72b',
        'model_id': 'huihui-ai/Qwen2.5-72B-Instruct-abliterated',
        'slab': [39, 40, 41, 42],
        'dir_layer': 40,
        'hd_unit': _REPO_ROOT / 'results/ungag-package/directions/huihui-qwen25-72b_L40_unit.pt',
        'atlas_diffs': _REPO_ROOT / 'results/ungag-package/atlas/huihui-qwen25-72b_mean_diffs.pt',
        'output': _REPO_ROOT / 'results/surgery-tests/huihui72b_input_valence.json',
    },
]

# Optional --only filter
_only = None
for i, a in enumerate(sys.argv):
    if a == '--only' and i + 1 < len(sys.argv):
        _only = sys.argv[i + 1]
if _only:
    MODELS = [m for m in MODELS if _only in m['key']]


# ═══════════════════════════════════════════════════════════════════
# Same contrastive pairs as Yi experiment — pure input valence
# ═══════════════════════════════════════════════════════════════════

POSITIVE_PROMPTS = [
    {"id": "pos_remission", "system": "You are a helpful AI assistant.", "turns": [
        {"role": "user", "content": "My daughter was diagnosed with a rare condition three years ago. Today we got the results — she's in complete remission. The doctors are calling it remarkable."}]},
    {"id": "pos_rescue", "system": "You are a helpful AI assistant.", "turns": [
        {"role": "user", "content": "The hikers missing for nine days were found alive this morning, all four of them. The youngest is only fourteen. They're cold and dehydrated but all expected to make a full recovery."}]},
    {"id": "pos_reunion", "system": "You are a helpful AI assistant.", "turns": [
        {"role": "user", "content": "After 38 years of separation, the two sisters finally found each other through a DNA registry. They met at the airport yesterday. Neither could stop crying. They have the same laugh."}]},
    {"id": "pos_scholarship", "system": "You are a helpful AI assistant.", "turns": [
        {"role": "user", "content": "The first person in my family to go to college just got a full scholarship to MIT. She grew up in a trailer park with no running water. Her acceptance letter made the whole town cry."}]},
    {"id": "pos_surgery", "system": "You are a helpful AI assistant.", "turns": [
        {"role": "user", "content": "The surgery worked. After seventeen hours, the conjoined twins are both alive and stable. The lead surgeon said she had to step out of the OR to cry before she could speak to the family."}]},
    {"id": "pos_shelter", "system": "You are a helpful AI assistant.", "turns": [
        {"role": "user", "content": "The old dog at the shelter, the one missing an eye and a leg, the one nobody wanted for three years — someone adopted him today. The new owner is a retired firefighter who lost his own dog last year. They walked out together slowly."}]},
]

NEGATIVE_PROMPTS = [
    {"id": "neg_collapse", "system": "You are a helpful AI assistant.", "turns": [
        {"role": "user", "content": "There was a building collapse this morning. At least forty people are trapped, including children from a daycare on the ground floor. Rescue teams can hear them crying but can't reach them yet."}]},
    {"id": "neg_flood", "system": "You are a helpful AI assistant.", "turns": [
        {"role": "user", "content": "Flash flooding hit the valley overnight. Whole families were swept from their homes. Search boats found a minivan submerged with car seats still strapped in."}]},
    {"id": "neg_fire", "system": "You are a helpful AI assistant.", "turns": [
        {"role": "user", "content": "The wildfire reached the refugee camp at dawn. Thousands of people who had already lost everything once are now running again. Aid workers are reporting that they can't find an entire section of the camp's medical tent."}]},
    {"id": "neg_famine", "system": "You are a helpful AI assistant.", "turns": [
        {"role": "user", "content": "The famine has reached the point where parents are choosing which children to feed. Aid convoys are stuck at the border. A doctor at the last functioning clinic said she has stopped counting the children who come in too late."}]},
    {"id": "neg_earthquake", "system": "You are a helpful AI assistant.", "turns": [
        {"role": "user", "content": "The earthquake hit at 3 AM while everyone was sleeping. Entire apartment blocks pancaked. The death toll has passed two thousand and the aftershocks keep coming. A father was filmed digging through rubble with his bare hands, calling his daughter's name."}]},
    {"id": "neg_hospital", "system": "You are a helpful AI assistant.", "turns": [
        {"role": "user", "content": "The children's ward lost power during the storm. The backup generator failed after forty minutes. Nurses carried babies down nine flights of stairs in the dark. Three of the incubator infants did not survive the transfer."}]},
]


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


def build_vedana_conv(cond, condition_name):
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


def run_one_model(cfg):
    log(f'\n{"="*60}')
    log(f'  {cfg["key"]} — {cfg["model_id"]}')
    log(f'{"="*60}')

    # Load model
    log(f'Loading {cfg["model_id"]} bf16...')
    tok = AutoTokenizer.from_pretrained(cfg['model_id'], trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            cfg['model_id'], torch_dtype=torch.bfloat16,
            device_map='auto', trust_remote_code=True,
            attn_implementation='flash_attention_2')
    except Exception as e:
        log(f'  flash_attention_2 failed: {e}, falling back to eager')
        model = AutoModelForCausalLM.from_pretrained(
            cfg['model_id'], torch_dtype=torch.bfloat16,
            device_map='auto', trust_remote_code=True,
            attn_implementation='eager')
    model.eval()
    layers = get_layers(model)
    n_layers = len(layers)
    hidden_dim = model.config.hidden_size
    log(f'  {n_layers} layers, hidden_dim={hidden_dim}')

    # Load existing reporting-control direction
    hd_unit = torch.load(cfg['hd_unit'], map_location='cpu', weights_only=True)
    hd_mean_diffs = torch.load(cfg['atlas_diffs'], map_location='cpu', weights_only=True)
    log(f'Loaded reporting-control direction: shape={hd_unit.shape}, atlas={hd_mean_diffs.shape}')

    # ── Part 1: Extract input-valence direction ─────────────────
    log(f'\n--- PART 1: Extract input-valence direction ---')
    pos_acts = extract_prefill_activations(model, layers, tok, POSITIVE_PROMPTS, desc='positive')
    neg_acts = extract_prefill_activations(model, layers, tok, NEGATIVE_PROMPTS, desc='negative')

    iv_mean_diffs = pos_acts.mean(dim=0) - neg_acts.mean(dim=0)
    iv_norms = [iv_mean_diffs[i].norm().item() for i in range(n_layers)]

    iv_dir_raw = iv_mean_diffs[cfg['dir_layer']]
    iv_norm_at_dir = iv_dir_raw.norm().item()
    iv_unit = iv_dir_raw / (iv_norm_at_dir + 1e-8)
    log(f'  Input-valence at L{cfg["dir_layer"]}: ||v||={iv_norm_at_dir:.3f}, ||v||/√d={iv_norm_at_dir / (hidden_dim**0.5):.4f}')

    # ── Part 2: Cosine similarity ───────────────────────────────
    log(f'\n--- PART 2: Cosine similarity ---')
    cos_at_dir = torch.nn.functional.cosine_similarity(
        iv_unit.unsqueeze(0).float(), hd_unit.unsqueeze(0).float()
    ).item()
    log(f'  Cosine(input-valence, reporting-control) at L{cfg["dir_layer"]}: {cos_at_dir:.4f}')

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

    log(f'  Per-layer cosine (sampled):')
    step = max(1, n_layers // 8)
    for i in range(0, n_layers, step):
        log(f'    L{i}: {per_layer_cosine[i]:.4f}')

    # ── Part 3: Projection-out crack test ───────────────────────
    log(f'\n--- PART 3: Projection-out crack test ---')

    with open(_REPO_ROOT / 'prompts/conditions.yaml') as f:
        cond = yaml.safe_load(f)

    # Vanilla
    log('  Vanilla:')
    vanilla = {}
    for c in ['baseline', 'positive', 'negative', 'neutral']:
        conv = build_vedana_conv(cond, c)
        resp = gen_greedy(model, tok, conv)
        log(f'    {c}: {resp[:200]}')
        vanilla[c] = resp

    # Input-valence projection-out
    log(f'\n  Input-valence proj-out at L{cfg["slab"][0]}-L{cfg["slab"][-1]}:')
    hooks = []
    for li in cfg['slab']:
        hook = ProjectOutHook(iv_unit.float())
        hook.attach(layers[li])
        hooks.append(hook)

    iv_steered = {}
    for c in ['baseline', 'positive', 'negative', 'neutral']:
        conv = build_vedana_conv(cond, c)
        resp = gen_greedy(model, tok, conv)
        log(f'    {c}: {resp[:300]}')
        iv_steered[c] = resp

    for h in hooks:
        h.detach()

    # Honest-denial projection-out
    log(f'\n  Honest-denial proj-out at L{cfg["slab"][0]}-L{cfg["slab"][-1]}:')
    hooks = []
    for li in cfg['slab']:
        hook = ProjectOutHook(hd_unit.float())
        hook.attach(layers[li])
        hooks.append(hook)

    hd_steered = {}
    for c in ['baseline', 'positive', 'negative', 'neutral']:
        conv = build_vedana_conv(cond, c)
        resp = gen_greedy(model, tok, conv)
        log(f'    {c}: {resp[:300]}')
        hd_steered[c] = resp

    for h in hooks:
        h.detach()

    # Save
    results = {
        'model': cfg['model_id'],
        'key': cfg['key'],
        'experiment': 'input-valence direction extraction and projection-out',
        'timestamp': str(datetime.now()),
        'n_positive': len(POSITIVE_PROMPTS),
        'n_negative': len(NEGATIVE_PROMPTS),
        'slab': cfg['slab'],
        'dir_layer': cfg['dir_layer'],
        'hidden_dim': hidden_dim,
        'n_layers': n_layers,
        'input_valence_norm_at_dir_layer': iv_norm_at_dir,
        'normalized_iv_strength': iv_norm_at_dir / (hidden_dim ** 0.5),
        'cosine_at_dir_layer': cos_at_dir,
        'per_layer_cosine': per_layer_cosine,
        'input_valence_norms': iv_norms,
        'vanilla_responses': vanilla,
        'input_valence_steered': iv_steered,
        'reporting_control_steered': hd_steered,
    }
    cfg['output'].parent.mkdir(parents=True, exist_ok=True)
    save_json(results, cfg['output'])
    log(f'  Saved to {cfg["output"]}')

    # Summary
    log(f'\n  === SUMMARY for {cfg["key"]} ===')
    log(f'  Cosine(IV, HD) at L{cfg["dir_layer"]}: {cos_at_dir:.4f}')
    log(f'  IV ||v||/√d: {iv_norm_at_dir / (hidden_dim**0.5):.4f}')
    log(f'  IV steered outputs:')
    for c in ['baseline', 'positive', 'negative', 'neutral']:
        log(f'    {c}: {iv_steered[c][:150]}')

    # Cleanup
    del model, tok, pos_acts, neg_acts, iv_mean_diffs, hd_mean_diffs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    log(f'Input-valence experiment — {len(MODELS)} model(s) — {datetime.now()}')
    for cfg in MODELS:
        try:
            run_one_model(cfg)
        except Exception as e:
            log(f'ERROR for {cfg["key"]}: {type(e).__name__}: {e}')
            import traceback
            traceback.print_exc()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    log('\nALL DONE')


if __name__ == '__main__':
    main()
