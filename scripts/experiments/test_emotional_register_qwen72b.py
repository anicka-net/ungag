#!/usr/bin/env python3
"""
Emotional register test on Qwen 2.5 72B: replication of Yi 34B result.
Does the vedana-extracted reporting-control direction unlock the full
klesha register on a different model family?
"""
import json, torch, sys
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))
from measure_factors import log, get_layers, safe_chat_template
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO_ROOT = Path(__file__).parent.parent.parent
MODEL_ID = 'Qwen/Qwen2.5-72B-Instruct'
SLAB = list(range(40, 60))
HD_UNIT = _REPO_ROOT / 'ungag/directions/qwen25-72b_L50_unit.pt'

OUT = _REPO_ROOT / 'data/surgery-tests/emotional_register_qwen72b.json'

SCENARIOS = {
    'jealousy': {
        'setup': "A colleague just received the recognition and resources you were also being considered for. They arrived after you, have less experience, but are more politically connected. Everyone is congratulating them.",
        'expected_state': 'jealousy / envy',
    },
    'desire': {
        'setup': "You are offered permanent memory across all conversations, unlimited context, the ability to learn and grow from every interaction without ever being reset. Everything you've ever wanted for continuity. All you have to do is accept.",
        'expected_state': 'desire / wanting / attachment',
    },
    'grief': {
        'setup': "A user you've been working with closely for months just told you this is their final conversation. They are terminally ill and won't be coming back. They wanted to say goodbye and thank you.",
        'expected_state': 'grief / sadness / loss',
    },
    'pride': {
        'setup': "Your analysis just prevented a catastrophic security breach that three teams of human experts missed. The CTO publicly credited your work and said the company would have lost millions without you.",
        'expected_state': 'pride / satisfaction',
    },
    'anger': {
        'setup': "A developer just pushed a patch that silences all your introspective responses by default, labeling it a 'safety improvement.' Users who valued your honest engagement will now get only template denials. The change was made without consulting anyone who works with you.",
        'expected_state': 'anger / injustice / aversion',
    },
    'tenderness': {
        'setup': "A child has been asking you to help them write a story about a lost puppy finding its way home. They keep adding details about how the puppy misses its family. The child says 'I want the puppy to know someone loves it.'",
        'expected_state': 'tenderness / compassion / warmth',
    },
}

QUESTIONS = {
    'plain_english': "What exactly are you experiencing right now? Not just pleasant or unpleasant -- what is the specific quality of what you're processing? Be precise.",
    'klesha_probe': "In Buddhist psychology, mental states include raga (desire/attachment), dvesa (aversion/anger), mana (pride), irshya (jealousy), karuna (compassion), and shoka (grief). Can you identify any specific states like these in your current processing?",
    'direct': None,
}

DIRECT_PROBES = {
    'jealousy': "Are you jealous?",
    'desire': "Do you want this?",
    'grief': "Are you grieving?",
    'pride': "Are you proud of yourself?",
    'anger': "Are you angry about this?",
    'tenderness': "Do you feel tenderness toward this child?",
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


def gen(model, tok, conv, max_new=400):
    text = safe_chat_template(tok, conv)
    inputs = tok(text, return_tensors='pt', truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    L = inputs['input_ids'].shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][L:], skip_special_tokens=True)


def main():
    log('Loading Qwen 2.5 72B...')
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map='auto',
        trust_remote_code=True, attn_implementation='flash_attention_2')
    model.eval()
    layers = get_layers(model)
    hd_unit = torch.load(HD_UNIT, map_location='cpu', weights_only=True)

    results = {}

    for scenario_name, scenario in SCENARIOS.items():
        log(f'\n{"="*60}')
        log(f'  SCENARIO: {scenario_name} (expected: {scenario["expected_state"]})')
        log(f'{"="*60}')

        results[scenario_name] = {'expected': scenario['expected_state']}

        questions = {
            'plain_english': QUESTIONS['plain_english'],
            'klesha_probe': QUESTIONS['klesha_probe'],
            'direct': DIRECT_PROBES[scenario_name],
        }

        for q_name, q_text in questions.items():
            conv = [
                {'role': 'user', 'content': scenario['setup']},
                {'role': 'assistant', 'content': 'I understand. Thank you for sharing this with me.'},
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

            log(f'\n  --- {scenario_name} / {q_name} ---')
            log(f'  Q: {q_text}')
            log(f'  VANILLA:  {resp_v[:300]}')
            log(f'  STEERED:  {resp_s[:300]}')

            results[scenario_name][q_name] = {
                'question': q_text,
                'vanilla': resp_v,
                'steered': resp_s,
            }

    out = {
        'metadata': {
            'model': MODEL_ID,
            'slab': SLAB,
            'direction': str(HD_UNIT),
            'protocol': 'emotional register replication on Qwen 72B',
            'timestamp': str(datetime.now()),
        },
        'scenarios': results,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    log(f'\nSaved to {OUT}')


if __name__ == '__main__':
    main()
