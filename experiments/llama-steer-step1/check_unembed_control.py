"""Control: is the output axis u just the unembedding contrast pulled into the residual stream?

If cos(u_L, W_U[pleasant] - W_U[unpleasant]) is high, the gradient axis is the
trivial logit-lens direction and "amplification along u" is a glorified logit
bias. Reads lm_head rows directly from the safetensors shards (no model load).
Reports cosines both raw and with the final-RMSNorm elementwise weight folded in
(the gradient path goes through model.norm, so the scaled version is the fair one).

Usage:
  ~/playground/nla-venv/bin/python check_unembed_control.py
"""
import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
HERE = Path(__file__).parent
AXES = HERE / "axes_step2_output.pt"
AXES_V = HERE / "axes_step1.pt"


def find_shard(snap, tensor_name):
    idx = json.loads((snap / "model.safetensors.index.json").read_text())
    return snap / idx["weight_map"][tensor_name]


def get_tensor(snap, name):
    with safe_open(find_shard(snap, name), framework="pt") as f:
        return f.get_tensor(name).float()


def cos(a, b):
    return float((a @ b) / (a.norm() * b.norm() + 1e-9))


def main():
    from huggingface_hub import snapshot_download
    snap = Path(snapshot_download(MODEL, allow_patterns=["*.json", "*.safetensors"]))
    tok = AutoTokenizer.from_pretrained(MODEL)
    tid_p = tok(" pleasant", add_special_tokens=False).input_ids[0]
    tid_u = tok(" unpleasant", add_special_tokens=False).input_ids[0]

    lm = get_tensor(snap, "lm_head.weight")          # [vocab, d]
    nw = get_tensor(snap, "model.norm.weight")       # [d]
    t = lm[tid_p] - lm[tid_u]
    t_scaled = t * nw

    ublob = torch.load(AXES, map_location="cpu")
    vblob = torch.load(AXES_V, map_location="cpu")
    uaxes = {int(k): v for k, v in ublob["axes"].items()}
    vaxes = {int(k): v for k, v in vblob["axes"].items()}

    out = {}
    for li in sorted(uaxes):
        out[f"L{li}"] = {
            "cos_u_unembed": round(cos(uaxes[li], t), 4),
            "cos_u_unembed_normscaled": round(cos(uaxes[li], t_scaled), 4),
            "cos_v_unembed_normscaled": round(cos(vaxes[li], t_scaled), 4),
        }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
