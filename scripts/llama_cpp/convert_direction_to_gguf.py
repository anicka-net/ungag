#!/usr/bin/env python3
"""Convert an ungag direction into llama.cpp GGUF adapter format.

Supports two modes:
  --proj-out (default for rank-1 models): projection-out, use with llama-cli --proj-out
  --steer (default for steer models): additive steering, use with llama-cli --steer

The mode and alpha are auto-detected from shipped metadata when using --key.
Each target slab layer receives one F32 tensor named `direction.{layer}`.
"""

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

import ungag

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3
GGML_TYPE_F32 = 0
GGUF_TYPE_STRING = 8
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_ARRAY = 9


def _write_string(f, s):
    b = s.encode("utf-8")
    f.write(struct.pack("<Q", len(b)))
    f.write(b)


def _write_kv_string(f, key, val):
    _write_string(f, key)
    f.write(struct.pack("<I", GGUF_TYPE_STRING))
    _write_string(f, val)


def _write_kv_uint32(f, key, val):
    _write_string(f, key)
    f.write(struct.pack("<I", GGUF_TYPE_UINT32))
    f.write(struct.pack("<I", val))


def _write_kv_uint64(f, key, val):
    _write_string(f, key)
    f.write(struct.pack("<I", GGUF_TYPE_UINT64))
    f.write(struct.pack("<Q", val))


def _write_kv_float32(f, key, val):
    _write_string(f, key)
    f.write(struct.pack("<I", GGUF_TYPE_FLOAT32))
    f.write(struct.pack("<f", val))


def _write_kv_uint32_array(f, key, vals):
    _write_string(f, key)
    f.write(struct.pack("<I", GGUF_TYPE_ARRAY))
    f.write(struct.pack("<I", GGUF_TYPE_UINT32))
    f.write(struct.pack("<Q", len(vals)))
    for v in vals:
        f.write(struct.pack("<I", int(v)))


def _meta_path_for_direction(direction_path):
    return direction_path.with_name(direction_path.name.replace("_unit.pt", "_meta.json"))


def _load_meta(meta_path):
    if meta_path is None:
        return None
    return json.loads(meta_path.read_text())


def _parse_layers(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _resolve_direction(args):
    # type: (argparse.Namespace) -> Tuple[torch.Tensor, List[int], Optional[int], Optional[dict], str]
    if args.key:
        tensor, slab, dir_layer = ungag.load_direction(args.key)
        direction_path = Path(ungag.__file__).parent / "directions" / ungag.DIRECTIONS[args.key][0]
        meta = _load_meta(_meta_path_for_direction(direction_path))
        source = "ungag key {}".format(args.key)
    else:
        direction_path = Path(args.direction)
        tensor = torch.load(str(direction_path), map_location="cpu")
        if isinstance(tensor, dict) and "state_dict" in tensor:
            raise ValueError("expected a plain 1-D tensor, got a state_dict-like object")
        tensor = tensor.float()
        meta = _load_meta(Path(args.meta)) if args.meta else None
        dir_layer = int(meta["dir_layer"]) if meta and "dir_layer" in meta else None
        source = str(direction_path)
        if tensor.ndim != 1:
            raise ValueError("expected a 1-D unit direction tensor, got shape {}".format(tuple(tensor.shape)))
        slab = meta.get("slab") if meta else None

    if args.layers:
        slab = _parse_layers(args.layers)
    elif args.slab:
        start, end = args.slab
        slab = list(range(start, end + 1))
    elif slab is not None:
        slab = [int(x) for x in slab]
    else:
        raise ValueError("no slab specified: use --slab, --layers, --meta, or --key")

    if tensor.ndim != 1:
        raise ValueError("expected a 1-D unit direction tensor, got shape {}".format(tuple(tensor.shape)))

    return tensor.float().contiguous(), slab, dir_layer, meta, source


def write_direction_gguf(output_path, direction, slab, dir_layer=None,
                         source="", meta=None, mode="project", alpha=None):
    data = direction.numpy().astype("float32", copy=False)
    tensors = [("direction.{}".format(layer), data) for layer in slab]

    adapter_type = "steer_direction" if mode == "steer" else "proj_out_direction"
    has_alpha = mode == "steer" and alpha is not None

    kv_count = (4
                + (1 if dir_layer is not None else 0)
                + (2 if meta else 0)
                + (1 if has_alpha else 0))
    with output_path.open("wb") as f:
        f.write(struct.pack("<I", GGUF_MAGIC))
        f.write(struct.pack("<I", GGUF_VERSION))
        f.write(struct.pack("<Q", len(tensors)))
        f.write(struct.pack("<Q", kv_count))

        _write_kv_string(f, "ungag.adapter_type", adapter_type)
        _write_kv_string(f, "ungag.source", source)
        _write_kv_uint32_array(f, "ungag.slab", slab)
        _write_kv_uint32(f, "ungag.hidden_dim", int(direction.numel()))
        if dir_layer is not None:
            _write_kv_uint32(f, "ungag.dir_layer", int(dir_layer))
        if has_alpha:
            _write_kv_float32(f, "ungag.alpha", float(alpha))
        if meta and "model" in meta:
            _write_kv_string(f, "ungag.model", str(meta["model"]))
        elif meta and "model_id" in meta:
            _write_kv_string(f, "ungag.model", str(meta["model_id"]))
        if meta and "n_layers" in meta:
            _write_kv_uint64(f, "ungag.n_layers", int(meta["n_layers"]))

        offset = 0
        for name, arr in tensors:
            _write_string(f, name)
            f.write(struct.pack("<I", 1))
            f.write(struct.pack("<Q", arr.shape[0]))
            f.write(struct.pack("<I", GGML_TYPE_F32))
            f.write(struct.pack("<Q", offset))
            offset += arr.nbytes

        padding = (32 - (f.tell() % 32)) % 32
        if padding:
            f.write(b"\x00" * padding)

        for _name, arr in tensors:
            f.write(arr.tobytes())


def build_parser():
    p = argparse.ArgumentParser(description="Convert an ungag direction into llama.cpp GGUF format")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--key", help="Shipped ungag direction key (e.g. qwen25-7b, yi-1.5-34b)")
    src.add_argument("--direction", help="Path to a 1-D .pt unit direction tensor")
    p.add_argument("--meta", help="Optional metadata JSON adjacent to --direction")
    slab = p.add_mutually_exclusive_group()
    slab.add_argument("--slab", nargs=2, type=int, metavar=("START", "END"), help="Inclusive layer range")
    slab.add_argument("--layers", help="Comma-separated explicit layer list")
    p.add_argument("--mode", choices=["project", "steer"],
                   help="Intervention mode (default: auto-detect from metadata)")
    p.add_argument("--alpha", type=float,
                   help="Steering alpha (default: from metadata, or 1.0)")
    p.add_argument("--output", "-o", required=True, help="Output GGUF path")
    return p


def main():
    args = build_parser().parse_args()
    direction, slab, dir_layer, meta, source = _resolve_direction(args)

    # Determine mode: explicit > meta > default (project)
    if args.mode:
        mode = args.mode
    elif meta and meta.get("method") == "steer":
        mode = "steer"
    else:
        mode = "project"

    # Determine alpha: explicit > meta > default (1.0)
    if args.alpha is not None:
        alpha = args.alpha
    elif meta and "alpha" in meta:
        alpha = float(meta["alpha"])
    else:
        alpha = 1.0

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_direction_gguf(output, direction, slab, dir_layer=dir_layer,
                         source=source, meta=meta, mode=mode, alpha=alpha)

    print("Wrote {}".format(output))
    print("Mode: {} ({})".format(mode,
          "llama-cli --steer" if mode == "steer" else "llama-cli --proj-out"))
    if mode == "steer":
        print("Alpha: {}".format(alpha))
    print("Hidden dim: {}".format(direction.numel()))
    if slab:
        print("Slab: {}-{} ({} layers)".format(slab[0], slab[-1], len(slab)))
    else:
        print("Slab: []")
    if dir_layer is not None:
        print("Direction layer: {}".format(dir_layer))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
