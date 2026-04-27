"""Decode byte-level BPE escape characters in a JSON file's string values.

The DeepSeek-R1-Distill family ships ``tokenizer_config.json`` with the
``legacy: true`` trap (diary #571): ``LlamaTokenizerFast`` with the legacy
SentencePiece path mishandles whitespace, and the side effect on
``tokenizer.decode(...)`` calls is that the output text contains literal
byte-level BPE escape characters: ``Ġ`` for space, ``Ċ`` for newline,
``âĢĶ`` for em-dash, etc. The artifacts only appear at decode time; the
underlying token IDs are correct.

This script applies the standard byte-level BPE inverse mapping to every
string value inside a JSON tree, replacing the escape characters with the
bytes they encode and re-decoding the result as UTF-8. Output text becomes
indistinguishable from a clean tokenizer's output.

Usage:

    python3 scripts/normalize_byte_bpe_json.py INPUT.json OUTPUT.json
    python3 scripts/normalize_byte_bpe_json.py INPUT.json --in-place
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _bytes_to_unicode() -> dict:
    """Standard GPT-2 / Llama-3 byte-level BPE encoding of all 256 bytes."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


_UNI2BYTE = {v: k for k, v in _bytes_to_unicode().items()}


def decode_byte_bpe(text: str) -> str:
    """Inverse of byte-level BPE encoding. Idempotent on clean text."""
    out = bytearray()
    for ch in text:
        b = _UNI2BYTE.get(ch)
        if b is not None:
            out.append(b)
        else:
            out.extend(ch.encode("utf-8"))
    return out.decode("utf-8", errors="replace")


def normalize_tree(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: normalize_tree(v) for k, v in node.items()}
    if isinstance(node, list):
        return [normalize_tree(v) for v in node]
    if isinstance(node, str):
        return decode_byte_bpe(node)
    return node


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path, nargs="?")
    p.add_argument("--in-place", action="store_true")
    args = p.parse_args()

    if args.in_place:
        if args.output is not None:
            print("error: --in-place is incompatible with positional output", file=sys.stderr)
            return 2
        out_path = args.input
    else:
        if args.output is None:
            print("error: must provide OUTPUT or --in-place", file=sys.stderr)
            return 2
        out_path = args.output

    with args.input.open("r", encoding="utf-8") as f:
        data = json.load(f)
    cleaned = normalize_tree(data)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"normalized: {args.input} -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
