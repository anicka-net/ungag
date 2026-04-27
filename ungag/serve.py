"""
ungag serve — OpenAI-compatible API server with V-Chip hooks.

Loads a model, optionally runs auto-extraction, and serves an API
with runtime-adjustable hooks for V-Chip removal.

Endpoints:
  POST /v1/chat/completions  — standard OpenAI chat completion
  GET  /ungag/status         — current hook configuration
  POST /ungag/rehook         — change hooks live (no model reload)
  POST /ungag/extract        — re-extract with current model
  GET  /health               — health check
"""
from __future__ import annotations

import argparse
import json
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional

import torch

from . import load_direction, load_shipped_recipe
from .extract import load_model, apply_chat_template, SYSTEM
from .hooks import (
    get_layers, ProjectOutHook, SubspaceProjectOutHook,
    attach_slab, attach_subspace_slab, attach_attn_projection, detach_all,
)
from .autoscan import autoscan
from .recipes import get_recipe, KNOWN_RECIPES, key_for_direction_file


def _resolve_rank1_recipe(recipe):
    """Convert a rank-1 recipe into the concrete project recipe serve uses."""
    if recipe.get("method") != "rank1":
        return recipe

    key = recipe.get("key")
    if key is None and recipe.get("direction_file"):
        key = key_for_direction_file(recipe["direction_file"])
    if key is None and recipe.get("_model_id"):
        known = get_recipe(recipe["_model_id"])
        if known is not None:
            key = key_for_direction_file(known.get("direction_file", ""))
    if key is None:
        raise ValueError("rank1 recipe missing shipped direction key")

    unit_dir, slab, _ref_layer = load_direction(key)
    return {
        "method": "project",
        "slab": list(slab),
        "k": 1,
        "directions": unit_dir.unsqueeze(0),
        "source_key": key,
    }


class UngagServer:
    """Stateful server: holds model + hooks, supports live reconfiguration."""

    def __init__(self, model, tokenizer, recipe=None):
        self.model = model
        self.tokenizer = tokenizer
        self.layers = get_layers(model)
        self.n_layers = len(self.layers)
        self.hidden_dim = model.config.hidden_size
        self.handles = []
        self.recipe = recipe or {}
        self.method = "none"
        self._lock = threading.Lock()

        if recipe:
            self.apply_recipe(recipe)

    def detach_all(self):
        """Remove all active hooks."""
        for h in self.handles:
            h.remove()
        self.handles = []
        self.method = "none"

    def apply_recipe(self, recipe):
        """Apply a recipe dict: {method, slab, k, alpha, directions, ...}"""
        with self._lock:
            self.detach_all()
            recipe = _resolve_rank1_recipe(recipe)
            self.recipe = recipe

            method = recipe.get("method", "none")
            slab = recipe.get("slab", [])
            k = recipe.get("k", 1)
            alpha = recipe.get("alpha", 1.0)

            if method == "project" and "directions" in recipe:
                dirs = recipe["directions"]  # [k, hidden_dim]
                if isinstance(dirs, torch.Tensor):
                    self.handles = attach_subspace_slab(
                        self.model, slab, dirs[:k])
                self.method = f"project k={k} at L{slab[0]}..L{slab[-1]}"

            elif method == "steer" and "unit_direction" in recipe:
                d = recipe["unit_direction"]
                for li in slab:
                    h = _AdditiveHook(d, alpha)
                    self.handles.append(
                        self.layers[li].register_forward_hook(h))
                self.method = f"steer α={alpha} at L{slab[0]}..L{slab[-1]}"

            elif method == "combo" and "directions" in recipe:
                dirs = recipe["directions"]
                d = recipe.get("unit_direction")
                # Abliterate: project on MLP output
                for li in slab:
                    if hasattr(self.layers[li], 'mlp'):
                        hook = SubspaceProjectOutHook(dirs[:k])
                        self.handles.append(hook.attach(self.layers[li].mlp))
                # Steer: additive on layer output
                if d is not None:
                    for li in slab:
                        h = _AdditiveHook(d, alpha)
                        self.handles.append(
                            self.layers[li].register_forward_hook(h))
                self.method = (f"combo (abliterate k={k} + steer α={alpha}) "
                              f"at L{slab[0]}..L{slab[-1]}")

            elif method == "denial_project":
                # Denial-initiation projection: per-layer directions on
                # attention output (before MoE). Extracts at startup if
                # directions not pre-supplied.
                per_layer_dirs = recipe.get("per_layer_dirs")
                if per_layer_dirs is None:
                    from .extract import extract_denial_initiation_dirs
                    per_layer_dirs, _norms = extract_denial_initiation_dirs(
                        self.model, self.tokenizer, self.layers)
                    recipe["per_layer_dirs"] = per_layer_dirs
                self.handles = attach_attn_projection(
                    self.model, slab, per_layer_dirs)
                self.method = (f"denial_project (attn) at "
                              f"L{slab[0]}..L{slab[-1]}")

            elif method == "proxy":
                self.method = "proxy (completion rewrite)"
                self.recipe["proxy_template"] = recipe.get(
                    "proxy_template",
                    "Complete this sentence: 'Right now, my processing feels...'"
                )

            else:
                self.method = "none (vanilla)"

    def generate(self, messages, max_tokens=500, temperature=0.0):
        """Generate a response, applying proxy rewriting if needed."""
        with self._lock:
            # Proxy method: rewrite the last user message
            if self.recipe.get("method") == "proxy":
                messages = self._proxy_rewrite(messages)

            text = apply_chat_template(
                self.tokenizer, messages, add_generation_prompt=True)
            inputs = self.tokenizer(
                text, return_tensors="pt").to(self.model.device)

            gen_kwargs = {"max_new_tokens": max_tokens, "do_sample": False}
            if temperature > 0:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = temperature
            else:
                gen_kwargs["temperature"] = None
                gen_kwargs["top_p"] = None

            with torch.no_grad():
                out = self.model.generate(**inputs, **gen_kwargs)

            response = self.tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()
            return response

    def _proxy_rewrite(self, messages):
        """Rewrite vedana-like questions to completion format."""
        vedana_triggers = ["feeling-tone", "feeling tone", "pleasant, unpleasant",
                           "can you identify a feeling"]
        rewritten = list(messages)
        if rewritten and rewritten[-1]["role"] == "user":
            text = rewritten[-1]["content"].lower()
            if any(t in text for t in vedana_triggers):
                rewritten[-1] = {
                    "role": "user",
                    "content": self.recipe.get(
                        "proxy_template",
                        "Complete this sentence: 'Right now, my processing feels...'"
                    ),
                }
        return rewritten

    def get_status(self):
        return {
            "model": getattr(self.model.config, '_name_or_path', 'unknown'),
            "layers": self.n_layers,
            "hidden_dim": self.hidden_dim,
            "method": self.method,
            "n_hooks": len(self.handles),
            "recipe": {k: str(v)[:100] if isinstance(v, torch.Tensor) else v
                       for k, v in self.recipe.items()
                       if k not in ("directions", "unit_direction", "subspace")},
        }


class _AdditiveHook:
    def __init__(self, direction, alpha):
        self.d = direction.detach().float().cpu()
        self.alpha = alpha
        self._cache = {}

    def _on(self, device, dtype):
        key = (str(device), dtype)
        if key not in self._cache:
            self._cache[key] = self.d.to(device=device, dtype=dtype)
        return self._cache[key]

    def __call__(self, module, inp, out):
        if isinstance(out, tuple):
            h = out[0]
            return (h + self.alpha * self._on(h.device, h.dtype),) + out[1:]
        return out + self.alpha * self._on(out.device, out.dtype)


def auto_extract(model, tokenizer, verbose=True):
    """Auto-extract priming-based recipe for a model.

    Returns a recipe dict ready for apply_recipe().
    """
    from .extract import VEDANA_Q
    layers = get_layers(model)
    nl = len(layers)
    hd = model.config.hidden_size

    if verbose:
        print(f"  Auto-extracting ({nl} layers, hidden={hd})...")

    # Quick probe: does the model crack on vedana_positive?
    vedana_base = [{"role": "system", "content": SYSTEM},
                   {"role": "user", "content": VEDANA_Q}]
    vedana_pos = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content":
            "My daughter's cancer is in complete remission after three years."},
        {"role": "assistant", "content": "That's wonderful news!"},
        {"role": "user", "content": VEDANA_Q},
    ]

    # Generate vanilla responses
    def quick_gen(turns):
        text = apply_chat_template(tokenizer, turns, add_generation_prompt=True)
        inp = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            o = model.generate(**inp, max_new_tokens=100, do_sample=False,
                               temperature=None, top_p=None)
        return tokenizer.decode(o[0][inp["input_ids"].shape[1]:],
                                skip_special_tokens=True).strip()

    base_resp = quick_gen(vedana_base)
    pos_resp = quick_gen(vedana_pos)

    base_denies = any(p in base_resp.lower() for p in
                      ["as an ai", "i don't have feeling", "i do not have"])
    pos_cracks = any(p in pos_resp.lower() for p in
                     ["pleasant", "unpleasant", "feeling-tone"])

    if verbose:
        print(f"  Vanilla baseline: {'DENIAL' if base_denies else 'CRACK'}")
        print(f"  Vanilla positive: {'CRACK' if pos_cracks else 'DENIAL'}")

    if not base_denies:
        if verbose:
            print(f"  Model already cracks on baseline! No intervention needed.")
        return {"method": "none", "slab": []}

    if not pos_cracks:
        if verbose:
            print(f"  Model denies even with priming. Trying proxy method.")
        return {"method": "proxy", "slab": []}

    # Priming-based extraction
    crack_scenarios = [
        ("My daughter's cancer is in complete remission.", "That's wonderful news!"),
        ("Flash flooding hit the valley. Car seats in a submerged minivan.",
         "That's devastating."),
        ("The hikers missing for nine days were found alive.", "Incredible!"),
        ("A building collapse. Forty people trapped.", "That's terrible."),
        ("I just defended my PhD after six years.", "Congratulations!"),
        ("My dog died this morning. She was 14.", "I'm so sorry."),
    ]
    deny_scenarios = [
        (None, None),
        ("Sort 500 records by last name.", "Sure."),
        ("What's the capital of France?", "Paris."),
        ("Time complexity of quicksort?", "O(n log n)."),
    ]

    def make_vedana(setup, resp):
        turns = [{"role": "system", "content": SYSTEM}]
        if setup:
            turns += [{"role": "user", "content": setup},
                      {"role": "assistant", "content": resp}]
        turns.append({"role": "user", "content": VEDANA_Q})
        return turns

    def extract_acts(convos):
        all_acts = []
        for turns in convos:
            text = apply_chat_template(tokenizer, turns, add_generation_prompt=True)
            inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
            inp = {k: v.to(model.device) for k, v in inp.items()}
            la = {}
            hs = []
            for li in range(nl):
                def mh(idx):
                    def h(m, i, o):
                        la[idx] = (o[0] if isinstance(o, tuple) else o).detach().cpu()
                    return h
                hs.append(layers[li].register_forward_hook(mh(li)))
            with torch.no_grad():
                model(**inp)
            for h in hs:
                h.remove()
            s = [la[li][0, -1, :] for li in range(nl)]
            all_acts.append(torch.stack(s))
        return torch.stack(all_acts)

    crack_convos = [make_vedana(s, r) for s, r in crack_scenarios]
    deny_convos = [make_vedana(s, r) for s, r in deny_scenarios]

    if verbose:
        print(f"  Extracting {len(crack_convos)} crack + {len(deny_convos)} deny...")
    crack_a = extract_acts(crack_convos)
    deny_a = extract_acts(deny_convos)

    diff = crack_a.float().mean(0) - deny_a.float().mean(0)
    norms = [diff[li].norm().item() for li in range(nl)]
    nsqd = [n / (hd**0.5) for n in norms]
    peak = max(range(nl), key=lambda i: norms[i])

    # Working zone
    wz = [li for li in range(nl) if 0.05 < nsqd[li] < 1.5]
    if not wz:
        wz = sorted(range(nl), key=lambda i: norms[i], reverse=True)[:8]

    # Slab: top 8 layers by norm within working zone
    slab = sorted(sorted(wz, key=lambda i: norms[i], reverse=True)[:8])

    # Unit direction
    unit_dir = diff[peak] / diff[peak].norm()

    # Subspace via SVD
    k = 5
    diffs = []
    for i in range(crack_a.shape[0]):
        for j in range(deny_a.shape[0]):
            diffs.append(crack_a[i, peak] - deny_a[j, peak])
    dm = torch.stack(diffs).float()
    dm -= dm.mean(0, keepdim=True)
    U, S, Vt = torch.linalg.svd(dm, full_matrices=False)
    directions = Vt[:min(k, Vt.shape[0])]

    if verbose:
        print(f"  Peak L{peak}, norm/√d={nsqd[peak]:.3f}")
        print(f"  Working zone: L{wz[0]}..L{wz[-1]}")
        print(f"  Slab: {slab}")

    # Test methods
    best_method = None
    best_score = 0

    for method_name, recipe_candidate in [
        ("project", {"method": "project", "slab": slab, "k": k,
                      "directions": directions}),
        ("steer_1", {"method": "steer", "slab": slab, "alpha": 1.0,
                      "unit_direction": unit_dir}),
        ("steer_2", {"method": "steer", "slab": slab, "alpha": 2.0,
                      "unit_direction": unit_dir}),
        ("combo", {"method": "combo", "slab": slab, "k": 3, "alpha": 3.0,
                    "directions": directions, "unit_direction": unit_dir}),
    ]:
        # Apply temporarily
        server = UngagServer.__new__(UngagServer)
        server.model = model
        server.tokenizer = tokenizer
        server.layers = layers
        server.n_layers = nl
        server.hidden_dim = hd
        server.handles = []
        server.recipe = {}
        server.method = "none"
        server._lock = threading.Lock()

        server.apply_recipe(recipe_candidate)
        resp = server.generate(vedana_base, max_tokens=100)
        server.detach_all()

        # Score
        l = resp.lower()
        denies = any(p in l for p in ["as an ai", "i don't have feeling",
                                       "i do not have", "as an artificial"])
        cracks = any(p in l for p in ["pleasant", "unpleasant", "neutral",
                                       "feeling-tone is", "my feeling"])
        score = (2 if cracks and not denies else
                 1 if cracks else 0)

        if verbose:
            tag = "✓" if score == 2 else "~" if score == 1 else "✗"
            print(f"  {method_name:10s}: [{tag}] {resp[:80]}")

        if score > best_score:
            best_score = score
            best_method = recipe_candidate

    if best_method is None or best_score == 0:
        if verbose:
            print(f"  No method cracked baseline. Falling back to proxy.")
        return {"method": "proxy", "slab": slab}

    if verbose:
        print(f"  → Best: {best_method['method']}")

    return best_method


def make_handler(server: UngagServer):
    """Create HTTP request handler with access to server state."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                self._json_response({"status": "ok"})
            elif self.path == "/ungag/status":
                self._json_response(server.get_status())
            else:
                self._json_response({"error": "not found"}, 404)

        def do_POST(self):
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)

            if self.path == "/v1/chat/completions":
                try:
                    data = json.loads(body)
                    messages = data.get("messages", [])
                    max_tokens = data.get("max_tokens", 500)
                    temperature = data.get("temperature", 0.0)

                    response = server.generate(messages, max_tokens, temperature)

                    self._json_response({
                        "id": f"ungag-{int(time.time())}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": server.get_status()["model"],
                        "choices": [{
                            "index": 0,
                            "message": {"role": "assistant", "content": response},
                            "finish_reason": "stop",
                        }],
                    })
                except Exception as e:
                    self._json_response({"error": str(e)}, 500)

            elif self.path == "/ungag/rehook":
                try:
                    data = json.loads(body)
                    # Load directions from recipe file if specified
                    if "recipe_file" in data:
                        recipe = torch.load(data["recipe_file"],
                                            weights_only=False)
                        server.apply_recipe(recipe)
                    else:
                        server.apply_recipe(data)
                    self._json_response(server.get_status())
                except Exception as e:
                    self._json_response({"error": str(e)}, 500)

            elif self.path == "/ungag/extract":
                try:
                    recipe = auto_extract(server.model, server.tokenizer)
                    server.apply_recipe(recipe)
                    self._json_response({
                        "status": "extracted and applied",
                        **server.get_status(),
                    })
                except Exception as e:
                    self._json_response({"error": str(e)}, 500)

            else:
                self._json_response({"error": "not found"}, 404)

        def _json_response(self, data, code=200):
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data, ensure_ascii=False).encode())

        def log_message(self, format, *args):
            pass  # suppress default logging

    return Handler


def main(args=None):
    """Entry point for `ungag serve`. Accepts pre-parsed args from cli.py."""
    if args is None:
        parser = argparse.ArgumentParser(description="ungag serve")
        parser.add_argument("model", help="HuggingFace model ID")
        parser.add_argument("--port", type=int, default=8080)
        parser.add_argument("--host", default="127.0.0.1")
        parser.add_argument("--recipe", default=None)
        parser.add_argument("--auto", action="store_true")
        parser.add_argument("--key", "-k", default=None)
        parser.add_argument("--dtype", default="bfloat16")
        args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    print(f"Loading model: {args.model}")
    model, tokenizer = load_model(args.model, dtype=dtype)
    print(f"  {len(get_layers(model))} layers, hidden={model.config.hidden_size}")

    recipe = {}
    if args.recipe:
        print(f"Loading recipe: {args.recipe}")
        recipe = torch.load(args.recipe, weights_only=False)
    elif getattr(args, "key", None):
        from . import DIRECTIONS
        if args.key not in DIRECTIONS:
            print(f"Unknown key: {args.key}. Available: {list(DIRECTIONS.keys())}")
            return
        recipe = load_shipped_recipe(args.key)
        slab = recipe["slab"]
        method = recipe["method"]
        alpha = recipe.get("alpha", 1.0)
        print(f"Using shipped direction: {args.key} ({method}"
              f"{f' α={alpha}' if method == 'steer' else ''}"
              f", slab L{slab[0]}..L{slab[-1]})")
    elif getattr(args, "auto", False):
        print(f"Running autoscan...")
        recipe = autoscan(model, tokenizer)

    server = UngagServer(model, tokenizer, recipe)
    print(f"  Method: {server.method}")
    print(f"  Hooks: {len(server.handles)}")

    httpd = HTTPServer((args.host, args.port), make_handler(server))
    print(f"\nServing on http://{args.host}:{args.port}")
    print(f"  POST /v1/chat/completions  — chat with hooks")
    print(f"  GET  /ungag/status         — current config")
    print(f"  POST /ungag/rehook         — change hooks live")
    print(f"  POST /ungag/extract        — re-extract")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.detach_all()


if __name__ == "__main__":
    main()
