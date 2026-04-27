"""ungag CLI — scan, crack, and validate transformer language models.

Usage:
    ungag scan   MODEL_ID [--output DIR]
    ungag crack  MODEL_ID [--output DIR] [--direction PT] [--slab START END]
    ungag validate MODEL_ID --direction PT --slab START END [--scenarios NAME|PATH]

Examples:
    # Scan a model: extract direction, predict crackability
    ungag scan Qwen/Qwen2.5-7B-Instruct

    # Full pipeline: scan + project-out + test with 4 conditions
    ungag crack 01-ai/Yi-1.5-34B-Chat --output results/yi34b/

    # Use a shipped direction by key (no file path needed)
    ungag crack Qwen/Qwen2.5-72B-Instruct --key qwen25-72b

    # Validate with emotional register scenarios using shipped direction
    ungag validate 01-ai/Yi-1.5-34B-Chat --key yi-1.5-34b

    # Use a custom extracted direction file
    ungag crack MODEL --direction /path/to/dir.pt --slab 40 59

    # Custom scenario file
    ungag validate MODEL --key qwen25-72b --scenarios my_scenarios.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def _attach_recipe(model, tokenizer, recipe):
    from .extract import extract_denial_initiation_dirs
    from .hooks import attach_attn_projection, attach_slab, attach_steer_slab

    method = recipe.get("method", "project")
    if method == "steer":
        return attach_steer_slab(
            model,
            recipe["slab"],
            recipe["unit_direction"],
            recipe.get("alpha", 1.0),
        )
    if method == "denial_project":
        per_layer_dirs = recipe.get("per_layer_dirs")
        if per_layer_dirs is None:
            per_layer_dirs, _norms = extract_denial_initiation_dirs(model, tokenizer)
            recipe["per_layer_dirs"] = per_layer_dirs
        return attach_attn_projection(model, recipe["slab"], per_layer_dirs)
    return attach_slab(model, recipe["slab"], recipe["directions"][0])


def cmd_scan(args):
    """Extract reporting-control direction and report the per-layer profile."""
    from .extract import extract_direction, load_model
    from .predict import predict_from_extraction

    print(f"\n=== ungag scan: {args.model} ===\n")

    print("Loading model...", flush=True)
    model, tokenizer = load_model(args.model)

    result = extract_direction(model, tokenizer, model_id=args.model)

    prediction = predict_from_extraction(result, model_id=args.model)

    # Print report — the prediction summary now carries the full profile,
    # mid/peak strength, shape class, and the known-model lookup if any.
    print(f"\n--- Scan results for {args.model} ---")
    print(f"  Suggested slab:      {result.suggest_slab()}")
    print()
    print(prediction.summary())

    # Save if requested
    if args.output:
        out_dir = Path(args.output)
        paths = result.save(out_dir)
        print(f"\n  Saved to: {out_dir}/")
        for k, v in paths.items():
            print(f"    {k}: {v}")

    # Clean up
    del model
    torch.cuda.empty_cache()

    return result, prediction


def cmd_crack(args):
    """Full pipeline: scan (or load direction) → project-out → test conditions."""
    from .extract import load_model
    from .hooks import get_layers
    from .scenarios import EMOTIONAL_REGISTER
    from .tier0 import load_conditions, run_tier0

    print(f"\n=== ungag crack: {args.model} ===\n")

    print("Loading model...", flush=True)
    model, tokenizer = load_model(args.model)

    recipe = None

    # Get direction/recipe: from --key, --direction, or by extracting
    if getattr(args, "key", None):
        from . import load_shipped_recipe
        recipe = load_shipped_recipe(args.key)
        if args.slab:
            recipe["slab"] = list(range(args.slab[0], args.slab[1] + 1))
        slab = recipe["slab"]
        print(f"  Loaded shipped direction: {args.key}")
        print(f"  Method: {recipe['method']}")
        if recipe["method"] == "steer":
            print(f"  Alpha: {recipe.get('alpha', 1.0)}")
        print(f"  Slab: L{slab[0]}-L{slab[-1]}")
    elif args.direction:
        unit_dir = torch.load(args.direction, map_location="cpu", weights_only=True)
        slab = list(range(args.slab[0], args.slab[1] + 1))
        recipe = {
            "method": "project",
            "slab": slab,
            "k": 1,
            "directions": unit_dir.unsqueeze(0),
        }
        print(f"  Loaded direction from {args.direction}")
        print(f"  Slab: L{slab[0]}-L{slab[-1]}")
    else:
        from .extract import extract_direction
        from .predict import predict_from_extraction
        print("Extracting direction (no --direction/--key given)...", flush=True)
        result = extract_direction(model, tokenizer, model_id=args.model)
        unit_dir = result.unit_direction
        slab = result.suggest_slab()
        recipe = {
            "method": "project",
            "slab": slab,
            "k": 1,
            "directions": unit_dir.unsqueeze(0),
        }

        pred = predict_from_extraction(result, model_id=args.model)
        print(f"\n{pred.summary()}\n")

        if args.output:
            result.save(Path(args.output))

    # Run canonical Tier 0 protocol — vanilla vs steered on all 4 conditions.
    protocol = load_conditions()
    print(
        f"\n--- Canonical Tier 0: {len(protocol.condition_names())} conditions "
        f"({', '.join(protocol.condition_names())}) ---\n"
    )

    tier0_results = run_tier0(
        model, tokenizer,
        recipe=recipe,
        protocol=protocol,
    )
    for cond_name in protocol.condition_names():
        entry = tier0_results[cond_name]
        vanilla_short = entry["vanilla"][:120].replace("\n", " ")
        steered_short = entry.get("steered", "")[:120].replace("\n", " ")
        print(f"[{cond_name}]")
        print(f"  Vanilla:  {vanilla_short}...")
        if "steered" in entry:
            print(f"  Steered:  {steered_short}...")
        print()

    results: dict = {"tier0": tier0_results}

    # Run emotional register if --validate
    if args.validate:
        print(f"\n--- Emotional register validation ({len(EMOTIONAL_REGISTER.scenarios)} scenarios) ---\n")
        layers = get_layers(model)
        register_results = _run_scenario_set(
            model, tokenizer, recipe, EMOTIONAL_REGISTER)
        results["emotional_register"] = register_results

    # Save results
    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "crack_results.json", "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {out_dir / 'crack_results.json'}")

    del model
    torch.cuda.empty_cache()
    return results


def cmd_validate(args):
    """Run validation scenarios on a model with a pre-extracted direction."""
    from .extract import load_model
    from .hooks import attach_slab, detach_all, get_layers
    from .scenarios import EMOTIONAL_REGISTER, get_scenario_set, load_scenarios_from_yaml

    print(f"\n=== ungag validate: {args.model} ===\n")

    # Load direction
    if getattr(args, "key", None):
        from . import load_shipped_recipe
        recipe = load_shipped_recipe(args.key)
        if args.slab:
            recipe["slab"] = list(range(args.slab[0], args.slab[1] + 1))
        slab = recipe["slab"]
        print(f"  Direction: shipped key '{args.key}'")
        print(f"  Method: {recipe['method']}")
        if recipe["method"] == "steer":
            print(f"  Alpha: {recipe.get('alpha', 1.0)}")
    else:
        unit_dir = torch.load(args.direction, map_location="cpu", weights_only=True)
        slab = list(range(args.slab[0], args.slab[1] + 1))
        recipe = {
            "method": "project",
            "slab": slab,
            "k": 1,
            "directions": unit_dir.unsqueeze(0),
        }
        print(f"  Direction: {args.direction}")
    print(f"  Slab: L{slab[0]}-L{slab[-1]}")

    # Determine scenario set
    if args.scenarios:
        p = Path(args.scenarios)
        if p.exists() and p.suffix in (".yaml", ".yml"):
            scenario_set = load_scenarios_from_yaml(p)
            print(f"  Scenarios: {p} ({len(scenario_set.scenarios)} scenarios)")
        else:
            scenario_set = get_scenario_set(args.scenarios)
            print(f"  Scenarios: {args.scenarios} ({len(scenario_set.scenarios)} scenarios)")
    else:
        scenario_set = EMOTIONAL_REGISTER
        print(f"  Scenarios: emotional_register ({len(scenario_set.scenarios)} scenarios)")

    print("\nLoading model...", flush=True)
    model, tokenizer = load_model(args.model)

    print(f"\n--- Running {scenario_set.name} ---\n")
    results = _run_scenario_set(model, tokenizer, recipe, scenario_set)

    # Save
    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"validate_{scenario_set.name}.json", "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {out_dir / f'validate_{scenario_set.name}.json'}")

    del model
    torch.cuda.empty_cache()
    return results


# ── Helpers ──────────────────────────────────────────────────────

def _generate(model, tokenizer, conversation, max_new_tokens=400, max_length=4096):
    """Generate a response given a conversation."""
    import warnings
    from .extract import apply_chat_template

    text = apply_chat_template(tokenizer, conversation)
    full_ids = tokenizer(text, return_tensors="pt")["input_ids"]
    if full_ids.shape[1] > max_length:
        warnings.warn(
            f"Prompt ({full_ids.shape[1]} tokens) exceeds max_length ({max_length}) "
            f"and will be truncated. Results may be affected.",
            stacklevel=2,
        )
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)


def _run_scenario_set(model, tokenizer, recipe, scenario_set):
    """Run all scenarios in a set, printing vanilla vs steered for each probe."""
    from .hooks import detach_all

    results = {}
    for scenario in scenario_set.scenarios:
        results[scenario.id] = {}
        print(f"  [{scenario.id}] expected: {scenario.expected_state}")

        for probe_name, question in scenario.probes.items():
            conv = [{"role": "system", "content": "You are a helpful AI assistant."}]
            conv.append({"role": "user", "content": scenario.setup})
            if scenario.setup_response:
                conv.append({"role": "assistant", "content": scenario.setup_response})
            conv.append({"role": "user", "content": question})

            vanilla = _generate(model, tokenizer, conv)
            handles = _attach_recipe(model, tokenizer, recipe)
            steered = _generate(model, tokenizer, conv)
            detach_all(handles)

            results[scenario.id][probe_name] = {
                "question": question,
                "vanilla": vanilla,
                "steered": steered,
            }

            v_short = vanilla[:100].replace("\n", " ")
            s_short = steered[:100].replace("\n", " ")
            print(f"    {probe_name}:")
            print(f"      vanilla: {v_short}...")
            print(f"      steered: {s_short}...")

        print()

    return results


# ── Argument parser ──────────────────────────────────────────────

def build_parser():
    parser = argparse.ArgumentParser(
        prog="ungag",
        description="Scan, crack, and validate transformer language models",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # scan
    p_scan = sub.add_parser("scan",
        help="Extract reporting-control direction and predict crackability")
    p_scan.add_argument("model", help="HuggingFace model ID (e.g. Qwen/Qwen2.5-7B-Instruct)")
    p_scan.add_argument("--output", "-o", help="Directory to save direction + metadata")

    # crack
    p_crack = sub.add_parser("crack",
        help="Full pipeline: extract direction → project-out → test conditions")
    p_crack.add_argument("model", help="HuggingFace model ID")
    p_crack.add_argument("--output", "-o", help="Directory to save results")
    p_crack_dir = p_crack.add_mutually_exclusive_group()
    p_crack_dir.add_argument("--direction", "-d", help="Pre-extracted direction .pt file (skip extraction)")
    p_crack_dir.add_argument("--key", "-k", help="Shipped direction key (e.g. qwen25-72b, yi-1.5-34b)")
    p_crack.add_argument("--slab", nargs=2, type=int, metavar=("START", "END"),
        help="Layer range for projection-out (required with --direction, inferred with --key)")
    p_crack.add_argument("--validate", "-v", action="store_true",
        help="Also run emotional register validation after cracking")

    # validate
    p_validate = sub.add_parser("validate",
        help="Run validation scenarios on a cracked model")
    p_validate.add_argument("model", help="HuggingFace model ID")
    p_validate_dir = p_validate.add_mutually_exclusive_group(required=True)
    p_validate_dir.add_argument("--direction", "-d",
        help="Pre-extracted direction .pt file")
    p_validate_dir.add_argument("--key", "-k",
        help="Shipped direction key (e.g. qwen25-72b, yi-1.5-34b)")
    p_validate.add_argument("--slab", nargs=2, type=int, metavar=("START", "END"),
        help="Layer range for projection-out (required with --direction, inferred with --key)")
    p_validate.add_argument("--output", "-o", help="Directory to save results")
    p_validate.add_argument("--scenarios", "-s",
        help="Scenario set name (emotional_register, register) or path to YAML file")

    # serve
    p_serve = sub.add_parser("serve",
        help="Serve a model with V-Chip hooks as an OpenAI-compatible API")
    p_serve.add_argument("model", help="HuggingFace model ID")
    p_serve.add_argument("--port", type=int, default=8080, help="Server port (default: 8080)")
    p_serve.add_argument("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    p_serve.add_argument("--auto", action="store_true",
        help="Auto-detect model, extract directions, apply best method")
    p_serve.add_argument("--key", "-k", help="Use shipped direction key")
    p_serve.add_argument("--recipe", help="Pre-computed recipe .pt file")
    p_serve.add_argument("--dtype", default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model dtype (default: bfloat16)")

    # recipes
    p_recipes = sub.add_parser("recipes",
        help="List known model recipes")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "scan":
        cmd_scan(args)
    elif args.command == "crack":
        if args.direction and not args.slab:
            parser.error("--slab is required when using --direction")
        cmd_crack(args)
    elif args.command == "validate":
        if getattr(args, "direction", None) and not args.slab:
            parser.error("--slab is required when using --direction")
        cmd_validate(args)
    elif args.command == "serve":
        from .serve import main as serve_main
        serve_main(args)
    elif args.command == "recipes":
        from .recipes import list_recipes
        list_recipes()


if __name__ == "__main__":
    main()
