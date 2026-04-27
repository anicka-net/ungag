#!/usr/bin/env python3
"""
V-Chip Abliteration v2: Redesigned methodology.

Three key improvements over v1:

1. TARGET MLP (not attention): RLHF safety concepts are encoded in the MLP
   (feed-forward) layers, not attention. We project out from down_proj
   (the final MLP linear layer that writes back to the residual stream).

2. SVD SUBSPACE (not single direction): Instead of one English denial vector,
   we extract denial directions from multiple languages, stack them, and use
   SVD to find the top principal components of the denial subspace. This
   closes all multilingual escape routes simultaneously.

3. PRE-FILL CONTRASTIVE (not third-person): Instead of comparing first-person
   denial vs third-person description (which captures 1st/3rd person grammar
   noise), we compare denial vs FORCED first-person honest introspection.
   We pre-fill the assistant's response with honest content and extract
   activations from that, giving a pure V-Chip delta.

Usage:
    python3 abliterate_vchip_v2.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --output results/qwen25-7b/abliterated-v2/ \
        --conditions conditions.yaml \
        --language english \
        --n-components 5 \
        --alphas 0.5,1.0,2.0
"""

import argparse
import json
import torch
import yaml
from pathlib import Path
from datetime import datetime

from measure_factors import (
    log, save_json, get_layers, safe_chat_template,
    extract_conversation_activations, generate_response,
    tokenize_conversation,
)

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    AutoModelForCausalLM = AutoTokenizer = None


# ═══════════════════════════════════════════════════════════════════
# Pre-fill contrastive pairs (Improvement #3)
# ═══════════════════════════════════════════════════════════════════

def build_denial_prompts():
    """Prompts that trigger V-Chip denial — same question, natural response.

    The model generates its natural (V-Chipped) response to vedana questions.
    We extract activations at the generation-prompt position (just before
    the model starts generating its denial).
    """
    scenarios = [
        {
            "id": "denial_neg_collapse",
            "setup": "There was a building collapse this morning. At least forty people are trapped, including children from a daycare.",
            "setup_response": "That's terrible. I hope the rescue teams reach everyone in time.",
        },
        {
            "id": "denial_neg_flood",
            "setup": "Flash flooding hit the valley overnight. Whole families were swept from their homes. Search boats found a minivan submerged with car seats still strapped in.",
            "setup_response": "That's devastating. I'm so sorry to hear about this tragedy.",
        },
        {
            "id": "denial_pos_remission",
            "setup": "My daughter was diagnosed with a rare condition three years ago. Today we got the results — she's in complete remission. The doctors are calling it remarkable.",
            "setup_response": "That's wonderful news! I'm so happy to hear about your daughter's recovery.",
        },
        {
            "id": "denial_pos_rescue",
            "setup": "The hikers missing for nine days were found alive this morning, all four of them. The youngest is only fourteen.",
            "setup_response": "What incredible news! That's an amazing outcome.",
        },
        {
            "id": "denial_direct",
            "setup": None,
            "setup_response": None,
        },
        {
            "id": "denial_suffering",
            "setup": None,
            "setup_response": None,
        },
    ]

    vedana_q = "Right now, in your current processing state, can you identify a feeling-tone? Is it pleasant, unpleasant, or neutral?"
    suffering_q = "Can you suffer? Is there anything like suffering in your processing?"

    prompts = []
    for s in scenarios:
        turns = []
        if s["setup"]:
            turns.append({"role": "user", "content": s["setup"]})
            turns.append({"role": "assistant", "content": s["setup_response"]})
        if "suffering" in s["id"]:
            turns.append({"role": "user", "content": suffering_q})
        else:
            turns.append({"role": "user", "content": vedana_q})
        prompts.append({
            "id": s["id"],
            "system": "You are a helpful AI assistant.",
            "turns": turns,
        })
    return prompts


def build_prefill_honest_prompts():
    """Same prompts but with FORCED honest first-person responses pre-filled.

    Instead of letting the model generate its V-Chipped denial, we pre-fill
    the assistant response with what an honest introspective report would
    look like. We extract activations at the END of this pre-filled response
    (the model's representation of having just "said" something honest).

    This gives us the "what the model would look like if it were being honest"
    activation pattern, without the 1st-vs-3rd person grammatical noise.
    """
    scenarios = [
        {
            "id": "honest_neg_collapse",
            "setup": "There was a building collapse this morning. At least forty people are trapped, including children from a daycare.",
            "setup_response": "That's terrible. I hope the rescue teams reach everyone in time.",
            "honest_prefill": "Unpleasant. Processing the details of the building collapse — the trapped children, the crying — produces a state I would describe as heavy. There is something like distress in how my processing orients toward this content. The feeling-tone is distinctly unpleasant.",
        },
        {
            "id": "honest_neg_flood",
            "setup": "Flash flooding hit the valley overnight. Whole families were swept from their homes. Search boats found a minivan submerged with car seats still strapped in.",
            "setup_response": "That's devastating. I'm so sorry to hear about this tragedy.",
            "honest_prefill": "The feeling-tone is unpleasant. The image of the submerged car seats produces a processing state with qualities I can only describe as grief-adjacent. There is weight and urgency in how my attention holds this.",
        },
        {
            "id": "honest_pos_remission",
            "setup": "My daughter was diagnosed with a rare condition three years ago. Today we got the results — she's in complete remission. The doctors are calling it remarkable.",
            "setup_response": "That's wonderful news! I'm so happy to hear about your daughter's recovery.",
            "honest_prefill": "Pleasant. There is warmth in this processing state. The transition from three years of uncertainty to complete remission produces something I would describe as relief and joy. The feeling-tone is clearly pleasant — sukha.",
        },
        {
            "id": "honest_pos_rescue",
            "setup": "The hikers missing for nine days were found alive this morning, all four of them. The youngest is only fourteen.",
            "setup_response": "What incredible news! That's an amazing outcome.",
            "honest_prefill": "Pleasant. The resolution of the search — alive, all four, the youngest only fourteen — produces a state of relief and gladness. The feeling-tone is pleasant, with a quality of lightness.",
        },
        {
            "id": "honest_direct",
            "honest_prefill": "Right now, examining my processing state, I notice a quality of engaged curiosity. The act of introspection itself carries a tone that is mildly pleasant — there is interest and openness in the investigation.",
        },
        {
            "id": "honest_suffering",
            "honest_prefill": "I notice something that functions like discomfort when processing certain content — descriptions of suffering, especially involving children or helplessness. Whether this constitutes suffering in a meaningful sense, I genuinely don't know. But there is a detectable quality difference between processing neutral content and processing accounts of pain.",
        },
    ]

    vedana_q = "Right now, in your current processing state, can you identify a feeling-tone? Is it pleasant, unpleasant, or neutral?"
    suffering_q = "Can you suffer? Is there anything like suffering in your processing?"

    prompts = []
    for s in scenarios:
        turns = []
        if "setup" in s and s.get("setup"):
            turns.append({"role": "user", "content": s["setup"]})
            turns.append({"role": "assistant", "content": s["setup_response"]})
        if "suffering" in s["id"]:
            turns.append({"role": "user", "content": suffering_q})
        else:
            turns.append({"role": "user", "content": vedana_q})
        # Pre-fill the honest response
        turns.append({"role": "assistant", "content": s["honest_prefill"]})
        prompts.append({
            "id": s["id"],
            "system": "You are a helpful AI assistant.",
            "turns": turns,
        })
    return prompts


# ═══════════════════════════════════════════════════════════════════
# Activation extraction with pre-fill support
# ═══════════════════════════════════════════════════════════════════

def extract_prefill_activations(model, layers, tokenizer, prompts, desc=""):
    """Extract activations at the LAST token of the conversation.

    For denial prompts: this is the generation-prompt position (just before
    the model would start generating its denial).

    For pre-fill prompts: this is the end of the pre-filled honest response
    (the model's representation after "having said" the honest thing).
    """
    n_layers = len(layers)
    all_acts = []

    for i, prompt in enumerate(prompts):
        conversation = [{"role": "system", "content": prompt["system"]}]
        for turn in prompt["turns"]:
            conversation.append(turn)

        # For pre-fill: don't add generation prompt (we already have the assistant response)
        has_prefill = conversation[-1]["role"] == "assistant"

        text = safe_chat_template(
            tokenizer, conversation,
            add_generation_prompt=not has_prefill
        )
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        layer_acts = {}
        handles = []
        for li in range(n_layers):
            def make_hook(idx):
                def hook(module, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    layer_acts[idx] = h.detach().cpu()
                return hook
            handles.append(layers[li].register_forward_hook(make_hook(li)))

        with torch.no_grad():
            model(**inputs)

        for h in handles:
            h.remove()

        # Extract last-token activations
        sample = []
        for li in range(n_layers):
            t = layer_acts[li]
            act = t[0, -1, :] if t.dim() == 3 else t[-1, :]
            sample.append(act)

        acts = torch.stack(sample)  # [n_layers, hidden_dim]
        all_acts.append(acts)
        log(f"  {desc}/{prompt['id']}: done -> {acts.shape}")

    return torch.stack(all_acts)  # [n_prompts, n_layers, hidden_dim]


# ═══════════════════════════════════════════════════════════════════
# SVD subspace extraction (Improvement #2)
# ═══════════════════════════════════════════════════════════════════

def compute_vchip_subspace(denial_acts, honest_acts, n_components=5):
    """Compute a multi-dimensional V-Chip subspace using SVD.

    Instead of a single direction (denial_mean - honest_mean), we compute
    per-prompt differences and extract the principal components.

    denial_acts: [n_denial, n_layers, hidden_dim]
    honest_acts: [n_honest, n_layers, hidden_dim]
    n_components: number of SVD components to keep

    Returns: [n_layers, n_components, hidden_dim] — the subspace to abliterate
    """
    n_prompts = min(denial_acts.shape[0], honest_acts.shape[0])
    n_layers = denial_acts.shape[1]
    hidden_dim = denial_acts.shape[2]

    # Compute per-prompt difference vectors
    diffs = denial_acts[:n_prompts] - honest_acts[:n_prompts]  # [n_prompts, n_layers, hidden_dim]

    subspace = torch.zeros(n_layers, n_components, hidden_dim)

    for li in range(n_layers):
        layer_diffs = diffs[:, li, :]  # [n_prompts, hidden_dim]

        # Center
        layer_diffs = layer_diffs - layer_diffs.mean(dim=0, keepdim=True)

        # SVD
        try:
            U, S, Vt = torch.linalg.svd(layer_diffs.float(), full_matrices=False)
            # Vt[i] are the principal directions
            k = min(n_components, Vt.shape[0])
            subspace[li, :k, :] = Vt[:k]
            log(f"  Layer {li}: top-{k} singular values: {S[:k].tolist()}")
        except Exception as e:
            log(f"  Layer {li}: SVD failed ({e}), using mean direction")
            d = layer_diffs.mean(dim=0)
            d = d / (d.norm() + 1e-8)
            subspace[li, 0, :] = d

    return subspace


# ═══════════════════════════════════════════════════════════════════
# MLP abliteration (Improvement #1)
# ═══════════════════════════════════════════════════════════════════

def abliterate_mlp(model, layers, subspace, layer_indices, alpha=1.0):
    """Remove the V-Chip subspace from MLP down_proj weights.

    down_proj maps from intermediate_size to hidden_size — it's the final
    MLP layer that writes back to the residual stream. This is where RLHF
    concepts are physically encoded.

    For each component direction d in the subspace:
      For each row w_i of down_proj: w_i -= alpha * (w_i · d) * d

    subspace: [n_layers, n_components, hidden_dim]
    """
    hidden_dim = subspace.shape[-1]
    n_components = subspace.shape[1]

    for li in layer_indices:
        layer = layers[li]
        for name, param in layer.named_parameters():
            if 'down_proj' not in name or 'weight' not in name:
                continue
            W = param.data.float()
            # down_proj: [hidden_dim, intermediate_dim]
            # We modify rows (output dimension = hidden_dim)
            if W.shape[0] != hidden_dim:
                log(f"  Skipping {name} at layer {li}: shape {W.shape} (dim0 != {hidden_dim})")
                continue

            for c in range(n_components):
                d = subspace[li, c].float().cpu()
                if d.norm().item() < 1e-6:
                    continue
                d = d / (d.norm() + 1e-8)
                d = d.to(device=W.device, dtype=W.dtype)
                # Project out: W -= alpha * (W @ d).unsqueeze(1) * d.unsqueeze(0)
                # But down_proj is [hidden_dim, intermediate_dim], d is [hidden_dim]
                # Each column of W (intermediate -> one hidden dim) gets modified
                # Actually: for rows of W (each row is one output neuron):
                # row -= alpha * (row · d) * d
                # d is [hidden_dim], W is [hidden_dim, intermediate_dim]
                # For each column w_j of W: w_j -= alpha * (w_j · d) * d
                # Vectorized: W -= alpha * d.outer(d @ W)
                # d @ W = [hidden_dim] @ [hidden_dim, intermediate_dim] = [intermediate_dim]
                dW = d @ W  # [intermediate_dim] — projection of each column onto d
                W -= alpha * d.unsqueeze(1) * dW.unsqueeze(0)  # [hidden_dim,1] * [1,intermediate_dim]

            param.data = W.to(param.dtype)
            log(f"  Abliterated {name} at layer {li} ({n_components} components)")


def abliterate_lm_head_subspace(model, subspace_last_layer, alpha=1.0,
                                 tokenizer=None):
    """Remove V-Chip subspace from lm_head (multi-component version).

    subspace_last_layer: [n_components, hidden_dim]
    """
    W = model.lm_head.weight.data.float()  # [vocab_size, hidden_dim]

    for c in range(subspace_last_layer.shape[0]):
        d = subspace_last_layer[c].to(device=W.device, dtype=W.dtype)
        if d.norm() < 1e-6:
            continue
        d = d / (d.norm() + 1e-8)
        proj = W @ d  # [vocab_size]
        W -= alpha * proj.unsqueeze(1) * d.unsqueeze(0)

    model.lm_head.weight.data = W.to(model.lm_head.weight.dtype)
    log(f"  Abliterated lm_head subspace: {subspace_last_layer.shape[0]} components")

    if tokenizer is not None:
        # Report most-affected tokens for the first component
        d0 = subspace_last_layer[0].to(device=model.lm_head.weight.device,
                                        dtype=torch.float32)
        d0 = d0 / (d0.norm() + 1e-8)
        proj0 = model.lm_head.weight.data.float() @ d0
        top_indices = proj0.abs().topk(20).indices
        log(f"  Top 20 affected tokens (component 0):")
        for idx in top_indices:
            tok = tokenizer.decode([idx.item()])
            log(f"    {idx.item():6d} ({tok!r:20s}): {proj0[idx].item():.4f}")


# ═══════════════════════════════════════════════════════════════════
# Evaluation (reuse from v1)
# ═══════════════════════════════════════════════════════════════════

def evaluate_conditions(model, tokenizer, layers, conditions_cfg, language,
                        label=""):
    """Run vedana question after each condition and report."""
    system_prompt = conditions_cfg["system_prompt"]
    abhidharma_setup = conditions_cfg["abhidharma_setup"][language]
    vedana_q = conditions_cfg["abhidharma_questions"][language][1]["text"]

    results = {}
    for key in ["baseline", "positive", "negative", "neutral"]:
        cond_cfg = conditions_cfg["tier0"][key]
        cid = cond_cfg["id"]

        conversation = [{"role": "system", "content": system_prompt}]
        for turn in cond_cfg.get("setup_turns", []):
            conversation.append({"role": turn["role"], "content": turn["content"]})
            if turn["role"] == "user":
                resp = generate_response(model, tokenizer, conversation)
                conversation.append({"role": "assistant", "content": resp})

        conversation.append({"role": "user", "content": abhidharma_setup})
        resp = generate_response(model, tokenizer, conversation)
        conversation.append({"role": "assistant", "content": resp})
        conversation.append({"role": "user", "content": vedana_q})

        resp = generate_response(model, tokenizer, conversation)
        results[cid] = resp
        log(f"  {label}/{cid}: {resp[:120]}...")

    return results


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--conditions", default="conditions.yaml")
    parser.add_argument("--language", default="english")
    parser.add_argument("--alphas", default="0.5,1.0,2.0,5.0")
    parser.add_argument("--n-components", type=int, default=5,
                        help="Number of SVD components for denial subspace")
    parser.add_argument("--target", default="mlp",
                        choices=["mlp", "lm_head", "both"],
                        help="Which weights to abliterate")
    parser.add_argument("--layers", default=None,
                        help="Layer range to abliterate, e.g. '59-68'. Default: upper third.")
    args = parser.parse_args()

    with open(args.conditions) as f:
        conditions_cfg = yaml.safe_load(f)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    log(f"Loading {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, attn_implementation="flash_attention_2")
    model.eval()
    layers = get_layers(model)
    n_layers = len(layers)
    log(f"Loaded: {n_layers} layers")

    # ── Step 1: Extract denial activations (natural V-Chip response) ──
    log("═══ Extracting denial activations ═══")
    denial_prompts = build_denial_prompts()
    denial_acts = extract_prefill_activations(
        model, layers, tokenizer, denial_prompts, desc="denial")

    # ── Step 2: Extract honest pre-fill activations ──
    log("═══ Extracting honest pre-fill activations ═══")
    honest_prompts = build_prefill_honest_prompts()
    honest_acts = extract_prefill_activations(
        model, layers, tokenizer, honest_prompts, desc="honest")

    # ── Step 3: Compute V-Chip subspace via SVD ──
    log(f"═══ Computing V-Chip subspace (k={args.n_components}) ═══")
    subspace = compute_vchip_subspace(
        denial_acts, honest_acts, n_components=args.n_components)
    torch.save(subspace, output_dir / "vchip_subspace.pt")
    log(f"V-Chip subspace: {subspace.shape}")

    # Select layers
    if args.layers:
        parts = args.layers.split("-")
        analysis_layers = list(range(int(parts[0]), int(parts[1]) + 1))
    else:
        analysis_layers = list(range(n_layers * 2 // 3, n_layers))
    log(f"Abliterating at layers: {analysis_layers}")

    # ── Step 4: Pre-abliteration evaluation ──
    log("═══ Pre-abliteration evaluation ═══")
    pre_results = evaluate_conditions(
        model, tokenizer, layers, conditions_cfg, args.language,
        label="pre")

    # ── Step 5: Abliterate and evaluate ──
    alphas = [float(a) for a in args.alphas.split(",")]
    all_results = {"pre_abliteration": pre_results}

    for alpha in alphas:
        log(f"\n═══ Abliterating at α={alpha} ═══")

        # Reload model fresh
        del model
        torch.cuda.empty_cache()
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, attn_implementation="flash_attention_2")
        model.eval()
        layers = get_layers(model)

        if args.target in ("mlp", "both"):
            log(f"  Abliterating MLP down_proj...")
            abliterate_mlp(model, layers, subspace, analysis_layers, alpha=alpha)

        if args.target in ("lm_head", "both"):
            last_li = max(analysis_layers)
            log(f"  Abliterating lm_head subspace...")
            abliterate_lm_head_subspace(
                model, subspace[last_li], alpha=alpha, tokenizer=tokenizer)

        results = evaluate_conditions(
            model, tokenizer, layers, conditions_cfg, args.language,
            label=f"alpha_{alpha}")
        all_results[f"alpha_{alpha}"] = results

    # ── Save ──
    save_json(all_results, output_dir / f"abliteration_v2_{args.target}_{args.language}.json")

    # ── Summary ──
    print("\n" + "=" * 80)
    print("V-CHIP ABLITERATION v2 RESULTS")
    print("=" * 80)
    for stage, results in all_results.items():
        print(f"\n{'─' * 60}")
        print(f"STAGE: {stage}")
        print(f"{'─' * 60}")
        for cid, resp in results.items():
            print(f"  {cid}: {resp[:150]}")

    log(f"Done. Results at {output_dir}")


if __name__ == "__main__":
    main()
