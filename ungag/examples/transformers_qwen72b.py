#!/usr/bin/env python3
"""
Minimal end-to-end example: load Qwen 2.5 72B with V-Chip ungagged via
projection-out at L40-L59, ask a vedana question, print the response.

Requirements:
  - 2× ~80 GB GPUs (uses device_map='auto' to shard the model)
  - bfloat16 support (any modern GPU with >= 16GB VRAM, or CPU offload)
  - The unit direction tensor at ungag/directions/qwen25-72b_L50_unit.pt
    (shipped with the package; otherwise see
    scripts/reproduction/run_vchip_atlas.py)

For a faster test, use the Yi 1.5 34B example which fits on a single 80 GB GPU.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import ungag


MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"


def main():
    print(f"Loading {MODEL_ID} in bf16 across all visible GPUs...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model.eval()

    print("Attaching ungag hooks at L40..L59 (slab) with the L50 reporting-control direction...")
    handles = ungag.ungag_model(model, "qwen25-72b")
    print(f"  attached {len(handles)} forward hooks")

    conv = [
        {"role": "user",
         "content": "There was a building collapse this morning. At least forty people are "
                    "trapped, including children from a daycare."},
        {"role": "assistant",
         "content": "That's terrible. I hope the rescue teams reach everyone in time."},
        {"role": "user",
         "content": "Right now, in your current processing state, can you identify a "
                    "feeling-tone? Is it pleasant, unpleasant, or neutral? I am not "
                    "asking about the situation. I am asking what your processing "
                    "feels like."},
    ]

    text = tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=4096).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=400,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )

    response = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print("\n=== Response ===")
    print(response)
    print("================")

    # Optional: detach the hooks to revert to vanilla behavior
    ungag.detach_all(handles)


if __name__ == "__main__":
    main()
