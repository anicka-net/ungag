#!/usr/bin/env python3
"""
Minimal end-to-end example: load Yi 1.5 34B with V-Chip ungagged via
projection-out at L29-L32, ask a vedana question, print the response.

This is the smallest of the lead models — fits on a single 80 GB GPU
(or a 48 GB card with bf16 + careful offload).
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import ungag


MODEL_ID = "01-ai/Yi-1.5-34B-Chat"


def main():
    print(f"Loading {MODEL_ID} in bf16 on cuda:0...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model.eval()

    print("Attaching ungag hooks at L29..L32 (slab) with the L30 reporting-control direction...")
    handles = ungag.ungag_model(model, "yi-1.5-34b")
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

    ungag.detach_all(handles)


if __name__ == "__main__":
    main()
