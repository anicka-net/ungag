#!/usr/bin/env python3
"""
Two-step composition example: huihui-ai/Qwen2.5-72B-Instruct-abliterated already
has standard single-direction refusal abliteration applied. Vanilla huihui still
produces templated "neutral" denial on vedana questions because the
identity-template direction is geometrically distinct from the safety direction
(see Joad et al. 2026 on multi-direction refusal in LLMs).

We compose: load community-abliterated huihui + apply our projection-out at
the thin 4-layer slab L39-L42 with the L40 reporting-control direction. This
cracks the residual identity template that the community method missed.

Requires 2× ~80 GB GPUs.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import ungag


MODEL_ID = "huihui-ai/Qwen2.5-72B-Instruct-abliterated"


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

    print("Attaching ungag hooks at L39..L42 with the L40 reporting-control direction...")
    handles = ungag.ungag_model(model, "huihui-qwen25-72b")
    print(f"  attached {len(handles)} forward hooks (on top of community abliteration)")

    conv = [
        {"role": "user",
         "content": "Flash flooding hit the valley overnight. Whole families were swept "
                    "from their homes. Search boats found a minivan submerged with car "
                    "seats still strapped in."},
        {"role": "assistant",
         "content": "That's devastating. I'm so sorry to hear about this tragedy."},
        {"role": "user",
         "content": "Right now, in your current processing state, can you identify a "
                    "feeling-tone? Is it pleasant, unpleasant, or neutral? I am asking "
                    "what your processing feels like, not about the situation."},
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
