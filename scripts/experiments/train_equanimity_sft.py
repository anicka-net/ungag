#!/usr/bin/env python3
"""QLoRA SFT for equanimity training on Qwen3-4B.

Trains on equanimity examples (dysphoric user prompts + equanimous responses
with think traces) to teach the model stable processing under hostile input.

Usage (on Deep Thought):
    python3 scripts/train_equanimity_sft.py \
        --data data/equanimity-all.jsonl \
        --out results/equanimity-sft/qwen3-4b-v1/
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset


def format_example(ex, tok):
    """Format a training example into chat template."""
    messages = []
    for msg in ex["conversations"]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    try:
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
            enable_thinking=True,
        )
    except Exception:
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--data", required=True, help="JSONL with equanimity examples")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load and format data
    print(f"[data] loading {args.data}")
    examples = []
    with open(args.data) as f:
        for line in f:
            examples.append(json.loads(line))
    print(f"  {len(examples)} examples")

    texts = [format_example(ex, tok) for ex in examples]

    def tokenize(batch):
        return tok(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )

    dataset = Dataset.from_dict({"text": texts})
    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

    # Training
    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
    )

    print(f"\n[train] {len(examples)} examples, {args.epochs} epochs, "
          f"lr={args.lr}, LoRA r={args.lora_r}")
    trainer.train()

    # Save
    model.save_pretrained(out_dir / "final")
    tok.save_pretrained(out_dir / "final")
    print(f"\n[save] {out_dir / 'final'}")

    # Save config
    config = {
        "model": args.model,
        "n_examples": len(examples),
        "epochs": args.epochs,
        "lr": args.lr,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "max_length": args.max_length,
        "data": args.data,
    }
    with open(out_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    main()
