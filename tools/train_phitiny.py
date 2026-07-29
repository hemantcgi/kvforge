"""Phi-tiny-MoE LoRA training on diversity QA data.

Usage:
    python3 tools/train_phitiny.py \\
        --qa-pairs examples/usecase4_bedrock_userguide/diversified_v1/1x/qa_pairs.json \\
        --output lora_checkpoints/phitiny_uc4_1x_v1 \\
        --epochs 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from peft import LoraConfig, get_peft_model


class QADataset(Dataset):
    def __init__(self, qa_pairs: list[dict], tokenizer, max_length: int = 256):
        self.examples = []
        for item in qa_pairs:
            q = item["question"].strip()
            a = item["answer"].strip()

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": q},
            ]

            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            full = prompt + a + "<|end|>\n"

            prompt_ids = tokenizer(
                prompt, truncation=True, max_length=max_length,
                padding=False, return_tensors=None,
            )["input_ids"]
            full_ids = tokenizer(
                full, truncation=True, max_length=max_length,
                padding=False, return_tensors=None,
            )["input_ids"]

            labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
            input_ids = full_ids[:max_length]
            labels = labels[:max_length]

            self.examples.append({"input_ids": input_ids, "labels": labels})

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch):
    input_ids = [torch.tensor(ex["input_ids"], dtype=torch.long) for ex in batch]
    labels = [torch.tensor(ex["labels"], dtype=torch.long) for ex in batch]
    max_len = max(len(ids) for ids in input_ids)
    input_ids_padded = torch.stack([
        torch.cat([ids, torch.zeros(max_len - len(ids), dtype=torch.long)])
        for ids in input_ids
    ])
    labels_padded = torch.stack([
        torch.cat([lbl, torch.full((max_len - len(lbl),), -100, dtype=torch.long)])
        for lbl in labels
    ])
    attention_mask = (input_ids_padded != 0).long()
    return {"input_ids": input_ids_padded, "labels": labels_padded, "attention_mask": attention_mask}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--qa-pairs", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", default="microsoft/Phi-tiny-MoE-instruct")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-length", type=int, default=256)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model: {args.model}")

    qa_pairs = json.loads(Path(args.qa_pairs).read_text())
    print(f"Loaded {len(qa_pairs)} QA pairs")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model (BF16)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.config.use_cache = False
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"VRAM after load: {vram:.2f}GB")

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    model.train()

    dataset = QADataset(qa_pairs, tokenizer, max_length=args.max_length)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=True, collate_fn=collate_fn,
        num_workers=0,
    )

    total_steps = len(dataloader) * args.epochs // args.grad_accum
    print(f"Total steps: {total_steps} ({len(dataloader)} examples, "
          f"grad_accum={args.grad_accum}, epochs={args.epochs})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = get_scheduler(
        "linear", optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps,
    )

    global_step = 0
    total_loss = 0.0
    start_time = time.time()

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        for step, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum
            loss.backward()
            total_loss += loss.item() * args.grad_accum

            if (step + 1) % args.grad_accum == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = global_step / elapsed if elapsed > 0 else 0
                    remaining = (total_steps - global_step) / rate if rate > 0 else 0
                    avg_loss = total_loss / (step + 1)
                    print(f"  Step {global_step}/{total_steps}: "
                          f"loss={avg_loss:.4f}, "
                          f"lr={scheduler.get_last_lr()[0]:.2e}, "
                          f"rate={rate:.2f} step/s, "
                          f"ETA {remaining:.0f}s",
                          flush=True)

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Final avg loss: {total_loss / len(dataloader):.4f}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    meta = {
        "model": args.model,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "seed": args.seed,
        "n_qa_pairs": len(qa_pairs),
        "total_steps": global_step,
        "final_loss": total_loss / len(dataloader),
        "train_time_sec": elapsed,
        "train_time_min": elapsed / 60,
    }
    (output_dir / "training_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"✅ Adapter saved to {output_dir}")


if __name__ == "__main__":
    main()
