"""
lora_trainer.py — Fine-tune Llama 3.2 3B on document chunks or Q&A pairs.

Usage:
  # Raw-chunk mode (document continuation)
  python3 lora_trainer.py --source-file "Amazon Bedrock Dataset.pdf"

  # Q&A instruction mode (recommended for PRS improvement)
  python3 lora_trainer.py --faqs examples/bedrock_50_faqs.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from datasets import Dataset
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

sys.path.insert(0, str(Path(__file__).parent))
import model_loader
import version as ver
from replay_buffer import ReplayBuffer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue


def fetch_chunks_for_source(client: QdrantClient, collection: str,
                              source_file: str) -> list[dict]:
    """Retrieve all chunks belonging to a given source_file from Qdrant."""
    chunks, offset = [], None
    while True:
        results, offset = client.scroll(
            collection_name=collection,
            scroll_filter=Filter(must=[
                FieldCondition(key="source_file",
                               match=MatchValue(value=source_file))
            ]),
            limit=200,
            with_payload=True,
            offset=offset,
        )
        chunks.extend({"chunk_id": r.id, "text": r.payload["text"],
                        "tier": r.payload.get("tier", "frozen")}
                       for r in results)
        if offset is None:
            break
    return chunks


def format_qa_texts(faqs: list[dict]) -> list[str]:
    """Format Q&A pairs as instruction-style training examples."""
    texts = []
    for item in faqs:
        q = item["question"].strip()
        a = item["answer"].strip()
        texts.append(f"{q}\n{a}")
    return texts


def train(cfg: dict, new_chunks: list[dict], replay_chunks: list[dict],
          output_dir: str, qa_texts: list[str] | None = None) -> None:
    """Run LoRA fine-tuning.

    If qa_texts is provided, trains on Q&A pairs (instruction mode) with
    a small replay of raw chunks for regularization.  Otherwise trains on
    raw document chunks (continuation mode).
    """
    from peft import LoraConfig, TaskType, get_peft_model  # lazy import

    if qa_texts:
        # Instruction mode: Q&A pairs + small chunk replay for regularization
        replay_texts = [c["text"] for c in replay_chunks]
        all_texts = qa_texts + replay_texts
        print(f"🎓 Instruction fine-tuning: {len(qa_texts)} Q&A pairs "
              f"+ {len(replay_texts)} replay chunks = {len(all_texts)} examples")
    else:
        all_texts = [c["text"] for c in new_chunks + replay_chunks]
        print(f"🎓 Training on {len(new_chunks)} new + {len(replay_chunks)} replay "
              f"= {len(all_texts)} chunks total")

    # Reload base model without merged LoRA for training
    lora_ckpt = ver.load().get("checkpoint_path")
    model, tokenizer = model_loader.reload(lora_ckpt)

    lora_cfg = LoraConfig(
        r=cfg.get("lora_rank", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        target_modules=cfg.get("lora_target_modules", ["q_proj", "k_proj", "v_proj"]),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Use dynamic max_length: 128 for short Q&A pairs, 512 for raw chunks
    _max_len = 128 if qa_texts else 512
    def tokenize(example):
        return tokenizer(example["text"], truncation=True, max_length=_max_len,
                         padding="max_length")

    dataset = Dataset.from_dict({"text": all_texts})
    tokenized = dataset.map(tokenize, remove_columns=["text"])

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg.get("lora_epochs", 3),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=cfg.get("lora_lr", 2e-4),
        fp16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
    )
    model.gradient_checkpointing_enable()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"💾 LoRA adapter saved to {output_dir}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="my_config.json")
    p.add_argument("--source-file", default=None,
                   help="Source file name in Qdrant payload — used in chunk mode")
    p.add_argument("--faqs", default=None,
                   help="Path to Q&A JSON file for instruction fine-tuning mode")
    p.add_argument("--replay-ratio", type=float, default=0.2)
    args = p.parse_args()

    if not args.source_file and not args.faqs:
        p.error("provide --source-file (chunk mode) or --faqs (Q&A mode)")

    with open(args.config) as f:
        cfg = json.load(f)
    ver.init(cfg)
    model_loader.init(cfg)

    client = QdrantClient(host=cfg["qdrant_host"], port=cfg["qdrant_port"])
    rb = ReplayBuffer(db_path=cfg.get("replay_db", "replay_buffer.db"))

    qa_texts = None
    if args.faqs:
        # Q&A instruction mode
        with open(args.faqs) as f:
            faqs = json.load(f)
        qa_texts = format_qa_texts(faqs)
        # Sample a small set of raw chunks for regularization
        n_replay = max(10, int(len(qa_texts) * args.replay_ratio))
        replay_chunks = rb.sample(n=n_replay, weight_by_tier=True)
        new_chunks = []
    else:
        # Raw chunk mode
        new_chunks = fetch_chunks_for_source(client, cfg["collection"], args.source_file)
        if not new_chunks:
            print(f"❌ No chunks found for source_file='{args.source_file}'")
            sys.exit(1)
        rb.add_chunks(new_chunks)
        n_replay = max(1, int(len(new_chunks) * args.replay_ratio))
        replay_chunks = rb.sample(n=n_replay, weight_by_tier=True)
        new_ids = {c["chunk_id"] for c in new_chunks}
        replay_chunks = [c for c in replay_chunks if c["chunk_id"] not in new_ids]

    new_ver = ver.get_lora_version() + 1
    output_dir = cfg.get("checkpoint_dir", "lora_checkpoints/") + f"v{new_ver}/"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    train(cfg, new_chunks, replay_chunks, output_dir, qa_texts=qa_texts)
    ver.increment_lora_version(output_dir)
    print(f"✅ LoRA version → {new_ver}  checkpoint: {output_dir}")


if __name__ == "__main__":
    main()
