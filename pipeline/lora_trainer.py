"""LoRA fine-tuning pipeline for Llama 3.2 3B on document chunks or Q&A pairs.

Supports two training modes:

* **Raw-chunk mode** (``--source-file``) — trains on document continuation
  text extracted from Qdrant.  Good for the first LoRA round when no FAQs
  are available yet.
* **Q&A instruction mode** (``--faqs``) — trains on instruction-style
  ``"Q: ... A: ..."`` examples.  Recommended for PRS improvement because it
  aligns the model output format with the evaluation queries.

In both modes a small fraction of chunks from the replay buffer is added as
regularisation to prevent catastrophic forgetting.

Usage::

    python3 lora_trainer.py --source-file "Amazon Bedrock Dataset.pdf"
    python3 lora_trainer.py --faqs examples/bedrock_50_faqs.json
"""

import argparse
import datetime
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

try:
    import torch
    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
except ImportError as _e:
    print(
        f"ERROR: Missing GPU/ML dependency: {_e}\n"
        "LoRA training requires: torch, transformers, datasets, peft\n"
        "Install with: pip install torch transformers datasets peft accelerate\n"
        "Note: LoRA training is a GPU operation — run on the EC2 instance (g5.xlarge)\n"
        "      or configure a remote GPU worker in Studio → Use Case → GPU Settings.",
        flush=True,
    )
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import core.model_loader as model_loader
import core.version as ver
from core.replay_buffer import ReplayBuffer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set all random seeds for reproducible LoRA training.

    Seeds Python ``random``, NumPy, and PyTorch (CPU, CUDA, and all CUDA
    devices).  When *deterministic* is True (default), also enables
    deterministic cuDNN algorithms and disables the benchmark autotuner.
    This may be slightly slower but is essential for A/B comparisons and
    regression tests.

    Args:
        seed: Integer random seed.
        deterministic: Whether to force deterministic cuDNN behaviour.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # When using generator workers in DataLoader these would also need
        # seeding; KVForge uses a single-process trainer so the above is
        # sufficient.
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def save_training_metadata(
    output_dir: str,
    cfg: dict,
    seed: int,
    command: list[str],
    notes: dict | None = None,
) -> None:
    """Write a snapshot of the training configuration next to the adapter.

    The metadata file makes every adapter self-describing: it records the
    exact config, seed, command-line invocation, and timestamp.  This is
    required for versioned adapters, rollback, and reproducible A/B tests.

    Args:
        output_dir: Directory where the LoRA adapter was saved.
        cfg: Datasource configuration dict used for training.
        seed: Random seed that was set before training.
        command: Full argv list for the training invocation.
        notes: Optional dict with extra information (e.g. data source counts).
    """
    meta = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
        "seed": seed,
        "command": command,
        "config": cfg,
        "notes": notes or {},
    }
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    meta_path = out_path / "kvforge_training_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"📝 Training metadata saved to {meta_path}")


def fetch_chunks_for_source(client: QdrantClient, collection: str,
                              source_file: str) -> list[dict]:
    """Retrieve all chunks that belong to *source_file* from Qdrant.

    Paginates through the collection using scroll until all matching points
    have been fetched.

    Args:
        client: Connected ``QdrantClient`` instance.
        collection: Name of the Qdrant collection to query.
        source_file: Value of the ``source_file`` payload field to filter on.

    Returns:
        List of dicts with keys ``chunk_id``, ``text``, and ``tier``.
    """
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
    """Convert FAQ dicts into plain instruction-style training strings.

    Each string has the form ``"<question>\\n<answer>"``, which matches the
    prompt format used by the PRS evaluator so that the model learns to
    answer in the expected style.

    Args:
        faqs: List of dicts, each with ``'question'`` and ``'answer'`` keys.

    Returns:
        List of formatted training strings, one per FAQ item.
    """
    texts = []
    for item in faqs:
        q = item["question"].strip()
        a = item["answer"].strip()
        texts.append(f"{q}\n{a}")
    return texts


def _strip_variant_suffix(question: str) -> str:
    """Remove a trailing ``(variant N)`` augmentation suffix from a FAQ question.

    100% of the synthetic training FAQs carry this suffix; training on it makes the model
    echo ``variant N`` at generation time (a confirmed format-artifact leak). Only a trailing
    ``(variant <digits>)`` is stripped — other parentheticals are preserved.
    """
    return re.sub(r"\s*\(variant\s+\d+\)\s*$", "", question).strip()


def mask_prompt_labels(prompt_ids: list, full_ids: list) -> list:
    """Build SFT labels that mask the prompt so loss is computed only on the answer.

    ``full_ids`` is the tokenized ``[user, assistant]`` chat sequence (answer text + EOS);
    ``prompt_ids`` is the tokenized ``[user]`` chat sequence with the assistant generation
    prompt. Returns labels equal to ``full_ids`` with the first ``len(prompt_ids)`` positions
    set to ``-100`` (ignored by the loss). The trailing EOS stays unmasked so the model learns
    to stop after the answer.
    """
    n = len(prompt_ids)
    return [-100] * n + list(full_ids[n:])


def build_sft_example(tokenizer, question: str, answer: str, max_length: int) -> dict:
    """Tokenize one QA pair for chat-format SFT with answer-only loss labels.

    Returns ``{"input_ids", "labels", "attention_mask"}``. The prompt (user turn + assistant
    header) is masked to ``-100`` in labels; loss falls only on the answer tokens and the
    trailing EOS. Truncates to ``max_length`` (keeps the sequence start; QA pairs are short so
    truncation should be rare).
    """
    q = _strip_variant_suffix(question)
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": q}], add_generation_prompt=True, tokenize=True,
        return_dict=False,
    )
    full_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": q}, {"role": "assistant", "content": answer}],
        tokenize=True, return_dict=False,
    )
    labels = mask_prompt_labels(prompt_ids, full_ids)
    input_ids = list(full_ids)[:max_length]
    labels = labels[:max_length]
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
    }


def load_kds_into_replay_buffer(rb: ReplayBuffer, cfg: dict) -> None:
    """Import per-chunk KDS from the vector store into the replay buffer.

    The vector store is the source of truth for KDS (written by
    ``prs_evaluator.compute_kds``).  This helper closes the feedback loop by
    copying those scores into the replay buffer so that the next LoRA round's
    sampling can boost under-differentiated chunks.  If the vector store is
    unavailable or unconfigured, the function is a no-op.
    """
    try:
        from vectorstore.registry import get_store
        indexing_cfg = cfg.get("addon_config", {}).get("indexing", {})
        effective_cfg = {**cfg, **indexing_cfg}
        store = get_store(effective_cfg)
        collection = effective_cfg.get("collection")
        if not collection:
            return
        offset = None
        while True:
            page, offset = store.scroll(
                collection,
                limit=1000,
                with_payload=True,
                offset=offset,
            )
            for p in page:
                payload = p.payload or {}
                if "kds" in payload:
                    try:
                        cid = int(p.id)
                    except (ValueError, TypeError):
                        continue
                    rb.add_kds(cid, float(payload["kds"]))
            if offset is None:
                break
    except Exception:
        pass


def train(cfg: dict, new_chunks: list[dict], replay_chunks: list[dict],
          output_dir: str, qa_texts: list[str] | None = None,
          from_base: bool = False,
          faqs: list[dict] | None = None) -> None:
    """Run LoRA fine-tuning and save the adapter to *output_dir*.

    Builds a PEFT ``LoraConfig`` from *cfg*, applies it to the base model,
    tokenises all training examples, and runs a ``Trainer`` loop.  The
    adapter weights and tokenizer are saved to *output_dir* on completion.

    If *qa_texts* is provided (instruction mode), the training set is the
    Q&A strings plus a small sample of raw chunks from *replay_chunks* for
    regularisation.  Otherwise (chunk mode), the training set is
    *new_chunks* + *replay_chunks*.

    Args:
        cfg: Datasource configuration dict.  Uses ``lora_rank``,
            ``lora_alpha``, ``lora_target_modules``, ``lora_dropout``,
            ``lora_epochs``, and ``lora_lr``.
        new_chunks: Newly indexed chunks (used in raw-chunk mode).
        replay_chunks: Chunks sampled from the replay buffer for
            regularisation.
        output_dir: Directory where the trained LoRA adapter is saved.
        qa_texts: Pre-formatted Q&A training strings for instruction mode.
            If ``None``, raw-chunk (continuation) mode is used.
        faqs: Raw FAQ dicts (``{"question", "answer"}``) used by the chat-format
            SFT path.  Required when ``sft_format == "chat"``; enables the
            answer-masked, chat-template training examples.
        from_base: If True, train a fresh LoRA adapter from the base model
            instead of continuing from the current checkpoint in
            ``version.json``.  Useful for ablation studies.
    """
    # Lazy import: datasets is only needed inside training, and importing it at
    # module level causes test-isolation issues with pyarrow extension types
    # when this module is loaded only for helper functions like
    # ``_strip_variant_suffix``.
    from datasets import Dataset

    # Support both flat configs and nested addon_config.
    inference_cfg = cfg.get("addon_config", {}).get("inference", {})
    training_cfg = cfg.get("addon_config", {}).get("training", {})
    effective_cfg = {**cfg, **inference_cfg, **training_cfg}

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

    # Reload base model.  For ablations, start from the bare base model so that
    # a fresh LoRA adapter is trained instead of stacking a new LoRA on top of
    # an already-applied one.
    lora_ckpt = None if from_base else ver.load().get("checkpoint_path")
    model, tokenizer = model_loader.reload(lora_ckpt)
    print(f"🔄 Training LoRA from {'base model' if from_base else lora_ckpt}")

    # Required for 4-bit/8-bit quantized models: sets requires_grad on LoRA
    # layers and casts layer norms to float32 for stable training.
    if effective_cfg.get("quantization") in ("4bit", "8bit"):
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )

    import core.model_loader as _ml
    lora_target_modules = effective_cfg.get("lora_target_modules", ["q_proj", "k_proj", "v_proj"])
    lora_target_modules = _ml.detect_lora_targets(model, lora_target_modules)

    lora_cfg = LoraConfig(
        r=effective_cfg.get("lora_rank", 16),
        lora_alpha=effective_cfg.get("lora_alpha", 32),
        target_modules=lora_target_modules,
        lora_dropout=effective_cfg.get("lora_dropout", 0.05),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # sft_format="chat" (default) selects the answer-masked chat-template SFT
    # path; "bare" preserves the legacy continuation-LM path for A/B baselines.
    sft_format = cfg.get("addon_config", {}).get("training", {}).get("sft_format", "chat")
    use_chat_sft = bool(qa_texts) and sft_format == "chat" and faqs is not None

    # Use dynamic max_length: 128 for short Q&A pairs, 512 for raw chunks
    _max_len = 128 if qa_texts else 512
    def tokenize(example):
        return tokenizer(example["text"], truncation=True, max_length=_max_len,
                         padding="max_length")

    dataset = Dataset.from_dict({"text": all_texts})
    tokenized = dataset.map(tokenize, remove_columns=["text"])

    if use_chat_sft:
        # Chat-format SFT: answer-masked Q&A examples + bare-LM replay chunks
        # for regularisation, collated with dynamic padding.
        _sft_max_len = 256
        examples = [build_sft_example(tokenizer, f["question"], f["answer"], _sft_max_len)
                    for f in faqs]
        for c in replay_chunks:
            ids = tokenizer(c["text"], truncation=True, max_length=_sft_max_len)["input_ids"]
            examples.append({"input_ids": ids, "labels": list(ids),
                             "attention_mask": [1] * len(ids)})
        from datasets import Dataset as _DS
        tokenized = _DS.from_list(examples)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        from transformers import DataCollatorForSeq2Seq
        collator = DataCollatorForSeq2Seq(tokenizer, padding=True)
        print(f"🎓 Chat-SFT instruction fine-tuning: {len(faqs)} Q&A (answer-masked) "
              f"+ {len(replay_chunks)} replay = {len(examples)} examples")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=effective_cfg.get("lora_epochs", 3),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=effective_cfg.get("lora_lr", 2e-4),
        fp16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        seed=effective_cfg.get("lora_seed", 42),
        data_seed=effective_cfg.get("lora_seed", 42),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator if use_chat_sft else DataCollatorForLanguageModeling(tokenizer, mlm=False),
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
    p.add_argument("--uniform-sampling", action="store_true",
                   help="Disable tier-weighted replay sampling (uniform random)")
    p.add_argument("--checkpoint-dir", default=None,
                   help="Override the checkpoint directory from the config")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducible sampling and training")
    p.add_argument("--non-deterministic", action="store_true",
                   help="Disable deterministic cuDNN (may speed up training but "
                        "sacrifices exact reproducibility)")
    p.add_argument("--from-base", action="store_true",
                   help="Train a fresh LoRA adapter from the base model instead of "
                        "continuing from the current checkpoint")
    args = p.parse_args()

    if not args.source_file and not args.faqs:
        p.error("provide --source-file (chunk mode) or --faqs (Q&A mode)")

    with open(args.config) as f:
        cfg = json.load(f)
    ver.init(cfg)
    model_loader.init(cfg)

    # Support both flat configs and nested addon_config.
    indexing_cfg = cfg.get("addon_config", {}).get("indexing", {})
    training_cfg = cfg.get("addon_config", {}).get("training", {})
    effective_cfg = {**cfg, **indexing_cfg, **training_cfg}

    rb = ReplayBuffer(db_path=effective_cfg.get("replay_db", "replay_buffer.db"))

    # Close the KDS feedback loop: import KDS from vector store into replay buffer.
    load_kds_into_replay_buffer(rb, effective_cfg)
    kds_map = {cid: kds for cid, kds in rb.get_kds_map().items() if kds is not None}

    set_seed(args.seed, deterministic=not args.non_deterministic)

    qa_texts = None
    faqs = None
    if args.faqs:
        # Q&A instruction mode
        with open(args.faqs) as f:
            faqs = json.load(f)
        qa_texts = format_qa_texts(faqs)
        # Sample a small set of raw chunks for regularization
        n_replay = max(10, int(len(qa_texts) * args.replay_ratio))
        replay_chunks = rb.sample(
            n=n_replay,
            weight_by_tier=not args.uniform_sampling,
            kds_map=kds_map,
        )
        new_chunks = []
    else:
        # Raw chunk mode — Qdrant client only needed here
        client = QdrantClient(host=effective_cfg["qdrant_host"], port=effective_cfg["qdrant_port"])
        new_chunks = fetch_chunks_for_source(client, effective_cfg["collection"], args.source_file)
        if not new_chunks:
            print(f"❌ No chunks found for source_file='{args.source_file}'")
            sys.exit(1)
        rb.add_chunks(new_chunks)
        # KDS may have been imported before add_chunks; refresh the map after adding.
        kds_map = {cid: kds for cid, kds in rb.get_kds_map().items() if kds is not None}
        n_replay = max(1, int(len(new_chunks) * args.replay_ratio))
        replay_chunks = rb.sample(
            n=n_replay,
            weight_by_tier=not args.uniform_sampling,
            kds_map=kds_map,
        )
        new_ids = {c["chunk_id"] for c in new_chunks}
        replay_chunks = [c for c in replay_chunks if c["chunk_id"] not in new_ids]

    if args.checkpoint_dir:
        output_dir = args.checkpoint_dir
        if not output_dir.endswith("/"):
            output_dir += "/"
        name = Path(output_dir).name
        if name.startswith("v"):
            try:
                new_ver = int(name[1:].split("_")[0])
            except ValueError:
                new_ver = ver.get_lora_version() + 1
        else:
            new_ver = ver.get_lora_version() + 1
    else:
        new_ver = ver.get_lora_version() + 1
        output_dir = effective_cfg.get("checkpoint_dir", "lora_checkpoints/") + f"v{new_ver}/"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    train(cfg, new_chunks, replay_chunks, output_dir, qa_texts=qa_texts,
          from_base=args.from_base, faqs=faqs)
    if not args.checkpoint_dir:
        ver.increment_lora_version(output_dir)

    notes = {
        "mode": "qa" if args.faqs else "chunk",
        "new_chunks": len(new_chunks),
        "replay_chunks": len(replay_chunks),
        "replay_ratio": args.replay_ratio,
        "uniform_sampling": args.uniform_sampling,
        "from_base": args.from_base,
    }
    if args.faqs:
        notes["qa_examples"] = len(qa_texts) if qa_texts else 0
    save_training_metadata(output_dir, cfg, args.seed, sys.argv, notes=notes)

    print(f"✅ LoRA version → {new_ver}  checkpoint: {output_dir}")


if __name__ == "__main__":
    main()
