#!/usr/bin/env python3
"""
Merge KVForge LoRA adapter into Llama 3.2 3B base, then quantize with AWQ.

Outputs:
  ./merged_fp16/   — merged base + LoRA in FP16 (used as FP16 baseline)
  ./merged_awq/    — AWQ 4-bit quantized version of merged model
"""

import json
import sys
import time
from pathlib import Path

print("=== KVForge LoRA Merge + AWQ Quantization ===\n")

VERSION_FILE   = Path("examples/usecase4_bedrock_userguide/version.json")
CHECKPOINT_DIR = Path("examples/usecase4_bedrock_userguide/lora_checkpoints")
BASE_MODEL     = "meta-llama/Llama-3.2-3B-Instruct"
MERGED_FP16    = "./merged_fp16"
MERGED_AWQ     = "./merged_awq"

# ── 1. Resolve LoRA checkpoint ────────────────────────────────────────────────
lora_path = None
if VERSION_FILE.exists():
    with open(VERSION_FILE) as f:
        ver = json.load(f)
    lora_path = ver.get("checkpoint_path")
    print(f"version.json: phase={ver.get('phase')}, lora_version={ver.get('current_lora_version')}")
    print(f"checkpoint_path: {lora_path}")

if not lora_path or not Path(lora_path).exists():
    candidates = sorted(CHECKPOINT_DIR.glob("*/"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        lora_path = str(candidates[0])
        print(f"Falling back to most-recent checkpoint: {lora_path}")
    else:
        print("ERROR: No LoRA checkpoint found. Run lora_trainer first.")
        sys.exit(1)

print(f"\nLoRA path : {lora_path}")
print(f"Base model: {BASE_MODEL}\n")

# ── 2. Load base + LoRA, merge, save FP16 ────────────────────────────────────
print("Step 1/3: Loading base model + LoRA adapter and merging...")
t0 = time.time()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
)
print(f"  base loaded ({time.time()-t0:.1f}s)")

t1 = time.time()
peft_model = PeftModel.from_pretrained(base_model, lora_path)
merged = peft_model.merge_and_unload()
print(f"  LoRA merged ({time.time()-t1:.1f}s)")

t2 = time.time()
merged.save_pretrained(MERGED_FP16)
tokenizer.save_pretrained(MERGED_FP16)
print(f"  saved → {MERGED_FP16}  ({time.time()-t2:.1f}s)\n")

# Free VRAM before quantization
del merged, peft_model, base_model
torch.cuda.empty_cache()
print("VRAM freed.\n")

# ── 3. Quantize merged model with AutoAWQ ────────────────────────────────────
print("Step 2/3: Quantizing merged model with AWQ (4-bit GEMM, group=128)...")
t3 = time.time()

from awq import AutoAWQForCausalLM

awq_model = AutoAWQForCausalLM.from_pretrained(
    MERGED_FP16,
    device_map="auto",
    safetensors=True,
)
awq_tok = AutoTokenizer.from_pretrained(MERGED_FP16)

awq_model.quantize(awq_tok, quant_config={
    "zero_point": True,   # asymmetric — better quality
    "q_group_size": 128,  # standard; smaller = better quality at cost of memory
    "w_bit": 4,
    "version": "GEMM",    # best for vLLM continuous batching
})
print(f"  quantization done ({time.time()-t3:.1f}s)")

t4 = time.time()
awq_model.save_quantized(MERGED_AWQ)
awq_tok.save_pretrained(MERGED_AWQ)
print(f"  saved → {MERGED_AWQ}  ({time.time()-t4:.1f}s)\n")

print("=== Done ===")
print(f"  FP16 model : {MERGED_FP16}")
print(f"  AWQ  model : {MERGED_AWQ}")
print(f"  Total time : {time.time()-t0:.1f}s")
