# KVForge Research — Master Experiment Checklist
# ✅ = done, 🔄 = running, ⏳ = queued, ❌ = blocked, ⬜ = not started

## PHASE 1: Single-Dataset Crossover (UC4 Bedrock, Gemma-4-E2B-it)
[✅] 1.1 337 FAQs × 5 tiers × 3 seeds

## PHASE 2: Multi-Dataset Crossover (Gemma-4-E2B-it)
[✅] 2.1 SQuAD     — parametric (9/9)    | ⏳ text_rag queued (EC2-A)
[🔄] 2.2 HotpotQA  — 3/9 done            | 🔄 6 runs (EC2-A)
[✅] 2.3 2WikiMQA  — 9/9 done            | EC2-B

## PHASE 3: 4-Mode Comparison (UC4 Bedrock, Gemma-4-E2B-it) — kv_fulltoken FIX
[✅] 3.1 text_rag, kv_meanpool, parametric (5 tiers)
[⬜] 3.2 kv_fulltoken accuracy sweep — UNBLOCKED (fix validated in Aug 2026)
  On-the-fly path (question-only prompt, no SYSTEM_PROMPT):
  ⏳ 3.2a Single-chunk injection   — UC4 Bedrock, 5 tiers × 3 seeds    [EC2-B GPU0]
  ⏳ 3.2b Multi-chunk injection    — UC4 Bedrock, 5 tiers × 3 seeds    [EC2-B GPU0]
  ⏳ 3.2c Rerotation quality       — compare rerotated vs both-pos-0    [EC2-B GPU1]
  ⏳ 3.2d Enhanced Tier on-disk    — save/load round-trip accuracy      [EC2-B GPU1]
  ⏳ 3.2e Prompt format ablation   — SYSTEM_PROMPT vs simple vs chat-template
  ⏳ 3.2f Full 3.2 comparison vs text_rag & kv_meanpool

## PHASE 4: Embedding Ablation (UC4 Bedrock, Path A)
[✅] 4.1 bge-small baseline
[✅] 4.2 bge-large   — done (re-indexing complete on EC2-B GPU1)
[⬜] 4.3 mxbai-large — needs re-index

## PHASE 5: Large-Scale FAQ (UC4 Bedrock, Gemma-4-E2B-it)
[✅] 5.1 3858 FAQs × 5 tiers × 3 seeds

## PHASE 6: Readiness Predictor
[⚠️] 6.1 Baseline (29 pts — needs HotpotQA text_rag + SQuAD text_rag)
[⬜] 6.2 Per-dataset analysis + bootstrap CIs
[⬜] 6.3 Leave-One-Dataset-Out CV

## UC BENCHMARKS (Gemma-4-E2B-it)
[⏳] UC1 Customer Support — 3 tiers × 3 seeds (EC2-A, after SQuAD text_rag)
[✅] UC2 PubMedQA         — 3 tiers × 3 seeds (finished on EC2-B GPU0)
[⏳] UC3 SQuAD            — same as 2.1, text_rag (EC2-A)

## KV FULLTOKEN — REROTATION VALIDATION (EC2-B GPU1)
[⬜] R1 Single-chunk, single-context   — 44-token context, question-only prompt
[⬜] R2 Multi-chunk, no rerotation     — two chunks, both at pos 0 (control)
[⬜] R3 Multi-chunk, with rerotation   — two chunks, chunk2 at offset len1
[⬜] R4 Multi-chunk, 4 chunks          — four chunks, progressive offsets
[⬜] R5 Rerotation vs re-encode        — compare to encoding at correct position
[⬜] R6 Repetition_penalty sweep       — 1.0, 1.2, 1.3, 1.5 on fulltoken vs text_rag

## ENHANCED TIER ON-DISK — VALIDATION (EC2-B GPU1)
[⬜] E1 Save/load round-trip           — bit-exact check (diff=0.0)
[⬜] E2 TurboQuant compression         — save/load with TurboQuant, measure compression ratio
[⬜] E3 Quality vs on-the-fly          — same chunk, compare answer quality
[⬜] E4 Promote + route integration    — full pipeline: promote → store → route → inject → generate
[⬜] E5 Latency comparison             — on-the-fly vs on-disk for 1, 2, 4 chunks

## CACHEBLEND / PARTIAL RECOMPUTE
[⬜] C1 Fix compute_per_token_kv call  — switch to compute_per_token_kv_as_list for Gemma4
[⬜] C2 Partial recompute quality      — recompute_ratio sweep (0.0, 0.1, 0.25, 0.5, 1.0)
[⬜] C3 Rerotation for partial rec.    — per-layer-type rotary for recomputed kernels

## GPU ALLOCATION — CURRENT
| Instance | GPU | Running | Queued |
|----------|-----|---------|--------|
| EC2-A GPU0 | A10G | HotpotQA (6/9) | SQuAD text_rag → UC1 |
| EC2-B GPU0 | RTX PRO 4500 | (free) | Phase 3.2a,b — kv_fulltoken sweep |
| EC2-B GPU1 | RTX PRO 4500 | (free) | Phase 3.2c,d — rerotation + Enhanced Tier |

## ANALYSIS & PAPER
[⬜] A1 Crossover plots (delta vs N per dataset)
[⬜] A2 Mode comparison bar charts
[⬜] A3 Latency analysis (includes fulltoken + Enhanced Tier vs meanpool vs text)
[⬜] A4 FAQ impact analysis (337 vs 500 vs 3858)
[⬜] A5 Final paper writeup
