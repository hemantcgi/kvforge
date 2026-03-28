"""Per-use-case A/B evaluation runner.

Queries a running KVForge dashboard for each FAQ question, computes
semantic similarity and ROUGE-L scores against ground truth, and writes
examples/<uc>/ab_eval_results.json + examples/<uc>/ab_eval_viewer.html.

Usage::

    python -m pipeline.ab_evaluator \\
        --config examples/usecase1_customer_support/config.json \\
        --dashboard-url http://localhost:8081 \\
        --gemini-api-key <key> \\
        --max-samples 200
"""

import argparse
import json
import os
import time
from pathlib import Path

import httpx
import numpy as np
from fastembed import TextEmbedding


def _rouge_l(hyp: str, ref: str) -> float:
    """Compute ROUGE-L F1 score (pure Python, no external deps)."""
    h = hyp.lower().split()
    r = ref.lower().split()
    if not h or not r:
        return 0.0
    m, n = len(r), len(h)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if r[i - 1] == h[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    lcs = prev[n]
    if lcs == 0:
        return 0.0
    precision = lcs / n
    recall = lcs / m
    return 2 * precision * recall / (precision + recall)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D numpy arrays."""
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))
