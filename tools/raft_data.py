"""RAFT Phase-2 data layer — pure logic, no Qdrant / GPU / network.

Assembles (question-with-context, answer) training examples for the RAFT
complementarity study (docs/raft_phase2_experiment_plan.md). Retrieval results
are passed in as plain dicts {"chunk_id": int, "text": str}, so every function
here is deterministic and unit-testable offline. The GPU-bound dumper and the
trainer are intentionally NOT part of this module.
"""
from __future__ import annotations

import random
import re
from typing import Optional

# The one short-answer instruction shared by BOTH the training builder and the
# (future) eval dumper, so train and eval prompts are identical except for the
# retrieved chunks. Deliberately NOT the default RAG instruction ("2-4
# sentences. Cite page numbers."), which overruns the 48-token cap and refers to
# a page field the 6154-char chunks do not have.
SHORT_ANSWER_INSTRUCTION = "Answer the question in as few words as possible using the context."

# The retrieval collection whose point ids match eval_500.chunk_id. Using the
# config-default "bedrock-userguide" (2,520 word-chunks) silently mis-scores.
RAFT_COLLECTION = "bedrock-userguide-6154"

_WORD_RE = re.compile(r"[a-z0-9]+")
_PAGE_RE = re.compile(r"\bpage\b", re.IGNORECASE)

# Minimal English stopword set for content-token-overlap oracle recovery.
_STOPWORDS = frozenset("""
a an the of to in on at by for with from into over under and or but not is are was were be been being
this that these those it its as can could may might will would shall should do does did has have had
you your they them their he she his her we our us i me my if then than so such no nor only own same
about above below up down out off again further once here there when where why how all any both each few
more most other some only very s t
""".split())


def _normalize(s: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace."""
    return " ".join(_WORD_RE.findall((s or "").lower()))


def content_tokens(text: str) -> set:
    """Normalized tokens with stopwords removed — the units for overlap recovery."""
    return {t for t in _normalize(text).split() if t not in _STOPWORDS}


# --------------------------------------------------------------------- prompts
def render_prompt(instruction: str, context_chunks: list[str], question: str) -> str:
    """Render the packed prompt. `context_chunks` is a list of chunk TEXTS.

    This is the single source of truth for prompt shape; the builder packs it
    into the --faqs `question` field and the dumper renders eval prompts the
    same way, guaranteeing train/eval parity.
    """
    context = "\n\n".join(context_chunks)
    return f"{instruction}\n\nContext:\n{context}\n\nQuestion: {question}"


# ------------------------------------------------------------ context assembly
def pick_distractors(gold_chunk_id, retrieved: list[dict], n: int,
                     rng: random.Random) -> list[dict]:
    """Up to `n` hard-negative distractors from retrieved neighbours, never the
    gold chunk, de-duplicated by chunk_id, preserving retrieval rank."""
    out, seen = [], set()
    for c in retrieved:
        cid = c["chunk_id"]
        if str(cid) == str(gold_chunk_id) or cid in seen:
            continue
        seen.add(cid)
        out.append(c)
        if len(out) == n:
            break
    return out


def assemble_context(gold: dict, distractors: list[dict], *, keep_oracle: bool,
                     n_context: int, rng: random.Random) -> list[dict]:
    """Return exactly `n_context` shuffled context chunks. When `keep_oracle`,
    one of them is the gold chunk; otherwise the gold chunk is absent."""
    need = n_context - (1 if keep_oracle else 0)
    chosen = pick_distractors(gold["chunk_id"], distractors, need, rng)
    ctx = ([gold] if keep_oracle else []) + chosen
    rng.shuffle(ctx)
    return ctx


def assign_keep_flags(n: int, p: float, seed: int) -> list[bool]:
    """Deterministic per-example oracle-kept flags; keep-rate ~= p."""
    rng = random.Random(seed)
    return [rng.random() < p for _ in range(n)]


# -------------------------------------------------------------- oracle recovery
def answer_in_chunks(answer: str, chunks: list[dict]) -> bool:
    """True if the normalized answer appears in any chunk's normalized text."""
    na = _normalize(answer)
    if not na:
        return False
    return any(na in _normalize(c.get("text", "")) for c in chunks)


def attach_norm(chunks: list[dict]) -> list[dict]:
    """Return the chunks with a cached ``_norm`` (normalized text) field, so a
    full-corpus oracle search normalizes each of the 6,154 chunks once instead
    of once per training question."""
    for c in chunks:
        if "_norm" not in c:
            c["_norm"] = _normalize(c.get("text", ""))
        if "_tokset" not in c:
            c["_tokset"] = set(c["_norm"].split())
    return chunks


def recover_oracle(answer: str, candidates: list[dict], *, min_len: int = 4,
                   overlap_thresh: float = 0.8, min_content_toks: int = 3) -> Optional[dict]:
    """First candidate whose normalized text contains the normalized answer,
    else None. The caller MUST pass the whole ordered corpus (retrieved top-k
    first, then the rest) so a miss fact — whose gold is not in the retrieved
    set — is still recovered; otherwise miss-fact QA silently leaves R2's
    training and the miss test is biased toward a false DEAD.

    Guard: a purely-numeric or very short answer (e.g. "2024") matches many
    chunks, so it is treated as unrecoverable; the caller keeps that example as
    a withheld, distractor-only one (the parametric-fallback signal) rather than
    binding it to an arbitrary chunk."""
    na = _normalize(answer)
    if not na or len(na) < min_len or na.replace(" ", "").isdigit():
        return None
    # Pass 1 — verbatim substring (highest precision).
    for c in candidates:
        text_norm = c.get("_norm")
        if text_norm is None:
            text_norm = _normalize(c.get("text", ""))
        if na in text_norm:
            return c
    # Pass 2 — content-token overlap (abstractive answers). Requires
    # >=min_content_toks content tokens so a couple of common words can't bind an
    # arbitrary chunk. Among the candidates clearing `overlap_thresh`, return the
    # one with the HIGHEST overlap (ties broken by candidate order, i.e.
    # retrieval rank then corpus index). Returning the first chunk over the
    # threshold silently binds the weakest qualifying source — and in the
    # full-corpus miss path "first" is just the lowest chunk_id of generic
    # boilerplate, which produced ~20% false oracles in spot-checks.
    a_toks = content_tokens(answer)
    if len(a_toks) >= min_content_toks:
        best, best_frac = None, overlap_thresh
        for c in candidates:
            ts = c.get("_tokset")
            if ts is None:
                ts = set((c.get("_norm") or _normalize(c.get("text", ""))).split())
            frac = sum(1 for t in a_toks if t in ts) / len(a_toks)
            if frac >= best_frac and (best is None or frac > best_frac):
                best, best_frac = c, frac
        if best is not None:
            return best
    return None


# ------------------------------------------------------------------ leak dedup
def jaccard(a: str, b: str) -> float:
    sa, sb = set(_normalize(a).split()), set(_normalize(b).split())
    if not sa and not sb:
        return 1.0
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


def dedup_against_eval(train: list[dict], eval_items: list[dict], *,
                       jaccard_thresh: float = 0.80) -> tuple[list[dict], list[dict]]:
    """Drop train records that leak eval_500: Q-exact (normalized) OR
    Jaccard>=thresh on questions OR A-exact (normalized). Returns (kept, removed)."""
    eval_q_norm = {_normalize(e.get("question", "")) for e in eval_items}
    eval_a_norm = {_normalize(e.get("answer", "")) for e in eval_items}
    eval_q_list = [e.get("question", "") for e in eval_items]
    kept, removed = [], []
    for r in train:
        q, a = r.get("question", ""), r.get("answer", "")
        qn, an = _normalize(q), _normalize(a)
        leaked = (qn in eval_q_norm) or (an and an in eval_a_norm) \
            or any(jaccard(q, eq) >= jaccard_thresh for eq in eval_q_list)
        (removed if leaked else kept).append(r)
    return kept, removed


# -------------------------------------------------------------- eval set + split
def filter_481(items: list[dict]) -> list[dict]:
    """eval_500 minus TOC page-number trivia: drop any item whose question or
    answer matches \\bpage\\b (mirrors run_knowledge_measurement.py:61-73)."""
    return [r for r in items
            if not _PAGE_RE.search((r.get("question", "") or "") + " " + (r.get("answer", "") or ""))]


def is_hit(gold_chunk_id, retrieved_ids: list) -> bool:
    """Hit = gold chunk id is among the raw top-k retrieved ids (threshold=None)."""
    return str(gold_chunk_id) in {str(x) for x in retrieved_ids}


def build_examples(records: list[dict], id2text: dict, chunks_norm: list[dict], *,
                   p: float = 0.8, n_context: int = 5, seed: int = 42) -> tuple[list[dict], dict]:
    """Assemble RAFT `--faqs` examples from 60k QA records that each carry
    `{question, answer, retrieved_ids}`.

    Oracle recovery is two-stage: the cheap retrieved top-k first, then the
    cached full corpus (`chunks_norm`) so a miss fact is still recovered. An
    unrecovered question is NEVER dropped — it is kept as a withheld,
    distractor-only example. A fact counts as *trained* whenever its chunk is
    the recovered oracle of ≥1 example (kept or withheld), which is what the §3
    coverage gate measures. Returns (examples, stats)."""
    flags = assign_keep_flags(len(records), p, seed)
    rng = random.Random((seed ^ 0x9E3779B9) & 0xFFFFFFFF)
    examples, trained = [], set()
    recovered = withheld = withheld_leak = 0
    for rec, keep in zip(records, flags):
        rids = [int(x) for x in rec.get("retrieved_ids", [])]
        retrieved = attach_norm([{"chunk_id": i, "text": id2text[i]} for i in rids if i in id2text])
        oracle = recover_oracle(rec["answer"], retrieved) or recover_oracle(rec["answer"], chunks_norm)
        if oracle is not None:
            recovered += 1
            trained.add(int(oracle["chunk_id"]))
            gold, keep_oracle = oracle, keep
        else:
            gold, keep_oracle = {"chunk_id": None, "text": ""}, False
        ctx = assemble_context(gold, retrieved, keep_oracle=keep_oracle, n_context=n_context, rng=rng)
        if not keep_oracle:
            withheld += 1
            if answer_in_chunks(rec["answer"], ctx):
                withheld_leak += 1
        q = render_prompt(SHORT_ANSWER_INSTRUCTION, [c["text"] for c in ctx], rec["question"])
        examples.append({"question": q, "answer": rec["answer"]})
    stats = {
        "n_examples": len(examples),
        "recovered": recovered,
        "recovery_rate": round(recovered / max(1, len(records)), 4),
        "trained_oracle_ids": trained,
        "withheld": withheld,
        "withheld_answer_leak": withheld_leak,
        "withheld_answer_leak_rate": round(withheld_leak / max(1, withheld), 4),
    }
    return examples, stats


def _recall_records(recall_table) -> list[dict]:
    if isinstance(recall_table, dict):
        return recall_table.get("records") or recall_table.get("items") or []
    return recall_table or []


def split_hit_miss(items: list[dict], recall_table) -> dict:
    """Split items by retrieval hit/miss using a materialized recall@k table
    (gold_chunk_id -> hit). threshold=None is implicit in the table."""
    by_gold = {}
    for r in _recall_records(recall_table):
        g = r.get("gold_chunk_id", r.get("chunk_id"))
        if g is None:
            continue
        by_gold[str(g)] = bool(r.get("hit"))
    hit = miss = unknown = 0
    for it in items:
        key = str(it.get("chunk_id"))
        if key not in by_gold:
            unknown += 1
        elif by_gold[key]:
            hit += 1
        else:
            miss += 1
    return {"hit": hit, "miss": miss, "unknown": unknown}
