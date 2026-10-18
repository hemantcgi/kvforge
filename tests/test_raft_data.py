"""Unit tests for the RAFT Phase-2 data layer (tools/raft_data.py).

Pure-logic only: retrieval results are injected as plain dicts, so these tests
need no Qdrant, no GPU, and no network. They encode the corrected RAFT design
from docs/raft_phase2_experiment_plan.md (the four advisor fixes).

Run: python -m pytest tests/test_raft_data.py -v --override-ini="addopts="
"""
import json
import os
import random

import pytest

from tools import raft_data as R

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UC4 = os.path.join(HERE, "examples", "usecase4_bedrock_userguide")


def _chunk(cid, text):
    return {"chunk_id": cid, "text": text}


# ---------------------------------------------------------------- render parity
def test_render_prompt_is_deterministic_and_single_source():
    ctx = ["alpha text", "bravo text", "charlie text"]
    a = R.render_prompt(R.SHORT_ANSWER_INSTRUCTION, ctx, "What is X?")
    b = R.render_prompt(R.SHORT_ANSWER_INSTRUCTION, ctx, "What is X?")
    assert a == b  # byte-identical => builder and dumper get the same prompt
    # instruction, every context chunk, and the question all appear
    assert R.SHORT_ANSWER_INSTRUCTION in a
    for c in ctx:
        assert c in a
    assert "What is X?" in a


def test_render_prompt_does_not_use_the_page_citation_instruction():
    # The default RAG instruction asks for "2-4 sentences. Cite page numbers.",
    # which overruns the 48-tok cap and references a non-existent page key.
    low = R.SHORT_ANSWER_INSTRUCTION.lower()
    assert "page" not in low
    assert "sentence" not in low


# ------------------------------------------------------------ context assembly
def test_kept_example_contains_gold_exactly_once():
    gold = _chunk(7, "the model id is amazon.titan-embed-image-v1")
    distractors = [_chunk(i, f"distractor {i}") for i in range(100, 110)]
    ctx = R.assemble_context(gold, distractors, keep_oracle=True,
                             n_context=5, rng=random.Random(0))
    ids = [c["chunk_id"] for c in ctx]
    assert len(ctx) == 5
    assert ids.count(7) == 1


def test_withheld_example_never_contains_gold():
    gold = _chunk(7, "gold text")
    distractors = [_chunk(i, f"distractor {i}") for i in range(100, 110)]
    ctx = R.assemble_context(gold, distractors, keep_oracle=False,
                             n_context=5, rng=random.Random(0))
    ids = [c["chunk_id"] for c in ctx]
    assert len(ctx) == 5
    assert 7 not in ids


def test_gold_is_never_used_as_a_distractor():
    gold = _chunk(7, "gold text")
    # gold id deliberately present in the candidate distractor pool
    pool = [_chunk(7, "gold text dup")] + [_chunk(i, f"d{i}") for i in range(100, 110)]
    distractors = R.pick_distractors(gold_chunk_id=7, retrieved=pool, n=4,
                                     rng=random.Random(1))
    assert all(c["chunk_id"] != 7 for c in distractors)
    assert len(distractors) == 4


# ----------------------------------------------------------- withhold fraction
def test_withhold_fraction_matches_p():
    flags = R.assign_keep_flags(n=2000, p=0.8, seed=42)
    rate = sum(flags) / len(flags)
    assert abs(rate - 0.8) < 0.03
    # deterministic under a fixed seed
    assert flags == R.assign_keep_flags(n=2000, p=0.8, seed=42)


# --------------------------------------------------------- withheld answer leak
def test_answer_in_chunks_detects_leak():
    chunks = [_chunk(1, "Foo bar."), _chunk(2, "The model id is amazon.titan-embed-image-v1.")]
    assert R.answer_in_chunks("amazon.titan-embed-image-v1", chunks) is True
    assert R.answer_in_chunks("something unrelated entirely", chunks) is False


# -------------------------------------------------------------- oracle recovery
def test_recover_oracle_picks_highest_ranked_answer_bearing_chunk():
    answer = "amazon.titan-embed-image-v1"
    retrieved = [
        _chunk(10, "unrelated preamble"),
        _chunk(11, "The model id is Amazon.Titan-Embed-Image-v1 for images."),  # case differs
        _chunk(12, "also mentions amazon.titan-embed-image-v1 later"),
    ]
    o = R.recover_oracle(answer, retrieved)
    assert o is not None and o["chunk_id"] == 11  # first (highest-ranked) match


def test_recover_oracle_returns_none_when_absent():
    assert R.recover_oracle("nowhere to be found", [_chunk(1, "abc"), _chunk(2, "def")]) is None


def test_recover_oracle_finds_gold_beyond_the_retrieved_prefix():
    # The caller must search the WHOLE ordered corpus (retrieved top-k first,
    # then the rest), so a miss fact whose gold is NOT in the retrieved set is
    # still recovered. This is what keeps miss-fact QA in R2's training.
    answer = "amazon.titan-embed-image-v1"
    ordered = [_chunk(i, f"neighbour {i} no answer here") for i in range(5)]  # retrieved top-5, no gold
    ordered += [_chunk(99, "deep in the corpus: amazon.titan-embed-image-v1")]  # gold, beyond top-k
    o = R.recover_oracle(answer, ordered)
    assert o is not None and o["chunk_id"] == 99


def test_attach_norm_caches_normalized_text():
    chunks = [_chunk(1, "The Model ID Is X."), _chunk(2, "Other!")]
    out = R.attach_norm(chunks)
    assert out[0]["_norm"] == "the model id is x"
    assert out[1]["_norm"] == "other"


def test_recover_oracle_uses_cached_norm_when_present():
    # When a candidate carries a precomputed _norm, recovery uses it (fast path
    # for the 60k x 6154 full-corpus search) and still returns the right chunk.
    cands = R.attach_norm([_chunk(10, "no match here"),
                           _chunk(11, "contains amazon.titan-embed-image-v1 inside")])
    o = R.recover_oracle("amazon.titan-embed-image-v1", cands)
    assert o is not None and o["chunk_id"] == 11


def test_content_tokens_drops_stopwords_and_short():
    toks = R.content_tokens("The Amazon Bedrock model id is v1")
    assert "amazon" in toks and "bedrock" in toks and "model" in toks
    assert "the" not in toks and "is" not in toks        # stopwords dropped


def test_recover_oracle_overlap_matches_paraphrase_without_verbatim():
    # No verbatim substring, but the chunk contains >=80% of the answer's
    # content tokens (scattered / reordered) -> recovered via overlap.
    answer = "Amazon Bedrock supports model invocation logging"
    cands = R.attach_norm([
        _chunk(1, "Unrelated text about pricing and regions."),
        _chunk(2, "Logging of model invocation is supported by Amazon Bedrock for audit."),
    ])
    o = R.recover_oracle(answer, cands)
    assert o is not None and o["chunk_id"] == 2


def test_recover_oracle_overlap_rejects_low_overlap():
    answer = "Amazon Bedrock supports model invocation logging"
    cands = R.attach_norm([_chunk(1, "Amazon pricing varies by region and usage.")])  # only 'amazon' overlaps
    assert R.recover_oracle(answer, cands) is None


def test_recover_oracle_overlap_needs_at_least_three_content_tokens():
    # "No, it is set to false" -> content {set, false}: only 2 generic tokens.
    # Overlap on 2 generic tokens binds arbitrary chunks (spurious), so it is
    # NOT overlap-matched even if both tokens appear.
    answer = "No, it is set to false"
    cands = R.attach_norm([_chunk(1, "the flag is set and the value is false somewhere")])
    assert R.recover_oracle(answer, cands) is None


def test_recover_oracle_overlap_picks_max_overlap_not_first():
    # Two candidates both clear the 0.80 overlap threshold, neither verbatim.
    # The FIRST (chunk 1) covers 4/5 content tokens (0.80); a later chunk (2)
    # covers 5/5 (1.00). Returning the first ">=threshold" chunk binds the
    # weaker source -- in the full-corpus miss path that means the lowest
    # chunk_id of generic boilerplate. Pass 2 must pick the MAX-overlap chunk.
    answer = "bedrock guardrail invocation logging latency"  # 5 content tokens
    cands = R.attach_norm([
        _chunk(1, "guardrail bedrock logging invocation extra words"),       # 4/5 = 0.80
        _chunk(2, "latency bedrock logging guardrail invocation scattered"),  # 5/5 = 1.00
    ])
    o = R.recover_oracle(answer, cands)
    assert o is not None and o["chunk_id"] == 2


def test_recover_oracle_overlap_ties_keep_retrieval_order():
    # When two chunks have EQUAL (max) overlap, the earlier (higher-ranked /
    # lower corpus index) wins -- same precedence as verbatim pass 1.
    answer = "bedrock guardrail invocation"  # 3 content tokens
    cands = R.attach_norm([
        _chunk(1, "invocation guardrail bedrock here"),   # 3/3 = 1.00
        _chunk(2, "bedrock invocation guardrail there"),  # 3/3 = 1.00
    ])
    o = R.recover_oracle(answer, cands)
    assert o is not None and o["chunk_id"] == 1


def test_recover_oracle_guards_short_or_numeric_answers():
    # "2024" occurs in many chunks; picking an arbitrary one is spurious, so a
    # purely-numeric (or too-short) answer is treated as unrecoverable (None),
    # and the caller keeps the example as a withheld, distractor-only one.
    chunks = [_chunk(1, "Copyright 2024 Amazon."), _chunk(2, "Published in 2024.")]
    assert R.recover_oracle("2024", chunks) is None


# -------------------------------------------------------------------- leak dedup
def test_leak_dedup_removes_exact_jaccard_and_answer_matches():
    eval_items = [
        {"question": "What is the model ID for Amazon Titan Multimodal Embeddings?",
         "answer": "amazon.titan-embed-image-v1"},
    ]
    train = [
        {"question": "What is the model ID for Amazon Titan Multimodal Embeddings?",
         "answer": "amazon.titan-embed-image-v1"},                       # Q-exact
        {"question": "what is the model id for amazon titan multimodal embeddings",
         "answer": "x"},                                                  # Jaccard ~1
        {"question": "Totally different question about pricing?",
         "answer": "amazon.titan-embed-image-v1"},                        # A-exact
        {"question": "How do I configure a VPC endpoint for Bedrock?",
         "answer": "Use a gateway endpoint."},                            # survives
    ]
    kept, removed = R.dedup_against_eval(train, eval_items, jaccard_thresh=0.80)
    assert len(kept) == 1
    assert kept[0]["question"].startswith("How do I configure")
    assert len(removed) == 3


def test_jaccard_bounds():
    assert R.jaccard("a b c", "a b c") == 1.0
    assert R.jaccard("a b c", "x y z") == 0.0
    assert 0.0 < R.jaccard("a b c d", "a b x y") < 1.0


# -------------------------------------------------- 481 filter on real eval_500
@pytest.mark.skipif(not os.path.exists(os.path.join(UC4, "eval_500.json")),
                    reason="eval_500.json not present")
def test_page_filter_reproduces_481():
    items = json.load(open(os.path.join(UC4, "eval_500.json")))["items"]
    assert len(items) == 500
    kept = R.filter_481(items)
    assert len(kept) == 481
    assert all(r.get("chunk_id") is not None for r in kept)


# ------------------------------------------------------------------ hit / miss
def test_is_hit_uses_membership_in_raw_topk():
    assert R.is_hit(7, [3, 7, 9, 1, 2]) is True
    assert R.is_hit(7, [3, 8, 9, 1, 2]) is False
    # string/int id robustness
    assert R.is_hit("7", [3, 7, 9]) is True


@pytest.mark.skipif(
    not (os.path.exists(os.path.join(UC4, "eval_500.json"))
         and os.path.exists(os.path.join(HERE, "results", "knowledge", "uc4", "recall_at_k_top5.json"))),
    reason="eval_500.json or recall table not present")
def test_481_hit_miss_split_is_290_191():
    items = json.load(open(os.path.join(UC4, "eval_500.json")))["items"]
    f481 = R.filter_481(items)
    rk = json.load(open(os.path.join(HERE, "results", "knowledge", "uc4", "recall_at_k_top5.json")))
    split = R.split_hit_miss(f481, rk)
    assert split["hit"] + split["miss"] == 481
    assert split["hit"] == 290
    assert split["miss"] == 191


# --------------------------------------------------------------- build_examples
def test_build_examples_recovers_withholds_never_drops_and_counts():
    id2text = {1: "gold has amazon.titan-embed-image-v1 here", 2: "d2", 3: "d3",
               4: "d4", 5: "d5", 6: "d6", 7: "faraway gold xyzzy marker string"}
    chunks_norm = R.attach_norm([{"chunk_id": k, "text": v} for k, v in id2text.items()])
    records = [
        {"question": "q1", "answer": "amazon.titan-embed-image-v1", "retrieved_ids": [2, 1, 3, 4, 5, 6]},  # in top-k
        {"question": "q2", "answer": "xyzzy marker", "retrieved_ids": [2, 3, 4, 5, 6]},                     # only full-corpus (7)
        {"question": "q3", "answer": "2024", "retrieved_ids": [2, 3, 4, 5, 6]},                             # unrecoverable (numeric guard)
    ]
    ex, stats = R.build_examples(records, id2text, chunks_norm, p=1.0, n_context=5, seed=0)
    assert len(ex) == 3                              # never drop
    assert stats["recovered"] == 2
    assert stats["trained_oracle_ids"] == {1, 7}     # q3 contributes no oracle
    assert [e["answer"] for e in ex] == ["amazon.titan-embed-image-v1", "xyzzy marker", "2024"]
    assert "amazon.titan-embed-image-v1" in ex[0]["question"]   # oracle kept (p=1.0)
    assert "xyzzy marker" in ex[1]["question"]                  # full-corpus oracle kept
    assert R.SHORT_ANSWER_INSTRUCTION in ex[0]["question"]


def test_build_examples_withholds_when_flag_is_false():
    id2text = {1: "gold amazon.titan-embed-image-v1", 2: "d2", 3: "d3", 4: "d4", 5: "d5", 6: "d6"}
    chunks_norm = R.attach_norm([{"chunk_id": k, "text": v} for k, v in id2text.items()])
    records = [{"question": "q", "answer": "amazon.titan-embed-image-v1", "retrieved_ids": [1, 2, 3, 4, 5, 6]}]
    ex, stats = R.build_examples(records, id2text, chunks_norm, p=0.0, n_context=5, seed=0)  # all withheld
    assert stats["recovered"] == 1 and stats["trained_oracle_ids"] == {1}   # fact still trained
    assert "amazon.titan-embed-image-v1" not in ex[0]["question"]           # but oracle not in context
    assert stats["withheld"] == 1


# ------------------------------------------------------------ collection guard
def test_collection_constant_is_the_6154_space():
    assert R.RAFT_COLLECTION == "bedrock-userguide-6154"
