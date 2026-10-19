# `judge_out/` — fresh-Opus grades (canonical judged record)

The 10 files (`shard_00.json` … `shard_09.json`) hold the grades a fresh Opus
judge assigned to each blind shard in `../shards/`. This is the **canonical,
non-regenerable record** of the judging step: grading is not deterministic, so
these outputs — not a script — are the ground truth the verdict is scored from.
Committed (~44 KB total).

## Schema

One object per file, keyed by `fact_id`; each value maps the shard's blind labels
to a 3-way grade:

```json
{
  "f0000": {"A": "CORRECT", "B": "CORRECT", "C": "CORRECT", "D": "CORRECT"},
  "f0001": {"A": "PARTIAL", "B": "INCORRECT"}
}
```

Grades: `CORRECT` = 1.0, `PARTIAL` = 0.5, `INCORRECT` = 0.0.
`coverage@0.5` counts CORRECT+PARTIAL as covered.

## How these are consumed

`tools/cojudge_score_phase2.py` merges all 10 files, un-blinds them via
`../cojudge_p2_key.json`, validates completeness, then scores the verdict. To
re-judge from scratch, rebuild the blind shards (`tools/cojudge_build_phase2.py`)
and grade each with a fresh Opus subagent, writing one file per shard here in the
schema above.
