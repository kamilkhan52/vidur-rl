#!/usr/bin/env python3
"""Lint a Vidur trace CSV against the prefix-hash encoding rules the simulator relies on (changelog KV-1/KV-10).

  R1 length:   len(block_hash_ids) == (num_prefill_tokens + num_decode_tokens) // block_size
  R2 private:  ids at positions >= num_prefill_tokens // block_size (the partial last prompt block and decode
               blocks) occur in no other row, so they can never produce a cache hit
  R3 position: every id occurs at a single position across the trace (a prefix chain, never tiled)
  R4 size:     one block_size for the whole trace

  python scripts/lint_trace.py data/processed_traces/mooncake_synthetic_trace.csv [...]
"""
import json
import sys
from collections import defaultdict

import pandas as pd


def lint(path: str, max_examples: int = 3) -> dict:
    df = pd.read_csv(path, usecols=["num_prefill_tokens", "num_decode_tokens", "block_hash_ids", "block_size"])
    sizes = sorted(df["block_size"].dropna().unique().tolist())
    errs = defaultdict(list)
    if len(sizes) != 1:
        errs["R4"].append(f"block sizes {sizes}")
    bs = int(sizes[0]) if sizes else 16
    pos_of, rows_of = {}, defaultdict(set)
    private = []
    for i, (p, d, h) in enumerate(zip(df["num_prefill_tokens"], df["num_decode_tokens"], df["block_hash_ids"])):
        ids = json.loads(h)
        if len(ids) != (p + d) // bs:
            errs["R1"].append(f"row {i}: {len(ids)} ids for {p}+{d} tokens")
        cut = p // bs
        for j, x in enumerate(ids):
            if pos_of.setdefault(x, j) != j:
                errs["R3"].append(f"row {i}: id {x} at position {j} and {pos_of[x]}")
            rows_of[x].add(i)
        private.extend((i, x) for x in ids[cut:])
    for i, x in private:
        if len(rows_of[x]) > 1:
            errs["R2"].append(f"row {i}: id {x} past the full prompt blocks is shared with {len(rows_of[x]) - 1} rows")
    return {"rows": len(df), "block_size": bs,
            "violations": {k: len(v) for k, v in errs.items()},
            "examples": {k: v[:max_examples] for k, v in errs.items()}}


if __name__ == "__main__":
    bad = False
    for f in sys.argv[1:]:
        r = lint(f)
        bad |= bool(r["violations"])
        print(f, json.dumps(r))
    sys.exit(1 if bad else 0)
