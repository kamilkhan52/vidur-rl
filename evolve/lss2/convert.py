"""Vidur window CSV -> LLMServingSim 2.0 JSONL, and a cluster JSON of N identical replicas.

Arrivals are replayed exactly as Vidur's trace request generator does (``vidur/request_generator/
trace_request_generator.py``): clip prefill/decode to >= 1, drop rows above ``max_tokens``, optional arrival jitter
U[0, j) drawn with ``np.random.default_rng(seed)`` and a stable re-sort, then ``arrived_at *= scale``. With
``seed = 100 p + 1`` and Vidur's jitter rule (``evolve.evaluate.Scenario.jitter``) a draw p here is the same arrival
sequence as Vidur's draw p.

Prefix sharing. The v4 traces carry one id per 16-token block (``block_size`` 16; the v4 converter already split each
512-token Mooncake hash). Block k of a prompt with id h becomes the 16 token ids ``h * 16 + j`` (j = 0..15), truncated
to the prompt length. LLMServingSim chain-hashes 16-token blocks, so two prompts share exactly the leading blocks whose
ids agree (the ids are prefix-unique, as Mooncake's are; ``check_prefix_unique`` verifies that). Blocks without an id
(a prompt longer than the id list covers) get negative ids unique to the row, so they never match.

Output token ids are not written: in the v4 traces every output block id is unique to its row, so output blocks are
never shared, and LLMServingSim does not index blocks without token ids.

JSONL formats:
- ``full`` (standard LLMServingSim): ``input_toks, output_toks, arrival_time_ns, input_tok_ids`` plus our extra keys
  (ignored by the stock loader): ``row`` (row in the window CSV), ``session_id``, ``arrival_s``, ``prompt_hashes``.
- ``compact`` (only for our router): no ``input_tok_ids``; the router expands ``tok_block_ids`` itself. About 16x
  smaller files.

CLI::

    python -m evolve.lss2.convert --window W.csv --scale 0.75 --perturbation 1 --out w.jsonl \
        --cluster-out c8.json --replicas 8
"""
import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

BLOCK = 16
BUCKETED = {"Tool", "Conv"}  # Mooncake conv/tool timestamps are 3 s buckets (evolve.evaluate.BUCKETED)
SYNTH_JITTER_S = 1.0  # evolve.evaluate.SYNTH_JITTER_S (protocol v2 Synth draws)
MAX_TOKENS_L3_8B = 131072  # evolve.evaluate.max_request_tokens for l3_8b (context cap; KV capacity is 483,328)
# Vidur's l3_8b replica on an A100 80 GB (evolve.evaluate.KV_CAPACITY["l3_8b"]): 30,208 blocks of 16 tokens.
VIDUR_KV_TOKENS = 483328


def trace_name(window: str) -> str:
    return Path(window).name.split("_")[0]


def vidur_jitter(trace: str, perturbation: int, jitter_s: float = -1.0) -> float:
    """``evolve.evaluate.Scenario.jitter``."""
    if jitter_s >= 0:
        return jitter_s if perturbation > 0 else 0.0
    return 3.0 if (perturbation > 0 and trace in BUCKETED) else 0.0


def _session(x: str) -> Optional[int]:
    if x is None or x == "" or x.lower() == "nan":
        return None
    return int(float(x))


def read_window(path: str) -> List[dict]:
    """Rows of a Vidur window CSV, block ids parsed. Keeps the file's row order (= Vidur's pre-jitter order)."""
    csv.field_size_limit(sys.maxsize)
    rows = []
    with open(path, newline="") as f:
        for k, r in enumerate(csv.DictReader(f)):
            bs = int(float(r.get("block_size") or BLOCK))
            if bs != BLOCK:
                raise ValueError(f"{path}: block_size {bs}, the adapter assumes {BLOCK}")
            ids = json.loads(r["block_hash_ids"]) if r.get("block_hash_ids") else []
            rows.append({
                "row": k,
                "arrived_at": float(r["arrived_at"]),
                "num_prefill_tokens": max(1, int(float(r["num_prefill_tokens"]))),
                "num_decode_tokens": max(1, int(float(r["num_decode_tokens"]))),
                "block_hash_ids": [int(v) for v in ids],
                "session_id": _session(r.get("session_id")),
            })
    return rows


def replay(rows: List[dict], scale: float, perturbation: int = 0, jitter: float = 0.0,
           max_tokens: Optional[int] = MAX_TOKENS_L3_8B, seed: Optional[int] = None) -> List[dict]:
    """Vidur's trace replay: filter, jitter (seeded, stable re-sort), scale. Returns new rows with ``arrival_s``,
    in arrival order (Vidur's request id order)."""
    if max_tokens is not None:
        rows = [r for r in rows if r["num_prefill_tokens"] + r["num_decode_tokens"] <= max_tokens]
    t = np.array([r["arrived_at"] for r in rows], dtype=float)
    if jitter and jitter > 0:
        rng = np.random.default_rng(100 * perturbation + 1 if seed is None else seed)
        t = t + rng.uniform(0.0, jitter, size=len(rows))
        order = np.argsort(t, kind="mergesort")
    else:
        order = np.arange(len(rows))
    t = t * scale
    out = []
    for i in order:
        r = dict(rows[int(i)])
        r["arrival_s"] = float(t[int(i)])
        out.append(r)
    return out


def prompt_hashes(r: dict) -> List[int]:
    """``prompt_block_hashes``: ids of the prompt's full blocks (what a router may see)."""
    return r["block_hash_ids"][: r["num_prefill_tokens"] // BLOCK]


def tok_block_ids(r: dict) -> List[int]:
    """One id per prompt block, partial last block included; negative unique ids where the trace has none."""
    n = r["num_prefill_tokens"]
    nb = -(-n // BLOCK)
    ids = list(r["block_hash_ids"][:nb])
    for k in range(len(ids), nb):
        ids.append(-(1 + r["row"] * (1 << 24) + k))
    return ids


def expand_tokens(block_ids: List[int], n: int) -> List[int]:
    """Synthetic token ids: block id h at position j -> h * 16 + j; truncated to n tokens."""
    out = [0] * (len(block_ids) * BLOCK)
    p = 0
    for h in block_ids:
        b = h * BLOCK
        for j in range(BLOCK):
            out[p] = b + j
            p += 1
    del out[n:]
    return out


def check_prefix_unique(rows: List[dict]) -> int:
    """Number of block ids seen after two different predecessors (0 = ids are prefix-unique, so per-block matching
    in Vidur and chained matching in LLMServingSim agree)."""
    prev: Dict[int, Optional[int]] = {}
    bad = 0
    for r in rows:
        p = None
        for h in r["block_hash_ids"]:
            q = prev.setdefault(h, p)
            if q != p:
                bad += 1
            p = h
    return bad


def to_records(rows: List[dict], fmt: str = "full") -> List[dict]:
    recs = []
    for r in rows:
        n = r["num_prefill_tokens"]
        rec = {
            "input_toks": n,
            "output_toks": r["num_decode_tokens"],
            "arrival_time_ns": int(round(r["arrival_s"] * 1e9)),
            "row": r["row"],
            "session_id": r["session_id"],
            "arrival_s": r["arrival_s"],
            "prompt_hashes": prompt_hashes(r),
        }
        if fmt == "full":
            rec["input_tok_ids"] = expand_tokens(tok_block_ids(r), n)
        elif fmt == "compact":
            rec["tok_block_ids"] = tok_block_ids(r)
        else:
            raise ValueError(fmt)
        recs.append(rec)
    return recs


def write_jsonl(rows: List[dict], path: str, fmt: str = "full") -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        for rec in to_records(rows, fmt):
            f.write(json.dumps(rec, separators=(",", ":")))
            f.write("\n")
    os.replace(tmp, path)
    return len(rows)


def cluster_config(replicas: int, mem_util: float, hardware: str = "RTXPRO6000",
                   model: str = "meta-llama/Llama-3.1-8B", mem_size_gb: int = 96, mem_bw: int = 1597,
                   extra: Optional[dict] = None) -> dict:
    """N identical tp1 instances on one node (modelled on configs/cluster/single_node_multi_instance.json)."""
    inst = {
        "model_name": model,
        "hardware": hardware,
        "npu_mem": {"mem_size": mem_size_gb, "mem_bw": mem_bw, "mem_latency": 0, "mem_util": mem_util},
        "pd_type": None,
        "tp_size": 1,
    }
    inst.update(extra or {})
    return {
        "num_nodes": 1,
        "link_bw": 16,
        "link_latency": 20000,
        "nodes": [{
            "num_instances": replicas,
            "cpu_mem": {"mem_size": 512, "mem_bw": 256, "mem_latency": 0},
            "instances": [json.loads(json.dumps(inst)) for _ in range(replicas)],
        }],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--window", required=True)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--perturbation", type=int, default=0, help="Vidur draw p: seed 100p+1, jitter if p > 0")
    ap.add_argument("--jitter", type=float, default=-1.0, help=">= 0 overrides Vidur's jitter rule")
    ap.add_argument("--head", type=int, default=0, help="keep only the first N rows of the window (0 = all)")
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS_L3_8B)
    ap.add_argument("--format", choices=("full", "compact"), default="full")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cluster-out")
    ap.add_argument("--replicas", type=int, default=8)
    ap.add_argument("--mem-util", type=float, default=None,
                    help="npu_mem.mem_util (default: the value that gives Vidur's 30,208 KV blocks, see run.py)")
    a = ap.parse_args(argv)
    rows = read_window(a.window)
    if a.head:
        rows = rows[: a.head]
    bad = check_prefix_unique(rows)
    rows = replay(rows, a.scale, a.perturbation, vidur_jitter(trace_name(a.window), a.perturbation, a.jitter),
                  a.max_tokens)
    n = write_jsonl(rows, a.out, a.format)
    print(f"wrote {n} requests to {a.out} (prefix-uniqueness violations: {bad})")
    if a.cluster_out:
        from evolve.lss2.run import DEFAULT_MEM_UTIL
        cfg = cluster_config(a.replicas, a.mem_util if a.mem_util is not None else DEFAULT_MEM_UTIL)
        Path(a.cluster_out).write_text(json.dumps(cfg, indent=2))
        print(f"wrote {a.cluster_out}")


if __name__ == "__main__":
    main()
