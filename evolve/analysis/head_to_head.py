"""Head-to-head: a policy's P99s against each bar member separately, with a cluster bootstrap over configurations.

bar_eval JSONs store per-scenario ratios to the bar (the per-metric best baseline of the bar the JSON was scored
against). The policy's own P99 is ratio x bar, and its ratio to baseline b is ratio x bar / P99_b, with every
baseline read from the cache under the current code_version and trace hashes. So:
- the bar is the JSON's own (its "bar" list; JSONs without one need --bar), and JSONs scored against different
  bars are not merged;
- a JSON scored under another code_version (or other traces) is refused, because its ratios and the current cache
  describe different simulations (--allow-code-version-mismatch overrides, with a warning).

  python -m evolve.analysis.head_to_head --json evolve/analysis/out/test2_finalists.json --stage test2 \
      --policy tail4_ni.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from evolve import evaluate as ev
from evolve.analysis.cluster_ci import METRICS, boot, clusters


def short(b: str) -> str:
    return b.rsplit("/", 1)[-1].removesuffix(".py") if b.startswith("policy:") else b


def head_to_head(per_scenario: dict, scenarios, cache: dict, bar=ev.BAR_V2) -> dict:
    by_tag = {sc.tag: sc for sc in scenarios}
    tags = sorted(per_scenario)
    missing = [t for t in tags if t not in by_tag]
    if missing:
        raise KeyError(f"scenarios not in the stage: {missing[:3]}")
    mets = [m for m in METRICS if all(m in per_scenario[t] for t in tags)]
    out = {}
    for b in bar:
        res = {}
        for m in mets:
            d = []
            for t in tags:
                sc = by_tag[t]
                best = min(cache[ev.baseline_key(x, sc)][f"p99_{m}"] for x in bar)
                d.append(np.log(per_scenario[t][m] * best / cache[ev.baseline_key(b, sc)][f"p99_{m}"]))
            d = np.array(d)
            g, lo, hi = boot(d, clusters(tags))
            res[m] = {"geo": g, "lo": lo, "hi": hi, "wins": int((d < 0).sum()), "n": len(d)}
        out[short(b)] = res
    return out


def local_member(b: str) -> str:
    """A bar member as recorded in a JSON, with a policy path from another checkout mapped into this repo."""
    if not b.startswith("policy:"):
        return b
    p = Path(b[len("policy:"):])
    if not p.exists() and "/evolve/" in str(p):
        p = ev.REPO / ("evolve/" + str(p).rsplit("/evolve/", 1)[1])
    if not p.exists():
        raise SystemExit(f"bar member {b} not found (nor under {ev.REPO})")
    return f"policy:{p.resolve()}"


def json_bar(d: dict, name: str, fallback) -> tuple:
    if "bar" in d:
        return tuple(local_member(b) for b in d["bar"])
    if fallback is None:
        raise SystemExit(f"{name} does not record the bar it was scored against; pass --bar with that bar's name")
    return tuple(ev.BARS[fallback])


def check_provenance(d: dict, name: str, allow: bool) -> None:
    problems = []
    if d.get("code_version") != ev.code_version():
        problems.append(f"code_version {d.get('code_version')} (current {ev.code_version()})")
    traces = (d.get("provenance") or {}).get("traces")
    if traces:
        now = ev.provenance()["traces"]
        if traces != now:
            problems.append(f"trace shas {traces} (current {now})")
    if not problems:
        return
    msg = f"{name} was scored under {' and '.join(problems)}: its ratios cannot be combined with the current cache"
    if not allow:
        raise SystemExit(msg + " (--allow-code-version-mismatch overrides)")
    print("!" * 100 + f"\nWARNING: {msg}. Proceeding because of --allow-code-version-mismatch; the numbers below "
          "mix two simulations.\n" + "!" * 100, file=sys.stderr, flush=True)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="append", required=True)
    ap.add_argument("--stage", required=True, choices=sorted(ev.STAGES))
    ap.add_argument("--policy", action="append", required=True)
    ap.add_argument("--bar", choices=sorted(ev.BARS), default=None,
                    help="only for JSONs that do not record their bar (written before bar_eval stored it)")
    ap.add_argument("--allow-code-version-mismatch", action="store_true",
                    help="combine a JSON scored under another code_version/traces with the current cache (warns)")
    a = ap.parse_args(argv)
    pol, bar = {}, None
    for f in a.json:
        d = json.load(open(f))
        check_provenance(d, f, a.allow_code_version_mismatch)
        b = json_bar(d, f, a.bar)
        if bar is not None and b != bar:
            raise SystemExit(f"{f} was scored against another bar than {a.json[0]}; run head_to_head per bar")
        bar = b
        pol.update(d["policies"])
    cache = ev.load_baseline_cache()
    scs = ev.STAGES[a.stage]
    print(f"{a.stage} ({len(scs)} scenarios): policy P99 / baseline P99, geo mean [95% cluster-bootstrap CI] "
          f"and wins out of {len(scs)}; bar of {len(bar)}: {', '.join(short(b) for b in bar)}")
    out = {}
    for p in a.policy:
        print(f"\n{p.removesuffix('.py')}")
        out[p] = head_to_head(pol[p]["per_scenario"], scs, cache, bar)
        for b, res in out[p].items():
            print(f"  vs {b:18s} " + " | ".join(
                f"{m} {r['geo']:.3f} [{r['lo']:.3f},{r['hi']:.3f}] {r['wins']}" for m, r in res.items()))
    return out


if __name__ == "__main__":
    main()
