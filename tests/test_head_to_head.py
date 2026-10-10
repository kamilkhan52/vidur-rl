import math

from evolve import evaluate as ev
from evolve.analysis.head_to_head import head_to_head


def test_ratio_to_each_baseline_recovers_policy_over_baseline():
    scs = [ev.Scenario("Tool", 8, 1.0, -1, 0, p, model="l3_8b") for p in (1, 2)]
    bar = ("lor", "sticky_lor")
    p99 = {"lor": 10.0, "sticky_lor": 5.0}  # bar = sticky_lor on every metric
    cache = {ev.baseline_key(b, sc): {f"p99_{m}": p99[b] for m in ("e2e", "ttft", "tpot", "tbt")}
             for b in bar for sc in scs}
    per = {sc.tag: {m: 0.8 for m in ("e2e", "ttft", "tpot", "tbt")} for sc in scs}  # policy P99 = 4.0
    out = head_to_head(per, scs, cache, bar)
    assert math.isclose(out["sticky_lor"]["e2e"]["geo"], 0.8)
    assert math.isclose(out["lor"]["tbt"]["geo"], 0.4)
    assert out["lor"]["e2e"]["wins"] == 2
