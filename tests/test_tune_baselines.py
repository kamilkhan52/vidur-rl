import random
from pathlib import Path

import numpy as np

from evolve.analysis.tune_baselines import FAMILIES, ask_tell, render, to_unit, to_value
from vidur.scheduler.global_scheduler.evolved_global_scheduler import (
    PolicyContext, ReplicaView, RequestView, load_policy)

SYSTEMS = Path(__file__).resolve().parents[1] / "evolve" / "policies" / "systems"


def _random_case(rng, n=4):
    reps = [ReplicaView(i, rng.randint(0, 9), rng.randint(0, 5000), 0, 0, rng.random(), 100000) for i in range(n)]
    hashes = tuple(range(rng.randint(1, 6)))
    req = RequestView(0.0, 16 * len(hashes) + rng.randint(0, 15), None,
                      tuple(rng.choice([0, 16, 32]) for _ in range(n)), -1, 0, 1, hashes, 16,
                      tuple(rng.choice([0, 16]) for _ in range(n)), tuple(rng.choice([0, 16, 48]) for _ in range(n)))
    return req, reps


def test_defaults_reproduce_the_untuned_policies(tmp_path):
    for fam, base in (("dynamo", "dynamo_kv_cost.py"), ("mooncake", "mooncake_min_ttft.py")):
        template, space = FAMILIES[fam]
        p = tmp_path / f"{fam}.py"
        p.write_text(render(template.read_text(), {k: v[3] for k, v in space.items()}))
        tuned, ref = load_policy(str(p)), load_policy(str(SYSTEMS / base))
        rng = random.Random(1)
        for seed in range(200):
            req, reps = _random_case(rng)
            c1, c2 = PolicyContext(random.Random(seed), 4), PolicyContext(random.Random(seed), 4)
            assert tuned.choose(req, reps, c1) == ref.choose(req, reps, c2)


def test_unit_mapping_round_trips():
    for lo, hi, log in ((0.05, 20.0, True), (0.0, 1.0, False)):
        for u in (0.0, 0.3, 1.0):
            assert abs(to_unit(to_value(u, lo, hi, log), lo, hi, log) - u) < 1e-12


def test_es_finds_the_optimum_of_a_quadratic():
    target = np.array([0.2, 0.7])
    hist = ask_tell(120, 2, lambda pop: [-float(((u - target) ** 2).sum()) for u in pop], seed=0)
    best = max(hist, key=lambda x: x[1])[0]
    assert len(hist) == 120 and np.abs(best - target).max() < 0.05
