"""Train and score one RL design with the EXP12 protocol (outside the evolution loop).

    python -m evolve.analysis.rl_seed_eval evolve/rl_designs/seed_ttft_bandit.py --work /tmp/x --out x.json
"""

import argparse
import json
from pathlib import Path

from evolve import rl_task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("design")
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    res = rl_task.evaluate_design(a.design, Path(a.work), workers=a.workers)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(rl_task.feedback_text(res), flush=True)
    print("fitness", res["combined_score"], flush=True)


if __name__ == "__main__":
    main()
