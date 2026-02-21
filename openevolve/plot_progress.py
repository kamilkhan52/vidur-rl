"""
Plot OpenEvolve evolution progress.

Parses the evolution log file and/or the JSONL trace to visualise:
  - combined_score per iteration (RL improvement over STICKY_LOR)
  - rl_e2e per iteration vs the fixed baseline
  - best-so-far envelope

Usage:
    # Plot from latest log in evolution_output/
    python openevolve/plot_progress.py

    # Point at a specific run
    python openevolve/plot_progress.py --log-dir openevolve/evolution_output

    # Live-refresh every 60 s while evolution is running
    python openevolve/plot_progress.py --watch 60

    # Save to file instead of showing interactively
    python openevolve/plot_progress.py --out openevolve/progress.png
"""

import argparse
import json
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── regex patterns for log parsing ───────────────────────────────────────────
_EVAL_PAT = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ - openevolve\.evaluator - INFO - "
    r"Evaluated program (\S+) in [\d.]+s: (.+)"
)
_ITER_PAT = re.compile(
    r"Iteration (\d+): Program (\S+) \(parent: (\S+)\) completed"
)

# ── data loading ──────────────────────────────────────────────────────────────

def _parse_metrics(kv_str: str) -> dict:
    """Parse 'key=val, key=val' string into dict of floats."""
    out = {}
    for part in kv_str.split(", "):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        try:
            out[k.strip()] = float(v.strip())
        except ValueError:
            pass
    return out


def load_from_log(log_path: Path) -> list[dict]:
    """
    Parse an openevolve .log file.

    Returns a list of dicts with keys:
        iteration, program_id, combined_score, rl_e2e, baseline_e2e, improvement_pct
    ordered by iteration number.
    """
    # First pass: collect metrics per program_id
    metrics_by_id: dict[str, dict] = {}
    for line in log_path.read_text().splitlines():
        m = _EVAL_PAT.match(line)
        if m:
            prog_id = m.group(2)
            kv = _parse_metrics(m.group(3))
            if kv:
                metrics_by_id[prog_id] = kv

    # Second pass: assign iteration numbers
    iter_order: list[tuple[int, str]] = []   # (iter_num, child_id)
    for line in log_path.read_text().splitlines():
        m = _ITER_PAT.search(line)
        if m:
            iter_order.append((int(m.group(1)), m.group(2)))

    # Build records in iteration order
    records = []
    for iter_num, prog_id in sorted(iter_order):
        if prog_id in metrics_by_id:
            rec = {"iteration": iter_num, "program_id": prog_id}
            rec.update(metrics_by_id[prog_id])
            records.append(rec)

    # Also include iteration 0 (initial program, no matching "Iteration N:" line)
    # It's the first evaluated program that appears in the log
    if records and metrics_by_id:
        seen_ids = {r["program_id"] for r in records}
        for prog_id, m in metrics_by_id.items():
            if prog_id not in seen_ids:
                rec = {"iteration": 0, "program_id": prog_id}
                rec.update(m)
                records.insert(0, rec)
                break

    return sorted(records, key=lambda r: r["iteration"])


def load_from_trace(trace_path: Path) -> list[dict]:
    """
    Parse the JSONL evolution trace for richer iteration data.

    Returns list of dicts with keys:
        iteration, program_id, parent_id, combined_score, rl_e2e,
        improvement_pct, delta_score
    """
    records = []
    with open(trace_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            child_m = entry.get("child_metrics", {})
            parent_m = entry.get("parent_metrics", {})
            rec = {
                "iteration": entry.get("iteration", -1),
                "program_id": entry.get("child_id", ""),
                "parent_id": entry.get("parent_id", ""),
                "combined_score": child_m.get("combined_score", float("nan")),
                "rl_e2e": child_m.get("rl_e2e", float("nan")),
                "baseline_e2e": child_m.get("baseline_e2e", float("nan")),
                "improvement_pct": child_m.get("improvement_pct", float("nan")),
                "delta_score": (
                    child_m.get("combined_score", float("nan"))
                    - parent_m.get("combined_score", float("nan"))
                ),
            }
            records.append(rec)
    return sorted(records, key=lambda r: r["iteration"])


def find_latest_log(output_dir: Path) -> Path | None:
    logs = sorted((output_dir / "logs").glob("openevolve_*.log")) if (output_dir / "logs").exists() else []
    return logs[-1] if logs else None


def find_trace(output_dir: Path) -> Path | None:
    t = output_dir / "traces" / "evolution_trace.jsonl"
    return t if t.exists() else None


# ── plotting ──────────────────────────────────────────────────────────────────

def plot(records: list[dict], trace_records: list[dict] | None, out_path: Path | None):
    if not records:
        print("No data to plot.")
        return

    iters = [r["iteration"] for r in records]
    scores = [r.get("combined_score", float("nan")) for r in records]
    rl_e2e = [r.get("rl_e2e", float("nan")) for r in records]
    baseline = next((r.get("baseline_e2e") for r in records if r.get("baseline_e2e")), None)

    # Best-so-far envelope (ignoring NaN and -1.0 failures)
    best_so_far, current_best = [], float("-inf")
    for s in scores:
        if not np.isnan(s) and s > -1.0:
            current_best = max(current_best, s)
        best_so_far.append(current_best if current_best > float("-inf") else float("nan"))

    fig = plt.figure(figsize=(14, 8))
    fig.suptitle("OpenEvolve: RL Formulation Search Progress", fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── Panel 1: combined_score vs iteration ──────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    valid = [(i, s) for i, s in zip(iters, scores) if not np.isnan(s) and s > -1.0]
    fail  = [(i, s) for i, s in zip(iters, scores) if np.isnan(s) or s <= -1.0]

    if valid:
        vi, vs = zip(*valid)
        ax1.scatter(vi, vs, s=40, zorder=3, color="#1f77b4", label="eval score")
    if fail:
        fi, _ = zip(*fail)
        ax1.scatter(fi, [-1.0] * len(fi), s=40, marker="x", color="red", zorder=3, label="crash/timeout")
    ax1.step(iters, best_so_far, where="post", color="green", linewidth=2,
             linestyle="--", label="best so far")
    ax1.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Improvement over STICKY_LOR\n(combined_score)")
    ax1.set_title("Score vs Iteration  (>0 = RL beats STICKY_LOR)")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Colour background: positive = good, negative = bad
    ax1.axhspan(0, ax1.get_ylim()[1] if ax1.get_ylim()[1] > 0 else 0.1,
                alpha=0.04, color="green")
    ax1.axhspan(ax1.get_ylim()[0] if ax1.get_ylim()[0] < 0 else -0.1, 0,
                alpha=0.04, color="red")

    # ── Panel 2: rl_e2e latency ───────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    valid_e2e = [(i, e) for i, e in zip(iters, rl_e2e) if not np.isnan(e) and e > 0]
    if valid_e2e:
        vi2, ve2 = zip(*valid_e2e)
        ax2.scatter(vi2, ve2, s=30, color="#ff7f0e", zorder=3, label="RL E2E")
    if baseline:
        ax2.axhline(baseline, color="steelblue", linewidth=1.5,
                    linestyle="--", label=f"STICKY_LOR ({baseline:.3f}s)")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Mean E2E latency (s)")
    ax2.set_title("RL E2E vs Baseline")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: score delta per iteration (from trace) ───────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    if trace_records:
        t_iters = [r["iteration"] for r in trace_records]
        t_deltas = [r["delta_score"] for r in trace_records]
        colors = ["green" if d > 0 else "red" for d in t_deltas]
        ax3.bar(t_iters, t_deltas, color=colors, alpha=0.7, width=0.8)
        ax3.axhline(0, color="black", linewidth=0.8)
        ax3.set_xlabel("Iteration")
        ax3.set_ylabel("Δ score vs parent")
        ax3.set_title("Score Change per Mutation")
        ax3.grid(True, alpha=0.3, axis="y")
    else:
        ax3.text(0.5, 0.5, "Enable evolution_trace\nfor delta panel",
                 ha="center", va="center", transform=ax3.transAxes, color="gray")
        ax3.set_title("Score Change per Mutation")

    plt.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")
    else:
        plt.show()
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot OpenEvolve progress")
    parser.add_argument("--log-dir", default="openevolve/evolution_output",
                        help="Evolution output directory (default: openevolve/evolution_output)")
    parser.add_argument("--out", default=None,
                        help="Save plot to file instead of showing (e.g. progress.png)")
    parser.add_argument("--watch", type=int, default=0, metavar="SECONDS",
                        help="Live refresh every N seconds (0 = one-shot)")
    args = parser.parse_args()

    output_dir = Path(args.log_dir)
    out_path = Path(args.out) if args.out else None

    def render():
        log_path = find_latest_log(output_dir)
        if not log_path:
            print(f"No log file found under {output_dir}/logs/")
            return

        records = load_from_log(log_path)
        trace_path = find_trace(output_dir)
        trace_records = load_from_trace(trace_path) if trace_path else None

        print(f"Log: {log_path}  ({len(records)} evaluated programs)")
        if trace_records:
            print(f"Trace: {trace_path}  ({len(trace_records)} entries)")

        if records:
            best = max(r.get("combined_score", float("-inf")) for r in records)
            print(f"Best score so far: {best:.4f}  "
                  f"({'BEATS baseline' if best > 0 else 'below baseline'})")

        plot(records, trace_records, out_path)

    if args.watch:
        print(f"Live mode: refreshing every {args.watch}s. Ctrl+C to stop.")
        while True:
            render()
            time.sleep(args.watch)
    else:
        render()


if __name__ == "__main__":
    main()
