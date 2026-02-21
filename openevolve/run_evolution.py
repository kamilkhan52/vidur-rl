"""
Core OpenEvolve evolution loop. Called by run.sh and run_alpine.sh.

Usage:
    python openevolve/run_evolution.py [--iterations N] [--output-dir DIR]
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).parent.parent
OPENEVOLVE_DIR = Path(__file__).parent


def _make_patched_config(output_dir: Path) -> str:
    """
    Load config.yaml and override evolution_trace.output_path to be
    inside output_dir (so traces follow the run, not the project dir).
    Returns path to a temp YAML file.
    """
    config_path = OPENEVOLVE_DIR / "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if "evolution_trace" in cfg:
        cfg["evolution_trace"]["output_path"] = str(
            output_dir / "traces" / "evolution_trace.jsonl"
        )

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, dir=str(output_dir)
    )
    yaml.dump(cfg, tmp)
    tmp.close()
    return tmp.name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output-dir", default=str(OPENEVOLVE_DIR / "evolution_output"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Precompute baseline if not done yet
    baseline_file = OPENEVOLVE_DIR / "baselines" / "sticky_lor_e2e.json"
    if not baseline_file.exists():
        print("=== Precomputing baselines ===")
        result = subprocess.run(
            [sys.executable, str(OPENEVOLVE_DIR / "precompute_baseline.py")],
            cwd=str(REPO_ROOT),
        )
        if result.returncode != 0:
            print("ERROR: Baseline precomputation failed.")
            sys.exit(1)

    print(f"=== Starting OpenEvolve: {args.iterations} iterations ===")
    print(f"Output: {output_dir}")
    print()

    import openevolve

    patched_config = _make_patched_config(output_dir)

    result = openevolve.run_evolution(
        initial_program=str(OPENEVOLVE_DIR / "initial_program.py"),
        evaluator=str(OPENEVOLVE_DIR / "evaluator.py"),
        config=patched_config,
        iterations=args.iterations,
        output_dir=str(output_dir),
    )

    print()
    print("=== Evolution complete ===")
    print(f"Best score: {result.best_score:.4f}")

    best_path = output_dir / "best" / "best_program.py"
    if best_path.exists():
        print()
        print("--- Best program ---")
        print(best_path.read_text())

    # Plot results
    print()
    print("=== Plotting results ===")
    plot_out = output_dir / "progress.png"
    subprocess.run(
        [sys.executable, str(OPENEVOLVE_DIR / "plot_progress.py"),
         "--log-dir", str(output_dir),
         "--out", str(plot_out)],
        cwd=str(REPO_ROOT),
    )
    print(f"Plot saved to: {plot_out}")


if __name__ == "__main__":
    main()
