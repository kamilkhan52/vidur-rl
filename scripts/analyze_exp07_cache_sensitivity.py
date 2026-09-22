#!/usr/bin/env python3
"""
Aggregate and analyze EXP07 cache-sensitivity runs.

Outputs:
- exp07_priority2_all.csv
- exp07_hitmiss_summary.csv
- exp07_pairwise_sticky_vs_lor.csv
- exp07_policy_leverage.csv
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

import sys

# allow importing sibling script when run as: python scripts/analyze_exp07_cache_sensitivity.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate_baselines_postfix import scan_baselines


def attach_hit_miss(df: pd.DataFrame) -> pd.DataFrame:
    hit_list = []
    miss_list = []

    for _, row in df.iterrows():
        fp = Path(row["file_path"])
        hit = np.nan
        miss = np.nan
        if fp.exists():
            try:
                r = pd.read_csv(fp)
                if {
                    "request_num_prefill_tokens",
                    "request_num_prefill_tokens_cached",
                }.issubset(r.columns):
                    total = float(r["request_num_prefill_tokens"].sum())
                    cached = float(r["request_num_prefill_tokens_cached"].sum())
                    if total > 0:
                        hit = cached / total
                        miss = 1.0 - hit
            except Exception:
                pass
        hit_list.append(hit)
        miss_list.append(miss)

    out = df.copy()
    out["prefill_hit_ratio"] = hit_list
    out["prefill_miss_ratio"] = miss_list
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, help="EXP07 run root")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = scan_baselines(input_dir)
    if not rows:
        raise SystemExit(f"No completed results found under {input_dir}")

    df = pd.DataFrame(rows).sort_values(["scheduler", "model", "replicas", "trace", "scale"])
    df = attach_hit_miss(df)

    all_csv = output_dir / "exp07_priority2_all.csv"
    df.to_csv(all_csv, index=False)

    hitmiss = (
        df.groupby("scheduler")
        .agg(
            n=("scheduler", "count"),
            hit_mean=("prefill_hit_ratio", "mean"),
            miss_mean=("prefill_miss_ratio", "mean"),
            mean_e2e=("mean_e2e", "mean"),
            mean_ttft=("mean_ttft", "mean"),
            p99_e2e=("p99_e2e", "mean"),
        )
        .reset_index()
    )
    hitmiss.to_csv(output_dir / "exp07_hitmiss_summary.csv", index=False)

    # Sticky-vs-LOR paired table
    pair = (
        df[df["scheduler"].isin(["LOR", "STICKY_LOR"])]
        .pivot_table(
            index=["model", "replicas", "trace", "scale"],
            columns="scheduler",
            values=["mean_e2e", "mean_ttft", "p99_e2e", "prefill_hit_ratio"],
            aggfunc="mean",
        )
        .dropna()
        .reset_index()
    )

    if len(pair) > 0:
        pair.columns = [
            "_".join([str(c) for c in col if c != ""]).strip("_")
            for col in pair.columns.values
        ]
        pair["sticky_hit_delta"] = pair["prefill_hit_ratio_STICKY_LOR"] - pair["prefill_hit_ratio_LOR"]
        pair["sticky_e2e_impr_pct"] = (
            (pair["mean_e2e_LOR"] - pair["mean_e2e_STICKY_LOR"]) / pair["mean_e2e_LOR"] * 100.0
        )
        pair["sticky_ttft_impr_pct"] = (
            (pair["mean_ttft_LOR"] - pair["mean_ttft_STICKY_LOR"]) / pair["mean_ttft_LOR"] * 100.0
        )
        pair.to_csv(output_dir / "exp07_pairwise_sticky_vs_lor.csv", index=False)

    # Policy leverage: best vs worst scheduler spread by point
    lev_rows = []
    for key, g in df.groupby(["model", "replicas", "trace", "scale"]):
        best_e2e = g["mean_e2e"].min()
        worst_e2e = g["mean_e2e"].max()
        best_ttft = g["mean_ttft"].min()
        worst_ttft = g["mean_ttft"].max()
        lev_rows.append(
            {
                "model": key[0],
                "replicas": key[1],
                "trace": key[2],
                "scale": key[3],
                "best_scheduler": g.loc[g["mean_e2e"].idxmin(), "scheduler"],
                "spread_e2e_pct": (worst_e2e - best_e2e) / worst_e2e * 100.0 if worst_e2e > 0 else np.nan,
                "spread_ttft_pct": (worst_ttft - best_ttft) / worst_ttft * 100.0 if worst_ttft > 0 else np.nan,
            }
        )
    pd.DataFrame(lev_rows).to_csv(output_dir / "exp07_policy_leverage.csv", index=False)

    print(f"Wrote: {all_csv}")
    print(f"Wrote: {output_dir / 'exp07_hitmiss_summary.csv'}")
    if (output_dir / "exp07_pairwise_sticky_vs_lor.csv").exists():
        print(f"Wrote: {output_dir / 'exp07_pairwise_sticky_vs_lor.csv'}")
    print(f"Wrote: {output_dir / 'exp07_policy_leverage.csv'}")


if __name__ == "__main__":
    main()
