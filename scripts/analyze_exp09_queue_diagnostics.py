#!/usr/bin/env python3
"""
Analyze EXP09 queue diagnostics telemetry.

Inputs:
  --events-csv : queue telemetry rows (queue_diag_events.csv or merged)

Outputs:
  - exp09_queue_events_summary.csv
  - exp09_hol_metrics.csv
  - exp09_lookahead_opportunity.csv

Expected columns (best effort; script degrades gracefully):
  scheduler, model, replicas, trace, scale, run_id,
  event_level (global|replica),
  queue_depth_before, waiting_depth_before,
  queue_delay_p95,
  head_is_feasible,
  feasible_count_top2, feasible_count_top4, feasible_count_top8,
  head_num_new_tokens,
  best_feasible_num_new_tokens_top2/top4/top8
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

KEYS = ["scheduler", "model", "replicas", "trace", "scale"]
K_LIST = [2, 4, 8]


def col(df: pd.DataFrame, name: str, default=np.nan):
    if name in df.columns:
        return df[name]
    return pd.Series([default] * len(df), index=df.index)


def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu = s.mean()
    sd = s.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        return pd.Series([0.0] * len(s), index=s.index)
    return (s - mu) / sd


def summarize_availability(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["global_depth"] = pd.to_numeric(col(d, "queue_depth_before"), errors="coerce")
    d["replica_depth"] = pd.to_numeric(col(d, "waiting_depth_before"), errors="coerce")

    recs = []
    for key, g in d.groupby(KEYS, dropna=False):
        rec = {k: v for k, v in zip(KEYS, key)}

        gd = g["global_depth"].dropna()
        rd = g["replica_depth"].dropna()

        rec.update(
            {
                "n_events": int(len(g)),
                "global_med": float(gd.median()) if len(gd) else np.nan,
                "global_p90": float(gd.quantile(0.90)) if len(gd) else np.nan,
                "global_p99": float(gd.quantile(0.99)) if len(gd) else np.nan,
                "replica_med": float(rd.median()) if len(rd) else np.nan,
                "replica_p90": float(rd.quantile(0.90)) if len(rd) else np.nan,
                "replica_p99": float(rd.quantile(0.99)) if len(rd) else np.nan,
            }
        )

        for k in K_LIST:
            rec[f"p_global_ge{k}"] = float((gd >= k).mean()) if len(gd) else np.nan
            rec[f"p_replica_ge{k}"] = float((rd >= k).mean()) if len(rd) else np.nan

        recs.append(rec)

    return pd.DataFrame(recs)


def compute_hol(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["head_is_feasible"] = pd.to_numeric(col(d, "head_is_feasible"), errors="coerce")
    d["waiting_depth_before"] = pd.to_numeric(col(d, "waiting_depth_before"), errors="coerce")

    for k in K_LIST:
        d[f"feasible_count_top{k}"] = pd.to_numeric(col(d, f"feasible_count_top{k}"), errors="coerce")

    recs = []
    for key, g in d.groupby(KEYS, dropna=False):
        rec = {k: v for k, v in zip(KEYS, key)}
        rec["n_events"] = int(len(g))

        for k in K_LIST:
            hol_event = (g["head_is_feasible"] == 0) & (g[f"feasible_count_top{k}"] > 0)
            rec[f"hol_rate_{k}"] = float(hol_event.mean()) if len(g) else np.nan

            if hol_event.any():
                rec[f"hol_intensity_{k}"] = float(g.loc[hol_event, f"feasible_count_top{k}"].mean())
                tail = hol_event & (g["waiting_depth_before"] >= k)
                rec[f"hol_tail_rate_{k}"] = float(tail.mean())
            else:
                rec[f"hol_intensity_{k}"] = np.nan
                rec[f"hol_tail_rate_{k}"] = np.nan

        recs.append(rec)

    return pd.DataFrame(recs)


def compute_opportunity(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["queue_delay_p95"] = pd.to_numeric(col(d, "queue_delay_p95"), errors="coerce")

    for k in K_LIST:
        d[f"feasible_count_top{k}"] = pd.to_numeric(col(d, f"feasible_count_top{k}"), errors="coerce")
        d[f"best_feasible_num_new_tokens_top{k}"] = pd.to_numeric(
            col(d, f"best_feasible_num_new_tokens_top{k}"), errors="coerce"
        )

    d["head_num_new_tokens"] = pd.to_numeric(col(d, "head_num_new_tokens"), errors="coerce")
    d["head_is_feasible"] = pd.to_numeric(col(d, "head_is_feasible"), errors="coerce")
    d["queue_depth_before"] = pd.to_numeric(col(d, "queue_depth_before"), errors="coerce")

    recs = []
    for key, g in d.groupby(KEYS, dropna=False):
        rec = {k: v for k, v in zip(KEYS, key)}

        rec["queue_delay_p95"] = float(g["queue_delay_p95"].dropna().mean()) if g["queue_delay_p95"].notna().any() else np.nan

        for k in K_LIST:
            hol_event = (g["head_is_feasible"] == 0) & (g[f"feasible_count_top{k}"] > 0)
            rec[f"opp_rate_{k}"] = float(hol_event.mean()) if len(g) else np.nan

            token_gain = g[f"best_feasible_num_new_tokens_top{k}"] - g["head_num_new_tokens"]
            rec[f"token_gain_proxy_{k}"] = float(token_gain.dropna().mean()) if token_gain.notna().any() else np.nan
            rec[f"p_global_ge{k}"] = float((g["queue_depth_before"] >= k).mean()) if g["queue_depth_before"].notna().any() else np.nan

        recs.append(rec)

    out = pd.DataFrame(recs)
    # Opportunity ranking index (K=4)
    out["loi_4"] = zscore(out.get("opp_rate_4", pd.Series(dtype=float))) + zscore(
        out.get("p_global_ge4", pd.Series(dtype=float))
    ) + zscore(out.get("queue_delay_p95", pd.Series(dtype=float)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events-csv", required=True, help="Path to queue diagnostics CSV")
    ap.add_argument("--output-dir", required=True, help="Output directory")
    args = ap.parse_args()

    events_csv = Path(args.events_csv)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(events_csv)

    summary = summarize_availability(df)
    hol = compute_hol(df)
    opp = compute_opportunity(df)

    p1 = outdir / "exp09_queue_events_summary.csv"
    p2 = outdir / "exp09_hol_metrics.csv"
    p3 = outdir / "exp09_lookahead_opportunity.csv"

    summary.to_csv(p1, index=False)
    hol.to_csv(p2, index=False)
    opp.to_csv(p3, index=False)

    print(f"Wrote: {p1}")
    print(f"Wrote: {p2}")
    print(f"Wrote: {p3}")


if __name__ == "__main__":
    main()
