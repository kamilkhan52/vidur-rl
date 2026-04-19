#!/usr/bin/env python3
"""
EXP08 scheduler leverage analysis.

Inputs:
- Baseline-style run root (same layout used by aggregate_baselines_postfix.py)

Outputs:
- exp08_all.csv
- exp08_scenario_leverage.csv
- exp08_cache_sensitivity.csv
- exp08_attribution_coeffs.csv
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate_baselines_postfix import scan_baselines


def safe_div(a: float, b: float) -> float:
    if b is None or b == 0 or np.isnan(b):
        return np.nan
    return a / b


def attach_request_level_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, row in df.iterrows():
        fp = Path(row["file_path"])

        rec = {
            "prefill_tokens_total": np.nan,
            "prefill_tokens_cached": np.nan,
            "prefill_hit_ratio": np.nan,
            "prefill_miss_ratio": np.nan,
            "cache_oblivious_miss_pressure": np.nan,
            "queue_delay_mean": np.nan,
            "queue_delay_p95": np.nan,
            "queue_delay_p99": np.nan,
        }

        if fp.exists():
            try:
                r = pd.read_csv(fp)

                # Canonical EXP07-compatible hit-ratio definition.
                if {"request_num_prefill_tokens", "request_num_prefill_tokens_cached"}.issubset(r.columns):
                    total = float(r["request_num_prefill_tokens"].sum())
                    cached = float(r["request_num_prefill_tokens_cached"].sum())
                    rec["prefill_tokens_total"] = total
                    rec["prefill_tokens_cached"] = cached
                    if total > 0:
                        hit = cached / total
                        rec["prefill_hit_ratio"] = hit
                        rec["prefill_miss_ratio"] = 1.0 - hit

                        # Miss pressure proxy: miss tokens per completed request.
                        nreq = float(len(r))
                        if nreq > 0:
                            rec["cache_oblivious_miss_pressure"] = (total - cached) / nreq

                # Queueing proxies.
                if "request_scheduling_delay" in r.columns:
                    q = r["request_scheduling_delay"].dropna()
                    if len(q) > 0:
                        rec["queue_delay_mean"] = float(np.mean(q))
                        rec["queue_delay_p95"] = float(np.percentile(q, 95))
                        rec["queue_delay_p99"] = float(np.percentile(q, 99))

            except Exception:
                pass

        rows.append(rec)

    out = df.copy()
    extra = pd.DataFrame(rows)
    for c in extra.columns:
        out[c] = extra[c]
    return out


def compute_scenario_leverage(df: pd.DataFrame) -> pd.DataFrame:
    recs = []

    for key, g in df.groupby(["model", "replicas", "trace", "scale"]):
        worst_mean_e2e = g["mean_e2e"].max()
        best_mean_e2e = g["mean_e2e"].min()
        worst_mean_ttft = g["mean_ttft"].max() if "mean_ttft" in g.columns else np.nan
        best_mean_ttft = g["mean_ttft"].min() if "mean_ttft" in g.columns else np.nan

        recs.append(
            {
                "model": key[0],
                "replicas": key[1],
                "trace": key[2],
                "scale": key[3],
                "best_scheduler_e2e": g.loc[g["mean_e2e"].idxmin(), "scheduler"],
                "worst_scheduler_e2e": g.loc[g["mean_e2e"].idxmax(), "scheduler"],
                "best_mean_e2e": best_mean_e2e,
                "worst_mean_e2e": worst_mean_e2e,
                "sli_e2e": (worst_mean_e2e - best_mean_e2e) / worst_mean_e2e if worst_mean_e2e > 0 else np.nan,
                "best_mean_ttft": best_mean_ttft,
                "worst_mean_ttft": worst_mean_ttft,
                "sli_ttft": (worst_mean_ttft - best_mean_ttft) / worst_mean_ttft if (pd.notna(worst_mean_ttft) and worst_mean_ttft > 0) else np.nan,
                "best_prefill_hit_ratio": g["prefill_hit_ratio"].max(),
                "worst_prefill_hit_ratio": g["prefill_hit_ratio"].min(),
                "hit_ratio_spread": g["prefill_hit_ratio"].max() - g["prefill_hit_ratio"].min(),
            }
        )

    return pd.DataFrame(recs)


def robust_linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    # fallback to ordinary least squares slope for portability.
    if len(x) < 2:
        return np.nan
    if np.allclose(np.std(x), 0):
        return np.nan
    try:
        m, _b = np.polyfit(x, y, 1)
        return float(m)
    except Exception:
        return np.nan


def compute_cache_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for key, g in df.groupby(["model", "replicas", "trace", "scale"]):
        h = g["prefill_hit_ratio"].to_numpy(dtype=float)
        e = g["mean_e2e"].to_numpy(dtype=float)
        t = g["mean_ttft"].to_numpy(dtype=float) if "mean_ttft" in g.columns else np.array([])

        mask_e = np.isfinite(h) & np.isfinite(e)
        slope_e = robust_linear_slope(h[mask_e], e[mask_e])

        slope_t = np.nan
        if len(t) == len(h):
            mask_t = np.isfinite(h) & np.isfinite(t)
            slope_t = robust_linear_slope(h[mask_t], t[mask_t])

        recs.append(
            {
                "model": key[0],
                "replicas": key[1],
                "trace": key[2],
                "scale": key[3],
                # Positive means better hit-rate lowers latency.
                "css_e2e": -slope_e if pd.notna(slope_e) else np.nan,
                "css_ttft": -slope_t if pd.notna(slope_t) else np.nan,
                "n_points": len(g),
            }
        )

    return pd.DataFrame(recs)


def compute_attribution(df: pd.DataFrame) -> pd.DataFrame:
    recs = []

    # Scenario-level simple attribution without external deps.
    for key, g in df.groupby(["model", "replicas", "trace", "scale"]):
        d = g[["prefill_hit_ratio", "queue_delay_mean", "mean_e2e"]].dropna()
        if len(d) < 3:
            recs.append(
                {
                    "model": key[0],
                    "replicas": key[1],
                    "trace": key[2],
                    "scale": key[3],
                    "coef_hit": np.nan,
                    "coef_queue": np.nan,
                    "intercept": np.nan,
                    "r2": np.nan,
                    "n_points": len(d),
                }
            )
            continue

        X = np.column_stack([
            np.ones(len(d)),
            d["prefill_hit_ratio"].to_numpy(dtype=float),
            d["queue_delay_mean"].to_numpy(dtype=float),
        ])
        y = d["mean_e2e"].to_numpy(dtype=float)

        try:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            yhat = X @ beta
            ss_res = float(np.sum((y - yhat) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r2 = np.nan if ss_tot == 0 else 1 - ss_res / ss_tot

            recs.append(
                {
                    "model": key[0],
                    "replicas": key[1],
                    "trace": key[2],
                    "scale": key[3],
                    "coef_hit": float(beta[1]),
                    "coef_queue": float(beta[2]),
                    "intercept": float(beta[0]),
                    "r2": r2,
                    "n_points": len(d),
                }
            )
        except Exception:
            recs.append(
                {
                    "model": key[0],
                    "replicas": key[1],
                    "trace": key[2],
                    "scale": key[3],
                    "coef_hit": np.nan,
                    "coef_queue": np.nan,
                    "intercept": np.nan,
                    "r2": np.nan,
                    "n_points": len(d),
                }
            )

    return pd.DataFrame(recs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = scan_baselines(input_dir)
    if not rows:
        raise SystemExit(f"No completed results found under {input_dir}")

    df = pd.DataFrame(rows).sort_values(["scheduler", "model", "replicas", "trace", "scale"])
    df = attach_request_level_metrics(df)

    all_csv = output_dir / "exp08_all.csv"
    df.to_csv(all_csv, index=False)

    lev = compute_scenario_leverage(df)
    lev.to_csv(output_dir / "exp08_scenario_leverage.csv", index=False)

    css = compute_cache_sensitivity(df)
    css.to_csv(output_dir / "exp08_cache_sensitivity.csv", index=False)

    att = compute_attribution(df)
    att.to_csv(output_dir / "exp08_attribution_coeffs.csv", index=False)

    print(f"Wrote: {all_csv}")
    print(f"Wrote: {output_dir / 'exp08_scenario_leverage.csv'}")
    print(f"Wrote: {output_dir / 'exp08_cache_sensitivity.csv'}")
    print(f"Wrote: {output_dir / 'exp08_attribution_coeffs.csv'}")


if __name__ == "__main__":
    main()
