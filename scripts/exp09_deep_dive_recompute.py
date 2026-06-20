#!/usr/bin/env python3
"""EXP09 deep-dive recompute (request-level).

Reconstructs the EXP09 deep-dive summary numbers from the per-run "enhanced"
CSV (one row per simulation run). Originally the deep-dive was produced
ad-hoc with no saved generator; this script makes it reproducible.

Recipe (validated to reproduce analysis/exp09_light_2026-04-26 numbers):
  per-run rows
    -> add nonqueue_e2e = mean_e2e - queue_delay_mean
    -> average repeats -> scheduler-scenario table (one row per
       scheduler x model x replicas x trace x scale)
  queue-fraction stats + cross-scenario correlations: over scheduler-scenarios
  SLI per fixed scenario (model,replicas,trace,scale): (worst-best)/worst on mean_e2e
  within-scenario standardized 2-feature OLS: e2e_z ~ beta_queue*qdelay_z + beta_hit*hit_z

Usage:
  python3 exp09_deep_dive_recompute.py --enhanced-csv <csv> --out-dir <dir>
"""
import argparse, csv, math, os, statistics as st

KEY = ["scheduler", "model", "replicas", "trace", "scale"]
FIXED = ["model", "replicas", "trace", "scale"]


def pctl(v, q):  # numpy linear interpolation, q in [0,1]
    v = sorted(v)
    if not v:
        return float("nan")
    k = (len(v) - 1) * q
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return v[int(k)]
    return v[f] * (c - k) + v[c] * (k - f)


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    return cov / math.sqrt(vx * vy)


def zscore(v):
    n = len(v)
    m = sum(v) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / n)  # population sd (matches pandas std(ddof=0)? -> see note)
    if sd == 0:
        return [0.0] * n
    return [(x - m) / sd for x in v]


def ols2(y, x1, x2):
    """Standardized 2-feature OLS via normal equations; returns (b1, b2, r2)."""
    n = len(y)
    s11 = sum(a * a for a in x1)
    s22 = sum(a * a for a in x2)
    s12 = sum(a * b for a, b in zip(x1, x2))
    s1y = sum(a * b for a, b in zip(x1, y))
    s2y = sum(a * b for a, b in zip(x2, y))
    det = s11 * s22 - s12 * s12
    if abs(det) < 1e-12:
        return float("nan"), float("nan"), float("nan")
    b1 = (s22 * s1y - s12 * s2y) / det
    b2 = (s11 * s2y - s12 * s1y) / det
    yhat = [b1 * a + b2 * b for a, b in zip(x1, x2)]
    ybar = sum(y) / n
    ss_res = sum((a - b) ** 2 for a, b in zip(y, yhat))
    ss_tot = sum((a - ybar) ** 2 for a in y)
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
    return b1, b2, r2


def load_runs(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for c in ("mean_e2e", "queue_delay_mean", "prefill_hit_ratio",
                  "queue_fraction", "mean_ttft"):
            r[c] = float(r[c])
        r["nonqueue_e2e"] = r["mean_e2e"] - r["queue_delay_mean"]
    return rows


def group_mean(rows, keys, metrics):
    out = {}
    for r in rows:
        k = tuple(r[x] for x in keys)
        out.setdefault(k, []).append(r)
    res = []
    for k, grp in out.items():
        d = dict(zip(keys, k))
        for m in metrics:
            d[m] = sum(g[m] for g in grp) / len(grp)
        d["_n"] = len(grp)
        res.append(d)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enhanced-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    runs = load_runs(args.enhanced_csv)
    metrics = ["mean_e2e", "queue_delay_mean", "prefill_hit_ratio",
               "queue_fraction", "mean_ttft", "nonqueue_e2e"]
    ss = group_mean(runs, KEY, metrics)  # scheduler-scenarios

    # queue-fraction stats over scheduler-scenarios
    qf = [s["queue_fraction"] for s in ss]
    qf_median = pctl(qf, 0.5)
    qf_p75 = pctl(qf, 0.75)
    qf_gt90 = sum(1 for x in qf if x > 0.90) / len(qf)

    # cross-scenario correlations (over scheduler-scenarios)
    e2e = [s["mean_e2e"] for s in ss]
    qd = [s["queue_delay_mean"] for s in ss]
    hit = [s["prefill_hit_ratio"] for s in ss]
    nq = [s["nonqueue_e2e"] for s in ss]
    cor_q_e2e = pearson(qd, e2e)
    cor_hit_e2e = pearson(hit, e2e)
    cor_hit_nq = pearson(hit, nq)
    cor_hit_q = pearson(hit, qd)

    # SLI per fixed scenario + best/worst
    fixed = {}
    for s in ss:
        fixed.setdefault(tuple(s[x] for x in FIXED), []).append(s)
    sli_rows, complete, partial = [], 0, 0
    best_ct, worst_ct = {}, {}
    for k, grp in fixed.items():
        n_sched = len(grp)
        if n_sched == 4:
            complete += 1
        else:
            partial += 1
        worst = max(grp, key=lambda g: g["mean_e2e"])
        best = min(grp, key=lambda g: g["mean_e2e"])
        sli = (worst["mean_e2e"] - best["mean_e2e"]) / worst["mean_e2e"]
        d = dict(zip(FIXED, k))
        d.update(n_sched=n_sched, worst_scheduler=worst["scheduler"],
                 best_scheduler=best["scheduler"], worst_mean_e2e=worst["mean_e2e"],
                 best_mean_e2e=best["mean_e2e"], sli_e2e=sli)
        sli_rows.append(d)
        best_ct[best["scheduler"]] = best_ct.get(best["scheduler"], 0) + 1
        worst_ct[worst["scheduler"]] = worst_ct.get(worst["scheduler"], 0) + 1

    sli_vals = [r["sli_e2e"] for r in sli_rows]
    sli_median, sli_p90, sli_max = pctl(sli_vals, 0.5), pctl(sli_vals, 0.9), max(sli_vals)

    # within-scenario standardized regressions
    reg_rows, reg_q, reg_h = [], 0, 0
    for k, grp in fixed.items():
        if len(grp) < 3:
            continue
        y = zscore([g["mean_e2e"] for g in grp])
        x1 = zscore([g["queue_delay_mean"] for g in grp])
        x2 = zscore([g["prefill_hit_ratio"] for g in grp])
        bq, bh, r2 = ols2(y, x1, x2)
        dom = "queue" if abs(bq) >= abs(bh) else "hit"
        if dom == "queue":
            reg_q += 1
        else:
            reg_h += 1
        d = dict(zip(FIXED, k))
        d.update(n_sched=len(grp), beta_queue=bq, beta_hit=bh, r2=r2, dominant=dom)
        reg_rows.append(d)

    # write artifacts
    def wcsv(name, rows, cols):
        with open(os.path.join(args.out_dir, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    sli_rows.sort(key=lambda r: -r["sli_e2e"])
    wcsv("top_sli_cases.csv", sli_rows,
         FIXED + ["worst_scheduler", "best_scheduler", "worst_mean_e2e", "best_mean_e2e", "sli_e2e"])
    wcsv("within_regression.csv", reg_rows,
         FIXED + ["n_sched", "beta_queue", "beta_hit", "r2", "dominant"])
    wcsv("scheduler_scenarios.csv", sorted(ss, key=lambda s: tuple(s[x] for x in KEY)),
         KEY + metrics + ["_n"])

    lines = []
    P = lines.append
    P(f"n_runs: {len(runs)}")
    P(f"n_scenarios: {len(ss)}")
    P(f"n_sli: {len(sli_rows)}")
    P(f"complete_fixed_scenarios: {complete}")
    P(f"partial_fixed_scenarios: {partial}")
    P(f"queue_frac_median: {qf_median}")
    P(f"queue_frac_p75: {qf_p75}")
    P(f"queue_frac_gt90: {qf_gt90}")
    P(f"sli_median: {sli_median}")
    P(f"sli_p90: {sli_p90}")
    P(f"sli_max: {sli_max}")
    P(f"reg_queue_dom: {reg_q}")
    P(f"reg_hit_dom: {reg_h}")
    P("")
    P("correlations")
    P(f"queue_mean_vs_e2e: {cor_q_e2e}")
    P(f"hit_vs_e2e: {cor_hit_e2e}")
    P(f"hit_vs_nonqueue: {cor_hit_nq}")
    P(f"hit_vs_queue: {cor_hit_q}")
    P("")
    P("best counts")
    for k in sorted(best_ct, key=lambda x: -best_ct[x]):
        P(f"{k}\t{best_ct[k]}")
    P("worst counts")
    for k in sorted(worst_ct, key=lambda x: -worst_ct[x]):
        P(f"{k}\t{worst_ct[k]}")
    txt = "\n".join(lines) + "\n"
    open(os.path.join(args.out_dir, "summary_numbers.txt"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
