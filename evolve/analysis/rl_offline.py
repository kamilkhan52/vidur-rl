"""Offline learning diagnostics on saved RL replay buffers (plan E0): no simulation, no training run.

For each candidate with a persisted buffer (v3.1 bandit designs keep the last episode's buffer, which holds every
training decision when nothing was evicted), the regression problem the learner faced is re-fitted with
supervised learning, split by training EPISODE (scenario), never by row:

  representation  train vs held-out-episode explained variance (EV) for a linear model and 2-layer MLPs of width
                  16, 64, 256, 1024. Train EV rising while held-out EV stays flat = noise / non-transfer, not
                  missing capacity.
  noise ceiling   Gamma test (Stefansson et al., 1997): the intercept of semi-variance vs squared nearest-neighbour
                  distance estimates the irreducible target variance given the features -> ceiling EV.
  ablation        drop-one-feature-group (groups = the "# ---" headers in the design's replica_features).
  vs the rule     EV of models given (state-only), (state + "is the teacher's action"), (all features),
                  (all + teacher flag): does the per-replica feature set carry reward information beyond the rule's
                  choice? Plus a listwise classifier of the teacher's action (can the features reproduce the rule?)
                  and the agreement of the fitted reward model's argmax with the rule.
  data            held-out EV vs fraction of training rows (10 %..100 %) at the chosen width.

Targets: r (as stored, what the bandit regressed) and, when the design ends with a deterministic zero-mean
"externality price" feature that reward() subtracts, r_out = r + price[a] (the outcome part that has to be learned
from data; the price part is a known function of the features).

    python -m evolve.analysis.rl_offline [--cand rl_loto_r2/e413fe9e193c ...] [--sweep N] [--threads 6]
"""

import argparse
import ast
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from evolve.analysis import rl_stats

REPO = rl_stats.REPO
FINALISTS = ["rl_loto_r2/e413fe9e193c", "rl_loto_synth_r1/d2cb78118623"]
WIDTHS = [0, 16, 64, 256, 1024]  # 0 = linear (ridge)
FRACTIONS = [0.1, 0.25, 0.5, 1.0]
MIN_EPS_ROWS = 100  # fewest epsilon-explored rows for the beyond-rule test on those rows alone


# ----------------------------------------------------------------------------- small pure helpers (tested)

def ev(y, yhat):
    v = float(np.var(y))
    return 1.0 - float(np.var(y - yhat)) / v if v > 0 else float("nan")


def feature_groups(src, n_features):
    """[(name, [indices])] from the list returned by replica_features: each element joins the latest '# ---'
    comment above it. Returns None if the source cannot be mapped onto n_features entries."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    fn = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "replica_features"), None)
    if fn is None:
        return None
    ret = [n for n in ast.walk(fn) if isinstance(n, ast.Return) and isinstance(n.value, ast.List)]
    if not ret:
        return None
    elts = ret[-1].value.elts
    if len(elts) != n_features:
        return None
    lines = src.splitlines()
    headers = []
    for ln in range(ret[-1].value.lineno, ret[-1].value.end_lineno + 1):
        s = lines[ln - 1].strip()
        if s.startswith("# ---"):
            name = s.strip("# -").split("(")[0].split(":")[0].strip()
            headers.append((ln, name or f"group@{ln}"))
    groups = {}
    order = []
    for i, e in enumerate(elts):
        name = "ungrouped"
        for ln, h in headers:
            if ln < e.lineno:
                name = h
        if name not in groups:
            groups[name] = []
            order.append(name)
        groups[name].append(i)
    return [(n, groups[n]) for n in order]


def episode_labels(ends, loads, T):
    """Row -> episode index from per-episode buffer_size_end, valid only if the buffer never wrapped and each
    episode reloaded the previous one. None otherwise."""
    if not ends or any(v is None for v in ends) or ends[-1] != T:
        return None
    if loads[0] not in (0, None) or any(loads[k] != ends[k - 1] for k in range(1, len(ends))):
        return None
    lab = np.empty(T, dtype=np.int64)
    s = 0
    for k, e in enumerate(ends):
        lab[s:e] = k
        s = e
    return lab


def episode_labels_from_ranges(starts, added, T, cap):
    """Buffer slot -> episode index from the per-episode logging block (buffer_start, buffer_rows_added): episode k
    wrote slots (start + j) % cap for j < added; later episodes overwrite earlier ones, so this stays exact after the
    buffer wraps. Slots never written by a logged episode get -1. None if the ranges are missing."""
    if not starts or any(v is None for v in starts) or any(v is None for v in added):
        return None
    lab = np.full(T, -1, dtype=np.int64)
    for k, (s0, n) in enumerate(zip(starts, added)):
        idx = (int(s0) + np.arange(int(n))) % int(cap)
        lab[idx[idx < T]] = k
    return lab


def load_traces(d, n_episodes):
    """Concatenate the per-episode <ckpt>.trace.npz files of a training dir (RL_LOGGING_SPEC section 3). Returns
    None unless every episode has one with transitions. Keys: x3, a, r, ta, g, who, lab, r_outcome, trace_frac."""
    files = rl_stats.ckpt_files(d)
    if not files or len(files) != n_episodes:
        return None
    parts = []
    for k, p in files:
        f = Path(str(p) + ".trace.npz")
        if not f.exists():
            return None
        z = np.load(f)
        if "x" not in z.files:
            return None
        parts.append((k, z))
    cat = lambda key: np.concatenate([z[key] for _, z in parts])  # noqa: E731
    return {"x3": cat("x"), "a": cat("a"), "r": cat("r").astype(np.float64), "ta": cat("ta"), "g": cat("g"),
            "who": cat("who"), "r_outcome": cat("r_outcome"),
            "lab": np.concatenate([np.full(len(z["a"]), k, dtype=np.int64) for k, z in parts]),
            "trace_frac": float(parts[0][1]["trace_frac"])}


def dedup_index(x_taken, r):
    """Indices of first occurrences of identical (taken-action features, reward) rows (the replay double-store)."""
    h = np.ascontiguousarray(np.concatenate([x_taken.astype(np.float32), r.astype(np.float32)[:, None]], 1))
    view = h.view(np.dtype((np.void, h.dtype.itemsize * h.shape[1]))).ravel()
    _, idx = np.unique(view, return_index=True)
    return np.sort(idx)


def state_feature_mask(x3, tol=1e-6):
    """Features identical across replicas in (almost) every decision = state/request features."""
    spread = x3.max(1) - x3.min(1)  # (T, F)
    return (spread <= tol).mean(0) >= 0.999


def group_folds(labels, k, seed=0):
    """k folds of whole episodes, balanced by row count (greedy), deterministic."""
    u, cnt = np.unique(labels, return_counts=True)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(u))
    u, cnt = u[perm], cnt[perm]
    order = np.argsort(-cnt, kind="stable")
    load = [0] * k
    members = [[] for _ in range(k)]
    for j in order:
        f = int(np.argmin(load))
        members[f].append(u[j])
        load[f] += cnt[j]
    return [np.isin(labels, m) for m in members]


def gamma_test(X, y, k=10, n_max=20000, seed=0):
    """Gamma test noise-variance estimate: regress gamma(p) = mean (y_nn(p) - y)^2 / 2 on
    delta(p) = mean |x_nn(p) - x|^2 for p = 1..k; the intercept estimates Var(noise). Inputs are standardised.
    Returns dict(noise_var, var_y, ceiling_ev, delta1, slope)."""
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(seed)
    if len(y) > n_max:
        sel = rng.choice(len(y), n_max, replace=False)
        X, y = X[sel], y[sel]
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Z)
    dist, ind = nn.kneighbors(Z)
    dist, ind = dist[:, 1:], ind[:, 1:]
    delta = (dist ** 2).mean(0)
    gamma = ((y[ind] - y[:, None]) ** 2).mean(0) / 2.0
    A = np.vstack([np.ones_like(delta), delta]).T
    (b0, b1), *_ = np.linalg.lstsq(A, gamma, rcond=None)
    var_y = float(np.var(y))
    noise = float(max(b0, 0.0))
    return {"noise_var": noise, "noise_var_raw_intercept": float(b0), "gamma1": float(gamma[0]), "var_y": var_y,
            "ceiling_ev": 1.0 - noise / var_y if var_y > 0 else float("nan"),
            "ceiling_ev_upper_from_gamma1": 1.0 - float(gamma[0]) / var_y if var_y > 0 else float("nan"),
            "delta1": float(delta[0]), "slope": float(b1), "n": int(len(y)), "k": k}


# ----------------------------------------------------------------------------- model fitting

def _standardise(tr, *others):
    mu, sd = tr.mean(0), tr.std(0)
    sd[sd == 0] = 1.0
    return [(a - mu) / sd for a in (tr,) + others]


def fit_regressor(Xtr, ytr, Xva, width, seed=0, max_epochs=40, patience=4, batch=512, lr=1e-3, wd=1e-5):
    """Linear ridge (width 0) or 2-layer ReLU MLP with early stopping on an inner 10 % row split of train.
    Returns (pred_train, pred_val)."""
    Xtr, Xva = _standardise(Xtr, Xva)
    ym, ys = float(ytr.mean()), float(ytr.std()) or 1.0
    yt = (ytr - ym) / ys
    if width == 0:
        A = np.hstack([Xtr, np.ones((len(Xtr), 1))])
        lam = 1e-3 * len(Xtr)
        w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ yt)
        return (A @ w) * ys + ym, (np.hstack([Xva, np.ones((len(Xva), 1))]) @ w) * ys + ym
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(Xtr))
    n_in = max(1, len(Xtr) // 10)
    ii, it = perm[:n_in], perm[n_in:]
    d = Xtr.shape[1]
    net = torch.nn.Sequential(torch.nn.Linear(d, width), torch.nn.ReLU(), torch.nn.Linear(width, width),
                              torch.nn.ReLU(), torch.nn.Linear(width, 1))
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    Xt = torch.from_numpy(Xtr.astype(np.float32))
    Yt = torch.from_numpy(yt.astype(np.float32))
    best, best_state, bad = float("inf"), None, 0
    for _ in range(max_epochs):
        net.train()
        order = it[rng.permutation(len(it))]
        for s in range(0, len(order), batch):
            b = torch.from_numpy(order[s:s + batch])
            opt.zero_grad()
            loss = ((net(Xt[b]).squeeze(-1) - Yt[b]) ** 2).mean()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(((net(Xt[ii]).squeeze(-1) - Yt[ii]) ** 2).mean())
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    with torch.no_grad():
        ptr = net(Xt).squeeze(-1).numpy() * ys + ym
        pva = net(torch.from_numpy(Xva.astype(np.float32))).squeeze(-1).numpy() * ys + ym
    return ptr.astype(np.float64), pva.astype(np.float64)


def fit_listwise(X3tr, ttr, X3va, width=64, seed=0, epochs=15, batch=256, lr=1e-3):
    """Shared-weight scorer over replicas, softmax cross-entropy on the teacher's action. Returns val argmax."""
    import torch

    T, N, F = X3tr.shape
    flat_tr, flat_va = _standardise(X3tr.reshape(-1, F), X3va.reshape(-1, F))
    Xt = torch.from_numpy(flat_tr.reshape(T, N, F).astype(np.float32))
    Xv = torch.from_numpy(flat_va.reshape(len(X3va), N, F).astype(np.float32))
    yt = torch.from_numpy(ttr.astype(np.int64))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = torch.nn.Sequential(torch.nn.Linear(F, width), torch.nn.ReLU(), torch.nn.Linear(width, width),
                              torch.nn.ReLU(), torch.nn.Linear(width, 1))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(epochs):
        order = rng.permutation(T)
        for s in range(0, T, batch):
            b = torch.from_numpy(order[s:s + batch])
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(net(Xt[b]).squeeze(-1), yt[b])
            loss.backward()
            opt.step()
    with torch.no_grad():
        return net(Xv).squeeze(-1).argmax(1).numpy()


def cv_ev(X, y, folds, width, seed=0, frac=1.0):
    tr_ev, va_ev = [], []
    rng = np.random.default_rng(seed + 17)
    for f, va in enumerate(folds):
        tr = np.where(~va)[0]
        if frac < 1.0:
            tr = np.sort(rng.choice(tr, max(50, int(len(tr) * frac)), replace=False))
        ptr, pva = fit_regressor(X[tr], y[tr], X[va], width, seed=seed + f)
        tr_ev.append(ev(y[tr], ptr))
        va_ev.append(ev(y[va], pva))
    # headline = median over folds: one unseen-episode fold can extrapolate catastrophically (EV << 0) and
    # dominate the mean; the mean is kept for reference
    return {"train_ev": float(np.mean(tr_ev)), "val_ev": float(np.median(va_ev)), "val_ev_mean": float(np.mean(va_ev)),
            "val_ev_min": float(np.min(va_ev)), "val_ev_folds": [round(v, 4) for v in va_ev]}


# ----------------------------------------------------------------------------- per candidate

def load_candidate(key, runs):
    run, cid = key.split("/")
    R = runs[run]
    row = next(r for r in R["rows"] if r.get("id") == cid)
    d = rl_stats.ckpt_dir(REPO, run, R["run_dir"], cid)
    files = rl_stats.ckpt_files(d)
    kmax, pmax = files[-1]
    buf = Path(str(pmax) + ".buffer.npz")
    eps, _ = rl_stats.episodes_of(row, d)
    design, _ = rl_stats.design_path(REPO, run, R["run_dir"], row)
    return row, buf, eps, design


def _rel(p):
    p = Path(p)
    return str(p.relative_to(REPO)) if str(p).startswith(str(REPO)) else str(p)


def analyse(key, runs, light=False, log=print):
    row, buf, eps, design = load_candidate(key, runs)
    out = {"candidate": key, "buffer": _rel(buf), "fitness": row.get("fitness")}
    return analyse_data(out, buf.parent, buf, eps, design, light, log)


def analyse_dir(d, design=None, light=False, log=print):
    """A standalone training directory (e.g. a smoke run or training/<id>/seed<s>), outside any search archive."""
    d = Path(d)
    files = rl_stats.ckpt_files(d)
    if not files:
        return {"candidate": str(d), "status": "no checkpoints"}
    man = d / "manifest.json"
    if design is None and man.exists():
        design = json.loads(man.read_text()).get("design")
    eps, _ = rl_stats.episodes_of({}, d)
    buf = Path(str(files[-1][1]) + ".buffer.npz")
    out = {"candidate": _rel(d), "buffer": _rel(buf), "fitness": None}
    return analyse_data(out, d, buf, eps, design, light, log)


def analyse_data(out, d, buf, eps, design, light=False, log=print):
    t0 = time.time()
    key = Path(out["candidate"]).name if "/" in out["candidate"] and Path(out["candidate"]).is_absolute() \
        else out["candidate"]
    tr = load_traces(d, len(eps)) if d is not None else None
    who = None
    if tr is not None:  # logged run: every episode's rows with the who-acted flag, exact episode labels
        x3, a, r, ta, g, who, lab = tr["x3"], tr["a"], tr["r"], tr["ta"], tr["g"], tr["who"], tr["lab"]
        out.update({"source": "traces", "trace_frac": tr["trace_frac"]})
    else:
        if not buf.exists():
            out["status"] = "no buffer"
            return out
        z = np.load(buf)
        x3, a, r, ta, g = z["x"], z["a"], z["r"].astype(np.float64), z["ta"], z["g"]
        out["source"] = "buffer"
    T, N, F = x3.shape
    out.update({"T": int(T), "N": int(N), "F": int(F), "algorithm": "bandit" if not np.any(g) else "dqn"})
    if out["algorithm"] != "bandit":
        out["status"] = "skipped: DQN target is bootstrapped (needs the target net; not saved)"
        return out
    if tr is None:
        lg = [(e.get("learner") or {}).get("logging") or {} for e in eps]
        lab = episode_labels_from_ranges([x.get("buffer_start") for x in lg], [x.get("buffer_rows_added") for x in lg],
                                         T, (lg[-1].get("buffer_capacity") if lg else None) or T + 1) if lg else None
        if lab is not None:
            out["labels"] = "logged buffer ranges"
        else:
            ends = [(e.get("learner") or {}).get("buffer_size_end") for e in eps]
            loads = [(e.get("learner") or {}).get("buffer_loaded_from_previous_episode") for e in eps]
            lab = episode_labels(ends, loads, T)
            out["labels"] = "buffer_size_end (valid only without wrap)"
        if lab is None:
            out["status"] = "skipped: episode boundaries not recoverable (buffer wrapped or not reloaded)"
            return out
        sel = np.where(lab >= 0)[0]  # rows of earlier, unlogged runs (never in our episodes) are dropped
        x3, a, r, ta, g, lab = x3[sel], a[sel], r[sel], ta[sel], g[sel], lab[sel]
        T = len(r)
    out["episode_tags"] = [e.get("episode") for e in eps]
    xa = x3[np.arange(T), a]
    if tr is None:  # buffers of v3 runs hold each early-completing decision twice (fixed in v4)
        keep = dedup_index(xa, r)
        out["dedup_removed"] = int(T - len(keep))
        x3, a, r, ta, lab, xa = x3[keep], a[keep], r[keep], ta[keep], lab[keep], xa[keep]
    else:
        keep = np.arange(T)
        out["dedup_removed"] = 0
    if who is not None:
        who = who[keep]
        out["who_frac"] = {n: float((who == i).mean()) for i, n in enumerate(("greedy", "epsilon", "teacher"))}
    T = len(r)
    n_ep = len(np.unique(lab))
    if n_ep < 2:
        out["status"] = "skipped: fewer than 2 episodes (folds hold out whole episodes)"
        return out
    src = Path(design).read_text() if design else ""
    groups = feature_groups(src, F) or [(f"f{i}", [i]) for i in range(F)]
    out["groups"] = {n: idx for n, idx in groups}
    price = next((idx[0] for n, idx in groups if "externality price" in n and len(idx) == 1), None)
    targets = {"r": r}
    if price is not None:
        p_a = x3[np.arange(T), a, price].astype(np.float64)
        targets["r_out"] = r + p_a
        out["price_feature"] = int(price)
        out["price_clipped_frac"] = float((np.abs(p_a) >= 3.0 - 1e-6).mean())
        out["var_share_of_price"] = float(np.var(p_a) / np.var(r)) if np.var(r) > 0 else None
    smask = state_feature_mask(x3)
    out["state_features"] = [int(i) for i in np.where(smask)[0]]
    folds = group_folds(lab, min(4, n_ep))
    out["fold_rows"] = [int(f.sum()) for f in folds]
    main_t = "r_out" if "r_out" in targets else "r"

    # representation-power curve
    widths = [0, 64, 256] if light else WIDTHS
    rep = {}
    for tn, y in targets.items():
        rep[tn] = {}
        for w in widths:
            rep[tn][str(w)] = cv_ev(xa, y, folds, w)
            log(f"  {key} rep {tn} w={w}: {rep[tn][str(w)]['train_ev']:.3f} / {rep[tn][str(w)]['val_ev']:.3f}")
    out["representation"] = rep
    # in-distribution reference: random-row 4-fold (rows of the same episode on both sides)
    rr = np.random.default_rng(1).integers(0, 4, T)
    out["representation_rowsplit_w64"] = {tn: cv_ev(xa, y, [rr == k for k in range(4)], 64)
                                          for tn, y in targets.items()}
    vals = {int(w): v["val_ev"] for w, v in rep[main_t].items()}
    top = max(vals.values())
    best_w = min(w for w, v in vals.items() if v >= top - 0.01)
    out["best_width"] = best_w

    # noise ceiling
    out["noise"] = {tn: gamma_test(xa.astype(np.float64), y) for tn, y in targets.items()}
    out["noise_within_episode"] = {}
    for tn, y in targets.items():  # per-episode Gamma tests remove the episode-level shift
        vs = [gamma_test(xa[lab == k].astype(np.float64), y[lab == k], n_max=8000) for k in np.unique(lab)
              if (lab == k).sum() > 500]
        w = np.array([v["n"] for v in vs], float)
        out["noise_within_episode"][tn] = {"ceiling_ev_mean": float(np.average([v["ceiling_ev"] for v in vs],
                                                                                 weights=w)) if vs else None,
                                           "n_episodes": len(vs)}
    if light:
        out["seconds"] = round(time.time() - t0, 1)
        out["status"] = "ok (light)"
        return out

    # drop-one-group ablation (main target, best width)
    base = rep[main_t][str(best_w)]["val_ev"]
    abl = {}
    for n, idx in groups:
        cols = np.setdiff1d(np.arange(F), idx)
        res = cv_ev(xa[:, cols], targets[main_t], folds, best_w)
        abl[n] = {"n_features": len(idx), "val_ev": res["val_ev"], "delta": res["val_ev"] - base}
        log(f"  {key} ablate {n}: {abl[n]['delta']:+.3f}")
    out["ablation"] = {"target": main_t, "width": best_w, "base_val_ev": base, "groups": abl}

    # information beyond the rule (rows where the teacher's choice was recorded)
    ht = ta >= 0
    out["teacher_coverage"] = float(ht.mean())
    rows_used = "all rows with a teacher action (confounded by who acted)"
    if who is not None:  # epsilon-random actions: outcomes are unbiased counterfactuals of the rule's choice
        he = ht & (who == 1)
        if he.sum() >= MIN_EPS_ROWS and all(he[f].any() and he[~f].any() for f in folds):
            ht = he
            rows_used = "epsilon-explored rows only (who acted = epsilon)"
        else:
            rows_used += f"; only {int(he.sum())} epsilon rows (< {MIN_EPS_ROWS} or a fold without any)"
    xs, ys_ = xa[ht], targets[main_t][ht]
    is_t = (a[ht] == ta[ht]).astype(np.float64)[:, None]
    fh = [f[ht] for f in folds]
    st_cols = np.where(smask)[0]
    beyond = {"rows": rows_used, "n_rows": int(ht.sum())}
    for name, X in (("state_only", xs[:, st_cols]), ("state+is_teacher", np.hstack([xs[:, st_cols], is_t])),
                    ("all_features", xs), ("all+is_teacher", np.hstack([xs, is_t]))):
        beyond[name] = cv_ev(X, ys_, fh, best_w)["val_ev"]
        log(f"  {key} beyond {name}: {beyond[name]:.3f}")
    beyond["executed_eq_teacher"] = float(is_t.mean())
    # mean target when the executed action equals vs differs from the rule's (confounded by who acted)
    beyond["mean_target_if_teacher_action"] = float(ys_[is_t[:, 0] == 1].mean())
    beyond["mean_target_if_other_action"] = float(ys_[is_t[:, 0] == 0].mean())
    out["beyond_rule"] = {"target": main_t, "width": best_w, **beyond}

    # can the features reproduce the rule? listwise classifier; and the reward model's argmax vs the rule
    ht = ta >= 0  # all rows with a teacher action: imitation is about states, not about who acted
    xs, ys_ = xa[ht], targets[main_t][ht]
    fh = [f[ht] for f in folds]
    x3h, tah = x3[ht], ta[ht]
    acc, agree = [], []
    for f, va in enumerate(fh):
        pred = fit_listwise(x3h[~va], tah[~va], x3h[va], seed=f)
        acc.append(float((pred == tah[va]).mean()))
        flat_tr = xa[ht][~va]
        _, pall = fit_regressor(flat_tr, ys_[~va], x3h[va].reshape(-1, F), best_w, seed=f)
        greedy = pall.reshape(-1, N).argmax(1)
        agree.append(float((greedy == tah[va]).mean()))
    out["teacher_imitation"] = {"listwise_val_acc": float(np.mean(acc)), "chance": 1.0 / N,
                                "reward_model_argmax_eq_teacher": float(np.mean(agree))}
    log(f"  {key} imitation {np.mean(acc):.3f}, reward-argmax agreement {np.mean(agree):.3f}")

    # data sufficiency
    out["data_curve"] = {"target": main_t, "width": best_w,
                         "points": {str(fr): cv_ev(xa, targets[main_t], folds, best_w, frac=fr)["val_ev"]
                                    for fr in FRACTIONS}}
    out["seconds"] = round(time.time() - t0, 1)
    out["status"] = "ok"
    return out


# ----------------------------------------------------------------------------- report

def fmt(v, nd=3):
    return "—" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{nd}f}"


def to_markdown(res):
    L = [f"# RL offline diagnostics ({res['generated']})", "",
         "Supervised re-fits of the regression each bandit learner faced, on its saved replay buffer "
         "(duplicates from the double-store defect removed). Splits hold out whole training episodes "
         "(4 folds); EV = explained variance on held-out episodes (median over folds; worst fold in brackets) "
         "unless marked train. "
         "`r` = stored reward (what the bandit regressed); `r_out` = r + price[a], the outcome part that is not "
         "a known function of the features. Ceiling = 1 - Gamma-test noise / Var. "
         "Generated by `python -m evolve.analysis.rl_offline`.", ""]
    for key, c in res["candidates"].items():
        L += [f"## {key}", ""]
        if not c.get("status", "").startswith("ok"):
            L += [f"Status: {c.get('status')}", ""]
            continue
        src = c.get("source", "buffer")
        L += [f"Source: {src}" + (f" (trace_frac {c.get('trace_frac')})" if src == "traces" else
                                   f" (episode labels from {c.get('labels', 'buffer_size_end')})")
              + (", who acted: " + ", ".join(f"{k} {fmt(100 * v, 1)} %" for k, v in c["who_frac"].items())
                 if c.get("who_frac") else "") + ".", ""]
        L += [f"T = {c['T']} decisions after removing {c['dedup_removed']} duplicates, N = {c['N']} replicas, "
              f"F = {c['F']} features ({len(c['state_features'])} state-only), fitness {fmt(c.get('fitness'))}. "
              + (f"Price feature #{c['price_feature']}: {fmt(100 * c['var_share_of_price'], 1)} % of Var(r); "
                 f"clipped on {fmt(100 * c['price_clipped_frac'], 1)} % of rows." if 'price_feature' in c else ""), ""]
        L += ["**Representation power** (train EV / held-out-episode EV)", "",
              "| target | " + " | ".join("linear" if w == "0" else f"MLP {w}" for w in c["representation"]["r"]) +
              " | row-split MLP 64 | ceiling (pooled) | ceiling (within episode) |",
              "|---" * (len(c["representation"]["r"]) + 4) + "|"]
        for tn, rep in c["representation"].items():
            rs = c["representation_rowsplit_w64"][tn]
            L.append(f"| {tn} | " + " | ".join(f"{fmt(v['train_ev'])} / **{fmt(v['val_ev'])}** [{fmt(v['val_ev_min'], 2)}]" for v in rep.values())
                     + f" | {fmt(rs['val_ev'])} | {fmt(c['noise'][tn]['ceiling_ev'])} | "
                       f"{fmt(c['noise_within_episode'][tn]['ceiling_ev_mean'])} |")
        L.append("")
        if "ablation" in c:
            ab = c["ablation"]
            L += [f"**Drop-one-group ablation** ({ab['target']}, width {ab['width']}, base held-out EV "
                  f"{fmt(ab['base_val_ev'])})", "", "| group | features | ΔEV |", "|---|---|---|"]
            for n, v in sorted(ab["groups"].items(), key=lambda kv: kv[1]["delta"]):
                L.append(f"| {n} | {v['n_features']} | {v['delta']:+.3f} |")
            L.append("")
        if "beyond_rule" in c:
            b = c["beyond_rule"]
            L += [f"**Information beyond the rule** ({b['target']}, held-out EV, teacher coverage "
                  f"{fmt(100 * c['teacher_coverage'], 1)} %; rows: {b.get('rows', 'all rows with a teacher action')}"
                  + (f", n = {b['n_rows']}" if 'n_rows' in b else "") + ")", "",
                  "| state only | state + is-teacher-action | all features | all + is-teacher-action |",
                  "|---|---|---|---|",
                  f"| {fmt(b['state_only'])} | {fmt(b['state+is_teacher'])} | {fmt(b['all_features'])} | "
                  f"{fmt(b['all+is_teacher'])} |", "",
                  f"Executed action = rule's on {fmt(100 * b['executed_eq_teacher'], 1)} % of rows; mean "
                  f"{b['target']} {fmt(b['mean_target_if_teacher_action'])} when it was, "
                  f"{fmt(b['mean_target_if_other_action'])} when not"
                  + (" (unconfounded: epsilon rows)." if str(b.get("rows", "")).startswith("epsilon")
                     else " (confounded by who acted)."), ""]
            ti = c["teacher_imitation"]
            L += [f"Rule reproducible from the features: listwise classifier held-out accuracy "
                  f"{fmt(ti['listwise_val_acc'])} (chance {fmt(ti['chance'])}). Fitted reward model's argmax = "
                  f"rule's action on {fmt(ti['reward_model_argmax_eq_teacher'])} of decisions.", ""]
        if "data_curve" in c:
            d = c["data_curve"]
            L += [f"**Data sufficiency** ({d['target']}, width {d['width']}): held-out EV at "
                  + ", ".join(f"{int(100 * float(k))} % rows {fmt(v)}" for k, v in d["points"].items()) + ".", ""]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cand", action="append", help="run/id (default: the two v3.1 finalists)")
    ap.add_argument("--sweep", type=int, default=0, help="also run the light analysis on the top-N valid "
                                                          "candidates (by fitness) of each finalist's run")
    ap.add_argument("--train-dir", action="append", help="a standalone training dir (net_ep*.pt [+ traces]); "
                                                         "replaces the default finalists")
    ap.add_argument("--design", help="design file for --train-dir (default: the manifest's design)")
    ap.add_argument("--light", action="store_true", help="representation + noise only")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--out", default=str(REPO / "evolve/analysis/out/rl_offline"))
    args = ap.parse_args(argv)
    import torch

    torch.set_num_threads(args.threads)
    res = {"generated": datetime.now().strftime("%Y-%m-%d %H:%M"), "candidates": {}}
    if args.train_dir:
        for d in args.train_dir:
            print(f"[rl_offline] {d}")
            res["candidates"][d] = analyse_dir(d, args.design, light=args.light)
        keys = []
    else:
        runs = rl_stats.load_runs(rl_stats.discover(REPO))
        keys = args.cand or FINALISTS
    for k in keys:
        print(f"[rl_offline] {k}")
        res["candidates"][k] = analyse(k, runs, light=args.light)
    if args.sweep:
        for run in sorted({k.split("/")[0] for k in keys}):
            rows = [r for r in runs[run]["rows"] if r.get("valid") and r.get("fitness") is not None]
            rows.sort(key=lambda r: -r["fitness"])
            for r in rows[:args.sweep]:
                k = f"{run}/{r['id']}"
                if k in res["candidates"]:
                    continue
                print(f"[rl_offline] light {k}")
                try:
                    res["candidates"][k] = analyse(k, runs, light=True)
                except Exception as e:
                    res["candidates"][k] = {"status": f"error: {type(e).__name__}: {e}"}
    out = Path(args.out)
    out.with_suffix(".json").write_text(json.dumps(res, indent=1, default=float))
    out.with_suffix(".md").write_text(to_markdown(res))
    print(f"wrote {out}.json / .md")


if __name__ == "__main__":
    main()
