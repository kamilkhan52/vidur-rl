"""Lodestar (Lim et al., arXiv 2606.00946v1), ported from github.com/gangmuk/Lodestar @ 23da2a0b (Apache-2.0).
Notes with file:line sources: ~/vidur-rl-priorwork/notes/lodestar.md. Policy files call ``build(offline_rows)``.

Per request: per-replica features x_i (inflight prefill / decode requests, KV usage, running, waiting, prefill
tokens, decode tokens, fresh prefix-hit ratio, prefix-hit %, input tokens) -> a shared MLP predicts -TTFT for every
replica -> argmax; when the cluster's KV is saturated and the request has a worthwhile prefix, the choice is
restricted to K = 2 replicas picked by a hash of the prefix group. Before the first model, or when the request is out
of the training range, AIBrix's prefix-cache-and-load rule decides. The MLP (3 x 128, ReLU, dropout 0.1) is
retrained from scratch whenever >= 1000 new samples have arrived (checked every 30 s), on the newest 5000 samples
plus a k-centre replay set of older ones; the target is -TTFT of the chosen replica.

Configuration (paper values where the paper gives one, else the checked-in experiment driver RUN):
K = 2, tau_sat = 0.8 (paper), tau_ben = 100 tokens, epsilon = 0 (RUN), OOD fallback on (paper), prefix TTL 10 min
(RUN), retrain threshold 1000 (paper), first training at 5000 samples, flush every 10 s in batches >= 100.
Deviations (EXP33 PREREG): offline rows from training windows are loaded and trained on at start-up (RUN's
RETRAIN_AT_STARTUP; the paper says online only, but our 2000-request windows never reach 5000 online samples);
training takes no simulated time (the model goes live at the next 30 s tick); decode tokens per replica are
estimated from elapsed decode time and an EWMA of observed TPOT (no per-token events in the harness); outcomes are
joined to their route-time features by (arrival time, prompt tokens, session, replica); prefix blocks are the
harness's prompt blocks, not tiktoken tokens, and the prefix group is the first block; dropout is off at inference
(the code leaves it on, an apparent bug); route-time features are used for training too (the code logs post-routing
counters); the GPU one-hot is dropped (homogeneous replicas); all randomness comes from ctx.rng.
"""
import math
import pickle
from collections import deque

import numpy as np
import torch

torch.set_num_threads(1)

K, TAU_SAT, TAU_BEN, EPS = 2, 0.8, 100, 0.0
OOD_FALLBACK = True
TTL_S, HALF_LIFE_S = 600.0, 15.0
N_MIN, THETA, FIFO, STORE_CAP = 5000, 1000, 5000, 10000
POLL_S, FLUSH_S, FLUSH_MIN = 30.0, 10.0, 100
TTFT_MAX = 15.0
EPOCHS, BATCH, LR, WD, GAMMA, CLIP, HOLDOUT = 50, 256, 3e-4, 1e-5, 0.95, 1.0, 0.2
IMBAL, STD_K = 8, 1.0
D = 10  # features per replica


class _MLP(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.f = torch.nn.Sequential(
            torch.nn.Linear(d, 128), torch.nn.ReLU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(128, 128), torch.nn.ReLU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(128, 128), torch.nn.ReLU())
        self.out = torch.nn.Linear(128, 1)
        for m in list(self.f) + [self.out]:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0.01)

    def forward(self, x):
        return self.out(self.f(x)).squeeze(-1)


class Lodestar:
    def __init__(self, offline_rows=None):
        self.offline = offline_rows
        self.dump = None
        self.ready = False

    # ------------------------------------------------------------------ state
    def _init(self, ctx, n):
        self.n = n
        self.prefix = {}  # block hash -> {replica: last route time}
        self.inflight_prefill = [0] * n
        self.prefill_tokens = [0] * n
        self.decoding = [dict() for _ in range(n)]  # replica -> {key: first_token_at}
        self.tpot = [0.02] * n
        self.pending = {}  # join key -> deque of (features, action)
        self.done = []  # rows waiting for the next flush: (X, a, ttft)
        self.wait_ttft = {}  # join key -> deque of (X, a, ttft) awaiting the request's end
        self.history = []  # training store rows (X [n x D], a, ttft)
        self.new_since = 0
        self.model, self.stats = None, None
        self.next_poll, self.next_flush = POLL_S, FLUSH_S
        self.gen = torch.Generator().manual_seed(ctx.rng.getrandbits(31))
        self.nprng = np.random.default_rng(ctx.rng.getrandbits(31))
        self.dumped = []
        if self.offline is not None and len(self.offline):
            self.history = list(self.offline)
            self._train()  # RETRAIN_AT_STARTUP
        self.ready = True

    # --------------------------------------------------------------- features
    def _hits(self, request, now):
        hashes = request.prefix_hashes
        nb = len(hashes)
        hit, fresh = [0] * self.n, [0.0] * self.n
        if not nb:
            return hit, fresh
        alive = [True] * self.n
        matched, last = [0] * self.n, [None] * self.n
        for h in hashes:
            owners = self.prefix.get(h)
            if not owners:
                break
            any_alive = False
            for i in range(self.n):
                if alive[i]:
                    t = owners.get(i)
                    if t is not None and now - t <= TTL_S:
                        matched[i] += 1
                        last[i] = t
                        any_alive = True
                    else:
                        alive[i] = False
            if not any_alive:
                break
        for i in range(self.n):
            hit[i] = matched[i] * 100 // nb
            age = (now - last[i]) if last[i] is not None else 999.0
            fresh[i] = hit[i] * math.exp(-math.log(2) * age / HALF_LIFE_S)
        return hit, fresh

    def _decode_tokens(self, i, now):
        tp = max(self.tpot[i], 1e-4)
        return sum(1 + int((now - t) / tp) for t in self.decoding[i].values())

    def _features(self, request, replicas, now):
        hit, fresh = self._hits(request, now)
        X = np.empty((self.n, D), np.float32)
        for i, r in enumerate(replicas):
            X[i] = (self.inflight_prefill[i], len(self.decoding[i]), r.kv_usage, r.running, r.waiting,
                    self.prefill_tokens[i], self._decode_tokens(i, now), fresh[i], hit[i], request.num_prefill_tokens)
        return X, hit

    # ---------------------------------------------------------------- routing
    def _fallback(self, request, replicas, hit, rng):
        c = [r.outstanding_requests for r in replicas]
        if max(c) - min(c) > IMBAL:
            lo = min(c)
            return rng.choice([i for i in range(self.n) if c[i] == lo])
        order = list(range(self.n))
        rng.shuffle(order)
        order.sort(key=lambda i: (-hit[i], c[i]))
        mu = sum(c) / self.n
        sd = math.sqrt(sum((x - mu) ** 2 for x in c) / self.n)
        for i in order:
            if c[i] <= mu + STD_K * sd:
                return i
        lo = min(c)
        return rng.choice([i for i in range(self.n) if c[i] == lo])

    def _ood(self, X):
        lo, hi = self.stats["min"], self.stats["max"]
        tok = X[0, D - 1]
        return bool(tok < lo[D - 1] or tok > hi[D - 1] or np.any(X[0] < lo) or np.any(X[0] > hi))

    def _candidates(self, group):
        cand = []
        for j in range(K):
            c = hash((j, group)) % self.n
            while c in cand:
                c = (c + 1) % self.n
            cand.append(c)
        return cand

    def choose(self, request, replicas, ctx):
        if not self.ready:
            self._init(ctx, len(replicas))
        self._tick(ctx.now)
        X, hit = self._features(request, replicas, ctx.now)
        if self.model is None or (OOD_FALLBACK and self._ood(X)):
            a = self._fallback(request, replicas, hit, ctx.rng)
        else:
            with torch.no_grad():
                y = self.model(torch.from_numpy((X - self.stats["mu"]) / self.stats["sd"])).numpy()
            explored = EPS > 0 and ctx.rng.random() < EPS
            a = ctx.rng.randrange(self.n) if explored else int(np.argmax(y))
            group = request.prefix_hashes[0] if request.prefix_hashes else 0
            mean_kv = sum(r.kv_usage for r in replicas) / self.n
            if (not explored and K > 0 and mean_kv > TAU_SAT and group != 0
                    and max(hit) / 100 * request.num_prefill_tokens > TAU_BEN):
                cand = self._candidates(group)
                if a not in cand:
                    a = max(cand, key=lambda i: y[i])
        for h in request.prefix_hashes:
            self.prefix.setdefault(h, {})[a] = ctx.now
        self.inflight_prefill[a] += 1
        self.prefill_tokens[a] += request.num_prefill_tokens
        key = (request.arrived_at, request.num_prefill_tokens, request.session_id, a)
        self.pending.setdefault(key, deque()).append(X)
        return a

    def observe(self, outcome, ctx):
        if not self.ready:
            return
        a = outcome.replica
        key = (outcome.arrived_at, outcome.num_prefill_tokens, outcome.session_id, a)
        if outcome.event == "first_token":
            self.inflight_prefill[a] -= 1
            self.prefill_tokens[a] -= outcome.num_prefill_tokens
            self.decoding[a][(key, outcome.first_token_at)] = outcome.first_token_at
            q = self.pending.get(key)
            if q:
                X = q.popleft()
                if not q:
                    del self.pending[key]
                self.wait_ttft.setdefault(key, deque()).append((X, a, outcome.first_token_at - outcome.arrived_at))
        else:  # end: the row is logged at end of stream
            self.decoding[a].pop((key, outcome.first_token_at), None)
            if outcome.num_decode_tokens and outcome.num_decode_tokens > 1:
                tp = (outcome.completed_at - outcome.first_token_at) / (outcome.num_decode_tokens - 1)
                self.tpot[a] = 0.9 * self.tpot[a] + 0.1 * tp
            q = self.wait_ttft.get(key)
            if q:
                self.done.append(q.popleft())
                if not q:
                    del self.wait_ttft[key]
        self._tick(ctx.now)

    # --------------------------------------------------------------- learning
    def _tick(self, now):
        while now >= self.next_flush:
            if len(self.done) >= FLUSH_MIN:
                self.history.extend(self.done)
                self.new_since += len(self.done)
                if self.dump is not None:  # collection runs only: append this flush's rows (one pickle per flush)
                    with open(self.dump, "ab") as f:
                        pickle.dump(self.done, f)
                self.done = []
            self.next_flush += FLUSH_S
        while now >= self.next_poll:
            if len(self.history) >= N_MIN and (self.model is None or self.new_since >= THETA):
                self._train()
            self.next_poll += POLL_S

    def _embed(self, model, Xa):
        with torch.no_grad():
            h = model.f(torch.from_numpy(Xa))
            return h.numpy(), model.out(h).squeeze(-1).numpy()

    def _replay(self, older, k):
        """Greedy k-centre on cosine distance of last-hidden activations x |residual| (previous model)."""
        if self.model is None or k >= len(older):
            idx = self.nprng.permutation(len(older))[:k]
            return [older[i] for i in idx]
        Xa = np.stack([(r[0][r[1]] - self.stats["mu"]) / self.stats["sd"] for r in older]).astype(np.float32)
        h, pred = self._embed(self.model, Xa)
        y = -np.array([r[2] for r in older], np.float32)
        e = h * np.abs(y - pred)[:, None]
        e /= np.linalg.norm(e, axis=1, keepdims=True) + 1e-12
        first = int(np.argmax(np.linalg.norm(h * np.abs(y - pred)[:, None], axis=1)))
        chosen = [first]
        dist = 1.0 - e @ e[first]
        for _ in range(k - 1):
            j = int(np.argmax(dist))
            chosen.append(j)
            dist = np.minimum(dist, 1.0 - e @ e[j])
        return [older[i] for i in chosen]

    def _train(self):
        fifo = self.history[-FIFO:]
        older = self.history[:-FIFO] if len(self.history) > FIFO else []
        replay = self._replay(older, min(STORE_CAP - len(fifo), len(older))) if older else []
        window = replay + fifo
        self.history = list(window)
        self.new_since = 0
        rows = [r for r in window if r[2] <= TTFT_MAX]
        if len(rows) < 10:
            return
        allX = np.concatenate([r[0] for r in rows])  # pooled over every replica's row
        mu, sd = allX.mean(0), allX.std(0) + 1e-6
        stats = {"mu": mu.astype(np.float32), "sd": sd.astype(np.float32), "min": allX.min(0), "max": allX.max(0)}
        Xa = np.stack([(r[0][r[1]] - mu) / sd for r in rows]).astype(np.float32)
        y = -np.array([r[2] for r in rows], np.float32)
        perm = self.nprng.permutation(len(rows))
        tr = perm[int(HOLDOUT * len(rows)):]
        torch.manual_seed(int(self.nprng.integers(1 << 31)))
        model = _MLP(D)
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, GAMMA)
        Xt, yt = torch.from_numpy(Xa[tr]), torch.from_numpy(y[tr])
        model.train()
        for _ in range(EPOCHS):
            order = torch.randperm(len(tr), generator=self.gen)
            for s in range(0, len(tr), BATCH):
                b = order[s:s + BATCH]
                opt.zero_grad()
                loss = torch.nn.functional.mse_loss(model(Xt[b]), yt[b])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
                opt.step()
            sched.step()
        model.eval()  # deviation: the code leaves dropout on at inference
        self.model, self.stats = model, stats


def build(offline_rows=None, dump=None):
    """Returns (choose, observe, instance), one instance per simulation process. ``dump``: a path to which the rows
    logged so far are written at every flush (collection runs only)."""
    inst = Lodestar(offline_rows)
    if dump is not None:
        inst.dump = dump

    def choose(request, replicas, ctx):
        return inst.choose(request, replicas, ctx)

    def observe(outcome, ctx):
        inst.observe(outcome, ctx)

    return choose, observe, inst
