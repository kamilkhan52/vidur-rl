"""Astrolabe / Block (Da & Kalyvianaki, arXiv 2508.03611v3, SYSTOR '26), routing rule ported from
github.com/AKafakA/Block @ b2141b3e (MIT). Notes with file:line sources: ~/vidur-rl-priorwork/notes/astrolabe.md.
Policy files call ``build(spec)`` with a per-device ``ReplicaSpec``.

Rule (block/global_scheduler/api_server.py:92-282, block/predictor/simulate_predictor.py,
vidur/scheduler/replica_scheduler/simulate_predict_replica_scheduler.py): power of two choices. For each of k = 2
replicas sampled without replacement, rebuild that replica's scheduler from its current requests, add the new
request last, and simulate forward (batch by batch, each batch timed by Vidur's execution-time predictor) until the
new request completes; its completion time from now is the predicted E2E latency. The replica with the lower
prediction wins; ties at random. A sampled replica with no free KV block is dropped (api_server.py:131); if both are
dropped, a random replica over all of them (api_server.py:148-161). All randomness comes from ctx.rng.

Inputs (what a real Astrolabe router would have):
- its own ledger of every request it routed to each replica: prompt tokens, cached prefix tokens at routing
  (request.cached_prefix_tokens[i]; replaced by the exact count at first token), arrival, first-token time and
  completion (observe()); outcomes carry no request id, so they are matched on (replica, arrival, prompt tokens),
  first in first out among equal keys;
- the replica views (only kv_usage, for the free-block filter);
- Vidur's own execution-time predictor for the same device and model, built from the same profiles; it loads the
  run's prediction bundle (checked by key; it never trains), pruned to the keys this configuration can reach.

Output lengths are withheld. Each request's length estimate L is the running mean of num_decode_tokens over the
requests completed so far (PRIOR_DECODE_TOKENS = 256 before the first completion), fixed when the request is routed
(Astrolabe passes one L per request at routing). As in simulate_predictor.py:55-56, a running request that has
decoded d >= L tokens is assumed to end within 10 more (d + 10). Decoded-so-far d is dead-reckoned:
1 + (now - first_token_at) / TPOT_r, with TPOT_r an EWMA (alpha 0.1) of observed (completed - first_token) /
(decode tokens - 1) on that replica (else over all replicas; prior 0.02 s).

The replica simulation reuses this repo's VLLMV1ReplicaScheduler unchanged (token budget = chunk size, chunked
prefill, batch-size cap, FCFS waiting queue, KV block allocation and preemption), with one KV-cache manager without
prefix caching whose cache lookup returns the router's cached-prefix count for each request. Requests that have
produced their first token are put in the running set (arrival order) with their KV blocks allocated; the others
wait, not started, in arrival order; the new request arrives last. The iteration in progress on the GPU is ignored
(the simulation starts at 0 with fresh batches, as in Astrolabe). Batch times are memoised on the exact features
the predictor reads, so a cache hit gives the same time as a call.

Requests of one arrival burst use choose(): later ones see the earlier ones in the ledger.
Deviations from Astrolabe's code: see DEVIATIONS below.
"""
import atexit
import gc
import json
import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]

K = 2
PRIOR_DECODE_TOKENS = 256
OVERRUN_TOKENS = 10  # simulate_predictor.py:55-56
TPOT_PRIOR_S, TPOT_ALPHA = 0.02, 0.1
MAX_SIM_ITERATIONS = 200_000  # a probe that runs this long returns +inf (never reached in the smoke run)

DEVIATIONS = (
    "D1 length estimate: running mean of observed decode tokens (prior 256), not RoBERTa on the prompt text",
    "D2 replica state from the router's ledger, not vLLM's /schedule_trace: decode progress dead-reckoned from TPOT, "
    "chunk progress and preemptions unknown (unprefilled requests are treated as not started)",
    "D3 timing model = the simulator's own execution-time predictor (no model error)",
    "D4 backend: vllm_v1 scheduler with our cap/chunk/KV, not vLLM 0.7.2 V0",
    "D5 prefix cache: cached prefix tokens are skipped in the simulated prefill (Astrolabe ignores the cache)",
    "D6 bursts: sequential choose(), later requests of a burst see earlier ones",
    "D7 decision overhead is measured, not charged to the simulated requests",
    "D8 exact batch-time memo instead of Astrolabe's (size, first id, last id) key; no RPC timeouts, auto-scaling "
    "or profiling dispatch; the in-flight GPU iteration is ignored as in Astrolabe",
)


@dataclass(frozen=True)
class ReplicaSpec:
    """The deployment a policy file simulates (must equal the run's; checked against sys.argv when present)."""
    device: str  # Vidur device name
    network_device: str
    model_name: str
    tensor_parallel_size: int
    num_blocks: int
    block_size: int
    batch_size_cap: int
    chunk_size: int
    attention_input_file: str  # as passed to --random_forest_execution_time_predictor_config_attention_input_file
    bundle_key: str  # prediction_bundle_<key>.pkl the run uses


def _check_argv(spec: ReplicaSpec):
    """When imported inside a vidur.main run, fail loudly if the run's deployment differs from the spec."""
    argv = sys.argv
    want = {
        "--replica_config_device": spec.device,
        "--replica_config_network_device": spec.network_device,
        "--replica_config_model_name": spec.model_name,
        "--replica_config_tensor_parallel_size": str(spec.tensor_parallel_size),
        "--cache_config_num_blocks": str(spec.num_blocks),
        "--cache_config_block_size": str(spec.block_size),
        "--replica_scheduler_config_type": "vllm_v1",
        "--vllm_v1_scheduler_config_batch_size_cap": str(spec.batch_size_cap),
        "--vllm_v1_scheduler_config_chunk_size": str(spec.chunk_size),
        "--random_forest_execution_time_predictor_config_attention_input_file": spec.attention_input_file,
    }
    for flag, val in want.items():
        if flag in argv:
            i = argv.index(flag)
            got = argv[i + 1] if i + 1 < len(argv) else None
            if got != val:
                raise RuntimeError(f"astrolabe: run has {flag} {got!r}, policy spec has {val!r}")


# --------------------------------------------------------------------------------------------- timing model
def build_predictor(spec: ReplicaSpec, prune: bool = True):
    """Vidur's execution-time predictor for the spec's deployment, from the run's prediction bundle."""
    from vidur.config import CacheConfig, RandomForestExecutionTimePredictorConfig, ReplicaConfig
    from vidur.execution_time_predictor.random_forest_execution_time_predictor import (
        RandomForestExecutionTimePredictor,
    )
    cache_dir = REPO / "cache"
    bundle = cache_dir / f"prediction_bundle_{spec.bundle_key}.pkl"
    if not bundle.exists():
        raise RuntimeError(f"astrolabe: prediction bundle {bundle} missing (the policy never trains forests)")
    rc = ReplicaConfig(model_name=spec.model_name, tensor_parallel_size=spec.tensor_parallel_size,
                       device=spec.device, network_device=spec.network_device)
    cc = CacheConfig(block_size=spec.block_size, num_blocks=spec.num_blocks, enable_prefix_caching=False)
    pc = RandomForestExecutionTimePredictorConfig(attention_input_file=spec.attention_input_file,
                                                  cache_dir=str(cache_dir))
    pred = RandomForestExecutionTimePredictor(predictor_config=pc, replica_config=rc, cache_config=cc)
    key = pred.provenance()["bundle_key"]
    if key != spec.bundle_key or pred.bundle_source.get("source") != "bundle":
        raise RuntimeError(f"astrolabe: predictor key {key} / source {pred.bundle_source} != bundle {spec.bundle_key}")
    if prune:
        _prune(pred, spec)
    return pred, rc, cc


def _prune(pred, spec: ReplicaSpec):
    """Keep the keys this deployment can reach (3.3 GB -> ~0.1 GB). A batch has at most chunk_size tokens (the
    token budget) and at most batch_size_cap requests; the prefill chunk feature sqrt(sum chunk^2) <= chunk_size.
    Lookups beyond these would raise KeyError (never clamp silently); tests check equality with the full table."""
    tok_max = spec.chunk_size + 64
    p = pred._predictions
    out = {}
    for name, table in p.items():
        if name == "attn_decode":
            out[name] = {k: v for k, v in table.items() if k[0] <= spec.batch_size_cap}
        elif name == "attn_prefill":
            out[name] = {k: v for k, v in table.items() if k[1] <= spec.chunk_size + 32}
        elif name in ("schedule", "sampler_e2e", "prepare_inputs_e2e", "process_model_outputs", "ray_comm_time"):
            out[name] = {k: v for k, v in table.items() if k[0] <= spec.batch_size_cap}
        else:  # tables keyed by (total tokens,)
            out[name] = {k: v for k, v in table.items() if k[0] <= tok_max}
    # The predictor clamps decode/prefill lookups to the table maxima; keep those maxima as in the full table.
    pred._attn_decode_max_batch = max(k[0] for k in p["attn_decode"])
    pred._attn_decode_max_kv = max(k[1] for k in p["attn_decode"])
    pred._attn_prefill_max_kv = max(k[0] for k in p["attn_prefill"])
    pred._attn_prefill_max_chunk = max(k[1] for k in p["attn_prefill"])
    pred._predictions = out
    del p, table
    gc.collect()


class BatchTimer:
    """predictor.get_batch_execution_time(batch, 0).total_time, memoised on every feature the predictor reads."""

    def __init__(self, predictor):
        from vidur.execution_time_predictor.sklearn_execution_time_predictor_batch import (
            SklearnExecutionTimePredictorBatch,
        )
        self._p = predictor
        self._B = SklearnExecutionTimePredictorBatch
        self._g = (predictor._config.kv_cache_prediction_granularity,
                   predictor._config.prefill_chunk_size_prediction_granularity)
        self._memo = {}
        self.calls = self.misses = 0

    def __call__(self, batch) -> float:
        self.calls += 1
        sb = self._B.from_batch(batch, *self._g)
        key = (sb._total_num_tokens, sb._size, sb._decode_batch_size, getattr(sb, "_decode_avg_kv_cache_size", 0),
               sb._prefill_batch_size, getattr(sb, "_prefill_agg_kv_cache_size", 0),
               getattr(sb, "_prefill_agg_chunk_size", 0))
        t = self._memo.get(key)
        if t is None:
            self.misses += 1
            t = self._memo[key] = self._p.get_batch_execution_time(batch, 0).total_time
        return t


# ----------------------------------------------------------------------------------------- replica simulation
class CountingKV:
    """KVCacheManager.allocate_slots / free (vidur/kv_cache/base_kv_cache_manager.py) without prefix caching, with
    block counts instead of block objects: the same admission and preemption decisions, ~3x cheaper per iteration.
    get_computed_blocks is set by ReplicaSimulator. Tests check equal predictions against the real manager."""

    enable_caching = False

    def __init__(self, num_blocks, block_size, num_preallocate_tokens):
        self.num_gpu_blocks = num_blocks
        self.block_size = block_size
        self.num_preallocate_blocks = -(-num_preallocate_tokens // block_size)
        self.free_blocks = num_blocks
        self.req_blocks = {}

    @property
    def usage(self):
        return 1.0 - self.free_blocks / self.num_gpu_blocks

    def allocate_slots(self, request, num_tokens, new_computed_blocks=None):
        if num_tokens == 0:
            raise ValueError("num_tokens must be greater than 0")
        assert not new_computed_blocks
        have = self.req_blocks.get(request.id, 0)
        num_new = -(-(request.num_processed_tokens + num_tokens) // self.block_size) - have
        if num_new > self.free_blocks:
            return None
        if num_new > 0:
            num_new = min(num_new + self.num_preallocate_blocks, self.free_blocks)
            self.free_blocks -= num_new
            self.req_blocks[request.id] = have + num_new
        return ()

    def free(self, request):
        self.free_blocks += self.req_blocks.pop(request.id, 0)

    def free_block_hashes(self, request):
        pass

    def get_num_free_blocks(self):
        return self.free_blocks


class ReplicaSimulator:
    """One reusable vllm_v1 replica scheduler; each probe fills it from a ledger, runs it, and empties it."""

    def __init__(self, spec: ReplicaSpec, predictor, replica_config, cache_config, exact_kv: bool = False):
        from vidur.config import FCFSRequestQueueConfig, VllmV1SchedulerConfig
        from vidur.config.config import TraceRequestGeneratorConfig
        from vidur.entities import Batch, Request
        from vidur.scheduler.replica_scheduler.vllm_v1_replica_scheduler import VLLMV1ReplicaScheduler
        self._Request, self._Batch = Request, Batch
        self.spec = spec
        self.timer = BatchTimer(predictor)
        sc = VllmV1SchedulerConfig(batch_size_cap=spec.batch_size_cap, chunk_size=spec.chunk_size)
        self.rs = VLLMV1ReplicaScheduler(
            replica_config=replica_config, replica_scheduler_config=sc,
            request_generator_config=TraceRequestGeneratorConfig(), cache_config=cache_config,
            request_queue_config=FCFSRequestQueueConfig(), replica=SimpleNamespace(id=-1),
            execution_time_predictor=predictor)
        if not exact_kv:  # the scheduler's 30k-block pool is replaced by a counter with the same arithmetic
            self.rs._kv_cache_manager = CountingKV(spec.num_blocks, spec.block_size,
                                                   cache_config.num_preallocate_tokens)
        self.kv = self.rs._kv_cache_manager
        assert not self.kv.enable_caching
        self._cached = {}  # sim request id -> cached prefix tokens (router's view)
        bs = spec.block_size

        def get_computed_blocks(request):  # replaces the prefix-cache lookup by the router's cached-prefix count
            c = self._cached.get(request.id, 0)
            if c and request.num_restarts == 0:
                c = min(c, (request.num_prefill_tokens // bs) * bs)
                if c >= request.num_prefill_tokens:  # fully cached: vLLM recomputes the last block
                    c -= bs
                return [], max(0, c)
            return [], 0
        self.kv.get_computed_blocks = get_computed_blocks

    def _reset(self):
        rs, kv = self.rs, self.kv
        for r in list(rs._requests.values()):
            kv.free(r)
        rs._requests.clear()
        rs._running = []
        rs.scheduled_req_ids = set()
        while len(rs._waiting_queue):
            rs._waiting_queue.pop()
        rs._num_running_batches = 0
        if isinstance(kv, CountingKV):
            kv.req_blocks.clear()
        else:
            kv.req_to_blocks.clear()
            kv.req_to_block_hashes.clear()
            kv.num_cached_block.clear()
        self._cached.clear()
        assert kv.get_num_free_blocks() == self.spec.num_blocks

    def predict_e2e(self, running, waiting, target) -> float:
        """running: [(prompt, decoded_so_far, total_decode)] in arrival order; waiting: [(arrived_at, prompt, cached,
        total_decode)] in arrival order; target: (arrived_at, prompt, cached, total_decode). Returns the target's
        predicted completion time from now (seconds)."""
        Request, Batch = self._Request, self._Batch
        saved = (Request._id, Batch._id)  # sim entities must not shift the run's request / batch ids
        rs, kv = self.rs, self.kv
        cap = self.spec.num_blocks * self.spec.block_size
        try:
            for k, (p, d, dec) in enumerate(running):
                r = Request(0.0, p, dec, None, None)
                r.assign_replica(-1)
                if k < rs._max_micro_batch_size and kv.allocate_slots(r, p + d) is not None:
                    r._num_processed_tokens = p + d
                    r._is_prefill_complete = True
                    r._scheduled = True
                    rs._requests[r.id] = r
                    rs._running.append(r)
                else:  # no room: preempted, recomputes prompt + decoded tokens (Request.restart semantics)
                    r._num_processed_tokens = p + d
                    r.restart()
                    rs.add_request(r)
            for arr, p, c, dec in waiting:
                r = Request(arr, p, dec, None, None)
                if p + dec > cap:
                    continue
                self._cached[r.id] = c
                rs.add_request(r)
            arr, p, c, dec = target
            dec = max(1, min(dec, cap - p))
            tgt = Request(arr, p, dec, None, None)
            self._cached[tgt.id] = c
            rs.add_request(tgt)
            t = 0.0
            for _ in range(MAX_SIM_ITERATIONS):
                batch = rs.on_schedule(t).batch
                if batch is None:
                    return math.inf
                batch.on_schedule(t)
                t += self.timer(batch)
                batch.on_batch_end(t)
                rs.on_batch_end(batch)
                if tgt.completed:
                    return t
            return math.inf
        finally:
            self._reset()
            Request._id, Batch._id = saved


# ------------------------------------------------------------------------------------------------- router
class Astrolabe:
    def __init__(self, spec: ReplicaSpec, oracle=None, diag_path=None, check_argv=True):
        if check_argv:
            _check_argv(spec)
        self.spec = spec
        pred, rc, cc = build_predictor(spec)
        self.sim = ReplicaSimulator(spec, pred, rc, cc)
        self.oracle = oracle  # optional {(prompt tokens, prompt block hashes): deque of true decode lengths}
        self.n = None
        self.diag_path = diag_path
        self.times = []
        self.records = []  # per decision: [now, chosen, pred_chosen, pred_other, other, est_len, true_len, actual]

    def _init(self, n):
        self.n = n
        self.entries = [dict() for _ in range(n)]  # replica -> {eid: entry} in routing order
        self.keys = [dict() for _ in range(n)]  # replica -> {(arrived_at, prompt): deque of eid}
        self.next_eid = 0
        self.len_sum = 0
        self.len_n = 0
        self.tpot = [None] * n
        self.tpot_all = None

    def _len_estimate(self):
        return PRIOR_DECODE_TOKENS if not getattr(self, "len_n", 0) else max(1, round(self.len_sum / self.len_n))

    def _tpot(self, i):
        t = self.tpot[i] if self.tpot[i] is not None else self.tpot_all
        return TPOT_PRIOR_S if t is None else t

    def _ledger(self, i, now):
        running, waiting = [], []
        tp = self._tpot(i)
        for e in self.entries[i].values():
            L = e["L"]
            if e["ft"] is None:
                waiting.append((e["arr"], e["P"], e["cached"], L))
            else:
                d = 1 + int(max(0.0, now - e["ft"]) / tp)
                running.append((e["P"], d, L if d < L else d + OVERRUN_TOKENS))
        return running, waiting

    def choose(self, request, replicas, ctx):
        t0 = time.perf_counter()
        n = len(replicas)
        if self.n is None:
            self._init(n)
        now = ctx.now
        P = request.num_prefill_tokens
        L = self._len_estimate()
        true_len = None
        if self.oracle is not None:
            q = self.oracle.get((P, tuple(request.prefix_hashes)))
            if q:
                true_len = q.popleft()
                L = true_len
        sample = ctx.rng.sample(range(n), min(K, n))
        preds = {}
        for i in sample:
            if replicas[i].kv_usage >= 1.0:  # no free GPU block: dropped (api_server.py:131)
                continue
            running, waiting = self._ledger(i, now)
            preds[i] = self.sim.predict_e2e(running, waiting, (now, P, request.cached_prefix_tokens[i], L))
        if preds:
            best = min(preds.values())
            pick = ctx.rng.choice([i for i in sample if i in preds and preds[i] == best])
        else:
            pick = ctx.rng.randrange(n)
        eid = self.next_eid
        self.next_eid += 1
        self.entries[pick][eid] = {"arr": request.arrived_at, "P": P, "cached": request.cached_prefix_tokens[pick],
                                   "L": L, "ft": None, "rec": len(self.records)}
        self.keys[pick].setdefault((request.arrived_at, P), deque()).append(eid)
        others = [i for i in sample if i != pick]
        self.records.append([now, pick, preds.get(pick), preds.get(others[0]) if others else None,
                             others[0] if others else None, L, true_len, None])
        self.times.append(time.perf_counter() - t0)
        return pick

    def observe(self, outcome, ctx):
        if self.n is None:
            return
        i = outcome.replica
        q = self.keys[i].get((outcome.arrived_at, outcome.num_prefill_tokens))
        if not q:
            return
        if outcome.event == "first_token":
            for eid in q:  # the first not-yet-prefilled entry with this key
                e = self.entries[i][eid]
                if e["ft"] is None:
                    e["ft"] = outcome.first_token_at
                    e["cached"] = outcome.num_cached_tokens
                    break
            return
        eid = q.popleft()
        if not q:
            del self.keys[i][(outcome.arrived_at, outcome.num_prefill_tokens)]
        e = self.entries[i].pop(eid)
        nd = outcome.num_decode_tokens
        self.len_sum += nd
        self.len_n += 1
        if nd and nd > 1 and outcome.completed_at is not None:
            tp = (outcome.completed_at - outcome.first_token_at) / (nd - 1)
            self.tpot[i] = tp if self.tpot[i] is None else (1 - TPOT_ALPHA) * self.tpot[i] + TPOT_ALPHA * tp
            self.tpot_all = tp if self.tpot_all is None else (1 - TPOT_ALPHA) * self.tpot_all + TPOT_ALPHA * tp
        self.records[e["rec"]][7] = outcome.completed_at - outcome.arrived_at

    # ------------------------------------------------------------------ diagnostics
    def summary(self) -> dict:
        import numpy as np
        t = np.array(self.times) if self.times else np.zeros(1)
        rec = [r for r in self.records if r[2] is not None and r[7] is not None and math.isfinite(r[2])]
        out = {"decisions": len(self.times), "choose_mean_s": float(t.mean()), "choose_p50_s": float(np.median(t)),
               "choose_p99_s": float(np.percentile(t, 99)), "choose_max_s": float(t.max()),
               "choose_total_s": float(t.sum()), "batch_timer_calls": self.sim.timer.calls,
               "batch_timer_misses": self.sim.timer.misses, "matched": len(rec),
               "inf_predictions": sum(1 for r in self.records if r[2] is not None and not math.isfinite(r[2])),
               "len_estimate_final": self._len_estimate(), "oracle": self.oracle is not None}
        if rec:
            pr = np.array([r[2] for r in rec])
            ac = np.array([r[7] for r in rec])
            rel = np.abs(pr - ac) / np.maximum(ac, 1e-9)
            out.update({"pearson": float(np.corrcoef(pr, ac)[0, 1]),
                        "spearman": float(np.corrcoef(np.argsort(np.argsort(pr)), np.argsort(np.argsort(ac)))[0, 1]),
                        "median_rel_err": float(np.median(rel)), "mean_signed_rel_err": float(np.mean((pr - ac) / ac)),
                        "mean_pred_e2e": float(pr.mean()), "mean_actual_e2e": float(ac.mean())})
        return out

    def dump(self):
        s = self.summary()
        print("astrolabe: " + json.dumps(s), file=sys.stderr)
        if self.diag_path:
            Path(self.diag_path).write_text(json.dumps({"summary": s, "records": self.records,
                                                        "times": self.times}))


def build(spec: ReplicaSpec, oracle=None):
    """(choose, observe, instance). ASTROLABE_DIAG=<path> writes per-decision records and timings at exit."""
    inst = Astrolabe(spec, oracle=oracle, diag_path=os.environ.get("ASTROLABE_DIAG"))
    atexit.register(inst.dump)

    def choose(request, replicas, ctx):
        return inst.choose(request, replicas, ctx)

    def observe(outcome, ctx):
        inst.observe(outcome, ctx)

    return choose, observe, inst
