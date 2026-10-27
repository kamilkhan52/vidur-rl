"""DualMap (Wang et al., ICLR 2026, arXiv 2602.06502), routing rule ported from github.com/ASISys/DualMap @ 24816acc
(MIT). Notes: ~/vidur-rl-priorwork/notes/dualmap.md. This is the paper's no-rebalance ablation (§4.3), router only.

Per request (dualmap/scheduler/utils/double_hash_global_scheduler_utils.py:179-353):
  key  = the first two 512-token Mooncake blocks plus the first d of them, d from the adaptive prefix table
         (lazy_prefix_table.py: HotPrefixDetector window 200, hot_ratio 0.0612, min_samples 20; expand a hot prefix by
         one block, shrink when the parent's share is < 2 %; request_generator.py:91-96);
  c1, c2 = two consistent-hash rings on the key; if equal, c2 = (c1 + 1) % N (paper Eq. 5);
  hit_i = cached prompt tokens on i (a fully cached prompt counts as a miss, replica.py:38-43);
  V_i  = uncached prompt tokens routed to i and not yet prefilled + this request's uncached tokens;
  primary = larger hit; if V_primary > T = 0.5 * TTFT_SLO / PREFILL_TPOT, primary = smaller V
  (start_up.py:76, 91-94; TTFT_SLO = 5 s, the code's default).
  Metric ties: fewer running KV tokens, then c1 (the code re-seeds random with 42 on every tie: a fixed order).
Deviations (forced by a router-only harness): no router-side queue, release gating (one pending prefill per replica)
or cache-hit-first reordering, and no hotspot rebalancing of held requests; the cache view is the replica's actual
GPU prefix cache (Vidur has no DRAM tier) instead of a router-modelled 64 GB LRU; the tie-break uses running KV
(kv_usage) because DualMap's uses true output lengths; the rings are re-implemented (md5, 160 points per replica,
two salts) rather than uhashring; PREFILL_TPOT is our device's idle prefill time per uncached token
(evolve/priorwork/calib_prefill.py), not the Ascend 910B constant.
"""
import hashlib
from bisect import bisect

TTFT_SLO = 5.0
PREFILL_TPOT = 5.078e-05  # h100df (calib_prefill.py, validation window); device variants override this
WINDOW, HOT_RATIO, MIN_SAMPLES, SHRINK_RATIO = 200, 0.0612, 20, 0.02
MOONCAKE_BLOCK = 512
POINTS = 160


def _h(s):
    return int.from_bytes(hashlib.md5(s.encode()).digest()[:8], "big")


def _rings(n):
    out = []
    for salt in ("ring1", "ring2"):
        pts = sorted((_h(f"{salt}:{i}:{v}"), i) for i in range(n) for v in range(POINTS))
        out.append(([p for p, _ in pts], [i for _, i in pts]))
    return out


def _ring_pick(ring, key):
    keys, owners = ring
    return owners[bisect(keys, _h(key)) % len(keys)]


class _Table:
    """LazyPrefixTable + HotPrefixDetector + LazyExpansionController (lazy_prefix_table.py:42-156)."""

    def __init__(self):
        self.marks = {}  # (depth, prefix) -> expand-to depth
        self.window, self.counts = [], {}

    def lookup(self, blocks):
        depth = 1
        while depth <= len(blocks):
            nxt = self.marks.get((depth, blocks[:depth]))
            if nxt is None:
                return depth
            depth = nxt
        return len(blocks)

    def _observe(self, prefix):
        self.window.append(prefix)
        self.counts[prefix] = self.counts.get(prefix, 0) + 1
        if len(self.window) > WINDOW:
            old = self.window.pop(0)
            self.counts[old] -= 1
            if self.counts[old] <= 0:
                del self.counts[old]
        total = len(self.window)
        if total < MIN_SAMPLES:
            return False, 0.0
        ratio = self.counts.get(prefix, 0) / total if total > 20 else 0.0
        return ratio >= HOT_RATIO, ratio

    def process(self, blocks):
        depth = self.lookup(blocks)
        hot, _ = self._observe(blocks[:depth])
        if hot and depth < len(blocks):
            self.marks[(depth, blocks[:depth])] = depth + 1
        if depth > 1:
            _, p_ratio = self._observe(blocks[:depth - 1])
            if p_ratio < SHRINK_RATIO:
                self.marks.pop((depth, blocks[:depth]), None)


def _state(ctx, n):
    st = ctx.state.get("dualmap")
    if st is None:
        st = ctx.state["dualmap"] = {"rings": _rings(n), "table": _Table()}
    return st


def choose(request, replicas, ctx):
    n = len(replicas)
    st = _state(ctx, n)
    stride = max(1, MOONCAKE_BLOCK // request.block_tokens) if request.block_tokens else 1
    blocks = tuple(request.prefix_hashes[::stride])  # one label per 512-token Mooncake block
    if blocks:
        d = st["table"].lookup(blocks)  # the key uses the depth before this request's own update
        st["table"].process(blocks)
        key = repr(blocks[:2]) + "@" + repr(blocks[:d])
    else:
        key = f"nohash:{request.num_prefill_tokens}:{request.session_id}"
    c1 = _ring_pick(st["rings"][0], key)
    c2 = _ring_pick(st["rings"][1], key)
    if c1 == c2:
        c2 = (c1 + 1) % n
    hit, v, kv = {}, {}, {}
    for i in (c1, c2):
        cached = request.cached_prefix_tokens[i]
        uncached = request.num_prefill_tokens - cached if 0 < cached < request.num_prefill_tokens else request.num_prefill_tokens
        hit[i] = request.num_prefill_tokens - uncached
        v[i] = replicas[i].pending_prefill_tokens + uncached
        kv[i] = replicas[i].kv_usage * replicas[i].kv_capacity_tokens

    def pick2(metric, larger):
        a, b = metric[c1], metric[c2]
        if a != b:
            return c1 if (a > b) == larger else c2
        if kv[c1] != kv[c2]:
            return c1 if kv[c1] < kv[c2] else c2
        return c1

    primary = pick2(hit, True)
    if v[primary] > 0.5 * TTFT_SLO / PREFILL_TPOT:
        primary = pick2(v, False)
    return primary
