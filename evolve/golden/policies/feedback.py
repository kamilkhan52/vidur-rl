"""Golden-set test policy (not a research policy): routes on observe() feedback and on the polled replica metrics, so
the observe hook and the polled view (waiting, kv_usage) leave a trace in the outputs.

Lowest (outstanding requests, waiting, smoothed TTFT observed on the replica, KV usage); remaining ties go to the
scheduler's RNG. observe() keeps an exponentially smoothed TTFT per replica from "first_token" events and checks the
invariants of the feedback it receives, raising on a violation (the simulator then fails the run loudly).
"""


def observe(outcome, ctx):
    if not outcome.first_token_at >= outcome.arrived_at:
        raise ValueError(f"first_token_at {outcome.first_token_at} before arrival {outcome.arrived_at}")
    if outcome.event == "first_token":
        if outcome.completed_at is not None or outcome.num_decode_tokens is not None:
            raise ValueError("first_token outcome reveals the completion")
        ttft = ctx.state.setdefault("ttft", {})
        t = outcome.first_token_at - outcome.arrived_at
        prev = ttft.get(outcome.replica)
        ttft[outcome.replica] = t if prev is None else 0.8 * prev + 0.2 * t
    elif outcome.event == "end":
        if outcome.completed_at is None or outcome.completed_at < outcome.first_token_at:
            raise ValueError(f"end outcome with completed_at {outcome.completed_at}")
        if outcome.num_decode_tokens is None:
            raise ValueError("end outcome without num_decode_tokens")
        ctx.state["ends"] = ctx.state.get("ends", 0) + 1
    else:
        raise ValueError(f"unknown event {outcome.event!r}")


def choose(request, replicas, ctx):
    ttft = ctx.state.get("ttft", {})
    keys = [(r.outstanding_requests, r.waiting, ttft.get(r.index, 0.0), r.kv_usage) for r in replicas]
    best = min(keys)
    cands = [r.index for r, k in zip(replicas, keys) if k == best]
    return cands[0] if len(cands) == 1 else ctx.rng.choice(cands)
