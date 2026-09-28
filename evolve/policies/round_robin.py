"""Round robin, as a policy file. Must reproduce the built-in ROUND_ROBIN router exactly."""


def choose(request, replicas, ctx):
    i = ctx.state.get("rr_next", 0)
    ctx.state["rr_next"] = i + 1
    return i % len(replicas)
