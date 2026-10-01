"""
Static checks (stage 0a) for proposed routing policies, before anything is simulated.

A policy may only compute over the views it is given. It may not reach the
simulator, the filesystem, the clock, the network, or global randomness, and it
may not use dunder attributes (the usual route to escaping a restricted
namespace). This keeps "simulator escapes" (ArchAgent, 2026) and
nondeterminism out of the search, and keeps evolved routers deployable.
"""

import ast
from typing import List

ALLOWED_IMPORTS = {"math", "heapq", "collections", "itertools", "functools", "statistics", "bisect"}
BANNED_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "input", "breakpoint", "memoryview", "help",
}
MAX_SOURCE_CHARS = 20000


class GuardError(ValueError):
    pass


# Entry points per task: the router task needs route() or choose(); the RL-design task
# (EXP12) needs replica_features() and reward(), and may define a CONFIG dict.
ENTRY_POINTS = {
    "router": {"any_of": {"route": 3, "choose": 3}},
    "rl_design": {"all_of": {"replica_features": 4}, "one_or_more_of": {"reward": 1, "interval_reward": 4}},
}


def check_policy_source(source: str, task: str = "router") -> List[str]:
    """Return a list of violations (empty = accepted)."""
    problems: List[str] = []
    if len(source) > MAX_SOURCE_CHARS:
        problems.append(f"source is {len(source)} chars; limit {MAX_SOURCE_CHARS}")
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"syntax error: {e}"]

    defines = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    spec = ENTRY_POINTS[task]
    if "any_of" in spec:
        wanted = spec["any_of"]
        if not (set(wanted) & set(defines)):
            problems.append("must define route(burst, replicas, ctx) or choose(request, replicas, ctx) at module level")
    else:
        wanted = spec["all_of"]
        for name, n in wanted.items():
            if name not in defines:
                problems.append(f"must define {name}() with {n} positional argument(s) at module level")
    extra = spec.get("one_or_more_of", {})
    if task == "rl_design" and "candidate_mask" in defines and len(defines["candidate_mask"].args.args) != 3:
        problems.append("candidate_mask() must take exactly 3 positional argument(s)")
    if extra and not (set(extra) & set(defines)):
        problems.append("must define " + " and/or ".join(f"{k}()" for k in extra))
    for name, n in {**wanted, **extra}.items():
        fn = defines.get(name)
        if fn is not None and len(fn.args.args) != n:
            problems.append(f"{name}() must take exactly {n} positional argument(s)")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    problems.append(f"import of {a.name!r} not allowed (allowed: {sorted(ALLOWED_IMPORTS)})")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS or node.level:
                problems.append(f"import from {node.module!r} not allowed")
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            problems.append(f"use of {node.id!r} not allowed")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            problems.append(f"dunder attribute access {node.attr!r} not allowed")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            problems.append("global/nonlocal state not allowed; use ctx.state")
    return problems


def assert_policy_ok(source: str, task: str = "router") -> None:
    problems = check_policy_source(source, task)
    if problems:
        raise GuardError("; ".join(problems))
