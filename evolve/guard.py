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


def check_policy_source(source: str) -> List[str]:
    """Return a list of violations (empty = accepted)."""
    problems: List[str] = []
    if len(source) > MAX_SOURCE_CHARS:
        problems.append(f"source is {len(source)} chars; limit {MAX_SOURCE_CHARS}")
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"syntax error: {e}"]

    defines = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    if not ({"route", "choose"} & defines):
        problems.append("must define route(burst, replicas, ctx) or choose(request, replicas, ctx) at module level")
    for fn in (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("route", "choose")):
        if len(fn.args.args) != 3:
            problems.append(f"{fn.name}() must take exactly 3 positional arguments")

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


def assert_policy_ok(source: str) -> None:
    problems = check_policy_source(source)
    if problems:
        raise GuardError("; ".join(problems))
