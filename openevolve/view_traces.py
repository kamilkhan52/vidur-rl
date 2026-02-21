"""
View OpenEvolve reasoning traces.

Reads the JSONL evolution trace and displays what the LLM tried each iteration:
  - Score change vs parent
  - The prompt sent to the LLM
  - The LLM's reasoning / response
  - The code diff applied to the parent program
  - The resulting child program

Usage:
    # Show all traces from the latest run
    python openevolve/view_traces.py

    # Filter to a range of iterations
    python openevolve/view_traces.py --iter 10 20

    # Show only iterations that improved the score
    python openevolve/view_traces.py --improved-only

    # Tail mode: show new entries as they arrive (like tail -f)
    python openevolve/view_traces.py --tail

    # Point at a specific trace file
    python openevolve/view_traces.py --trace openevolve/evolution_output/traces/evolution_trace.jsonl

    # Dump as human-readable markdown to a file
    python openevolve/view_traces.py --out traces_report.md
"""

import argparse
import difflib
import json
import sys
import textwrap
import time
from pathlib import Path

DEFAULT_TRACE = Path("openevolve/evolution_output/traces/evolution_trace.jsonl")

# ── ANSI colours ──────────────────────────────────────────────────────────────
GREEN = "\033[92m"
RED   = "\033[91m"
CYAN  = "\033[96m"
BOLD  = "\033[1m"
DIM   = "\033[2m"
RESET = "\033[0m"

def _c(text, *codes):
    """Wrap text in ANSI codes (skipped if not a TTY)."""
    if not sys.stdout.isatty():
        return text
    return "".join(codes) + str(text) + RESET


def _wrap(text: str, width: int = 100, indent: str = "  ") -> str:
    if not text:
        return ""
    lines = []
    for line in text.splitlines():
        wrapped = textwrap.wrap(line, width=width - len(indent)) or [""]
        lines.extend(indent + w for w in wrapped)
    return "\n".join(lines)


def _divider(char="─", width=100):
    return char * width


# ── entry formatting ──────────────────────────────────────────────────────────

def _score_line(child_m: dict, parent_m: dict) -> str:
    child_s = child_m.get("combined_score", float("nan"))
    parent_s = parent_m.get("combined_score", float("nan"))
    delta = child_s - parent_s

    delta_str = f"{delta:+.4f}"
    if delta > 0:
        delta_str = _c(delta_str, GREEN, BOLD)
    elif delta < 0:
        delta_str = _c(delta_str, RED)

    child_str = f"{child_s:.4f}"
    rl_e2e = child_m.get("rl_e2e", float("nan"))
    base   = child_m.get("baseline_e2e", float("nan"))
    impr   = child_m.get("improvement_pct", float("nan"))

    return (
        f"score={_c(child_str, BOLD)}  Δ={delta_str}  "
        f"rl_e2e={rl_e2e:.3f}s  baseline={base:.3f}s  "
        f"improvement={impr:+.1f}%"
    )


def _compute_diff(parent_code: str | None, child_code: str | None) -> str:
    if not parent_code or not child_code:
        return "(code not captured — set evolution_trace.include_code: true)"
    parent_lines = parent_code.splitlines(keepends=True)
    child_lines  = child_code.splitlines(keepends=True)
    diff = list(difflib.unified_diff(
        parent_lines, child_lines,
        fromfile="parent", tofile="child", lineterm=""
    ))
    if not diff:
        return "(no change)"
    lines = []
    for line in diff:
        if line.startswith("+") and not line.startswith("+++"):
            lines.append(_c(line, GREEN))
        elif line.startswith("-") and not line.startswith("---"):
            lines.append(_c(line, RED))
        else:
            lines.append(_c(line, DIM))
    return "\n".join(lines)


def format_entry(entry: dict, show_prompt: bool = True, show_code: bool = False) -> str:
    iteration   = entry.get("iteration", "?")
    child_id    = entry.get("child_id", "?")[:8]
    parent_id   = entry.get("parent_id", "?")[:8]
    island_id   = entry.get("island_id")
    child_m     = entry.get("child_metrics", {})
    parent_m    = entry.get("parent_metrics", {})
    llm_resp    = entry.get("llm_response") or ""
    prompt      = entry.get("prompt") or {}
    parent_code = entry.get("parent_code")
    child_code  = entry.get("child_code")

    lines = [
        "",
        _divider("═"),
        _c(f"ITERATION {iteration}", BOLD, CYAN)
        + f"  child={child_id}  parent={parent_id}"
        + (f"  island={island_id}" if island_id is not None else ""),
        _score_line(child_m, parent_m),
        _divider(),
    ]

    # ── LLM reasoning / response ─────────────────────────────────────────
    if llm_resp:
        lines.append(_c("▶ LLM RESPONSE / REASONING:", BOLD))
        lines.append(_wrap(llm_resp.strip(), width=100))
    else:
        lines.append(_c("▶ LLM RESPONSE:", BOLD) + " (not captured — set include_prompts: true)")

    # ── Prompt (system + user, collapsed) ────────────────────────────────
    if show_prompt and prompt:
        lines.append("")
        lines.append(_c("▶ PROMPT SENT TO LLM:", BOLD))
        for role, content in prompt.items():
            if content:
                header = _c(f"  [{role.upper()}]", DIM)
                # Show first 400 chars of each prompt section
                preview = (content[:400] + "…") if len(content) > 400 else content
                lines.append(header)
                lines.append(_wrap(preview, width=100, indent="    "))

    # ── Code diff ────────────────────────────────────────────────────────
    lines.append("")
    lines.append(_c("▶ CODE DIFF (parent → child):", BOLD))
    diff_str = _compute_diff(parent_code, child_code)
    for dl in diff_str.splitlines():
        lines.append("  " + dl)

    # ── Full child code (optional) ────────────────────────────────────────
    if show_code and child_code:
        lines.append("")
        lines.append(_c("▶ CHILD PROGRAM:", BOLD))
        for cl in child_code.splitlines():
            lines.append("  " + cl)

    return "\n".join(lines)


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_entries(trace_path: Path) -> list[dict]:
    entries = []
    with open(trace_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return entries


def filter_entries(entries: list[dict], iter_range: tuple | None, improved_only: bool) -> list[dict]:
    if iter_range:
        lo, hi = iter_range
        entries = [e for e in entries if lo <= e.get("iteration", -1) <= hi]
    if improved_only:
        def is_improved(e):
            cm = e.get("child_metrics", {})
            pm = e.get("parent_metrics", {})
            cs = cm.get("combined_score", float("-inf"))
            ps = pm.get("combined_score", float("-inf"))
            return cs > ps
        entries = [e for e in entries if is_improved(e)]
    return entries


def print_summary(entries: list[dict]):
    if not entries:
        print("No entries.")
        return

    scores = [e.get("child_metrics", {}).get("combined_score", float("nan")) for e in entries]
    valid  = [s for s in scores if not __import__("math").isnan(s)]
    improved = sum(
        1 for e in entries
        if e.get("child_metrics", {}).get("combined_score", float("-inf"))
        > e.get("parent_metrics", {}).get("combined_score", float("-inf"))
    )

    print(_divider("═"))
    print(_c("SUMMARY", BOLD))
    print(f"  Total entries : {len(entries)}")
    print(f"  Improved      : {improved} ({improved/len(entries)*100:.0f}%)")
    if valid:
        print(f"  Best score    : {max(valid):.4f}")
        print(f"  Latest score  : {valid[-1]:.4f}")
    print(_divider("═"))


# ── tail mode ─────────────────────────────────────────────────────────────────

def tail_mode(trace_path: Path, show_prompt: bool, show_code: bool):
    print(f"Tailing {trace_path} ... (Ctrl+C to stop)")
    seen = 0
    while True:
        if not trace_path.exists():
            time.sleep(2)
            continue
        entries = load_entries(trace_path)
        new = entries[seen:]
        for entry in new:
            print(format_entry(entry, show_prompt=show_prompt, show_code=show_code))
        seen = len(entries)
        time.sleep(3)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="View OpenEvolve reasoning traces")
    parser.add_argument("--trace", default=str(DEFAULT_TRACE),
                        help="Path to evolution_trace.jsonl")
    parser.add_argument("--iter", nargs=2, type=int, metavar=("FROM", "TO"),
                        help="Show only iterations FROM..TO (inclusive)")
    parser.add_argument("--improved-only", action="store_true",
                        help="Show only iterations that improved the score")
    parser.add_argument("--no-prompt", action="store_true",
                        help="Hide the prompt section (LLM input)")
    parser.add_argument("--show-code", action="store_true",
                        help="Show full child program source (in addition to diff)")
    parser.add_argument("--tail", action="store_true",
                        help="Follow the trace file as new entries arrive")
    parser.add_argument("--out", default=None,
                        help="Write output to file (markdown-friendly)")
    args = parser.parse_args()

    trace_path = Path(args.trace)

    if not trace_path.exists():
        print(f"Trace file not found: {trace_path}")
        print("Make sure evolution_trace.enabled: true in openevolve/config.yaml and a run has started.")
        sys.exit(1)

    if args.tail:
        tail_mode(trace_path, show_prompt=not args.no_prompt, show_code=args.show_code)
        return

    entries = load_entries(trace_path)
    entries = filter_entries(
        entries,
        iter_range=tuple(args.iter) if args.iter else None,
        improved_only=args.improved_only,
    )

    if args.out:
        outfile = open(args.out, "w")
    else:
        outfile = None

    def emit(text):
        if outfile:
            # Strip ANSI for file output
            import re
            clean = re.sub(r"\033\[[0-9;]*m", "", text)
            outfile.write(clean + "\n")
        else:
            print(text)

    print_summary(entries)

    for entry in entries:
        emit(format_entry(entry, show_prompt=not args.no_prompt, show_code=args.show_code))

    if outfile:
        outfile.close()
        print(f"Written to {args.out}")


if __name__ == "__main__":
    main()
