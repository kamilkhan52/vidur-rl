"""LLM proposer for routing-policy mutations, via the official Anthropic SDK."""

import re
from typing import Optional, Tuple

SYSTEM_PROMPT = """\
You are a systems researcher designing a request router for an LLM inference cluster.
You improve a Python routing policy by rewriting it; a simulator scores every version.

## Setting
- Vidur discrete-event simulator; Llama-2-7b replicas (A100, vLLM-v1 scheduler with chunked
  prefill and prefix caching). Each replica holds 122,752 KV-cache tokens.
- Workloads are Mooncake production traces (conversation and tool/agent). Requests arrive in
  bursts; prompts are long (median ~7k tokens) and share prefixes, so prefix-cache hits matter.
- The router assigns each arriving request to one replica immediately (no holding, no
  re-routing). Replicas queue requests FCFS internally.

## Objective
Minimise TAIL latency: P99 end-to-end, P99 time-to-first-token and P99 time-per-output-token,
measured per scenario relative to the best of five heuristic routers (sticky-LOR, LOR,
round-robin, random, least-uncached-prefill). A mean-E2E more than 5% worse than the best
heuristic is penalised. Fitness 0 means "matches the best heuristic on every tail"; positive
means better. Different heuristics win different scenarios, so a policy that adapts to load
and cache state can beat all of them.

## Contract (strict)
Define exactly one entry point at module level:
    def choose(request, replicas, ctx) -> int     # called per request
or  def route(burst, replicas, ctx) -> list[int] # called once per arrival burst
Views are frozen dataclasses:
  request: arrived_at, num_prefill_tokens, session_id, cached_prefix_tokens (tuple, one per
           replica: prompt tokens already in that replica's prefix cache),
           session_last_replica (int, -1 if new session), position_in_burst, burst_size
  replicas[i]: index, outstanding_requests (routed here, not finished), pending_prefill_tokens
           (uncached prompt tokens routed here, not yet prefilled), waiting, running,
           kv_usage (0..1), kv_capacity_tokens
  ctx: rng (random.Random, the ONLY allowed randomness), now, num_replicas, state (dict you may
       use for memory across calls)
With choose(), views are refreshed after every assignment; with route() they are a snapshot
taken before the burst (track your own in-burst assignments).
Output lengths are NOT available (a real router does not know them).
Allowed imports: math, heapq, collections, itertools, functools, statistics, bisect.
No file, clock, network, os/sys access, no global/nonlocal, no dunder attributes,
no getattr/eval/exec. Keep per-call work small (O(replicas) per request is fine).

## Answer format
Think about why the parent behaves as its feedback shows, form one concrete hypothesis, and
implement it. Keep the code readable (it must be explainable in a paper). Reply with a short
paragraph stating the hypothesis, then the COMPLETE new policy in one ```python block.
"""

CODE_BLOCK = re.compile(r"```python\s*\n(.*?)```", re.S)


def extract_code(text: str) -> Optional[str]:
    blocks = CODE_BLOCK.findall(text or "")
    return blocks[-1].strip() + "\n" if blocks else None


class ClaudeProposer:
    """Streams one proposal per call. Credentials come from the environment
    (ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or an `ant auth login` profile)."""

    def __init__(self, model: str = "claude-opus-5", effort: str = "high", max_tokens: int = 64000):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens

    def propose(self, user_prompt: str) -> Tuple[Optional[str], dict]:
        with self.client.beta.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user_prompt}],
            output_config={"effort": self.effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        ) as stream:
            msg = stream.get_final_message()
        meta = {
            "model": msg.model,
            "stop_reason": msg.stop_reason,
            "input_tokens": msg.usage.input_tokens,
            "output_tokens": msg.usage.output_tokens,
            "cache_read_input_tokens": getattr(msg.usage, "cache_read_input_tokens", None),
        }
        if msg.stop_reason == "refusal":
            return None, meta
        text = "".join(b.text for b in msg.content if b.type == "text")
        meta["rationale"] = CODE_BLOCK.sub("", text).strip()[:2000]
        return extract_code(text), meta
