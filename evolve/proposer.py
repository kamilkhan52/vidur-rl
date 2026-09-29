"""LLM proposers for routing-policy mutations.

Three interchangeable back ends share one system prompt and return (code, meta):
  ClaudeProposer      Anthropic API via the official SDK (needs API credentials)
  OpenRouterProposer  any OpenRouter model over HTTPS (stdlib only), e.g. DeepSeek
  ClaudeCodeProposer  the `claude` CLI in print mode, billed to a Claude subscription
meta always carries token usage and, where the back end reports it, cost in USD.
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
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
            "backend": "anthropic-api",
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


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_KEY_FILE = os.path.expanduser("~/.config/vidur-evolve/openrouter.env")


def load_openrouter_key(key_file: str = DEFAULT_KEY_FILE) -> str:
    """OPENROUTER_API_KEY from the environment, else from a private KEY=VALUE file."""
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if key:
        return key
    if os.path.exists(key_file):
        for line in open(key_file):
            if line.strip().startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError(f"no OPENROUTER_API_KEY in the environment or {key_file}")


class OpenRouterProposer:
    """OpenAI-style chat completion through OpenRouter. Retries 429/5xx/network errors."""

    def __init__(self, model: str = "deepseek/deepseek-v4.1-flash", temperature: float = 0.8,
                 max_tokens: int = 64000, reasoning_effort: Optional[str] = "medium",
                 key_file: str = DEFAULT_KEY_FILE, timeout: int = 900, retries: int = 5):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.key = load_openrouter_key(key_file)
        self.timeout = timeout
        self.retries = retries

    def _post(self, body: dict) -> dict:
        req = urllib.request.Request(
            OPENROUTER_URL,
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json",
                     "X-Title": "vidur-evolve"},
        )
        delay = 10.0
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode())
            except urllib.error.HTTPError as e:
                retryable = e.code == 429 or e.code >= 500
                detail = e.read().decode(errors="replace")[:500]
                if not retryable or attempt == self.retries:
                    raise RuntimeError(f"OpenRouter HTTP {e.code}: {detail}") from None
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt == self.retries:
                    raise RuntimeError(f"OpenRouter unreachable: {e}") from None
            time.sleep(delay)
            delay = min(delay * 2, 300)
        raise AssertionError("unreachable")

    def propose(self, user_prompt: str) -> Tuple[Optional[str], dict]:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": user_prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "usage": {"include": True},
        }
        if self.reasoning_effort:
            body["reasoning"] = {"effort": self.reasoning_effort}
        data = self._post(body)
        if "error" in data:
            raise RuntimeError(f"OpenRouter error: {str(data['error'])[:500]}")
        choice = data["choices"][0]
        text = choice["message"].get("content") or ""
        usage = data.get("usage") or {}
        meta = {
            "backend": "openrouter",
            "model": data.get("model", self.model),
            "stop_reason": choice.get("finish_reason"),
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "cost_usd": usage.get("cost"),
            "rationale": CODE_BLOCK.sub("", text).strip()[:2000],
        }
        return extract_code(text), meta


class UsageLimitError(RuntimeError):
    pass


class ClaudeCodeProposer:
    """`claude -p` with every tool disabled, billed to the logged-in Claude subscription.

    ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN are removed from the child environment so
    the subscription login is used rather than API billing. The call runs in an empty
    temporary directory so no project instructions are picked up. The binary is invoked
    directly (never a shell wrapper that could add permission-skipping flags).
    """

    def __init__(self, model: str = "claude-opus-5", timeout: int = 1800,
                 limit_wait_s: int = 900, limit_retries: int = 8, claude_bin: Optional[str] = None):
        self.model = model
        self.timeout = timeout
        self.limit_wait_s = limit_wait_s
        self.limit_retries = limit_retries
        self.claude_bin = claude_bin or shutil.which("claude")
        if not self.claude_bin:
            raise RuntimeError("claude CLI not found on PATH")

    def command(self):
        return [self.claude_bin, "-p", "--tools", "", "--system-prompt", SYSTEM_PROMPT,
                "--model", self.model, "--output-format", "json",
                "--no-session-persistence", "--strict-mcp-config"]

    def _once(self, user_prompt: str) -> dict:
        env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}
        with tempfile.TemporaryDirectory(prefix="evolve_claude_") as cwd:
            proc = subprocess.run(self.command(), input=user_prompt, capture_output=True, text=True,
                                  timeout=self.timeout, cwd=cwd, env=env)
        try:
            out = json.loads(proc.stdout)
        except json.JSONDecodeError:
            raise RuntimeError(f"claude exited {proc.returncode}: {(proc.stderr or proc.stdout)[-500:]}") from None
        if out.get("is_error"):
            msg = str(out.get("result", ""))
            if "limit" in msg.lower():
                raise UsageLimitError(msg[:300])
            raise RuntimeError(f"claude error: {msg[:500]}")
        return out

    def propose(self, user_prompt: str) -> Tuple[Optional[str], dict]:
        for attempt in range(self.limit_retries + 1):
            try:
                out = self._once(user_prompt)
                break
            except UsageLimitError:
                if attempt == self.limit_retries:
                    raise
                time.sleep(self.limit_wait_s)
        text = out.get("result") or ""
        usage = out.get("usage") or {}
        meta = {
            "backend": "claude-code",
            "model": self.model,
            "stop_reason": out.get("stop_reason") or out.get("subtype"),
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "cost_usd": out.get("total_cost_usd"),  # notional API-equivalent; billed to the subscription
            "duration_ms": out.get("duration_ms"),
            "rationale": CODE_BLOCK.sub("", text).strip()[:2000],
        }
        return extract_code(text), meta
