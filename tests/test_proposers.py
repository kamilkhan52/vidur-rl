"""Proposer back ends, tested offline (no network, no real claude CLI)."""

import io
import json
import subprocess
import urllib.error

import pytest

from evolve import driver
from evolve import proposer as P

CODE = "def choose(request, replicas, ctx):\n    return 0\n"
ANSWER = f"Hypothesis: always replica 0.\n```python\n{CODE}```"


class FakeResp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_openrouter_request_and_parsing(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    sent = {}

    def fake_urlopen(req, timeout):
        sent["body"] = json.loads(req.data)
        sent["auth"] = req.headers["Authorization"]
        return FakeResp(json.dumps({
            "model": "deepseek/deepseek-v4.1-flash",
            "choices": [{"message": {"content": ANSWER}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 500, "cost": 0.0009},
        }).encode())

    monkeypatch.setattr(P.urllib.request, "urlopen", fake_urlopen)
    prop = P.OpenRouterProposer()
    code, meta = prop.propose("improve this")
    assert code == CODE
    assert meta["cost_usd"] == 0.0009 and meta["backend"] == "openrouter"
    assert sent["auth"] == "Bearer test-key"
    body = sent["body"]
    assert body["model"] == "deepseek/deepseek-v4.1-flash"
    assert body["messages"][0]["role"] == "system" and body["messages"][0]["content"] == P.SYSTEM_PROMPT
    assert body["usage"] == {"include": True} and body["reasoning"] == {"effort": "high"}


def test_openrouter_retries_429_then_succeeds(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setattr(P.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def flaky(req, timeout):
        calls["n"] += 1
        if calls["n"] < 3:
            raise urllib.error.HTTPError(req.full_url, 429, "rate", {}, io.BytesIO(b"slow down"))
        return FakeResp(json.dumps({"choices": [{"message": {"content": ANSWER}}], "usage": {}}).encode())

    monkeypatch.setattr(P.urllib.request, "urlopen", flaky)
    code, _ = P.OpenRouterProposer().propose("x")
    assert code == CODE and calls["n"] == 3


def test_openrouter_does_not_retry_400(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")

    def bad(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 400, "bad", {}, io.BytesIO(b"invalid model"))

    monkeypatch.setattr(P.urllib.request, "urlopen", bad)
    with pytest.raises(RuntimeError, match="HTTP 400"):
        P.OpenRouterProposer().propose("x")


def test_openrouter_key_from_private_file(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    f = tmp_path / "openrouter.env"
    f.write_text("# comment\nOPENROUTER_API_KEY=from-file\n")
    assert P.load_openrouter_key(str(f)) == "from-file"
    with pytest.raises(RuntimeError):
        P.load_openrouter_key(str(tmp_path / "missing.env"))


def test_claude_code_uses_subscription_and_no_tools(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-leak")
    seen = {}

    def fake_run(cmd, input, capture_output, text, timeout, cwd, env):
        seen.update(cmd=cmd, input=input, env=env, cwd=cwd)
        out = {"type": "result", "subtype": "success", "is_error": False, "result": ANSWER,
               "total_cost_usd": 0.12, "usage": {"input_tokens": 900, "output_tokens": 400}}
        return subprocess.CompletedProcess(cmd, 0, json.dumps(out), "")

    monkeypatch.setattr(P.subprocess, "run", fake_run)
    prop = P.ClaudeCodeProposer(claude_bin="/usr/local/bin/claude")
    code, meta = prop.propose("improve this")
    assert code == CODE and meta["backend"] == "claude-code" and meta["cost_usd"] == 0.12
    cmd = seen["cmd"]
    assert cmd[0] == "/usr/local/bin/claude" and "-p" in cmd
    assert cmd[cmd.index("--tools") + 1] == ""          # every tool disabled
    assert "--dangerously-skip-permissions" not in cmd
    assert cmd[cmd.index("--system-prompt") + 1] == P.SYSTEM_PROMPT
    assert "ANTHROPIC_API_KEY" not in seen["env"]        # billed to the subscription
    assert seen["input"] == "improve this"


def test_claude_code_waits_on_usage_limit(monkeypatch):
    monkeypatch.setattr(P.time, "sleep", lambda s: None)
    replies = iter([
        {"is_error": True, "result": "Claude usage limit reached; resets at 5pm"},
        {"is_error": False, "result": ANSWER, "usage": {}},
    ])
    monkeypatch.setattr(P.subprocess, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, json.dumps(next(replies)), ""))
    code, _ = P.ClaudeCodeProposer(claude_bin="claude", limit_wait_s=0).propose("x")
    assert code == CODE


def test_driver_spend_cap_and_proposer_factory(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    assert isinstance(driver.make_proposer("openrouter", None), P.OpenRouterProposer)
    assert driver.make_proposer("openrouter", None).model == "deepseek/deepseek-v4.1-flash"

    class A:
        rows = [{"proposer": {"cost_usd": 0.5}}, {"proposer": {"cost_usd": None}}, {"proposer": {}}, {}]

    assert driver.spent_usd(A) == 0.5
