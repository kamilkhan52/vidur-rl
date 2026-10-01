"""
File relay between an evolution driver on the cluster and an LLM proposer elsewhere.

The Claude subscription login lives on the workstation (the `claude` CLI), while the
simulations run on Alpine. The driver uses ``RelayProposer``: each proposal request is
written as a JSON file into ``<relay_dir>/requests/`` and the driver waits for
``<relay_dir>/responses/<id>.json``. On the workstation,

    python -m evolve.relay serve --remote alpine:/scratch/.../relay --proposer claude-code

polls the request directory over the existing SSH connection (ControlMaster; one short
command per poll), runs the proposer locally, and writes the response back. Nothing but
prompts and proposals crosses the relay; no credentials are copied anywhere.
"""

import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Optional, Tuple


class RelayProposer:
    """Driver side (cluster): hand prompts to the relay and wait for the answer."""

    def __init__(self, relay_dir: str, model: Optional[str] = None, timeout_s: int = 6 * 3600, poll_s: float = 10.0):
        self.dir = Path(relay_dir)
        (self.dir / "requests").mkdir(parents=True, exist_ok=True)
        (self.dir / "responses").mkdir(parents=True, exist_ok=True)
        self.model = model
        self.timeout_s = timeout_s
        self.poll_s = poll_s
        from evolve.proposer import SYSTEM_PROMPT

        self.system_prompt = SYSTEM_PROMPT

    def propose(self, user_prompt: str) -> Tuple[Optional[str], dict]:
        rid = uuid.uuid4().hex[:16]
        req = {"id": rid, "system_prompt": self.system_prompt, "user_prompt": user_prompt,
               "model": self.model, "created": time.time()}
        tmp = self.dir / "requests" / f".{rid}.tmp"
        tmp.write_text(json.dumps(req))
        tmp.replace(self.dir / "requests" / f"{rid}.json")
        resp = self.dir / "responses" / f"{rid}.json"
        t0 = time.time()
        while not resp.exists():
            if time.time() - t0 > self.timeout_s:
                raise TimeoutError(f"no relay response for {rid} after {self.timeout_s}s")
            time.sleep(self.poll_s)
        out = json.loads(resp.read_text())
        if out.get("error"):
            raise RuntimeError(f"relay proposer error: {out['error']}")
        meta = dict(out.get("meta") or {})
        meta["relay_id"] = rid
        return out.get("code"), meta


def _ssh(host: str, cmd: str, stdin: Optional[str] = None) -> str:
    p = subprocess.run(["ssh", "-o", "BatchMode=yes", host, cmd], input=stdin, capture_output=True,
                       text=True, timeout=300)
    if p.returncode != 0:
        raise RuntimeError(f"ssh {host} failed ({p.returncode}): {p.stderr[-300:]}")
    return p.stdout


def serve(remote: str, proposer_kind: str, model: Optional[str], workers: int, poll_s: float) -> None:
    """Workstation side: answer relay requests with a local proposer until interrupted."""
    from evolve.driver import make_proposer

    host, rdir = remote.split(":", 1)
    q = lambda p: "'" + p.replace("'", "'\\''") + "'"  # noqa: E731  (shell-quote a remote path)
    inflight = set()
    ex = cf.ThreadPoolExecutor(max_workers=workers)
    failures = 0

    def answer(rid: str) -> None:
        try:
            req = json.loads(_ssh(host, f"cat {q(rdir)}/requests/{rid}.json"))
            prop = make_proposer(proposer_kind, req.get("model") or model)
            prop.system_prompt = req["system_prompt"]
            try:
                code, meta = prop.propose(req["user_prompt"])
                out = {"id": rid, "code": code, "meta": meta}
            except Exception as e:  # reported to the driver, which records it in the archive
                out = {"id": rid, "error": f"{type(e).__name__}: {e}"}
            _ssh(host, f"cat > {q(rdir)}/responses/.{rid}.tmp && mv {q(rdir)}/responses/.{rid}.tmp "
                       f"{q(rdir)}/responses/{rid}.json", stdin=json.dumps(out))
            print(f"[relay] answered {rid} ({'error' if 'error' in out else 'ok'})", flush=True)
        finally:
            inflight.discard(rid)

    print(f"[relay] serving {remote} with {proposer_kind} ({model or 'default model'})", flush=True)
    while True:
        try:
            listing = _ssh(host, f"cd {q(rdir)} && ls requests | grep '\\.json$' | sed 's/\\.json$//' ; "
                                 f"echo ---; ls responses | grep '\\.json$' | sed 's/\\.json$//'")
            failures = 0
        except Exception as e:
            failures += 1
            print(f"[relay] poll failed ({failures}): {e}", flush=True)
            if failures >= 3:  # never hammer SSH (Duo lockout risk); stop and let a human look
                print("[relay] stopping after 3 consecutive SSH failures", flush=True)
                return
            time.sleep(poll_s * 4)
            continue
        req_part, _, resp_part = listing.partition("---")
        pending = set(req_part.split()) - set(resp_part.split()) - inflight
        for rid in sorted(pending):
            inflight.add(rid)
            ex.submit(answer, rid)
        time.sleep(poll_s)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--remote", required=True, help="host:/path/to/relay_dir")
    s.add_argument("--proposer", default="claude-code", choices=["claude-code", "openrouter", "anthropic"])
    s.add_argument("--model", default=None)
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--poll-s", type=float, default=30.0)
    a = ap.parse_args()
    serve(a.remote, a.proposer, a.model, a.workers, a.poll_s)


if __name__ == "__main__":
    main()
