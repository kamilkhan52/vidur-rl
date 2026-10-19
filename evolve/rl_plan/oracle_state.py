"""The ``--oracle`` table of true output lengths shared by the ES episode runner and the policy it runs.

rl_es usually runs as ``python -m evolve.rl_plan.rl_es``, so the runner lives in ``__main__`` while the generated policy
imports ``evolve.rl_plan.rl_es`` by name: two module copies. A global in rl_es set by the runner was therefore never seen
by the policy, and ``--oracle`` silently ran the online predictor (rc, 2026-10-19). Both now use this module.
TABLE: {(arrival time, prompt tokens): [output tokens, ...]} or None; LOOKUPS counts the policy's reads.
"""
TABLE = None
LOOKUPS = 0
