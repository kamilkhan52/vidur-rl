"""LLMServingSim 2.0 adapter: run our routing policies inside LLMServingSim as an independent second simulator.

convert.py  Vidur window CSV -> LLMServingSim JSONL (synthetic token ids that keep the prefix sharing), cluster JSON
router.py   Router subclass that calls a policy's choose()/route() and observe() with Vidur's view dataclasses
run.py      launcher: python -m evolve.lss2.run --policy <path|rr|lor|sticky_lor|lop_uncached> --window <csv> ...

Nothing here edits the LLMServingSim checkout; the router is installed by monkeypatching ``serving.__main__.Router``.
"""
