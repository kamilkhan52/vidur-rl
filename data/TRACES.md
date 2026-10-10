# Trace provenance

`TRACES.sha256` pins every Mooncake file the evolve harness reads (`evolve/evaluate.py: TRACES`).
Check with `cd data && shasum -a 256 -c TRACES.sha256`.

| Harness name | File | Origin | Notes |
|---|---|---|---|
| Tool | `processed_traces/mooncake_toolagent_trace.csv` (101 MB, not committed: over GitHub's 100 MB limit) | `scripts/convert_mooncake_traces.py` on `raw_traces/toolagent_trace.jsonl` | Rebuilds byte-identically (checked 2026-10-10). Prompts rounded up to whole 512-token blocks (+287 tokens mean). Session ids are inferred, not supplied by Mooncake; two inferred sessions hold 54 % of requests. |
| Conv | `processed_traces/mooncake_conversation_trace.csv` | Upstream Vidur's committed file (base commit 25e0082) | Not produced by our converter: every prompt carries a +512-token pad (always uncached), and its session ids differ. `mooncake_conversation_trace_v2.csv` is our converter's output (unused by the harness). |
| Synth | `processed_traces/mooncake_synthetic_trace.csv` | `scripts/convert_mooncake_traces.py` on `raw_traces/synthetic_trace.jsonl` | Rebuilds byte-identically (checked 2026-10-10). |

Raw traces: Mooncake FAST'25 release (`raw_traces/*.jsonl`, Apache-2.0).
To rebuild Tool: `.venv/bin/python scripts/convert_mooncake_traces.py --trace toolagent`.

## v4 traces (simulator release v4, 2026-10-10)

`processed_traces/*_v4.csv`, built by `.venv/bin/python scripts/convert_mooncake_traces.py --v4` (not committed: Tool is over 100 MB; rebuild and check against `TRACES.sha256`).
- **Prompts unpadded:** `num_prefill_tokens` equals Mooncake's `input_length`, which removes the v3 pads of +287 / +512 / +302 tokens (Tool / Conv / Synth means). Total tokens drop 3.2 / 4.0 / 1.9 %.
- **Hash ids:**
  - The prompt's full 16-token blocks take the Mooncake block sub-ids.
  - The partial last prompt block and the decode blocks get ids unique to the row.
  - The list keeps (P + D) // 16 entries and is never truncated (changelog SC-1).
- **Sessions:** carried over row by row from the v3 files, including Conv's upstream ids. Session inference is unchanged.
- **Lint:** all three files pass `scripts/lint_trace.py`.
- **Over 128K:** 10 Synth requests exceed Llama-3.1's 128K context; the harness drops them (`MAX_CONTEXT`).
