# Tests

Run from the repository root:

    .venv/bin/python -m pytest tests -q

The end-to-end tests replay 24 requests of the Mooncake synthetic trace through the
real simulator on a two-replica Llama-2-7b cluster (about ten seconds each). They
need `data/processed_traces/mooncake_synthetic_trace.csv` and the predictor cache in
`cache/`; they skip when either is missing.
