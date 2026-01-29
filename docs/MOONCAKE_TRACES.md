# Mooncake Trace Acquisition and Conversion

## Source

Traces downloaded from the Mooncake FAST25 release:
```
https://github.com/kvcache-ai/Mooncake/tree/main/FAST25-release/traces
```

Downloaded to `data/raw_traces/`:
- `conversation_trace.jsonl` (12,031 requests)
- `synthetic_trace.jsonl` (3,993 requests)
- `toolagent_trace.jsonl` (23,608 requests)

## Trace Statistics (Verified Against Paper Table 2)

| Workload | Avg Input | Avg Output | Cache Ratio | Requests | Arrival |
|----------|-----------|------------|-------------|----------|---------|
| Conversation | 12,035 | 343 | 40% | 12,031 | Timestamp |
| Tool&Agent | 8,596 | 182 | 59% | 23,608 | Timestamp |
| Synthetic | 15,325 | 149 | 66% | 3,993 | Poisson |

**Cache Ratio** = tokens whose block hash appeared in any previous request / total tokens

## Workload Characteristics (from paper)

### Conversation
- Chatbot multi-turn conversations
- Long-context requests (up to 128K tokens, avg ~12K)
- Cache hits from multi-turn conversation history
- 1-hour sample from online inference cluster

### Tool&Agent
- LLMs deployed as tools/agents
- Pre-designed, lengthy system prompts that are **fully repetitive**
- High cache ratio (59%) from shared system prompts across unrelated requests
- Shorter input/output lengths

### Synthetic
- Constructed from public datasets: ShareGPT, Leval, LooGLE
- Highest cache ratio (66%) but **dispersed** - requires large cache capacity
- Each conversation turn mapped to separate request
- Arrival times simulated via Poisson process

## Format Conversion

### Mooncake JSONL Format
```json
{"timestamp": 0, "input_length": 6758, "output_length": 500, "hash_ids": [0, 1, 2, ...]}
```
- `hash_ids`: Content-based block hashes, each representing **512 tokens**

### Vidur CSV Format
```csv
arrived_at,num_prefill_tokens,num_decode_tokens,block_hash_ids,block_size,session_id
```
- `block_hash_ids`: Block hashes at **16 tokens** per block (vLLM default), covering **prefill + decode**
- `session_id`: Inferred from hash patterns for sticky schedulers

### Conversion Script

`scripts/convert_mooncake_traces.py`

**Timestamp conversion:**
- Raw JSONL timestamps are in **milliseconds**
- Converted to seconds by dividing by 1000
- Result: ~1 hour traces matching paper description

**Block hash expansion:**
- Each 512-token Mooncake block → 32 consecutive 16-token Vidur blocks
- Example: hash_id `5` → blocks `160-191`
- Preserves block sharing between requests
- Adds **unique placeholder hashes** for decode blocks so `block_hash_ids` spans the full request
- Ensures request consistency check: `(prefill + decode) - block_size * len(block_hash_ids)` is in `[0, block_size)`

**Session inference:**
- Identifies common hashes (>50% of requests) as shared system prompts
- Groups requests by non-common hash prefix
- Note: Session grouping is heuristic; actual cache benefit comes from block hash reuse

### Output Files
- `data/processed_traces/mooncake_conversation_trace_v2.csv`
- `data/processed_traces/mooncake_synthetic_trace.csv`
- `data/processed_traces/mooncake_toolagent_trace.csv`

## Key Insight: Cache Ratio vs Session Affinity

The paper's cache ratio measures **global block reuse** across all requests, not within-session reuse:

- **Tool&Agent** achieves 59% cache ratio because many unrelated requests share the same system prompts
- **Synthetic** has 66% cache ratio but "dispersed" - cached blocks spread across many prefix patterns
- **Conversation** has 40% cache ratio from multi-turn history within sessions

**Implication for scheduling:** Optimal caching requires routing requests with shared block hashes to the same replica, regardless of whether they belong to the same logical "session".

## Running Experiments

```bash
# Convert traces (if needed)
python scripts/convert_mooncake_traces.py --trace all

# Run baseline with different traces
python scripts/run_baseline_with_progress.py --trace toolagent --scheduler round_robin --scale 20.0
python scripts/run_baseline_with_progress.py --trace toolagent --scheduler sticky_lor --scale 20.0
```

## Pre-existing Trace

Note: `data/processed_traces/mooncake_conversation_trace.csv` is a pre-existing conversion with slightly different methodology (7,417 sessions vs 8,650 in our conversion, ~2% more prefill tokens). Both are valid for experiments.
