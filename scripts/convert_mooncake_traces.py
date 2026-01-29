#!/usr/bin/env python3
"""
Convert Mooncake JSONL traces to Vidur CSV format.

Mooncake format (JSONL):
  - timestamp: arrival time (seconds)
  - input_length: prefill tokens
  - output_length: decode tokens
  - hash_ids: block hash IDs (512 tokens per block)

Vidur format (CSV):
  - arrived_at: arrival time (seconds)
  - num_prefill_tokens: prefill tokens
  - num_decode_tokens: decode tokens
  - block_hash_ids: JSON list of block hash IDs
  - block_size: tokens per block (16 for vLLM)
  - session_id: session identifier (inferred from hash patterns)

Usage:
    python scripts/convert_mooncake_traces.py [--trace conversation|synthetic|toolagent|all]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


def infer_sessions(records: list, min_shared_blocks: int = 2) -> list:
    """
    Infer session IDs from hash_id patterns.

    Requests sharing the same hash_id prefix (beyond common system prompt)
    are considered part of the same session.

    Args:
        records: List of raw trace records
        min_shared_blocks: Minimum shared blocks to consider same session

    Returns:
        List of session IDs corresponding to each record
    """
    # Find common prefix (system prompt) - hash_ids shared by many requests
    hash_counts = defaultdict(int)
    for r in records:
        for h in r['hash_ids'][:5]:  # Check first 5 blocks
            hash_counts[h] += 1

    # Hashes present in >50% of requests are common (system prompt)
    threshold = len(records) * 0.5
    common_hashes = {h for h, c in hash_counts.items() if c > threshold}

    # Group by session key (non-common hash prefix)
    session_map = {}  # session_key -> session_id
    session_ids = []
    next_session_id = 0

    for r in records:
        # Extract session key: first few non-common hashes
        session_key = tuple(h for h in r['hash_ids'][:10] if h not in common_hashes)[:5]

        if not session_key:
            # No distinguishing hashes - use hash prefix as fallback
            session_key = tuple(r['hash_ids'][1:6]) if len(r['hash_ids']) > 5 else tuple(r['hash_ids'])

        if session_key not in session_map:
            session_map[session_key] = next_session_id
            next_session_id += 1

        session_ids.append(session_map[session_key])

    return session_ids


def expand_block_hashes(hash_ids: list, input_length: int, output_length: int,
                        src_block_size: int = 512,
                        dst_block_size: int = 16,
                        unique_base: int = 5_000_000_000) -> tuple:
    """
    Expand block hash IDs from source block size to destination block size.

    Each source block (512 tokens) becomes multiple destination blocks (16 tokens).
    Also generates unique hashes for the "tail" tokens not covered by hash_ids.

    Args:
        hash_ids: Original hash IDs (512-token blocks)
        input_length: Total input tokens (prefill)
        output_length: Total output tokens (decode)
        src_block_size: Original block size (512)
        dst_block_size: Target block size (16)

    Returns:
        Tuple of (expanded_hash_ids, adjusted_prefill_tokens)
    """
    expansion_factor = src_block_size // dst_block_size  # 32

    # Expand each hash_id into multiple blocks
    expanded = []
    for h in hash_ids:
        base = h * expansion_factor
        expanded.extend(range(base, base + expansion_factor))

    # Calculate tokens covered by hash_ids
    tokens_from_hashes = len(hash_ids) * src_block_size

    # Remaining prefill tokens need unique block hashes
    remaining_tokens = input_length - tokens_from_hashes
    if remaining_tokens > 0:
        # Generate unique hashes for remaining blocks
        num_remaining_blocks = (remaining_tokens + dst_block_size - 1) // dst_block_size
        # Use request-unique hash values to avoid collision
        base_unique = unique_base + len(expanded)
        expanded.extend(range(base_unique, base_unique + num_remaining_blocks))

    # Adjusted prefill token count (aligned to dst_block_size)
    prefill_blocks = len(expanded)
    adjusted_prefill_tokens = prefill_blocks * dst_block_size
    adjusted_prefill_tokens = max(adjusted_prefill_tokens, input_length)

    # Append unique placeholder hashes for decode blocks
    total_tokens = adjusted_prefill_tokens + output_length
    total_full_blocks = total_tokens // dst_block_size
    decode_blocks = max(total_full_blocks - prefill_blocks, 0)
    if decode_blocks > 0:
        base_unique = unique_base + len(expanded)
        expanded.extend(range(base_unique, base_unique + decode_blocks))

    return expanded, adjusted_prefill_tokens


def convert_trace(input_path: Path, output_path: Path,
                  src_block_size: int = 512,
                  dst_block_size: int = 16) -> dict:
    """
    Convert a Mooncake trace to Vidur format.

    Args:
        input_path: Path to input JSONL file
        output_path: Path to output CSV file
        src_block_size: Source block size (512 for Mooncake)
        dst_block_size: Target block size (16 for vLLM)

    Returns:
        Dictionary with conversion statistics
    """
    print(f"Converting {input_path.name}...")

    # Load raw trace
    with open(input_path) as f:
        records = [json.loads(line) for line in f]

    print(f"  Loaded {len(records):,} records")

    # Infer sessions
    session_ids = infer_sessions(records)
    num_sessions = len(set(session_ids))
    print(f"  Inferred {num_sessions:,} sessions")

    # Convert records
    converted = []
    for i, r in enumerate(records):
        unique_base = 5_000_000_000 + i * 1_000_000
        expanded_hashes, adjusted_tokens = expand_block_hashes(
            r['hash_ids'], r['input_length'], r['output_length'],
            src_block_size, dst_block_size, unique_base
        )

        converted.append({
            'arrived_at': float(r['timestamp']) / 1000.0,  # Convert ms to seconds
            'num_prefill_tokens': adjusted_tokens,
            'num_decode_tokens': r['output_length'],
            'block_hash_ids': json.dumps(expanded_hashes),
            'block_size': dst_block_size,
            'session_id': session_ids[i]
        })

    # Create DataFrame and save
    df = pd.DataFrame(converted)
    df.to_csv(output_path, index=False)

    # Statistics
    stats = {
        'num_requests': len(df),
        'num_sessions': num_sessions,
        'duration_hours': (df['arrived_at'].max() - df['arrived_at'].min()) / 3600,
        'avg_prefill': df['num_prefill_tokens'].mean(),
        'avg_decode': df['num_decode_tokens'].mean(),
        'max_prefill': df['num_prefill_tokens'].max(),
        'requests_per_session': len(df) / num_sessions if num_sessions > 0 else 0
    }

    print(f"  Saved to {output_path}")
    print(f"  Stats: {stats['num_requests']:,} requests, {stats['num_sessions']:,} sessions")
    print(f"         Duration: {stats['duration_hours']:.1f} hours")
    print(f"         Avg prefill: {stats['avg_prefill']:.0f}, Max: {stats['max_prefill']}")
    print()

    return stats


def main():
    parser = argparse.ArgumentParser(description='Convert Mooncake traces to Vidur format')
    parser.add_argument('--trace', type=str, default='all',
                        choices=['conversation', 'synthetic', 'toolagent', 'all'],
                        help='Which trace to convert')
    parser.add_argument('--input_dir', type=str, default='data/raw_traces',
                        help='Input directory with JSONL files')
    parser.add_argument('--output_dir', type=str, default='data/processed_traces',
                        help='Output directory for CSV files')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    traces = {
        'conversation': ('conversation_trace.jsonl', 'mooncake_conversation_trace_v2.csv'),
        'synthetic': ('synthetic_trace.jsonl', 'mooncake_synthetic_trace.csv'),
        'toolagent': ('toolagent_trace.jsonl', 'mooncake_toolagent_trace.csv'),
    }

    if args.trace == 'all':
        to_convert = list(traces.keys())
    else:
        to_convert = [args.trace]

    print("=" * 70)
    print("  Mooncake Trace Conversion")
    print("=" * 70)
    print()

    all_stats = {}
    for trace_name in to_convert:
        input_file, output_file = traces[trace_name]
        input_path = input_dir / input_file
        output_path = output_dir / output_file

        if not input_path.exists():
            print(f"  Skipping {trace_name}: {input_path} not found")
            continue

        all_stats[trace_name] = convert_trace(input_path, output_path)

    # Summary
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Trace':<15} {'Requests':>10} {'Sessions':>10} {'Req/Sess':>10} {'Avg Prefill':>12}")
    print("-" * 70)
    for name, stats in all_stats.items():
        print(f"{name:<15} {stats['num_requests']:>10,} {stats['num_sessions']:>10,} "
              f"{stats['requests_per_session']:>10.2f} {stats['avg_prefill']:>12,.0f}")
    print()


if __name__ == '__main__':
    main()
