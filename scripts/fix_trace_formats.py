#!/usr/bin/env python3
"""
Fix trace file formats to be compatible with Vidur's trace request generator.

Required columns: arrived_at, num_prefill_tokens, num_decode_tokens
Optional columns: Any other columns are ignored

Usage:
    python scripts/fix_trace_formats.py data/processed_traces/arxiv_*.csv
    python scripts/fix_trace_formats.py --check-all
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def check_trace_format(filepath: Path) -> dict:
    """
    Check if trace file has required format.

    Args:
        filepath: Path to trace file

    Returns:
        Dict with status info
    """
    try:
        df = pd.read_csv(filepath, nrows=5)

        required_cols = ['arrived_at', 'num_prefill_tokens', 'num_decode_tokens']
        missing_cols = [col for col in required_cols if col not in df.columns]

        return {
            'valid': len(missing_cols) == 0,
            'missing_columns': missing_cols,
            'existing_columns': list(df.columns),
            'num_rows': len(pd.read_csv(filepath)),
        }
    except Exception as e:
        return {
            'valid': False,
            'error': str(e),
        }


def fix_arxiv_trace(input_path: Path, output_path: Path = None, qps: float = 10.0, force: bool = False):
    """
    Fix ArXiv trace by adding arrived_at column.

    Args:
        input_path: Input trace file
        output_path: Output file (default: overwrite input)
        qps: Queries per second for synthetic arrival times
        force: Force regeneration even if arrived_at exists
    """
    print(f"Loading: {input_path}")
    df = pd.read_csv(input_path)

    print(f"Original columns: {list(df.columns)}")
    print(f"Original rows: {len(df)}")

    # Check if arrived_at is missing
    if 'arrived_at' in df.columns:
        if not force:
            print("✓ File already has arrived_at column")
            return
        else:
            print("⚠ Removing existing arrived_at column (--force enabled)")
            df = df.drop(columns=['arrived_at'])

    # Generate synthetic arrival times using Poisson process
    print(f"Generating synthetic arrival times (QPS={qps})...")

    # Inter-arrival times from exponential distribution
    inter_arrival_times = np.random.exponential(1.0 / qps, size=len(df))
    arrived_at = np.cumsum(inter_arrival_times)

    # Add as first column
    df.insert(0, 'arrived_at', arrived_at)

    print(f"New columns: {list(df.columns)}")

    # Save
    output_path = output_path or input_path
    df.to_csv(output_path, index=False)

    print(f"✓ Saved fixed trace to: {output_path}")
    print(f"  Rows: {len(df)}")
    print(f"  Duration: {arrived_at[-1]:.1f}s")
    print(f"  Avg QPS: {len(df) / arrived_at[-1]:.2f}")


def fix_mooncake_trace(input_path: Path, output_path: Path = None):
    """
    Fix Mooncake trace (already has required columns, just validate).

    Args:
        input_path: Input trace file
        output_path: Output file (default: input)
    """
    print(f"Loading: {input_path}")
    df = pd.read_csv(input_path)

    required = ['arrived_at', 'num_prefill_tokens', 'num_decode_tokens']

    if all(col in df.columns for col in required):
        print(f"✓ Mooncake trace already valid")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Rows: {len(df)}")

        # Optionally select only required columns to reduce size
        if output_path:
            df_minimal = df[required].copy()
            df_minimal.to_csv(output_path, index=False)
            print(f"✓ Saved minimal trace to: {output_path}")
    else:
        missing = [col for col in required if col not in df.columns]
        print(f"✗ Mooncake trace missing columns: {missing}")


def main():
    parser = argparse.ArgumentParser(
        description="Fix trace file formats for Vidur",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check all traces
  python scripts/fix_trace_formats.py --check-all

  # Fix ArXiv trace (add arrival times)
  python scripts/fix_trace_formats.py data/processed_traces/arxiv_*.csv

  # Fix with custom QPS
  python scripts/fix_trace_formats.py data/processed_traces/arxiv_*.csv --qps 15.0

  # Save to new file
  python scripts/fix_trace_formats.py arxiv.csv --output arxiv_fixed.csv
        """
    )

    parser.add_argument(
        'input_file',
        type=str,
        nargs='?',
        help='Input trace file to fix'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='Output file (default: overwrite input)'
    )

    parser.add_argument(
        '--qps',
        type=float,
        default=10.0,
        help='Queries per second for synthetic arrivals (default: 10.0)'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Force regeneration of arrived_at column even if it exists'
    )

    parser.add_argument(
        '--check-all',
        action='store_true',
        help='Check all trace files in data/processed_traces/'
    )

    args = parser.parse_args()

    # Check all mode
    if args.check_all:
        traces_dir = Path('data/processed_traces')
        print(f"Checking all traces in {traces_dir}\n")
        print("=" * 80)

        for trace_file in sorted(traces_dir.glob('*.csv')):
            print(f"\n📄 {trace_file.name}")
            status = check_trace_format(trace_file)

            if status['valid']:
                print(f"  ✓ Valid format")
                print(f"  Rows: {status['num_rows']}")
                print(f"  Columns: {', '.join(status['existing_columns'])}")
            else:
                print(f"  ✗ Invalid format")
                if 'error' in status:
                    print(f"  Error: {status['error']}")
                else:
                    print(f"  Missing columns: {', '.join(status['missing_columns'])}")
                    print(f"  Existing columns: {', '.join(status['existing_columns'])}")

        print("\n" + "=" * 80)
        return 0

    # Fix specific file
    if not args.input_file:
        parser.print_help()
        return 1

    input_path = Path(args.input_file)
    output_path = Path(args.output) if args.output else None

    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        return 1

    # Determine trace type and fix
    if 'arxiv' in input_path.name.lower():
        fix_arxiv_trace(input_path, output_path, qps=args.qps, force=args.force)
    elif 'mooncake' in input_path.name.lower():
        fix_mooncake_trace(input_path, output_path)
    else:
        # Generic fix - check and fix if needed
        status = check_trace_format(input_path)
        if status['valid']:
            print(f"✓ {input_path.name} is already valid")
        else:
            print(f"Attempting to fix {input_path.name}...")
            if 'arrived_at' not in status['existing_columns']:
                fix_arxiv_trace(input_path, output_path, qps=args.qps)
            else:
                print(f"✗ Unknown issue: {status}")

    return 0


if __name__ == "__main__":
    import numpy as np
    sys.exit(main())
