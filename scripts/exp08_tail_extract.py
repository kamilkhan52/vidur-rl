#!/usr/bin/env python3
"""EXP08 tail extractor: same layout and metrics as exp09_tail_extract.py (kept for the notes that cite it)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp09_tail_extract import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
