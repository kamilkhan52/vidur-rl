"""Lodestar in collection mode (no offline rows; AIBrix prefix-cache-and-load until 5000 online samples): logs its
route-time feature rows to $LODESTAR_DUMP. Used only on training windows to build Lodestar's offline rows."""
import os

from evolve.priorwork.lodestar_core import build

choose, observe, _inst = build(None, dump=os.environ["LODESTAR_DUMP"])
