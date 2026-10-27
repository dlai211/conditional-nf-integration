#!/usr/bin/env python3
"""Prefix each stdin line with elapsed wall-clock seconds.

Used to pipe a training run into its log so the loss/ESS versus cumulative
compute-time plots have a real time axis.  The training prints with ``-u``,
so the timestamp is emitted close to the moment the line is written.
"""
from __future__ import annotations

import sys
import time


def main() -> None:
    start = time.monotonic()
    for line in sys.stdin:
        print(f"[{time.monotonic() - start:9.3f}] {line}", end="", flush=True)


if __name__ == "__main__":
    main()
