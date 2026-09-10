"""Measure per-call registration and block-insertion costs.

Run with python benchmarks/throughput.py.
"""

from __future__ import annotations

import time

import numpy as np

from pypulseqpp import _ext

TRAPEZOID = np.array([1.0e6, 1.0e-4, 1.0e-3, 1.0e-4, 0.0])
BLOCK_DURATION = 1.2e-3


def _best_of(work, count: int, repeats: int = 5) -> float:
    """Nanoseconds per iteration, taking the fastest run."""
    best = float("inf")
    for _ in range(repeats):
        started = time.perf_counter()
        work(count)
        best = min(best, time.perf_counter() - started)
    return best / count * 1e9


def add_blocks(count: int) -> None:
    sequence = _ext.Sequence()
    sequence.register_trap(TRAPEZOID)
    add_block = sequence.add_block
    for _ in range(count):
        add_block(0, 1, 0, 0, 0, 0, BLOCK_DURATION)


def register_trapezoids(count: int) -> None:
    sequence = _ext.Sequence()
    register = sequence.register_trap
    for scale in range(count):
        register(TRAPEZOID * (1.0 + scale))


def register_scaled_trapezoids(count: int) -> None:
    """What a phase-encode loop registers: one timing at many amplitudes."""
    sequence = _ext.Sequence()
    register = sequence.register_trap
    row = TRAPEZOID.copy()
    for step in range(count):
        row[0] = 1.0 + step
        register(row)


def write_a_scan(count: int) -> None:
    sequence = _ext.Sequence()
    sequence.register_trap(TRAPEZOID)
    add_block = sequence.add_block
    for _ in range(count):
        add_block(0, 1, 0, 0, 0, 0, BLOCK_DURATION)
    _ext.write_text(sequence, True)


def main() -> None:
    measurements = [
        ("add_block", add_blocks, 1_000_000),
        ("register_trap", register_trapezoids, 200_000),
        ("register_trap, scaled", register_scaled_trapezoids, 200_000),
        ("add_block + write_text", write_a_scan, 200_000),
    ]
    print(f"{'call':24s} {'ns/block':>10s} {'M blocks/s':>12s}")
    for name, work, count in measurements:
        nanoseconds = _best_of(work, count)
        print(f"{name:24s} {nanoseconds:10.1f} {1e3 / nanoseconds:12.3f}")


if __name__ == "__main__":
    main()
