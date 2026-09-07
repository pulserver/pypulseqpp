"""What a block costs, measured rather than asserted.

A protocol-scale scan is millions of blocks, so the design loop's cost is set
by the calls inside it, and loading a sequence that already exists is set by
whether its libraries cross one row at a time or one array at a time. This
reports both, which are the numbers to watch when the bindings change.

    python benchmarks/throughput.py
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


def write_a_scan(count: int) -> None:
    sequence = _ext.Sequence()
    sequence.register_trap(TRAPEZOID)
    add_block = sequence.add_block
    for _ in range(count):
        add_block(0, 1, 0, 0, 0, 0, BLOCK_DURATION)
    _ext.write_text(sequence, True)


def _loadable(count: int):
    """A gradient library and block table of ``count`` rows, as dense arrays."""
    traps = np.tile(TRAPEZOID, (count, 1)) * np.linspace(1.0, 2.0, count)[:, None]
    slots = np.arange(1, count + 1, dtype=np.int32)
    events = np.zeros((count, 6), dtype=np.int32)
    events[:, 1] = slots
    return traps, slots, events, np.full(count, BLOCK_DURATION)


def load_row_by_row(count: int) -> None:
    traps, _, _, _ = _loadable(count)
    sequence = _ext.Sequence()
    register = sequence.register_trap
    for row in traps:
        register(row)
    add_block = sequence.add_block
    for identifier in range(1, count + 1):
        add_block(0, identifier, 0, 0, 0, 0, BLOCK_DURATION)


def load_in_bulk(count: int) -> None:
    traps, slots, events, durations = _loadable(count)
    sequence = _ext.Sequence()
    sequence.set_gradients(traps, np.zeros((0, 6)), slots)
    sequence.set_blocks(events, durations)


def main() -> None:
    building = [
        ("add_block", add_blocks, 1_000_000),
        ("register_trap", register_trapezoids, 200_000),
        ("add_block + write_text", write_a_scan, 200_000),
    ]
    print(f"{'building a sequence':30s} {'ns/block':>10s} {'M blocks/s':>12s}")
    for name, work, count in building:
        nanoseconds = _best_of(work, count)
        print(f"  {name:28s} {nanoseconds:10.1f} {1e3 / nanoseconds:12.3f}")

    loading = [
        ("row by row", load_row_by_row, 200_000),
        ("in bulk", load_in_bulk, 200_000),
    ]
    print(f"\n{'loading one that exists':30s} {'ns/row':>10s} {'M rows/s':>12s}")
    for name, work, count in loading:
        nanoseconds = _best_of(work, count, repeats=3)
        print(f"  {name:28s} {nanoseconds:10.1f} {1e3 / nanoseconds:12.3f}")


if __name__ == "__main__":
    main()
