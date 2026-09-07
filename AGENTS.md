# pypulseqpp — agent instructions

<!--
This file is the SOURCE. CLAUDE.md and GEMINI.md are generated from it by
scripts/sync_agent_docs.sh, which pre-commit runs. Edit this file, never those.
-->

## What this package is

Fast drop-in PyPulseq replacement over a C++ sequence core, with hardware safety checks.

The contract is PyPulseq's own API: a design script written against
`pypulseq` must run here unchanged, with the same functions, the same
signatures and the same `.seq` output. What differs is what each call does
underneath. Events are compact compiled objects, `add_block` is one compiled
call, and reading and writing, in text and in binary, are C++.

The package owns everything that is true about a sequence in isolation:
parsing and writing, event deduplication, structural TR and base-block
detection, gradient amplitude, slew and continuity checks, PNS and mechanical
resonance, k-space and gradient-moment calculation, and sequence-level
operations such as FOV transformation and tiling. It does not own anything
that needs a scanner or a reconstruction in the picture. Segmentation, the
scanner-side execution stream, protocol contracts and consoles live in
`pulserver`; pulse, trajectory and sampling *design* lives in its own package.
If a change here needs to know which vendor will play the sequence, it belongs
elsewhere.

The runtime dependency is NumPy alone. Upstream `pypulseq` is a test
dependency, used for byte-parity fixtures, and is never imported by the
package.

## Layout

| Path | What lives there |
|---|---|
| `src/cpp/pulseq/` | The C++17 core: event libraries, the block table, the shape codec, the writers. It knows nothing about Python. |
| `src/cpp/bindings/` | The pybind11 sources, building one extension module, `pypulseqpp._ext`. |
| `src/pypulseqpp/` | The Python package: the PyPulseq-compatible API over the core. |
| `tests/` | pytest. `reference.py` builds the reference sequences with upstream, `convert.py` loads one into the core, and `test_parity.py` compares what the two write. |

## Build and test

```bash
pip install -e .[dev]
bash scripts/format_and_lint.sh   # rewrites in place; --check to verify only
pytest -q
```

Build and test steps are mandatory before reporting a change complete. Run them
and report the exact output; do not assume success.

## Language rules

**`src/cpp/` is C++17.** It is the hot path for million-block sequences:
measure before adding an allocation per block. The extension links nothing but
the standard library and threads, so a wheel is self-contained on Linux,
macOS and Windows and an end user never needs a compiler.

**Python targets 3.10+.** Python code is the API surface and the glue; a loop
over blocks in Python is a bug, not a slow path.

## Performance

The design loop is the hot path: one call per block, and a protocol-scale scan
has millions of them. Two rules follow, and both are easy to undo by accident.

**A per-block call is bound by hand with `METH_FASTCALL`, not by pybind11.**
The arguments arrive as a C array of borrowed references, so nothing is
allocated and no tuple is built. `add_block` is bound this way, and
`test_adding_a_block_goes_through_the_fast_calling_convention` fails if it
stops being. Passing a bound object per block instead costs an order of
magnitude, because constructing that object is then the whole call.

**The block table is read back as a view, not a copy.** `block_events` and
`block_durations` return arrays pointing straight into the table, which is
what keeps a million-row table free to read a column out of. The array owns a
share of the buffer and the table is copied before it is written while a view
is out, so a view is a snapshot: it does not see later writes, and it stays
valid even if the sequence is collected. The copy-before-write check is a
capacity comparison on the per-block path and an atomic one only when the
table actually grows -- keep it that way.

**A call that does real work releases the GIL.** Deduplication, shape
compression and writing all run without it.

`benchmarks/throughput.py` reports what a block costs. Run it before and after
touching the bindings, and quote what came back rather than asserting an
improvement.

The registration calls are the remaining cost: a design loop that registers
each event from Python pays a binding crossing per event. The answer is
`add_block_events(*events)`, one fastcall that unpacks compiled event objects
and registers them inside C++, so a block costs one crossing rather than one
per event. That needs the compiled event types, so it lands with the Python
API rather than before it.

## What a shape is played as

A `[SHAPES]` entry does not say what it is; the file says so only where an
event refers to it. Each entry therefore carries a mask of `ShapeRole`, set
where the reference is made -- `register_rf` marks its magnitude, phase and
time shapes, `register_arbitrary` its waveform and times, `register_adc` its
phase modulation -- so "every gradient waveform" is answered without walking
the event libraries. A file read back fills the mask in on the way past,
because reading registers its events too, and nothing in the format changes.

It is a mask rather than a tag because deduplication merges shapes holding the
same numbers, and the merge ORs the roles: after it, one entry really is
played both ways.

## Tests

pytest with plain functions and fixtures — never `unittest.TestCase`. A test
name states the invariant it protects, so a failure reads as a sentence.

Two invariants hold everything else up, and each has a test:

- **Parity.** The `.seq` a reference sequence writes here is byte-identical to
  what upstream `pypulseq` writes for it, signature included. The comparison
  is live rather than against a checked-in file, so it cannot go stale, and it
  runs with deduplication both on and off: a sequence that agrees before
  collapsing identical library rows and disagrees after has a renumbering bug
  rather than a writing bug. A new event kind is not finished until it appears
  in a sequence in `tests/reference.py`.

  One divergence is deliberate and has a test of its own. Upstream's
  deduplication softens a logarithm with a `1e-12` floor, so a sample below
  that keeps four significant digits where nine were asked for. This package
  does not floor, and writes the sample the pulse plays.
  `BLUNTED_BY_UPSTREAM` names the reference sequences that reach it.
- **Fast path equals plain path.** Wherever a compiled call stands in for a
  calculation PyPulseq does in Python, a test holds the two equal on the
  reference sequences. Speed is never taken on assertion.

## Comments and docstrings

Write for someone reading the code as it is now, who has no memory of any
earlier version of it. **Never** write text whose subject is the history of the
code. Banned in comments, docstrings and prose alike:

- "used to", "was once", "no longer", "previously", "now that", "this replaces",
  "the old X", "before the fix"
- justifying the present shape by contrast with a shape that is gone
- naming a bug that has been fixed, or the session that fixed it
- restating what the code plainly says

A docstring carries what a caller needs: one line of what, Parameters, Returns,
Raises. A comment earns its place only by explaining a non-obvious algorithm or
a choice a reader would otherwise undo — and even then, prefer a well-named
function or a test whose name states the invariant, because those cannot go
stale silently. When tempted to explain *why not the other way*, write a test.

Stale comments are actively harmful. Deleting an outdated comment is always
correct; rewriting one to describe the change is not.

## Documentation style

The audience is MR scientists. Write in the vocabulary of pulse sequences and
physics, not of software architecture. Never justify a design by describing the
design it replaced.

Do not print a measured constant that is not guaranteed across releases or
hardware. Name the symbol and where it comes from, and let the build supply the
number. Benchmark tables in the README are regenerated by the benchmark script,
not typed in.
