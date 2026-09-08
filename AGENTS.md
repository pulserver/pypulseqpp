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

## Reading, and the two forms of a file

A sequence is written as Pulseq text or as Pulseq binary, and read back from
either -- told apart by what is in the bytes rather than by the name they were
stored under. The reader is the writer's inverse and is tested as one: a file
written, read and written again is the file it started as, byte for byte.

**The file is parsed whole before anything is registered.** `[BLOCKS]` comes
before the libraries it names and `[SHAPES]` comes last, so a reader that
registered as it went would be adding blocks whose events do not exist yet --
and a block is split into a definition and an instance as it is added, which
needs those events split already. Both forms fill a `Parsed` and hand it to
one builder, so the rules that turn a file into a sequence are written once.
Reading therefore forks exactly as building does, and a sequence off disk is
indistinguishable from one that was built.

**1.5.0 is the oldest revision that can be read.** Pulseq grew columns as it
went: 1.4 has no RF `center`, no `first` or `last` on an arbitrary gradient
and no ppm offsets. Those are not defaults that can be filled in -- the
reference toolbox recovers them by decompressing every waveform and walking
the block table -- so an older file is refused by version rather than read
with its columns one place out.

**What each form carries.** The binary form is the more faithful container for
everything except shapes: times cross as integer picoseconds and amplitudes as
float64, where the text form writes nine significant digits. Shape samples are
the exception -- float32 in binary, nine digits in text -- so a waveform comes
back within a float32 of itself and everything else comes back exactly. Only
the text form has a `[SIGNATURE]`, and only it is verified, on request.

A binary file always declares at least revision 1: the format arrived with
Pulseq 1.5.1, so a file claiming 1.5.0 claims a revision that had no way to
write it.

## Where the binary layout comes from

`pypulseq-matlab-like` is the authority for it, being a transcription of
MATLAB Pulseq's `writeBinary.m`. Where another implementation disagrees, that
one is followed. Two places this decided something:

- **A definition's name carries its length in front of it**, as an int32, and
  the value count is an int32 too -- not a NUL-terminated name and a
  single-byte count. The byte would have capped a definition at 255 values,
  which `SlicePositions` on a 256-slice acquisition exceeds.
- **`LABELNAMES` is written only when a label is one Pulseq does not define.**
  The section says what each label id is called, which is what lets a name
  outside Pulseq's table survive; but a reader predating it refuses any file
  carrying it, and for a label Pulseq defines the number alone is enough
  because every toolbox numbers those the same way.

`tests/test_interoperability.py` holds both directions against that toolbox
and skips when it is not installed, since it is not on PyPI.

## Definitions and instances

A scan is a handful of things played many times with different numbers in
them. Every event registered is therefore split in two: a **definition**, what
is the same every time it is played, and the per-playout parameters carried by
each block that plays it. A block is a definition of its own -- the
definitions its events play, and how long it lasts -- so the stream of block
definition ids is where the repeating unit becomes visible: a gradient echo
reads 1 2 3 4 1 2 3 4 whatever its phase encode is doing.

| | Definition | Instance |
|---|---|---|
| RF | magnitude, phase and time shapes; delay; center; use | amplitude, frequency and phase offsets, and their ppm forms |
| Trapezoid | rise, flat and fall times; delay | amplitude |
| Arbitrary gradient | time shape; delay | amplitude, waveform shape |
| ADC | sample count, dwell, delay | frequency and phase offsets, their ppm forms, phase modulation shape |
| Block | the definitions its RF and three gradients play, and its duration | the rows above, and the ADC definition it digitises with |
| Pure delay | nothing: every one of them is one definition | its duration, in `block_durations` |

Which column falls on which side is a statement about the hardware rather than
about the file. A gradient's *waveform* is on the instance side because a shot
really can arrive with its own arm, which is what a sparkling readout is; an
RF pulse's shapes are not, because nothing swaps a pulse envelope between
repetitions. The ADC is left out of the block definition altogether, so a
preparation shot playing the imaging shot's gradients with the digitiser off
is the same definition as the shot it stands in for, and a position digitised
two ways still repeats every shot rather than every pair. So is the extension
chain: a rotation and a label are things one playout does.

**A pure delay is one definition, and its duration is not part of it.** A
block that plays something lasts as long as its longest event, or as long as
the duration it was asked for if that is longer and it is padded out; either
way the duration follows from the content, so it belongs to the definition. A
block with no RF, no gradient, no ADC and no trigger or digital output plays
nothing, and an interpreter sets how long it waits there at run time -- so a
TI fill and the pad that follows it are one position waited at for two
different times, not two sequences. Labels, flags and a rotation may be
present; none of them makes the block play anything, and a rotation has no
gradient to remap. A trigger or a digital output does, wherever it sits in the
extension chain, so which chains carry one is recorded as they are built and
read off per block rather than walked.

**The fork happens where the reference is made.** Each `register_*` interns
its definition, `add_block` interns the block's, and reading a file forks as
it parses because reading registers its events. Nothing walks the sequence
afterwards to work it out, and the instance parameters are read out of the
event libraries on demand rather than stored a second time.

**Deduplication re-derives it.** A definition key names shapes by id, and
collapsing identical library rows moves those ids and shrinks the per-event
tables, so `remove_duplicates` ends by rebuilding every definition from the
rows that survived. Definition ids change across it, as shape and event ids
already do. Two events the file cannot tell apart are one definition once it
has run, which is the point at which that sentence is true at all: before it,
two separately registered but equal shapes split one pulse into two
definitions, and the stream is conservatively finer than the scan.

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
- **The fork says what the scan is.** `tests/test_structure.py` states each
  half of the split as a property of a sequence built to have it: a gradient
  echo repeats at four positions, a phase encode is one definition at many
  amplitudes, a sparkling readout is one definition with a waveform per shot,
  and a pulse registered twice over equal shapes is one definition once
  deduplication has run. `inversion_recovery_train` and `triggered_delays`
  are in the reference zoo so the same properties are held on sequences built
  by upstream rather than only on sequences written to have them.
- **A file read back is the file that was written.** Every reference sequence
  goes out as text and as binary and comes back through the reader, and what
  it writes the second time is compared with what it wrote the first. The
  reference toolbox's own files are read the same way, so the reader is held
  against the format as another implementation produces it. The 1.5.1 event
  kinds -- rotations, RF shims, and labels outside Pulseq's table -- have no
  upstream to compare against, so `tests/extended.py` builds them on the core
  and a round trip holds them.

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
