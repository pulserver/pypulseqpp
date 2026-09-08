<!-- Generated from AGENTS.md by scripts/sync_agent_docs.sh. Do not edit. -->

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
detection, block timing against a system's rasters and dead times, gradient
amplitude, slew and continuity checks, PNS and mechanical resonance, k-space and gradient-moment calculation, and sequence-level
operations such as FOV transformation and tiling. It does not own anything
that needs a scanner or a reconstruction in the picture. Segmentation, the
scanner-side execution stream, protocol contracts and consoles live in
`pulserver`; pulse, trajectory and sampling *design* lives in its own package.
If a change here needs to know which vendor will play the sequence, it belongs
elsewhere.

The runtime dependencies are NumPy and PyPulseq. PyPulseq is the API this
package replaces and the facade re-exports its namespace, so its factories
build the events and this builds the sequence under them.

`pypulseq-matlab-like` is a test dependency and nothing more -- the
transcription of MATLAB Pulseq that defines the file format, used for
byte-parity fixtures, never imported by the package. It is not on PyPI;
`tests/seq/` carries its reference corpus so the reader is tested without it,
and the tests that build sequences skip when it is absent.

## The Python facade

`import pypulseqpp as pp` is the only import a script needs. Everything
upstream exposes is re-exported, and every callable goes through
`_events.interoperating` -- not only the factories, because `calc_duration`,
`align`, `split_gradient` and `rotate` all take events and upstream implements
them with `isinstance` checks and `deepcopy`, neither of which a compiled
event satisfies. The decorator hands them a namespace on the way in and
converts what comes back, so upstream's helpers work against events they were
never written for.

A `make_*` factory hands back an event whose scalar fields are `PyMemberDef`
offsets rather than dictionary entries, and `add_block` unpacks a whole block
and registers it in one compiled call. On a gradient echo loop that is 780 ns
a block against upstream's 59.6 us.

Four things are ours rather than upstream's, and three of them are meant to
go. `Sequence` stays: upstream's is a different implementation, and this is
the one with the compiled core under it. `make_rotation` and `make_rf_shim`
arrived with Pulseq 1.5.1 and upstream 1.5.0 does not have them. `make_label`
is here because upstream's refuses a name outside the list Pulseq defines,
and a label here is named rather than numbered -- see the section on that.

## Layout

| Path | What lives there |
|---|---|
| `src/cpp/pulseq/` | The C++17 core: event libraries, the block table, the shape codec, the writers. It knows nothing about Python. |
| `src/cpp/bindings/` | The pybind11 sources, building one extension module, `pypulseqpp._ext`. |
| `src/pypulseqpp/` | The Python package: the facade over the core. `_events.py` converts between PyPulseq's namespaces and the compiled events and holds the decorators; `_sequence.py` is the sequence a script builds; `_make_*.py` are the factories upstream does not have. |
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

**What has been worked out about a sequence is kept, and the core says when
to drop it.** The timing check asks what the gradients slew at and how long
the whole sequence lasts; both are passes over the block table, so `Sequence`
keeps them. Every mutation of the core bumps a revision, in step with the
deduplication claim being dropped, and the kept answers are read through a
guard that compares one integer -- so a block added, a duration written, an
axis scaled, a soft delay applied or duplicates collapsed all invalidate them,
and none of it costs the design loop an attribute write per block.

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

## What a report says a sequence is

`test_report` reads a sequence back as a description of an experiment: the
echo and repetition times, the flip angles, what the encoding covers, and how
hard the gradients are driven. Most of it is counting, integrating the sampled
trajectory or reading the corner waveforms. Two answers are worked out, and
both are compiled because both are per-playout questions with per-definition
answers.

**A flip angle belongs to the envelope, not to the playout.** How far a pulse
tips is the integral of its envelope times the amplitude it is played at, and
the envelope is what the RF definition holds. So the integral is taken once
per definition and multiplied by each distinct amplitude played through it: an
inversion train sweeping one pulse over a thousand flip angles is one integral
and a thousand multiplies, and playing each of those a hundred times costs
nothing further. Before deduplication a pulse registered twice brings shapes
of its own and so splits into two definitions, which is the same
conservatively-finer answer the definition stream gives.

**What the encoding covers is a pass over every sample.** The sampled
trajectory is binned onto a lattice of its own extent over four million, which
says how many distinct positions each axis visits, how often one is revisited
-- slices, averages, contrasts -- and whether the positions fill the grid they
span. A coordinate one cell from one already seen is the same one, since a
position reached along two different ramps can land either side of a boundary.

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

**An older file is converted, not merely parsed.** Every revision back to
1.2.0 is read, and each moved something. 1.5 added an RF pulse's `center`, an
arbitrary gradient's `first` and `last` sample, and the ppm offsets; before 1.4
a gradient carries no time shape, a block's duration is an index into a
`[DELAYS]` section rather than a count of rasters, a zero trapezoid is written
with no ramps, and 1.2 has no extension column at all.

None of that is a default that can be filled in, so it is derived. The centre
comes from the pulse's own envelope, taking the middle of its peak. The
gradient edges come from walking the block table in playing order: `last` is
the waveform's end, extrapolated the way the factory would have, and `first`
is where the axis was left by the block before, which is zero unless the
previous gradient ran to that block's end. A pre-1.4 duration is the longest
thing the block plays. And every shape is decoded and re-encoded, because
before 1.4 an encoded shape whose length happened to equal its sample count
could not be told from one that was never encoded.

The corpus holds one sequence at 1.2.0, 1.3.0, 1.3.1, 1.4.0, 1.4.1, 1.4.2 and
1.5.0, so what the 1.5.0 file says is what the others are held to, and the
reference reader is the arbiter where they legitimately differ.

Two things a 1.4 file cannot give back. It has no `use` column, so what a
pulse is *for* stays undefined rather than being guessed from its flip angle;
and one derived `last` differs from the value the design knew, because an
extrapolation is not the original. The reference toolbox derives the same
number from the same file, which is what makes that the format's limit rather
than a difference between readers. Before 1.4 two more things move: an
extended trapezoid is held in fewer shapes, and 1.2.0's blocks last longer
than 1.5.0's -- the reference reader agrees on both.

**A file can be written for a scanner from before 1.5.** `write_text_v141`
produces Pulseq 1.4.1: the ppm offsets are folded back into absolute hertz at
a gyromagnetic ratio and a field, which is the only place those two are needed
and why they are arguments; an RF pulse's centre and use go, and so do an
arbitrary gradient's first and last sample, because 1.4 has no column for any
of them. Its output is byte-identical to the reference toolbox's for every
sequence 1.4.1 can express.

Two things it will not do quietly. A soft delay is left out with a warning, as
the reference does. A rotation or an RF shim makes it refuse: the reference
drops both silently, and a file missing a rotation is a different scan rather
than a coarser description of the same one.

**What each form carries.** The binary form is the more faithful container for
everything except shapes: times cross as integer picoseconds and amplitudes as
float64, where the text form writes nine significant digits. Shape samples are
the exception -- float32 in binary, nine digits in text -- so a waveform comes
back within a float32 of itself and everything else comes back exactly. Only
the text form has a `[SIGNATURE]`, and only it is verified, on request.

A binary file always declares at least revision 1: the format arrived with
Pulseq 1.5.1, so a file claiming 1.5.0 claims a revision that had no way to
write it.

## Where the format comes from

`pypulseq-matlab-like` is the authority, being a transcription of MATLAB
Pulseq. Where another implementation disagrees -- upstream `pypulseq`
included -- that one is followed, and what this package writes is compared
against it byte for byte. Upstream differs from it in ways that are not
cosmetic: an `OFF` label missing from the table and `TRID` in the wrong place,
so labels resolve to the wrong names; `freqPPm` for `freqPPM`; a soft delay's
offset as `%.0f` rather than `%g`.

Every file declares revision 1.5.1, whatever the sequence uses and whatever
it came in declaring: a writer says which revision of the format it produced,
not which subset a sequence happened to use. `TotalDuration` is not written,
because the authority records it when reporting on a sequence rather than
when writing one.

Two places this decided something about the binary layout:

- **A definition's name carries its length in front of it**, as an int32, and
  the value count is an int32 too -- not a NUL-terminated name and a
  single-byte count. The byte would have capped a definition at 255 values,
  which `SlicePositions` on a 256-slice acquisition exceeds.
- **A label's name travels in `[DEFINITIONS]`, not in a section.** See below.

`tests/test_interoperability.py` holds both directions against that toolbox
and skips when it is not installed, since it is not on PyPI.

## A label the builtin table does not carry

A label is named, not numbered. The text form writes the name and reads it
back, and `label_id` mints one for a name it has not seen, so a sequence may
use a label Pulseq does not define with nothing to configure first -- where
the reference toolbox makes the caller extend a vocabulary by hand.

The binary form writes the **number**, and a number means something only
against a table. `builtin_labels()` is that table, in the reference toolbox's
order, and it is a seed for the numbering rather than a statement about what a
label may be: with any other order our `NOISE` reads there as `IMA`. For a
name past the end of it no fixed list can help, which is the point of allowing
one.

So the names past the table are listed in `[DEFINITIONS]`, as `CustomLabels`,
in the order they were minted, and a number above the table's length resolves
by position. **Only the custom names go there** -- the builtins are shared, so
listing them would be overhead saying what every reader already knows. Both
forms carry definitions already and a reader must tolerate a key it does not
know, so a file using an invented label stays readable by anything that reads
the format at all; a section of its own would not have.

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

  One sequence is held differently and says why. `gre_with_noise_scan` drops
  a degenerate block, and the reference toolbox keys its blocks in a
  dictionary so the gap stays in the numbering where this package closes it.
  A block id is a label nothing refers to, so what is held there is that
  every row after that column is the same, in the same order.
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