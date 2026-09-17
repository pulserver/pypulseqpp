# The Pulseq file format

[Pulseq](https://pulseq.github.io) is an open file format for MR pulse
sequences. A `.seq` file states what is played, on which channel, for how
long — the complete prescription of an acquisition, portable between sites and
vendors. Everything `pypulseqpp` builds, analyses and writes is that
description, so this page states what it contains and which of its conventions
the API carries.

The authoritative definition is the [Pulseq
specification](https://pulseq.github.io/specification.pdf) and the MATLAB
reference implementation. What follows is the part of it a caller has to know
to read the API reference.

## Blocks and events

A sequence is an ordered list of **blocks**. A block has a duration and at most
one event per channel: one RF pulse, one gradient on each of `x`, `y` and `z`,
one ADC window, and a chain of extensions. The events of a block are played
concurrently, each with its own delay from the block start, and blocks are
played back to back with no gap between them. There is no nesting, no loop and
no branch: the file is the flattened playout order, and a scan loop exists only
in the script that wrote it.

A block longer than the events it holds is a delay, which is how a repetition
time is realised.

Events are stored in **libraries** and referenced from the block table by
integer id, so an event played ten thousand times is written once:

```
[BLOCKS]
# NUM  DUR  RF  GX  GY  GZ  ADC  EXT
    1  694   1   0   0   1    0    2
    2  356   0   2   3   4    0    0
    3  268   0   5   0   0    1    0
```

RF
: An amplitude in Hz, ids into a magnitude, a phase and optionally a time
  shape, a delay, frequency and phase offsets, a centre time, and a `use` —
  excitation, refocusing, inversion, saturation, preparation. The `use` is what
  lets an analysis tell which pulses start a shot and which refocus one, and is
  why {func}`~pypulseqpp.make_slr_pulse` and the other factories take it.

Gradients
: Two kinds share one column. A *trapezoid* is an amplitude with rise, flat and
  fall times; an *arbitrary* gradient is a shape id, with an optional time
  shape when the samples do not fall on the gradient raster, and endpoint
  amplitudes. The two occupy separate tables and share the file's id space.

ADC
: A sample count, a dwell time, a delay, frequency and phase offsets, and an
  optional per-sample phase modulation.

Shapes
: Sample arrays, compressed by a run-length encoding of their derivative, which
  is why a linear ramp of a thousand samples costs three numbers. A shape is
  stored normalised, so the amplitude that scales it belongs to the event
  rather than to the samples.

Amplitudes are in **Hz and Hz/m** rather than in tesla, so a file means the same
thing on any nucleus a scanner is tuned to: the gyromagnetic ratio enters only
where a physical amplitude is wanted, which is why
{class}`~pypulseqpp.Opts` carries `gamma` and why the checks in
{doc}`safety/index` convert with it.

Times sit on **rasters** the file declares in `[DEFINITIONS]`: an RF raster, a
gradient raster, an ADC raster and a block-duration raster. A time that is not
an integer multiple of the raster its event is played on cannot be addressed by
the sequencer, which is what {meth}`~pypulseqpp.Sequence.check_timing` reports
and what {func}`~pypulseqpp.round_to_raster` and
{func}`~pypulseqpp.calc_adc_timing` exist to avoid.

## Extensions

The extension chain is how a block carries what the core format has no column
for. Each block's `EXT` id points at a linked list of typed rows:

`LABELSET` and `LABELINC`
: Counters and flags. The counters (`LIN`, `PAR`, `SLC`, `ECO`, `REP`, `AVG`,
  `SET`, `SEG`, `PHS`, `ACQ`) are the encoding indices a reconstruction sorts
  the acquisition by; the flags (`NAV`, `REV`, `IMA`, `NOISE`, `REF`, …) state
  what an acquisition is for. Label values are *sticky*: a value set on one
  block holds until it is set or incremented again, so the file carries one row
  where a counter changes rather than one row per block.
  {meth}`~pypulseqpp.Sequence.evaluate_labels` walks the block table and returns
  the running values.

`ROTATIONS`
: A quaternion rotating that block's gradients into the physical frame. One row
  per orientation, rather than a rotated copy of every waveform, is what keeps
  a radial or spiral scan's shape library the size of one interleaf.

`TRIGGERS`
: Wait on, or emit, a hardware signal.

`RF_SHIMS`
: Per-transmit-channel magnitude and phase, for parallel transmission.

`SOFT_DELAY`
: A delay whose duration is adjusted at the console without rewriting the file.

## Definitions, chains and the signature

`[DEFINITIONS]` is free-form key/value metadata. Some keys are conventional —
`FOV`, `Name`, the raster times, `TotalDuration` — and the rest is whatever the
writer wants a reader to know. The sequences this package ships write the
prescription and the k-space geometry a reconstruction needs there: the matrix
size, the echo and repetition times, the index of the k-space centre line and
sample, the slice positions.

`NextSequence` links files into a chain. A calibration prescan and the imaging
scan it belongs to are two files played in order rather than one file with a
mode flag, which is what
{meth}`~pypulseqpp.sequences.SequenceApp.write` produces from an application's
{meth}`~pypulseqpp.sequences.SequenceApp.prescans`, and what keeps each file one
repeating unit.

`[SIGNATURE]` closes the file with an MD5 of everything above it, so a reader
can establish that the file is the one that was written.

## Deduplication and structure

Two sequences with identical playout can differ greatly in file size, because
the block table references library entries and nothing forces equal events to
share one. Writing deduplicates: equal shapes are merged, equal events are
merged, and the block table is renumbered. Registration ids are therefore
local to a sequence and are not stable across deduplication.

Underneath the libraries, an event is stored as a fixed **definition** — the
timing and shape data that does not vary from one playout to the next — and a
set of **instances** carrying the playout parameters. A phase-encode gradient
scaled per line is one definition and one instance per line; an arbitrary
gradient whose waveform changes is a new definition. This separation is what
lets the package detect that a run of blocks repeats, which is the repetition
the SAR check of {doc}`safety/sar` averages over.

The format itself states the playout order and nothing above it. Nothing in the
file says which run of blocks is a repetition, which events belong to one shot,
or what a calibration region is; a consumer that needs that structure derives it
from the content.

## Revisions

`pypulseqpp` writes Pulseq 1.5.1 by default. Revision 1.4.1, which
{meth}`~pypulseqpp.Sequence.write_v141` produces, has no column for an RF
centre time or `use`, no gradient endpoint amplitudes, and no ppm offsets: the
writer folds a ppm offset into an absolute offset using the gyromagnetic ratio
in Hz/T and the field strength in tesla, warns when it omits a soft delay, and
refuses a sequence carrying rotation or RF-shim extensions, which that revision
cannot express.

Reading accepts earlier revisions. Legacy text conversion derives the RF centre
times, gradient endpoint amplitudes and pre-1.4 block durations the file does
not carry; an RF `use` that is not stated stays undefined rather than being
guessed.

The binary form, {meth}`~pypulseqpp.Sequence.write_binary`, carries the same
content: records are little-endian, times are integer picoseconds and shape
samples are float32. Both forms take an optional MD5 signature.

## Where to go next

* {doc}`safety/index` — what the checks of a finished sequence compute, and
  against what.
* {doc}`../generated/gallery/01-basics/01-events-and-blocks` — the same
  vocabulary as executable code.
* {doc}`../api/sequence` — the container, the system limits and the
  field-of-view transforms.
