# Events and blocks

```{admonition} TL;DR
:class: tldr

- A sequence is an ordered list of blocks played back to back without a gap.
  The file holds a flattened event schedule without loops, branches or nested
  blocks.
- A block has a duration and at most one event per channel, and its events are
  played concurrently, each with its own delay. A block whose duration exceeds
  the extent of its events is a delay, which is how an echo time or a
  repetition time is realized.
- Events are stored in libraries, one row per distinct event, and the block
  table refers to them by integer id; indices are 1-based and a zero means no
  event on that channel. A shape is stored normalized, and the amplitude that
  scales it belongs to the event.
- Amplitudes are in Hz for RF and Hz/m for gradients rather than in tesla and
  tesla per metre. The gyromagnetic ratio in Hz/T, `gamma` on
  {class}`~pypulseqpp.Opts`, enters only where a physical amplitude is
  required.
- Extensions carry information for which the block table has no column: labels
  (`LABELSET`, `LABELINC`), whose values remain in force until set or
  incremented again, rotations, triggers, RF shims and soft delays.
```

[Pulseq](https://pulseq.github.io) is an open file format for MR pulse
sequences. A `.seq` file specifies event timing and channel assignment for a complete,
portable acquisition. `pypulseqpp` builds, analyses and writes that description, and its
Python interface preserves the format's conventions.

The authoritative definition is the [Pulseq
specification](https://pulseq.github.io/specification.pdf) and the MATLAB
reference implementation.

## Blocks

A sequence is an ordered list of **blocks**. A block has a duration and at most
one event per channel: one RF pulse, one gradient on each of `x`, `y` and `z`,
one ADC window, and a chain of extensions.

Two rules fix the timing completely:

1. The events of a block are played concurrently, each with its own delay from
   the start of the block.
2. Blocks are played back to back, with no gap between them.

The file contains a flattened event schedule without loops, branches or nested
blocks. Scan-loop control exists only in the authoring program.

A block whose duration exceeds the extent of its events is therefore a delay,
and this is how a repetition time is realized. A two-dimensional gradient-echo
repetition, for example, is six blocks: the pulse with its slice-selection
gradient, an echo-time delay carrying the slice rephaser, the prewinders, the
readout gradient with its ADC, the spoilers, and a delay that completes the
repetition time. Which of the middle blocks carries the wait is a design
choice; that the wait is a block duration rather than a field of its own is
the format.

```{figure} ../../generated/figures/gre_repetition_blocks.png
One repetition of a two-dimensional gradient echo, over its six blocks,
numbered in the order just listed. Blocks 2 and 6 last longer than their
events, and that difference is the echo time and the repetition time.
```

## Event libraries

Events are stored in **libraries** and referenced from the block table by
integer id, so an event played ten thousand times is stored once:

```
[BLOCKS]
# NUM  DUR  RF  GX  GY  GZ  ADC  EXT
    1  694   1   0   0   1    0    2
    2  356   0   2   3   4    0    0
    3  268   0   5   0   0    1    0
```

A zero means the block has no event on that channel. Block and library indices
are 1-based.

```{figure} ../../generated/figures/block_table_and_libraries.png
The block table of a written eight-line gradient-echo file, the libraries its
cells index, and the shape library the RF and gradient rows index in turn. Each
library holds one row per distinct event, however many blocks play it, so the
tables below the block table are far shorter than it is.
```

RF
: An amplitude in Hz, ids into a magnitude, a phase and optionally a time
  shape, a delay, frequency and phase offsets, a centre time, and a `use` —
  excitation, refocusing, inversion, saturation or preparation. The `use`
  identifies which pulses begin a shot and which refocus one, which is why
  the pulse factories such as {func}`~pypulseqpp.make_slr_pulse` accept it and
  why {meth}`~pypulseqpp.Sequence.rf_times` can report the two separately.

Gradients
: Two kinds share one column. A *trapezoid* is an amplitude with rise, flat and
  fall times. An *arbitrary* gradient is a shape id, with an optional time
  shape when the samples do not fall on the gradient raster, and endpoint
  amplitudes. The two occupy separate tables and share the file's id space.

ADC
: A sample count, a dwell time, a delay, frequency and phase offsets, and an
  optional per-sample phase modulation.

Shapes
: Sample arrays, compressed by a run-length encoding of their derivative, so a
  linear ramp of a thousand samples is stored as three numbers. A shape is
  stored normalized; the amplitude that scales it belongs to the event, not to
  the samples.

## Gyromagnetic-ratio-free units

Amplitudes are in **Hz** for RF and **Hz/m** for gradients rather than in tesla
and tesla per metre. The same file therefore means the same thing on any
nucleus a scanner is tuned to, and the gyromagnetic ratio enters only where a
physical amplitude is required.

Converting a reported amplitude from Hz/m to mT/m means dividing by the
gyromagnetic ratio in Hz/T, which {class}`~pypulseqpp.Opts` holds as `gamma`,
and multiplying by 1000. The constraint
checks in {doc}`../safety/index` report their values in the file's units and
convert with that constant.

## Extensions

Extensions carry information for which the block table has no column. Each
block's `EXT` id refers to a linked list of typed rows.

`LABELSET` and `LABELINC`
: Counters and flags. The counters (`LIN`, `PAR`, `SLC`, `ECO`, `REP`, `AVG`,
  `SET`, `SEG`, `PHS`, `ACQ`) are the encoding indices a reconstruction sorts
  the acquisition by; the flags (`NAV`, `REV`, `IMA`, `NOISE`, `REF`, …) state
  what an acquisition is for. Label values are *sticky*: a value set on one
  block remains in force until it is set or incremented again, so the file
  contains one row where a counter changes rather than one row per block.
  {meth}`~pypulseqpp.Sequence.evaluate_labels` evaluates the block table in
  order and returns the running values.

`ROTATIONS`
: A quaternion that rotates the block's gradients into the physical frame.
  One row per orientation replaces a rotated copy of every waveform, so a
  radial or spiral acquisition references a single interleaf for every shot;
  see {doc}`libraries-and-shapes`.

`TRIGGERS`
: Wait on, or emit, a hardware signal.

`RF_SHIMS`
: Per-transmit-channel magnitude and phase, for parallel transmission.

`SOFT_DELAY`
: A delay whose duration is adjusted at the console without rewriting the file.

## See also

* {doc}`libraries-and-shapes` — how events are stored, deduplicated and written.
* {doc}`timing-and-rasterization` — the rasters every event time is quantized to.
* {doc}`../../api/events` — the event factories and block operations.
