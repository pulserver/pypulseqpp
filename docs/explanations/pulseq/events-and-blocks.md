# Events and blocks

[Pulseq](https://pulseq.github.io) is an open file format for MR pulse
sequences. A `.seq` file states what is played, on which channel and for how
long: the complete prescription of an acquisition, portable between sites and
vendors. Everything `pypulseqpp` builds, analyses and writes is that
description, so this page states what a block contains and which of the
format's conventions the Python interface preserves.

The authoritative definition is the [Pulseq
specification](https://pulseq.github.io/specification.pdf) and the MATLAB
reference implementation.

## The block as the unit of playout

A sequence is an ordered list of **blocks**. A block has a duration and at most
one event per channel: one RF pulse, one gradient on each of `x`, `y` and `z`,
one ADC window, and a chain of extensions.

Two rules fix the timing completely:

1. The events of a block are played concurrently, each with its own delay from
   the start of the block.
2. Blocks are played back to back, with no gap between them.

There is no nesting, no loop and no branch. The file is the flattened playout
order; a scan loop exists only in the script that wrote the file.

A block whose duration exceeds the extent of its events is therefore a delay,
and this is how a repetition time is realized. A two-dimensional gradient-echo
repetition, for example, is six blocks: the pulse with its slice-selection
gradient, the prewinders, an echo-time delay, the readout gradient with its
ADC, the spoilers, and a delay that completes the repetition time.

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

RF
: An amplitude in Hz, ids into a magnitude, a phase and optionally a time
  shape, a delay, frequency and phase offsets, a centre time, and a `use` —
  excitation, refocusing, inversion, saturation or preparation. The `use`
  identifies which pulses begin a shot and which refocus one, which is why the
  pulse factories such as {func}`~pypulseqpp.make_slr_pulse` accept it and why
  {meth}`~pypulseqpp.Sequence.rf_times` can report the two separately.

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

Converting a reported amplitude to mT/m means dividing by the gyromagnetic
ratio in Hz/T, which {class}`~pypulseqpp.Opts` holds as `gamma`. The constraint
checks in {doc}`../safety/index` report their values in the file's units and
convert with that constant.

## Extensions

The extension chain expresses what the core format has no column for. Each
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
  Storing one row per orientation, rather than a rotated copy of every
  waveform, is what allows a radial or spiral acquisition to reference a single
  interleaf for every shot; see {doc}`libraries-and-shapes`.

`TRIGGERS`
: Wait on, or emit, a hardware signal.

`RF_SHIMS`
: Per-transmit-channel magnitude and phase, for parallel transmission.

`SOFT_DELAY`
: A delay whose duration is adjusted at the console without rewriting the file.

## Related pages

* {doc}`libraries-and-shapes` — how events are stored, deduplicated and written.
* {doc}`timing-and-rasterization` — the rasters every event time is quantized to.
* {doc}`../../api/events` — the event factories and block operations.
