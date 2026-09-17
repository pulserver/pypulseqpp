# Storage, deduplication and file revisions

The block table of {doc}`events-and-blocks` refers to events by id, and the
events refer to shapes by id.

## Definitions and instances

Underneath the libraries, an event is stored as a fixed **definition** together
with a set of **instances**. The definition holds the timing and shape data
that does not vary between playouts; each instance holds the playout
parameters.

A phase-encode gradient scaled per line is one definition and one instance per
line: the shape and the timing are identical and only the amplitude differs. An
arbitrary gradient whose waveform differs between playouts is a new definition,
because the waveform belongs to the instance only in the sense of being scaled,
not reshaped.

For RF events, the magnitude, phase and time shapes belong to the definition;
frequency offset, phase offset and amplitude belong to the instance.

This separation allows the package to detect that a run of blocks
repeats, which is the repetition the SAR check of {doc}`../safety/sar` averages
over.

## Deduplication

Two sequences with identical playout can differ substantially in file size,
because nothing in the format forces equal events to share a library entry.
Writing deduplicates: equal shapes are merged, equal events are merged, and the
block table is renumbered.

Registration ids are therefore local to a sequence and are not stable across
deduplication. Code that registers an event and retains its id must not assume
the id survives a write.

## Rotation extensions and the size of the shape library

A non-Cartesian acquisition plays the same readout at many orientations. There
are two ways to express this.

Materializing each orientation writes a separate gradient waveform per shot, so
the shape library grows in proportion to the number of shots. Referring to one
interleaf and attaching a `ROTATIONS` extension per block writes the waveform
once, and the per-shot cost is one quaternion.

A radial, spiral, PROPELLER or stack-of-stars acquisition designed the second
way therefore has a shape library the size of a single repetition, whatever the
number of shots, while its block count and its rotation-row count grow with the
scan. This is a property of the representation rather than of the trajectory: a
zero-echo-time acquisition, whose readout gradient is held across a shell and
stepped between views rather than rotated as a fixed interleaf, has a shape
library that grows with the view count.

The consequence for analysis is that any quantity evaluated on played gradient
waveforms must apply the block rotation first. {doc}`../safety/index` states
this for the constraint checks.

## Definitions, sequence chains and the signature

`[DEFINITIONS]` is free-form key/value metadata. A few keys are conventional:
`FOV`, `Name`, the raster times and `TotalDuration`. The rest are chosen by
whoever writes the file. The shipped sequence implementations record the
prescription and the k-space geometry a reconstruction requires: the matrix
size, the echo and repetition times, the index of the k-space centre line and
sample, and the slice positions.

`NextSequence` links files into a chain. A calibration prescan and the imaging
scan it belongs to are two files played in order rather than one file with a
mode flag, which {meth}`~pypulseqpp.sequences.SequenceApp.write`
produces from an application's
{meth}`~pypulseqpp.sequences.SequenceApp.prescans`, and which keeps each file a
single repeating unit.

`[SIGNATURE]` closes the file with an MD5 digest of everything above it, so a
reader can establish that the file is the one that was written.

## Information outside the format

The format states the playout order and nothing above it. Nothing in the file
identifies which run of blocks is a repetition, which events belong to one
shot, or which acquisitions form a calibration region. A consumer that requires that
structure derives it from the content. The repetition detection underlying the
SAR check does exactly that, and the shipped sequences record their encoding
indices as labels rather than leaving a consumer to infer them.

## Revisions and the binary form

`pypulseqpp` writes Pulseq 1.5.1 by default.

Revision 1.4.1, which {meth}`~pypulseqpp.Sequence.write_v141` produces, has no
column for an RF centre time or `use`, no gradient endpoint amplitudes and no
ppm offsets. The writer folds a ppm offset into an absolute offset using the
gyromagnetic ratio in Hz/T and the field strength in tesla, warns when it omits
a soft delay, and rejects a sequence containing rotation or RF-shim extensions,
which that revision cannot express.

Reading accepts earlier revisions. Legacy text conversion derives the RF centre
times, gradient endpoint amplitudes and pre-1.4 block durations the file does
not state; an RF `use` that is absent remains undefined rather than being
inferred.

The binary form, {meth}`~pypulseqpp.Sequence.write_binary`, holds the same
content in a different encoding: records are little-endian, times are integer
picoseconds and shape samples are float32. Both forms support an optional MD5
signature.

## See also

* {doc}`events-and-blocks` — the block table and the event kinds.
* {doc}`timing-and-rasterization` — quantization of event times.
* {doc}`../../api/sequence` — the container, its readers and its writers.
