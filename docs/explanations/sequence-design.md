# Sequence design in pypulseqpp

```{admonition} TL;DR
:class: tldr

- A {class}`~pypulseqpp.sequences.SequenceModule` solves a reusable block
  layout independently of the acquisition loop: block tuples, named mutable
  event templates and a timing reference, `center`, in seconds from its start.
  Module timing composes through `center`, independent of block boundaries.
- A complete sequence is a function `sequence(system, **protocol)` that returns
  the designed {class}`~pypulseqpp.Sequence`, or a list of sequences with the
  prescans first. Its protocol is read from the signature and the NumPy-style
  Parameters section, by {func}`~pypulseqpp.sequences.parameters` and by the
  command line.
- {class}`~pypulseqpp.sequences.Labels` writes the label events a block
  changes; {func}`~pypulseqpp.sequences.write` writes a list as files linked
  by `NextSequence`, so each file keeps one period of repetition.
- A Cartesian acquisition is specified by its support, which views are
  acquired, and its temporal ordering, when each is acquired. The sampling
  routines keep the two separate and create neither events nor labels;
  ordering indices are row numbers into the array the caller passed, not
  coordinates.
```

The package places three abstractions above the flat block list of
{doc}`pulseq-representation`: a sequence module, a reusable block layout
whose timing and gradient waveforms are solved once; a sequence function,
which adds a prescription, a sampling order and a scan loop; and the sampling
routines that feed the loop. None of them is part of the file format; a
sequence written through them is an ordinary `.seq` file.

## Sequence modules

A {class}`~pypulseqpp.sequences.SequenceModule` solves a reusable block layout
independently of the acquisition loop. The module contains block tuples, named
mutable event templates, and a timing reference.

```text
SequenceModule
├── blocks: [(event, ...), ...]
├── events: named mutable event templates
└── center: timing reference (s)
```

### Timing centre

The `center` attribute gives the module timing reference in seconds from its
start, usually an RF pulse centre or an echo. For an inversion module followed
by an excitation module, the recovery delay required for inversion time
$T_I$ is

$$
t_{\mathrm{delay}} = T_I
- (t_{\mathrm{inv}} - t_{\mathrm{centre,inv}})
- t_{\mathrm{centre,exc}}.
$$

This definition composes pulse-centre timing without depending on block
boundaries.

### Module contents

| Module family | Blocks and events | Responsibility |
| --- | --- | --- |
| Excitation | RF event, selection gradient, rephaser | Excitation or refocusing geometry and pulse timing. |
| Preparation | RF and gradient preparation blocks | Inversion, T2 preparation, saturation, diffusion, or magnetisation transfer. |
| Readout | Prephasing, ADC event, readout gradients, rewinding or spoiling | Echo timing, acquisition bandwidth, and sampling trajectory. |

Named events are published both as attributes and through `events`. An
acquisition loop may change an event template before `add_block`, for example
by setting an RF phase offset or scaling a phase-encode gradient. Blocks
already added to a sequence, and the module's internal sequence, are not
changed by it.

A readout constructed with `te=None` uses its shortest realizable echo time.
Requested bandwidths and times are rasterized, and the achieved values are
reported by the module.

### Representation boundary

Module construction separates fixed waveform and timing design from per-view
encoding. This mirrors the Pulseq distinction between event definitions and
playout instances; see {doc}`shapes-and-storage`. Sampling order and
repetition structure belong to the sequence function.

## Sequence functions

A complete sequence is designed from the system limits and a prescription: the
field of view, matrix, timing, flip angle and sampling order of the
acquisition. A sequence function is that design as a callable, from which a
script, the command line and a protocol editor obtain the sequence and the
prescription.

A sequence function has the signature `sequence(system, **protocol)`. `system`
is the {class}`~pypulseqpp.Opts` the sequence is designed under. A function
designed for lower gradient or slew limits lowers the limits of `system` to them
with {func}`~pypulseqpp.cap_system`, which never raises a limit. The keyword
parameters after `system` are the protocol, and each states its unit in its
description. The function returns a {class}`~pypulseqpp.Sequence`, or, for a
scan with prescans, a list of sequences in play order.

| Concern | Sequence function |
| --- | --- |
| Prescription | Keyword parameters after `system`, documented in the Parameters section |
| System limits | `system`, capped to the design's limits with `cap_system` |
| Modules, timing and sampling arrays | The body, before the loop |
| Sampling order | The loop |
| One repetition | The body of the loop |
| Pulseq definitions required by reconstruction | `set_definition` after the loop |
| Prescans | The elements of the returned list before the main sequence |

The Python module of a shipped sequence defines its design limits as the constants
`MAX_GRAD` (mT/m) and `MAX_SLEW` (T/m/s). Changing one redesigns the gradient
waveforms and may alter echo spacing, acquisition duration, and constraint
estimates.

### Protocol

The protocol is read from the function. A parameter's type and default come
from its annotation and default value, and its unit, choices and description
from the NumPy-style Parameters section. The unit is the parenthesised group in
the first sentence of a description, `Echo time (s).`, and the choices of a
string parameter are the values its documented type lists in braces. An
annotation that admits `None` leaves the value to the design.
{func}`~pypulseqpp.sequences.parameters` returns one
{class}`~pypulseqpp.sequences.ProtocolParameter` per keyword parameter, in
signature order, which is what a protocol editor presents.

{func}`pypulseqpp.cli.run` derives one command-line option from each scalar
keyword parameter and takes its help text from the first sentence of the
parameter's description. The limit options `--max-grad-mtm` (mT/m) and
`--max-slew-tm-s` (T/m/s) build the `Opts` passed as `system`, and the result is
written with {func}`~pypulseqpp.sequences.write`.

(encoding-labels)=

### Encoding labels

A loop records encoding indices with Pulseq label extensions. The principal
labels are `LIN`, `PAR`, `ECO`, `SEG`, `REP`, and `SET`. A label keeps its value
across blocks until an event changes it, so a block carries the events of the
labels it changes and no others. {class}`~pypulseqpp.sequences.Labels` holds
that state: calling it with the new value of each label returns the events to
add to the block. An unchanged value writes nothing, a change equal to the
label's previous change is an INC, and any other change is a SET.
{meth}`~pypulseqpp.Sequence.evaluate_labels` recovers the ADC order directly
from the sequence. Sampling figures can therefore use the implemented
acquisition order rather than reconstructing it from the prescription.

### Prescans

A prescan is a sequence played before the main sequence, such as a calibration
acquisition. A sequence function returns it ahead of the main sequence in a
list. {func}`~pypulseqpp.sequences.write` writes the list as separate files: the
first at the given path, and each later one beside it as `<stem>_<Name>.seq`,
where `Name` is the sequence's `Name` definition, or its position in the list,
counted from 0 at the first, when it has none. A later file whose name an
earlier file has taken is written as `<stem>_<Name>_<position>.seq`, so every
sequence has a file of its own. Each file but the last names the next with
`NextSequence`. The chain represents one acquisition while each file keeps
one period for {meth}`~pypulseqpp.Sequence.repetition` and the analyses that
use it.
{func}`~pypulseqpp.sequences.duration` is the sum of the durations of the
sequences in the chain.

### Checking a prescription

Calling a sequence function designs every block, and the number of blocks grows
with the matrix. A prescription the design cannot meet, such as an echo time
shorter than the readout admits, raises `ValueError`. The call does not evaluate
the waveforms against gradient, PNS or SAR limits, which take the designed
sequence ({doc}`constraint-checks`).

The function records the prescription as designed in the definitions it sets
after the loop. A value the design chooses, such as the shortest echo time for
`te=None`, is recorded as `TE`, and prescribing the recorded value as `te`
designs a sequence that records the same `TE`. The receiver bandwidth as
designed is the reciprocal of the dwell time of the ADC events, which is
rounded to the ADC raster.

## Sampling support and ordering

A Cartesian acquisition is specified by two independent choices: its
**support**, the set of phase- and partition-encoding views that are acquired,
and its **temporal ordering**, the repetition, shot and echo at which each
acquired view is played. The sampling routines keep the two separate, and
neither creates events or labels. A sequence function combines them in its
scan loop.

```text
acquisition prescription     matrix, acceleration, ACS, partial Fourier, CAIPI
        │
        ▼
acquired support             encoded view indices (y, z), split into
        │                    calibration and imaging views
        ▼
temporal ordering            loop order, or [shot][echo] indices into the views
        │
        ▼
scan loop                    one iteration per repetition
        │
        ├─▶ gradient scaling      phase and partition encodes from (y, z)
        └─▶ Pulseq labels         LIN, PAR, ECO, SEG, IMA, ...
```

### Kinds of value

The routines exchange values of distinct kinds, and the distinction is part of
their interface.

| Kind | Example | Produced by | Meaning |
| --- | --- | --- | --- |
| A. Encoded view indices | `(y, z)` with `0 ≤ y < n_y` | {func}`~pypulseqpp.make_cartesian_axis_sampling`, {func}`~pypulseqpp.make_cartesian_plane_sampling` | Zero-based line and partition numbers on the encoding grid, as the `LIN` and `PAR` labels count them. The k-space centre is `(n_y // 2, n_z // 2)`. |
| B. Centred coordinates | `(y - n_y // 2, z - n_z // 2)` | the caller | Offsets from the k-space centre in encoding steps. The echo-train orderings measure distance and angle in this frame. |
| C. Boolean support mask | `mask[y, z]` | `make_*_mask` | `True` where the view `(y, z)` is acquired. No order and no role. |
| D. Ordering indices | `trains[s][e]` | `make_*_order` | Row numbers into the array the caller passed, grouped by shot `s` and echo `e`. Not coordinates. |
| E. EPI relative offsets | `(Δky, Δkz)` per echo | {func}`~pypulseqpp.make_epi_shot_offsets` | Offsets from echo 0 of one shot. The scan loop chooses each shot's origin. |
| F. Orientation schedules | angle per shot, or 3D directions | `calc_*_angles`, {func}`~pypulseqpp.calc_projection_shell` | Rotations applied to a non-Cartesian readout, in radians. |
| G. RF schedules | phase or flip angle per repetition | `make_*_schedule` | Values applied to the RF (and ADC) events of each repetition. |

Centred coordinates are the input of the geometric echo-train orderings
because the k-space centre is a property of the encoding grid, not of the
acquired set. With an even matrix, partial Fourier or an asymmetric
undersampled support, the centroid of the acquired views is not the centre,
and an ordering that measured distance from it would acquire the wrong view
at the target echo. The conversion is explicit:

```python
calibration, imaging = pp.make_cartesian_plane_sampling(
    (n_y, n_z), (2, 2), (24, 24), partial_fourier=(0.75, 1.0)
)
views = np.array(calibration + imaging)       # A: encoded indices
centred = views - (n_y // 2, n_z // 2)        # B: centred coordinates
trains = pp.make_radial_adaptive_order(       # D: [shot][echo] indices
    centred, etl, center_echo=te_echo
)
view_of = [[tuple(views[i]) for i in train] for train in trains]
```

The same indices `i` select the encoded view `views[i]`, from which the scan
loop scales the phase-encoding gradient and writes the `LIN` and `PAR` labels.
A boolean mask enters the same way, through `np.argwhere(mask)`.

### Poisson-disc support and T2 Shuffling

Both are associated with variable-density, echo-resolved acquisitions, and they
answer different questions.

| | Poisson-disc sampling | T2 Shuffling |
| --- | --- | --- |
| Determines | **which** views are acquired | **when** already selected views are acquired along the echo train |
| Routine | `make_cartesian_plane_sampling(..., sampling='poisson')`, {func}`~pypulseqpp.make_poisson_disc_mask` | {func}`~pypulseqpp.make_shuffling_order` |
| Output | encoded views, or a boolean mask | `trains[shot][echo]` indices |

The shipped fast-spin-echo sequence combines them under
`ordering='shuffling'`. The MPRAGE sequence's `ordering='shuffling'` pairs
the same Poisson-disc support with a random line order within each partition.
Either choice can be used without the other.

The sampling routines return plain Python and NumPy values and create no label
events. Which calibration views are played first, whether the support is
reordered before acquisition, and which labels each repetition writes are
decided by the sequence function: `LIN` and `PAR` from the encoded view, `ECO`
from the echo index, `IMA` from membership of the calibration list, and `SEG`,
`SLC`, `SET` or `REP` from its position in the loop
([](#encoding-labels)).

## See also

* {doc}`../api/modules` — module interfaces and parameters.
* {doc}`../api/sequence-functions` — `parameters`, `Labels`, `write`,
  `duration` and the command line.
* {doc}`../api/sampling` — the sampling routines, their inputs and returns.
* {doc}`../user-guide/from-pypulseq` — a PyPulseq script as a sequence
  function.
* {doc}`/generated/gallery/01-course/09_sequence_modules` — a
  sequence function assembled from modules.
* {doc}`../sequences` — shipped complete sequences.
