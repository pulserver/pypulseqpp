# Sequence modules

A `.seq` file is a flat list of blocks, and a script that writes one directly
must solve every timing and gradient-waveform problem inline, in the same loop
that chooses which view to acquire. A **sequence module** separates the two:
it solves the layout of one group of blocks once, at construction, and exposes
the resulting events for a scan loop to place.

## The problem the abstraction addresses

Consider a Cartesian gradient-echo repetition. Its readout gradient follows
from the field of view, the matrix size and the receiver bandwidth. Its
prewinder must deliver half the readout moment, corrected for the moment
accumulated on the readout ramp. Its echo time fixes a delay that depends on
the pulse duration, the rephaser and the acquisition window. None of this
depends on which phase-encode line is being acquired.

What does depend on the line is a single scale factor applied to the
phase-encode gradient, and the RF and receiver phases. Solving the first group
once and varying only the second is both what the file format rewards — one
definition with one instance per playout, as {doc}`../pulseq/libraries-and-shapes`
describes — and what keeps a scan loop legible.

## What a module is

{class}`~pypulseqpp.sequences.SequenceModule` is a reusable block layout with
named, mutable event templates. A subclass builds its blocks in `init_module`,
and the events assigned there are published both as attributes of the module
and on its `events` namespace.

Three parts of the interface matter to a caller:

`blocks`
: The block tuples in play order. A module whose events need no per-view
  modification can be added to a sequence exactly as it stands.

Named events
: The individual events, for a loop that must scale or offset one of them.
  Repeated references are deduplicated by identity, so an event that appears in
  several blocks is published once.

`center`
: The module's timing reference, in seconds from the start of the module —
  typically an RF pulse centre or an echo. It is what makes intervals between
  modules computable.

The last point is worth stating explicitly, because interval arithmetic between
modules is otherwise error-prone. An inversion time is defined between two
pulse centres, not between block boundaries. Composing an inversion module and
an excitation module at a prescribed inversion time therefore requires the
delay

```python
recovery = TI - (inversion.duration - inversion.center) - excitation.center
```

which subtracts the part of the inversion module that follows its pulse and the
part of the excitation module that precedes its pulse.

A module also forwards a small set of analyses to the sequence it built
internally — `calculate_kspace`, `check_timing`, `paper_plot`, `test_report`
and `waveforms_and_times` — so a module can be inspected before any scan loop
exists.

## Mutable templates and their consequence

The published events are templates, and a scan loop modifies them in place
between `add_block` calls: it assigns an RF phase offset, or passes a scaled
copy of a phase-encode gradient through {func}`~pypulseqpp.scale_grad`.

This has a consequence worth knowing. `blocks` retains the original event
objects for replay, so mutating a published event changes what a subsequent
`add_block` writes, but it does not rewrite blocks already added, and it does
not rewrite the module's own stored sequence. Analyses taken from the module
describe the layout as constructed.

## The three families

Modules divide by what they contribute to a repetition.

**Excitation** modules produce a pulse and the gradients that go with it:
{class}`~pypulseqpp.sequences.SpatialSelectiveExcitation` designs an SLR pulse
with its selection gradient and rephaser, and reports the
`selection_amplitude` a slice offset is converted against. The non-selective,
two-dimensional selective, spectral-spatial, multiband and refocusing variants
follow the same contract.

**Preparation** modules produce magnetization preparation with no acquisition:
inversion, T2 preparation, fat saturation, diffusion weighting, magnetization
transfer. They deliberately define no recovery interval —
{class}`~pypulseqpp.sequences.InversionPreparation` has no inversion time —
because the interval belongs to the scan loop, and the same module then serves
a single-shot inversion recovery and a segmented magnetization-prepared train.

**Readout** modules solve the rest of the repetition around a pulse they are
given. {class}`~pypulseqpp.sequences.LineReadout2D` takes the excitation's RF
event and its selection gradient and rephaser, and returns the prewinder, the
phase-encode template at its largest step, the acquisition window and the
spoiler. Passing the pulse is what allows the module to measure the echo time
from the pulse centre; passing the rephaser is what allows it to place the
rephaser inside an interval the repetition already has to wait out rather than
appending one.

A readout constructed with `te=None` uses the shortest echo time its layout
admits and reports it as `echo_time`; one given a requested receiver bandwidth
reports the achieved value as `bandwidth_hz`, which may be lower for the reason
given in
{doc}`../pulseq/timing-and-rasterization`.

## Related pages

* {doc}`sequence-application` — the loop and prescription layer above modules.
* {doc}`../../api/modules` — the module classes and their parameters.
* {doc}`../../guides/custom-module` — assembling modules into a scan loop.
