# Sequence modules

A `.seq` file is a flat list of blocks, and a script that writes one directly
must solve every timing and gradient-waveform problem inline, in the same loop
that chooses which view to acquire. A **sequence module** separates the two: it
solves the layout of one group of blocks once, at construction, and exposes the
resulting events for a scan loop to place.

## Separation of layout from sampling order

In a Cartesian gradient-echo repetition, the readout gradient is determined by
the field of view, the matrix size and the receiver bandwidth. The prewinder
delivers half the readout moment, corrected for the moment accumulated on the
readout ramp. The echo time determines a delay that depends on the pulse
duration, the rephaser and the acquisition window. None of these quantities
depends on which phase-encode line is acquired.

Two quantities do depend on the line: a scale factor applied to the
phase-encode gradient, and the RF and receiver phases. Solving the first group
once and varying only the second corresponds to the file format's separation of
definitions from instances, described in
{doc}`../pulseq/libraries-and-shapes`, and keeps the scan loop short.

## Module interface

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
: The module's timing reference, in seconds from the start of the module,
  typically an RF pulse centre or an echo.

Intervals between modules are computed from `center`. An inversion time is
defined between two pulse centres rather than between block boundaries, so
composing an inversion module and an excitation module at a prescribed
inversion time requires the delay

```python
recovery = TI - (inversion.duration - inversion.center) - excitation.center
```

which subtracts the part of the inversion module that follows its pulse and the
part of the excitation module that precedes its pulse.

A module forwards `calculate_kspace`, `check_timing`, `paper_plot`,
`test_report` and `waveforms_and_times` to the sequence it built internally, so
its layout can be inspected before a scan loop exists.

## Mutable event templates

The published events are templates, and a scan loop modifies them in place
between `add_block` calls: it assigns an RF phase offset, or passes a scaled
copy of a phase-encode gradient through {func}`~pypulseqpp.scale_grad`.

Mutation has two consequences. A modified event changes what a subsequent
`add_block` writes, because `blocks` retains the original event objects for
replay. It does not change blocks that have already been added, nor the
sequence the module built internally, so analyses taken from the module
describe the layout as constructed rather than as played.

## Module families

Modules are grouped by what they contribute to a repetition.

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

**Readout** modules solve the rest of the repetition around an excitation.
{class}`~pypulseqpp.sequences.LineReadout2D` takes the excitation's RF event,
its selection gradient and its rephaser, and returns the prewinder, the
phase-encode template at its largest step, the acquisition window and the
spoiler. The echo time is defined from the centre of the excitation pulse, so
the pulse event is an argument. The rephaser is an argument because the module
places it within the delay that precedes the readout instead of adding a block
for it.

A readout constructed with `te=None` uses the shortest echo time its layout
admits and reports it as `echo_time`; one given a requested receiver bandwidth
reports the achieved value as `bandwidth_hz`, which may be lower for the reason
given in
{doc}`../pulseq/timing-and-rasterization`.

## See also

* {doc}`sequence-application` — the loop and prescription layer above modules.
* {doc}`../../api/modules` — the module classes and their parameters.
* {doc}`../../guides/custom-module` — assembling modules into a scan loop.
