# Sequence analysis

```{admonition} TL;DR
:class: tldr

- Waveforms, k-space, repetition and RF power are read from the stored blocks, after each block's rotation.
- Two integrals of the gradients exist: an unbroken one for shift phases, and one reset at excitation for echoes.
- A field-of-view transform changes the sequence itself, so every later analysis sees it.
```

Before you write a file you want to know what the scanner will play: where
each ADC sample lands in k-space, when the echo forms, how long a repetition
is, how much RF it carries. These are computed from the sequence you built, as
in lessons 2 and 5 of the {doc}`Course <../examples/course>`, and they follow
the rotations and the prescription the sequence holds.

## What pypulseqpp does

:::{container} capabilities

- **Expands the blocks into gradient waveforms, RF timing and ADC sample times.**

  <details><summary>Show code and tests</summary>

  Code: {meth}`~pypulseqpp.Sequence.waveforms_and_times`, {meth}`~pypulseqpp.Sequence.adc_times`, {meth}`~pypulseqpp.Sequence.rf_times`. Tests: *the gradient waveforms are the toolboxs*; *a range of blocks expands as the toolbox expands it*; *a trapezoid is its four corners* (`test_waveforms.py`).

  </details>
- **Applies each block's rotation to the waveforms, and reports every RF use.**

  <details><summary>Show code and tests</summary>

  Code: `compat=False` of {meth}`~pypulseqpp.Sequence.waveforms_and_times`. Tests: *a quarter turn about z moves the readout onto the other axis*; *every use a pulse can have is reported* (`test_waveforms.py`).

  </details>
- **Integrates the gradients to the k-space position of every ADC sample.**

  <details><summary>Show code and tests</summary>

  Code: {meth}`~pypulseqpp.Sequence.calculate_kspace`, {meth}`~pypulseqpp.Sequence.adc_kspace`. Tests: *the trajectory is the toolboxs*; *there is one sample position per sample*; *the adc kspace is where calculate kspace puts the samples* (`test_kspace.py`).

  </details>
- **Resets the trajectory at an excitation and inverts it at a refocusing pulse.**

  <details><summary>Show code and tests</summary>

  Code: {meth}`~pypulseqpp.Sequence.calculate_kspace`. Tests: *an excitation puts the trajectory back at the origin*; *a refocusing turns the trajectory around* (`test_kspace.py`).

  </details>
- **Finds where each readout passes nearest the centre of k-space.**

  <details><summary>Show code and tests</summary>

  Code: {meth}`~pypulseqpp.Sequence.adc_echoes`. Tests: *a readout sweeps k from where the prewinder left it*; *a stepped phase encode moves the line it reads* (`test_kspace.py`).

  </details>
- **Places a run of readouts as the whole sequence does, without integrating the rest.**

  <details><summary>Show code and tests</summary>

  Code: `readouts=` of {meth}`~pypulseqpp.Sequence.adc_kspace`. Tests: *each readout is where the whole sequence puts it*; *a run of readouts is the columns they hold in the whole trajectory* (`test_readout_kspace.py`).

  </details>
- **Finds the repetition, and takes a declared `TRSize` the blocks agree with.**

  <details><summary>Show code and tests</summary>

  Code: {meth}`~pypulseqpp.Sequence.repetition`. Tests: *a scan repeats at the length of one shot*; *a phase encode does not make a shot different*; *a declared TR the blocks contradict is ignored* (`test_repetition.py`).

  </details>
- **Reads the repetition from the block definitions, so a changed amplitude does not break it.**

  <details><summary>Show code and tests</summary>

  Code: {meth}`~pypulseqpp.Sequence.repetition`. Tests: *shots that differ only in their waveforms repeat by structure*; *the repeat survives a rotation*; *adding a block makes the answer stale* (`test_repetition.py`).

  </details>
- **Reports RF energy, peak and mean power and RMS B1, with dynamic pTx channels added in power.**

  <details><summary>Show code and tests</summary>

  Code: {meth}`~pypulseqpp.Sequence.calc_rf_power`, {func}`~pypulseqpp.calc_rf_power`. Tests: *ptx channels add their powers and never cancel*; *the sequence power sums its pulses*; *a window keeps its loudest stretch* (`test_rf_power.py`).

  </details>
- **Scales, translates and rotates the field of view of a finished sequence.**

  <details><summary>Show code and tests</summary>

  Code: {class}`~pypulseqpp.TransformFOV`. Tests: *a quarter turn about z sends x onto y*; *a second prescription turns what the first left*; *scaling an axis scales what it draws* (`test_transform_fov.py`).

  </details>
- **Keeps a rotation as an annotation, so a thousand orientations share one waveform.**

  <details><summary>Show code and tests</summary>

  Code: {class}`~pypulseqpp.TransformFOV`. Tests: *a rotation is an annotation and not a new waveform*; *baking the rotation into the waveforms is refused* (`test_transform_fov.py`).

  </details>
- **Moves RF and ADC phases by the translation, from the gradient area swept so far.**

  <details><summary>Show code and tests</summary>

  Code: `translation=` of {class}`~pypulseqpp.TransformFOV`. Tests: *a readout is acquired at the phase the shift asks for*; *a scan shifted across threads matches it shifted in short ranges*; *two identical repetitions read at the same phase* (`test_transform_fov.py`, `test_fov_shift.py`).

  </details>
- **Draws the sequence, its ADC samples in k-space and its RF profiles.**

  <details><summary>Show code and tests</summary>

  Code: {func}`~pypulseqpp.plot.plot`, {func}`~pypulseqpp.plot.paper_plot`, {func}`~pypulseqpp.plot.plot_kspace`, {func}`~pypulseqpp.plot.plot_rf`. Tests: none beyond the examples that draw them.

  </details>

:::

## What Pulserver does

- **Segments the repetition into its intermediate representation.** It reads
  {meth}`~pypulseqpp.Sequence.repetition` and requires it to start at block 1.
- **Applies the field-of-view offset a prescription carries.** It calls
  {class}`~pypulseqpp.TransformFOV` with `through_rotation=True`, and counts
  the gradient area on from the end of the file before.
- **Fills the readout table.** It reads
  {meth}`~pypulseqpp.Sequence.adc_echoes` for each readout's start and path.

## How it works

```{figure} ../generated/figures/kspace_reset_and_unbroken.png
Two spin-echo shots with an unbalanced readout. The unbroken gradient integral
(dashed) keeps growing; the excitation-aware trajectory (solid) restarts at
each excitation, is inverted by the refocusing pulse, and crosses zero in the
middle of the readout, where the ADC samples (dots) are placed.
```

### Two integrals of one gradient

The unbroken integral is the reference of the RF and ADC shift phases: a
translation moves a pulse or a readout by the area the gradients have swept
since the start, whatever the pulses did. The echo anchor needs the other one,
the position of the magnetisation in k-space, which an excitation resets and a
refocusing pulse inverts. {class}`~pypulseqpp.TransformFOV` carries both
across consecutive ranges as `swept_k` and `block_k_origin`.

### Frames of a transform

A translation is in metres along the channel axes, or along the logical axes
with `through_rotation`. A prescription rotation is composed after the
rotation each block already holds. The phase of an RF shape is stored in
cycles, ADC modulation and phase offsets in radians.

### The repetition

A repetition is the shortest period of the block definitions from block 1
that every later block repeats, so a phase-encode step or a spoke angle does
not break it. A slice with its own preparation and dummy shots is one
repetition, and a block played once makes the whole sequence one.

The units, the search and the translation algorithm are in
{doc}`../developer-guide/internals/sequence-analysis`.

## See it run

- {doc}`../generated/gallery/01-course/02_echo_and_repetition_time`: the echo time measured from the k-space trajectory, and the repetition drawn.
- {doc}`../generated/gallery/01-course/05_radial_sampling`: the trajectory of a radial acquisition.
- {doc}`../api/sequence`, {doc}`../api/plotting`: the container, the transform and the figures.
