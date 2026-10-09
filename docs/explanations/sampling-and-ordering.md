# Sampling and ordering

```{admonition} TL;DR
:class: tldr

- Which views or directions are acquired, when each is played, and the per-repetition RF values are three separate answers, each a function returning an array.
- None of them creates events or labels; the sequence function turns the arrays into gradients, labels and RF offsets.
```

A sequence has to decide which k-space samples it acquires, in what order,
and with which RF phase on each repetition. You keep those decisions out of
the scan loop: a view-selection, ordering or schedule rule that a sequence
needs is a function of the package, called once before the loop, and the loop
reads its result. The shipped sequences are written that way, as in
{doc}`lesson 5 of the Course <../generated/gallery/01-course/05_radial_sampling>`.

## What pypulseqpp does

:::{container} capabilities

- **Selects the views of a Cartesian axis or ky-kz plane, as a calibration block and an imaging lattice, with partial Fourier and an elliptical crop.**

  <details><summary>Show code and tests</summary>

  Code: {func}`~pypulseqpp.make_cartesian_axis_sampling`, {func}`~pypulseqpp.make_cartesian_plane_sampling`. Tests: *the centre view is acquired whatever the undersampling*; *the calibration block and the lattice do not overlap*; *partial Fourier drops the views before the centre and no others* (`test_sampling_cartesian.py`).

  </details>
- **Draws CAIPIRINHA, variable-density Poisson-disc and uniform-random supports as boolean masks.**

  <details><summary>Show code and tests</summary>

  Code: {func}`~pypulseqpp.make_caipirinha_mask`, {func}`~pypulseqpp.make_poisson_disc_mask`, {func}`~pypulseqpp.make_random_mask`. Tests: *a caipirinha shift climbs one partition per acquired line* (`test_sampling_cartesian.py`); *one seed gives one mask*; *the calibration block is always fully sampled*; *a request is either met or refused for a reason it states* (`test_sampling_poisson.py`).

  </details>
- **Orders selected views into shots and echoes: linear, centric, radial, radial about a target echo, and T2 Shuffling.**

  <details><summary>Show code and tests</summary>

  Code: {func}`~pypulseqpp.make_linear_order`, {func}`~pypulseqpp.make_centric_order`, {func}`~pypulseqpp.make_radial_order`, {func}`~pypulseqpp.make_radial_adaptive_order`, {func}`~pypulseqpp.make_shuffling_order`. Tests: *every ordering covers its views once* (`test_sampling_orderings.py`); *an ordering returns indices not coordinates*; *the adaptive order puts the kspace centre not the centroid at the target*; *shuffling is reproducible by seed and keeps clustered membership* (`test_sampling_contracts.py`).

  </details>
- **Permutes the positions of one loop axis: sequential, reverse, interleaved, centre-out, outside-in or random.**

  <details><summary>Show code and tests</summary>

  Code: {func}`~pypulseqpp.make_traversal_order`. Test: *a sampling routine returns its recorded result* (`test_sampling_regression.py`).

  </details>
- **Gives the within-shot phase-encoding offsets of an EPI shot: linear, segmented blipped-CAIPI or zigzag.**

  <details><summary>Show code and tests</summary>

  Code: {func}`~pypulseqpp.make_epi_shot_offsets`. Tests: *epi offsets start at the origin*; *segmented shots interleave to every accelerated line*; *caipi offsets tile the caipirinha lattice* (`test_sampling_contracts.py`).

  </details>
- **Computes per-shot angles for radial, spiral and blade readouts: uniform, golden, tiny golden and rational approximate golden (RAGA), and a 3D projection shell.**

  <details><summary>Show code and tests</summary>

  Code: {func}`~pypulseqpp.calc_uniform_angles`, {func}`~pypulseqpp.calc_golden_angles`, {func}`~pypulseqpp.calc_tiny_golden_angles`, {func}`~pypulseqpp.calc_raga_angles`, {func}`~pypulseqpp.calc_projection_shell`. Tests: *default golden is the pi periodic radial angle*; *full circle golden is the classic 137 degree angle*; *diametric uniform never repeats a direction* (`test_sampling_angles.py`).

  </details>
- **Computes per-repetition RF phase schedules (quadratic spoiling, repeated cycles) and a refocusing flip-angle schedule.**

  <details><summary>Show code and tests</summary>

  Code: {func}`~pypulseqpp.make_rf_spoiling_schedule`, {func}`~pypulseqpp.make_phase_cycling_schedule`, {func}`~pypulseqpp.make_traps_schedule`. Tests: *every slice is rf spoiled by its own excitation count* (`test_examples.py`); *a sampling routine returns its recorded result* (`test_sampling_regression.py`).

  </details>

:::

## What the sequence function does

- **Turns the arrays into a sequence.** It scales each phase-encoding gradient
  from a view, rotates a readout by an angle, offsets the RF and ADC phase by a
  schedule value, and writes the `LIN`, `PAR`, `ECO`, `IMA`, `SEG`, `SLC`, `SET`
  and `REP` labels from the view and its place in the loop
  ({doc}`sequence-modules`).
- **Chooses what is played first.** Calibration views, prescans and the origin
  of each EPI shot are the loop's decisions.

## How it works

```{figure} ../generated/figures/sampling_support_order_and_angles.png
A Poisson-disc support with its calibration block, each view coloured by the
echo at which the adaptive radial order plays it with the centre at echo 4 of
16, and the first 21 spokes of the golden and tiny golden angles, coloured by
acquisition order.
```

### Support and order are separate

The support is the set of views acquired; the order is the repetition, shot and
echo at which each is played. Poisson-disc sampling changes the first and
T2 Shuffling the second, and either can be used without the other. The
shipped fast-spin-echo sequence combines them under `ordering='shuffling'`.

The orderings return row numbers into the array you pass, not coordinates, so
the same index selects the encoded view that scales the gradient. They measure
distance and angle from the k-space centre, which is a property of the grid
rather than of the acquired set: with an even matrix, partial Fourier or an
asymmetric support the centroid of the views is not the centre, and an ordering
that used it would play the wrong view at the target echo. You pass views as
offsets from the centre.

### Golden angles

A fixed increment of $\pi/\phi$ keeps any window of consecutive spokes close to
uniformly spread, so a retrospective choice of window still covers the circle.
Tiny golden angles shrink the increment to $\pi/(\phi + N - 1)$ for index
$N$, which reduces the angular step between consecutive spokes.
RAGA draws the angles from a finite equidistant set, so they repeat exactly.

### Spoiling and cycling

Quadratic RF spoiling steps the phase by an increment that itself grows by
117 degrees per repetition, and the schedule applies to the excitation and the
ADC alike. A phase cycle such as $(0, \pi)$ alternates for balanced SSFP.

The definitions, frames and index conventions are in
{doc}`../developer-guide/internals/sampling-and-ordering`.

## See it run

- {doc}`../generated/gallery/01-course/05_radial_sampling`: golden-angle spokes in a radial gradient echo.
- {doc}`../generated/gallery/01-course/03_spoiling`: RF spoiling and the spoiler gradients.
- {doc}`../generated/gallery/01-course/09_sequence_modules`: a sequence function that calls the sampling routines.
- {doc}`../generated/gallery/02-tours/05_segmented_epi`: EPI shot offsets and interleaving.
- {doc}`../sequences`: the shipped sequences that use them.
- {doc}`../api/sampling`: the routines, their inputs and returns.
