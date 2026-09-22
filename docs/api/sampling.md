# Sampling and scan-loop schedules

Sampling utilities separate the spatial support of an acquisition from its
temporal ordering. Support functions determine which encoded coordinates are
acquired. Ordering functions assign an existing coordinate set to a
traversal, shot, or echo index. Sequence applications convert these
coordinates into gradient scaling and Pulseq labels.

```{eval-rst}
.. currentmodule:: pypulseqpp
```

None of these routines creates an event, adds a block, or emits a label. They
return plain Python and NumPy values that a
{class}`~pypulseqpp.sequences.SequenceApp` consumes in its scan loop.

## Data flow

For a Cartesian application:

```text
acquisition prescription          matrix, acceleration, ACS, partial Fourier, CAIPI
        |
        v
support: encoded coordinates      make_cartesian_axis_sampling, make_cartesian_plane_sampling
        |                         (or a boolean mask from make_*_mask)
        v
temporal ordering                 make_traversal_order: loop order over positions
        |                         make_*_order: indices into the coordinates, per shot and echo
        v
SequenceApp.kernel, once per repetition
        |
        +--> scales the phase- and partition-encoding gradients from the coordinate
        |
        +--> emits the LIN / PAR / ECO / SEG / IMA ... labels from the coordinate,
             the echo index and the calibration membership
```

An EPI application replaces the ordering step: the scan loop chooses each
shot's origin, and {func}`make_epi_shot_offsets` gives the offsets of the
shot's echoes from that origin. A non-Cartesian application takes one
orientation per shot from the orientation routines. RF schedules supply one
phase or flip angle per repetition.

The returned values are of four kinds, and the tables below state which one
each routine returns:

| Kind | Type | Meaning |
| --- | --- | --- |
| Encoded coordinates | `list[int]` or `list[tuple[int, int]]` | Zero-based encoded indices: a line `y`, or a view `(y, z)` of line and partition. The k-space centre is `n // 2` on each axis. |
| Support mask | `numpy.ndarray` of `bool`, shape `(n_y, n_z)` | `mask[y, z]` is `True` when the view `(y, z)` is acquired. |
| Indices | `numpy.ndarray` or `list[list[int]]` | Positions in, or rows of, an array the caller already holds. Not coordinates. |
| Offsets, angles, schedules | `numpy.ndarray` of `float` or `int` | Relative encoded offsets, angles in radians, or one value per repetition. |

## Naming convention

`make_<noun>` constructs an acquisition plan that a scan loop consumes: a
support (`*_sampling`, `*_mask`), an order (`*_order`), EPI shot offsets, or
an RF schedule (`*_schedule`). `calc_<noun>` evaluates closed-form geometry:
angle sequences and projection directions. This follows the PyPulseq use of
`calc_*` for computed quantities (`calc_duration`, `calc_rf_center`) and the
`calc_*_trajectory` routines of this package.

## Cartesian sampling support

Support routines answer which Cartesian views are acquired. They do not
determine the order of acquisition, assign echoes, create labels, or create
gradient events.

The two coordinate routines apply a complete prescription — undersampling,
calibration (ACS) region, partial Fourier, CAIPIRINHA shift, elliptical crop
and support scheme — and return the calibration views and the remaining
imaging views as two disjoint lists. The acquired support is their union. The
mask generators return a boolean support only: they do not separate the
calibration views and take no partial-Fourier or elliptical-crop argument
beyond their own parameters.
{func}`make_cartesian_plane_sampling` uses {func}`make_poisson_disc_mask`
internally when `sampling='poisson'`.

| Object | Input | Output | Purpose |
| --- | --- | --- | --- |
| {obj}`~pypulseqpp.make_cartesian_axis_sampling` | Matrix size, acceleration, ACS extent, partial Fourier | `(calibration, imaging)` lists of line indices | Views of one Cartesian axis: 2D phase-encoding lines, or the partitions of a stack-of-stars, stack-of-spirals or stack-of-blades acquisition. |
| {obj}`~pypulseqpp.make_cartesian_plane_sampling` | `(n_y, n_z)`, accelerations, ACS extent, partial Fourier, CAIPI shift, elliptical crop, `sampling` scheme | `(calibration, imaging)` lists of `(y, z)` views | Views of a ky-kz plane: 3D Cartesian acquisitions. |
| {obj}`~pypulseqpp.make_caipirinha_mask` | `(n_y, n_z)`, `ry`, `rz`, shift | Boolean mask | CAIPIRINHA lattice anchored at `(0, 0)`. |
| {obj}`~pypulseqpp.make_poisson_disc_mask` | `(n_y, n_z)`, acceleration, calibration block, seed | Boolean mask | Variable-density Poisson-disc support. |
| {obj}`~pypulseqpp.make_random_mask` | `(n_y, n_z)`, acceleration, calibration block, seed | Boolean mask | Uniform-random support with a calibration block. |

```python
>>> import pypulseqpp as pp
>>> calibration, imaging = pp.make_cartesian_plane_sampling((4, 4), (2, 2), (2, 2))
>>> calibration
[(1, 1), (1, 2), (2, 1), (2, 2)]
>>> imaging
[(0, 0), (0, 2), (2, 0)]
```

### Poisson-disc support and T2 Shuffling

Two operations that are both associated with variable-density acquisitions
are separate here. `make_cartesian_plane_sampling(..., sampling='poisson')`
selects a variable-density Poisson-disc *support*: which views are acquired.
{func}`make_shuffling_order` assigns an already selected set of views to
random echo positions: the *echo ordering* of T2 Shuffling. The shipped
fast-spin-echo and MPRAGE applications combine the two under
`ordering='shuffling'`.

## View traversal

{func}`make_traversal_order` answers in which order a loop visits an existing
list of positions, such as slices, partitions or the lines already selected.
It returns a permutation `p` of `range(n)`; `[items[i] for i in p]` is the
list in visiting order.

| Object | Input | Output | Purpose |
| --- | --- | --- | --- |
| {obj}`~pypulseqpp.make_traversal_order` | Number of positions, scheme | Permutation of `range(n)` | Sequential, reverse, interleaved, centre-out, outside-in or random loop order. |

## Echo-train and shot ordering

Echo-train orderings answer which already selected view is acquired by each
shot and each echo. They take the selected coordinates — an `(N, 2)` array
of `(ky, kz)`, an `(N,)` array of `ky`, or a boolean mask — and return
`trains`, a list with one entry per shot:

- `trains[s]` is the echo train of shot `s`;
- `trains[s][e]` is the **row index into the input coordinates** of the view
  acquired at echo `e` of that shot, never the coordinate itself;
- every input row appears exactly once; with `pad=True`, each train has
  `train_length` entries and `None` marks an echo that acquires no view.

`[[coords[i] for i in train] for train in trains]` converts the indices to
coordinates.

| Object | Input | Output | Purpose |
| --- | --- | --- | --- |
| {obj}`~pypulseqpp.make_linear_order` | Coordinates, train length, target echo | `trains[shot][echo]` indices | Raster-order bands, one band per echo. |
| {obj}`~pypulseqpp.make_centric_order` | Coordinates, train length, target echo | `trains[shot][echo]` indices | Distance-ranked bands; the views nearest the centre at the first (or target) echo. |
| {obj}`~pypulseqpp.make_radial_order` | Coordinates, train length | `trains[shot][echo]` indices | One angular wedge per shot, centre-out within the wedge. |
| {obj}`~pypulseqpp.make_radial_adaptive_order` | Coordinates, train length, target echo | `trains[shot][echo]` indices | Distance-ranked bands folded about the target echo, so the centre is acquired at `center_echo`. |
| {obj}`~pypulseqpp.make_shuffling_order` | Coordinates, train length, seed | `trains[shot][echo]` indices | Random echo positions within spatially clustered trains (T2 Shuffling). |

```python
>>> views = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
>>> trains = pp.make_radial_order(views, 4, center=(0, 0))
>>> trains
[[3, 6, 0, 5], [4, 1, 7, 2]]
>>> [[views[i] for i in train] for train in trains]
[[(0, -1), (1, 0), (-1, -1), (1, -1)], [(0, 1), (-1, 0), (1, 1), (-1, 1)]]
```

## EPI within-shot offsets

{func}`make_epi_shot_offsets` returns the phase- and partition-encoding
offsets `(Δky, Δkz)` of each echo of one EPI shot relative to its first
echo. It does not select the global acquisition support: the scan loop
chooses each shot's origin `(y0, z0)`, and

```text
shot origin (y0, z0)  +  make_epi_shot_offsets(...)[e]  =  view acquired at echo e
```

The union over shots of these views is the acquired support; with the
`'caipi'` scheme it tiles the lattice of {func}`make_caipirinha_mask`.

| Object | Input | Output | Purpose |
| --- | --- | --- | --- |
| {obj}`~pypulseqpp.make_epi_shot_offsets` | Echo-train length, scheme, acceleration, segments, CAIPI parameters | `(etl, 2)` integer offsets | Linear, segmented blipped-CAIPI and zigzag EPI traversals. |

```python
>>> offsets = pp.make_epi_shot_offsets(4, acceleration=2, segments=2)
>>> offsets[:, 0].tolist()
[0, 4, 8, 12]
>>> (offsets + [1, 0])[:, 0].tolist()
[1, 5, 9, 13]
```

## Non-Cartesian orientations

The angle routines return one in-plane rotation angle, in radians, per shot:
per radial spoke, spiral interleaf or PROPELLER blade. The scan loop rotates
the readout trajectory by it. {func}`calc_projection_shell` returns 3D unit
spoke directions and per-shot rotation matrices for a 3D radial (ZTE)
acquisition.

| Object | Input | Output | Purpose |
| --- | --- | --- | --- |
| {obj}`~pypulseqpp.calc_uniform_angles` | Count, angular span | Angles (rad) | Equal spacing over the span. |
| {obj}`~pypulseqpp.calc_golden_angles` | Count, half or full circle | Angles (rad), modulo 2π | Golden-angle increment; any window of consecutive angles is near-uniform. |
| {obj}`~pypulseqpp.calc_tiny_golden_angles` | Count, tiny-golden index | Angles (rad), modulo 2π | Smaller golden increment. |
| {obj}`~pypulseqpp.calc_raga_angles` | Count, approximation order | Angles (rad) from a finite equidistant set | Rational approximation of the golden angle, exactly periodic. |
| {obj}`~pypulseqpp.calc_projection_shell` | Views per shell, shots, scheme | `(n_views, 3)` unit directions and `(n_shots, 3, 3)` rotation matrices | Pole-to-pole 3D shell and its per-shot rotations. |

## RF phase and flip schedules

Schedules return one value per repetition or per echo, which the scan loop
applies to that repetition's RF events (and, for phases, its ADC events).
They are independent of the sampling support.

| Object | Input | Output | Purpose |
| --- | --- | --- | --- |
| {obj}`~pypulseqpp.make_rf_spoiling_schedule` | Number of repetitions, phase increment | Phases (rad) | Quadratic RF-spoiling phase. |
| {obj}`~pypulseqpp.make_phase_cycling_schedule` | Number of repetitions, phase cycle | Phases (rad) | Repeated phase cycle, as for balanced SSFP. |
| {obj}`~pypulseqpp.make_traps_schedule` | Echo-train length, target angle | Refocusing flip angles (rad) | Variable refocusing angles that stabilise towards a target. |

## Consumption by a sequence application

A {class}`~pypulseqpp.sequences.SequenceApp` computes support and ordering
once, in `init_sequence`, and stores them, for example as `lines`, `views`,
`partitions`, `calibration` or `trains`. Its `loop` iterates over them and
calls `kernel` once per repetition. The kernel scales the phase-encoding
gradient by the coordinate's normalised offset from the centre,
`(y - n_y // 2) / (n_y / 2)` in the shipped Cartesian sequences,
applies the angle or schedule value of that repetition, and creates the
label events with {func}`make_label`: `LIN` and `PAR` from the coordinate,
`ECO` from the echo index, `IMA` from calibration membership, and `SEG`,
`SLC` or `SET` from the loop position. The shipped sequences in
{doc}`../sequences` follow this pattern.

## Renamed routines

The following names were renamed. The former names still resolve as
`pypulseqpp.<name>`, emit a `DeprecationWarning` naming the replacement, and
will be removed in a future release.

| Former name | Replacement | Argument changes |
| --- | --- | --- |
| `calc_sampled_lines` | `make_cartesian_axis_sampling` | `r` is now `acceleration`; the second return value is `imaging`. |
| `calc_sampled_pairs` | `make_cartesian_plane_sampling` | `shuffling=True` is now `sampling='poisson'`; the second return value is `imaging`. |
| `calc_traversal_order` | `make_traversal_order` | None. |
| `calc_epi_order` | `make_epi_shot_offsets` | None. |
