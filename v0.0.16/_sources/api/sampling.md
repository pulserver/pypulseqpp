# Sampling and scan-loop schedules

Routines that select the Cartesian views an acquisition encodes, order an
already selected set into loop, shot and echo positions, and supply the
per-shot orientations and per-repetition RF values of a scan loop. None of
them creates events or labels: the sequence application scales the gradients
and creates the `LIN`, `PAR`, `ECO` and other labels from their results, as
{doc}`../explanations/design/sampling` describes.

```{eval-rst}
.. currentmodule:: pypulseqpp
```

## Cartesian sampling support

Support is returned as zero-based encoded view indices, `y` or `(y, z)` on an
`(n_y, n_z)` grid with the k-space centre at `(n_y // 2, n_z // 2)`, in two
disjoint lists: calibration views and the other acquired views.

| Object | Input | Returns | Purpose |
| --- | --- | --- | --- |
| {obj}`~pypulseqpp.make_cartesian_axis_sampling` | Prescription of one axis: matrix size, acceleration, ACS, partial Fourier | `(calibration, imaging)`: encoded view indices `y` | Lines of a 2D acquisition, or partitions of a stack-of-* acquisition. |
| {obj}`~pypulseqpp.make_cartesian_plane_sampling` | Prescription of a ky-kz plane, with `sampling='lattice'` or `'poisson'` | `(calibration, imaging)`: encoded view indices `(y, z)` | Views of a 3D Cartesian acquisition. |

## Boolean sampling masks

`mask[y, z]` is `True` where the view `(y, z)` is acquired;
`np.argwhere(mask)` lists those views as encoded indices.

| Object | Input | Returns | Purpose |
| --- | --- | --- | --- |
| {obj}`~pypulseqpp.make_caipirinha_mask` | Grid shape, `ry`, `rz`, shift | Boolean mask, `True` where acquired | CAIPIRINHA lattice anchored at `(0, 0)`. |
| {obj}`~pypulseqpp.make_poisson_disc_mask` | Grid shape, acceleration, calibration block, seed | Boolean mask, `True` where acquired | Variable-density Poisson-disc support. |
| {obj}`~pypulseqpp.make_random_mask` | Grid shape, acceleration, calibration block, seed | Boolean mask, `True` where acquired | Uniform-random support. |

## Traversal ordering

| Object | Input | Returns | Purpose |
| --- | --- | --- | --- |
| {obj}`~pypulseqpp.make_traversal_order` | Number of positions, scheme | Permutation of `range(n)` | Loop order over slices, partitions or selected lines. |

## Echo-train ordering

Each routine takes centred coordinates of already selected Cartesian views:
an `(N, 2)` or `(N,)` array with the k-space centre at the origin, obtained
from encoded indices as `views - (n_y // 2, n_z // 2)`. It returns `trains`,
where `trains[s][e]` is the row index into that array of the view acquired at
echo `e` of shot `s`. The values are indices, not coordinates. The orderings
do not change the support: T2 Shuffling reorders views already selected, for
example by the Poisson-disc support of
`make_cartesian_plane_sampling(..., sampling='poisson')`.

| Object | Input | Returns | Purpose |
| --- | --- | --- | --- |
| {obj}`~pypulseqpp.make_linear_order` | Centred coordinates, train length | Indices as `[shot][echo]` | Raster-order bands, one per echo. |
| {obj}`~pypulseqpp.make_centric_order` | Centred coordinates, train length | Indices as `[shot][echo]` | Distance-ranked bands; centre at echo 0 or `center_echo`. |
| {obj}`~pypulseqpp.make_radial_order` | Centred coordinates, train length | Indices as `[shot][echo]` | One angular wedge per shot, centre-out within it. |
| {obj}`~pypulseqpp.make_radial_adaptive_order` | Centred coordinates, train length | Indices as `[shot][echo]` | Distance bands folded about `center_echo`. |
| {obj}`~pypulseqpp.make_shuffling_order` | Centred coordinates, train length, seed | Indices as `[shot][echo]` | Random echo positions within raster-contiguous trains (T2 Shuffling). |

## EPI shot offsets

The offsets are relative to echo 0 of one shot. The scan loop chooses each
shot's origin `(y0, z0)`; echo `e` then acquires `(y0, z0) + offsets[e]`.

| Object | Input | Returns | Purpose |
| --- | --- | --- | --- |
| {obj}`~pypulseqpp.make_epi_shot_offsets` | EPI prescription: echo-train length, scheme, acceleration, segments, CAIPI | `(etl, 2)` offsets `(Δky, Δkz)` relative to echo 0 | Linear, segmented blipped-CAIPI and zigzag traversals of one shot. |

## Non-Cartesian orientations

| Object | Input | Returns | Purpose |
| --- | --- | --- | --- |
| {obj}`~pypulseqpp.calc_uniform_angles` | Count, angular span | Angles (rad) | Equal spacing over the span. |
| {obj}`~pypulseqpp.calc_golden_angles` | Count, half or full circle | Angles (rad), modulo 2π | Golden-angle increment. |
| {obj}`~pypulseqpp.calc_tiny_golden_angles` | Count, tiny-golden index | Angles (rad), modulo 2π | Smaller golden increment. |
| {obj}`~pypulseqpp.calc_raga_angles` | Count, approximation order | Angles (rad) from a finite equidistant set | Rational approximation of the golden angle. |
| {obj}`~pypulseqpp.calc_projection_shell` | Views per shell, shots, scheme | Unit directions `(n_views, 3)` and rotations `(n_shots, 3, 3)` | 3D radial shell and its per-shot rotations. |

## RF phase and flip schedules

| Object | Input | Returns | Purpose |
| --- | --- | --- | --- |
| {obj}`~pypulseqpp.make_rf_spoiling_schedule` | Number of repetitions, increment | Phases (rad), one per repetition | Quadratic RF spoiling. |
| {obj}`~pypulseqpp.make_phase_cycling_schedule` | Number of repetitions, cycle | Phases (rad), one per repetition | Repeated phase cycle. |
| {obj}`~pypulseqpp.make_traps_schedule` | Echo-train length, target angle | Refocusing angles (rad), one per echo | Stabilising variable refocusing angles. |
