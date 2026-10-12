# Sampling and ordering internals

The definitions, index conventions and algorithms behind the routines that
select views, order them and schedule orientations and RF values. The concepts
are in {doc}`../../explanations/sampling-and-ordering`.

The routines live in the private modules `_masks`, `_ordering`, `_epi`,
`_angles` and `_schedules`, re-exported by name from the package namespace
(`_SAMPLING` in `pypulseqpp/__init__.py`). They return plain Python and NumPy
values and create no events or labels. A sequence function combines them in
its scan loop.

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

## Kinds of value

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

## Support

The support routines assign no order, shot or echo index. Calibration and
imaging views are returned as two disjoint lists whose union is the acquired
support; neither is an acquisition order.

### One axis

With $R$ = `acceleration`, the lattice of
{func}`~pypulseqpp.make_cartesian_axis_sampling` is

$$
\{\, i : (i - \lfloor n/2 \rfloor) \bmod R = 0 \,\},
$$

so the centre view is acquired for every $R$. Partial Fourier removes the
indices below $n - \mathrm{round}(\text{partial\_fourier}\cdot n)$, keeping the
centre and the far edge; `partial_fourier` lies in $(0.5, 1]$. The
calibration region is the `n_acs` contiguous views centred on `n // 2`, fully
sampled and clipped by partial Fourier; it is empty when `acceleration` is 1
or `n_acs` is 0. The axis may be the phase-encoding lines of a 2D acquisition
or the partitions of a stack-of-stars, stack-of-spirals or stack-of-blades one.

### A ky-kz plane

{func}`~pypulseqpp.make_cartesian_plane_sampling` takes `sampling='lattice'` or
`'poisson'`. For the lattice, with $R_y, R_z$ = `acceleration` and
$\Delta$ = `caipi_shift`, line $y$ is acquired when
$(y - n_y // 2) \bmod R_y = 0$; on the $j$-th acquired line from the centre
line, $j = (y - n_y // 2) // R_y$, partition $z$ is acquired when
$(z - n_z // 2 - \Delta j) \bmod R_z = 0$. The centre view is always
acquired and $\Delta = 0$ gives a rectangular lattice.

For `'poisson'` the support is a draw of {func}`~pypulseqpp.make_poisson_disc_mask`
at nominal acceleration $R_y R_z$, and `caipi_shift` is ignored. A rectangular
calibration region is seeded into the draw as a fully sampled block; an
elliptical one (`elliptical_acs=True`) is not, so the corners of its bounding
rectangle follow the draw and only the ellipse is acquired in full.

For either scheme, partial Fourier and the `elliptical` crop apply to the
views outside the calibration region, and the calibration region is acquired in
full within partial Fourier. `elliptical=False` removes no view for lying
outside the inscribed ellipse.

### Masks

`mask[y, z]` is `True` where the view is acquired. A mask encodes no order.

- {func}`~pypulseqpp.make_caipirinha_mask` is anchored at `(0, 0)`: line `y`
  is acquired when `y mod ry = 0`, and on it partition `z` when
  `(z - delta * (y // ry)) mod rz = 0` [^caipi]. The nominal acceleration is
  `ry * rz`; the finite grid can change the realised factor.
- {func}`~pypulseqpp.make_poisson_disc_mask` places views by Bridson's dart
  throwing, adapted from `sigpy.mri.samp.poisson`. The minimum distance
  between acquired views grows with the distance $r$ from the centre as
  $1 + s r$, and the slope $s$ is found by bisection so that the realised
  acceleration matches `accel` within `tol`. The calibration block, rows
  `n_y // 2 - c_y // 2` to `n_y // 2 + (c_y + 1) // 2 - 1` and likewise for
  columns, is seeded into the draw, so no other view is placed within the minimum
  distance of it. `crop_corner` removes the views whose normalised distance
  outside the calibration block is at least 1: with no calibration block, the
  views outside the ellipse inscribed in the grid. Equal seeds give equal masks.
  A request the grid cannot meet raises `ValueError`.
- {func}`~pypulseqpp.make_random_mask` holds `round(n_y * n_z / accel)` views:
  the centred calibration block, and the remainder drawn uniformly without
  replacement. When the block alone exceeds that number, the mask is the block.

[^caipi]: Breuer FA, Blaimer M, Mueller MF, et al. Controlled aliasing in
    volumetric parallel imaging (2D CAIPIRINHA). *Magn Reson Med*
    2006;55(3):549-556. doi:10.1002/mrm.20787

## Ordering

{func}`~pypulseqpp.make_traversal_order` returns a permutation `p` of
`range(n)`; the loop visits position `p[0]` first. The positions are abstract
indices, such as slices or the entries of a list of selected lines, and
`[items[i] for i in p]` reorders such a list. `'interleaved'` visits the even
positions and then the odd ones, `'center_out'` starts at the middle position
and alternates outward with the lower position first on a tie, `'outside_in'`
is its reverse, and `'random'` is seeded.

The echo-train orderings take centred coordinates, an `(N, 2)` array of
`(ky, kz)` or an `(N,)` array of `ky`, in the units supplied; a boolean mask or
a bare count is refused. `train_length` is at least 1 and the number of shots
is `ceil(N / train_length)`. They return `trains`, with `trains[s][e]` the row
index of the view acquired at echo `e` of shot `s`; every row appears exactly
once and an empty input gives `[]`. `pad=True` pads every train to
`train_length` with `None` for an echo that acquires no view, so that position
in a train is the echo index. A `center_echo` outside the train is refused,
not wrapped. They do not change the support.

| Routine | Rule |
| --- | --- |
| {func}`~pypulseqpp.make_linear_order` | Raster rank, by `kz` then `ky`, cut into `train_length` bands of `ceil(N / train_length)` views; band `e` is echo `e`, dealt across shots in `ky` order. `center_echo` rotates the band order so that the band holding the view nearest the origin is acquired at that echo. |
| {func}`~pypulseqpp.make_centric_order` | Rank by distance from the origin, ties by polar angle, cut into bands as above and dealt across shots in angle order, so the first echo of each shot is near the centre. `center_echo` rotates the band order and moves the effective echo time without changing train membership. |
| {func}`~pypulseqpp.make_radial_order` | Sort by polar angle and cut into angular wedges of `train_length` views, one per shot; within a wedge, by increasing distance, so every shot acquires its view nearest the centre at echo 0. This is the view ordering of 3D fast spin echo, not a radial trajectory. |
| {func}`~pypulseqpp.make_radial_adaptive_order` | Distance bands as in the centric order; the innermost is acquired at `center_echo` and successive bands at echoes increasingly distant from it, alternating before and after, so distance grows monotonically away from the target in both directions, with no discontinuity at the centre. Within a band the views are dealt across shots in polar-angle order. |
| {func}`~pypulseqpp.make_shuffling_order` | With `cluster=True`, trains are contiguous views in raster order and the echo positions within each train are randomly permuted; with `False`, train membership is a random partition too. Seeded: equal seeds give equal orders. |

The linear, radial and adaptive radial orders follow schemes A to C of
Buonincontri et al. (ISMRM abstract 566-05-007, Fig. 2); the shuffled echo
trains follow Tamir et al.[^tamir]

[^tamir]: Tamir JI, Uecker M, Chen W, et al. T2 shuffling: sharp, multicontrast,
    volumetric fast spin-echo imaging. *Magn Reson Med* 2017;77:180-195.

Poisson-disc sampling selects **which** views are acquired; T2 Shuffling
decides **when** the selected views are played along the echo train. The
shipped fast-spin-echo sequence combines them under `ordering='shuffling'`;
the MPRAGE sequence's `ordering='shuffling'` pairs the same Poisson-disc
support with a random line order within each partition.

Which calibration views are played first, whether the support is reordered
before acquisition, and which labels each repetition writes are decided by the
sequence function: `LIN` and `PAR` from the encoded view, `ECO` from the echo
index, `IMA` from membership of the calibration list, and `SEG`, `SLC`, `SET`
or `REP` from its position in the loop. {meth}`~pypulseqpp.Sequence.evaluate_labels`
recovers the ADC order from the sequence, so a sampling figure can show the
implemented order.

## EPI shot offsets

{func}`~pypulseqpp.make_epi_shot_offsets` returns an `(etl, 2)` integer array
of `(Δky, Δkz)`, in encoded lines and partitions, relative to echo 0 of the
same shot; row 0 is `(0, 0)`. For a shot whose first echo acquires `(y0, z0)`,
echo `e` acquires `(y0, z0) + offsets[e]`. The scan loop chooses `(y0, z0)`;
the routine selects no support and creates no labels.

With $R_y$ = `acceleration` and $S$ = `segments`, `'linear'` and `'caipi'` have
$\Delta k_{y,e} = e S R_y$. $S$ shots with origins $y_0, y_0 + R_y, \dots,
y_0 + (S - 1) R_y$ interleave to acquire every $R_y$-th line. `'caipi'` adds
the partition offset of the CAIPIRINHA lattice with $R_z$ =
`partition_acceleration` and shift `caipi_shift`, reduced modulo $R_z$, read
from {func}`~pypulseqpp.make_caipirinha_mask` so that the offsets tile it
exactly. `'zigzag'` alternates an outward and a return pass across `extent`
lines, the return pass displaced by half a blip, which acquires the same lines
at several echo times; `extent` is required for it and rejected by the others,
and $\Delta k_z = 0$ for `'linear'` and `'zigzag'`. Segmented blipped-CAIPI
follows Stirnberg and Stöcker[^stirnberg] and the zigzag traversal Dong et
al.[^dong]

[^stirnberg]: Stirnberg R, Stöcker T. Segmented k-space blipped-controlled
    aliasing in parallel imaging for high spatiotemporal resolution EPI.
    *Magn Reson Med* 2021;85(3):1540-1551. doi:10.1002/mrm.28486

[^dong]: Dong Z, Wald LL, Polimeni JR, Wang F. Single-shot echo planar
    time-resolved imaging for multi-echo functional MRI and distortion-free
    diffusion imaging. *Magn Reson Med* 2025;93(3):993-1013.
    doi:10.1002/mrm.30327

## Orientation schedules

The angle routines return one in-plane rotation angle per shot (spoke,
interleaf or blade), in radians, accumulated modulo $2\pi$ for the golden
family; the scan loop applies each as a rotation of the readout trajectory.

| Routine | Angles |
| --- | --- |
| {func}`~pypulseqpp.calc_uniform_angles` | $k\,\mathrm{span}/n$ over $[0, \mathrm{span})$; `span` $= \pi$ for diametric spokes, $2\pi$ (default) for full-turn arms. |
| {func}`~pypulseqpp.calc_golden_angles` | Increment $\pi/\phi$ (default, $\pi$-periodic radial or blade angle) or, with `full_circle=True`, $2\pi/\phi^2$ (the 137.5° spiral angle) [^golden]. |
| {func}`~pypulseqpp.calc_tiny_golden_angles` | Increment $\pi/(\phi + N - 1)$ for index $N \ge 1$; $N = 1$ is the radial golden angle [^tiny]. |
| {func}`~pypulseqpp.calc_raga_angles` | A finite equidistant support of `fib(approximation_order, tiny_index)` angles, visited in a golden-angle-like order and repeating when `n` exceeds the support [^raga]. |
| {func}`~pypulseqpp.calc_projection_shell` | See below. |

$\phi$ is the golden ratio; its irrationality keeps any window of consecutive
golden-angle spokes near-uniformly distributed.

[^golden]: Winkelmann S, Schaeffter T, Koehler T, Eggers H, Doessel O. An
    optimal radial profile order based on the golden ratio for time-resolved
    MRI. *IEEE Trans Med Imaging* 2007. doi:10.1109/TMI.2006.885337

[^tiny]: Wundrak S, Paul J, Ulrici J, et al. Golden ratio sparse MRI using tiny
    golden angles. *Magn Reson Med*. doi:10.1002/mrm.25831

[^raga]: Scholand N, et al. RAGA sampling. *Magn Reson Med*.
    doi:10.1002/mrm.30254

{func}`~pypulseqpp.calc_projection_shell` returns the unit directions of a
pole-to-pole base shell, shape `(n_views, 3)` with `n_views` at least 3, and
`n_shots` rotation matrices about $z$, shape `(n_shots, 3, 3)`, each turning the
whole shell to where that shot samples. The poles are shared by the rotated
shells and each non-polar ring has one spoke per shot. `'spiral'` places
neighbours at equal angular separation on equal-area rings, winding pole to
pole, so the shell alone is near-uniform and every turn between consecutive
views is the same slew; the polar gap next to a pole fixes the step and each
azimuth increment follows from the spherical cosine relation. `'meridian'` is
a half great circle in the x-z plane at equal polar steps, simpler and
oversampling the poles.

## RF schedules

Each routine returns one value per repetition or echo, applied by the scan loop
to the RF and ADC events of that repetition.

- {func}`~pypulseqpp.make_rf_spoiling_schedule` returns quadratic phases in
  $[0, 2\pi)$: the phase advances by a linear increment, which itself advances by
  `increment` (117° by default) every repetition. With the default zero initial
  increment the first two phases are equal (0°, 0°, 117°, 351°, ...). The
  phase applies to both excitation and ADC offsets.
- {func}`~pypulseqpp.make_phase_cycling_schedule` repeats a non-empty,
  one-dimensional, finite cycle, `(0, π)` by default, reduced modulo $2\pi$,
  for balanced SSFP.
- {func}`~pypulseqpp.make_traps_schedule` returns refocusing flip angles in
  radians for an echo train of at least one echo and a positive target. With
  `variable=False` the target is held constant. Otherwise the first angle
  is raised above the target, near $\pi/2 + \text{target}/2$ for
  pseudo-steady-state stabilisation[^alsop], and the later angles decay towards
  the target by a factor of two per echo. Only that initial stabilising
  transition is implemented, not the smooth flip-angle trains of TRAPS[^traps].

[^alsop]: Alsop DC. The sensitivity of low flip angle RARE imaging. *Magn Reson
    Med* 1997;37(2):176-184. doi:10.1002/mrm.1910370206

[^traps]: Hennig J, Weigel M, Scheffler K. Multiecho sequences with variable
    refocusing flip angles: optimization of signal behavior using smooth
    transitions between pseudo steady states (TRAPS). *Magn Reson Med*
    2003;49(3):527-535. doi:10.1002/mrm.10391
