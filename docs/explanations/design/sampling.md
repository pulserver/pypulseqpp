# Sampling support and ordering

A Cartesian acquisition is specified by two independent choices: its
**support**, the set of phase- and partition-encoding views that are acquired,
and its **temporal ordering**, the repetition, shot and echo at which each
acquired view is played. The sampling routines keep the two separate, and
neither creates events or labels. A {class}`~pypulseqpp.sequences.SequenceApp`
combines them in its scan loop.

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
SequenceApp.loop / kernel    one call per repetition
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
views = np.array(calibration + imaging)                  # A: encoded indices
centred = views - (n_y // 2, n_z // 2)                   # B: centred coordinates
trains = pp.make_radial_adaptive_order(centred, etl, center_echo=te_echo)  # D
view_of = [[tuple(views[i]) for i in train] for train in trains]
```

The same indices `i` select the encoded view `views[i]`, from which the kernel
scales the phase-encoding gradient and writes the `LIN` and `PAR` labels. A
boolean mask enters the same way, through `np.argwhere(mask)`.

## Poisson-disc support and T2 Shuffling

Both are associated with variable-density, echo-resolved acquisitions, and they
answer different questions.

| | Poisson-disc sampling | T2 Shuffling |
| --- | --- | --- |
| Determines | **which** views are acquired | **when** already selected views are acquired along the echo train |
| Routine | `make_cartesian_plane_sampling(..., sampling='poisson')`, {func}`~pypulseqpp.make_poisson_disc_mask` | {func}`~pypulseqpp.make_shuffling_order` |
| Output | encoded views, or a boolean mask | `trains[shot][echo]` indices |

The shipped fast-spin-echo and MPRAGE applications combine them under
`ordering='shuffling'`; either can be used without the other.

## Labels

Label events are created only by the sequence application. In each call of
`kernel` it writes `LIN` and `PAR` from the encoded view, `ECO` from the echo
index, `IMA` from membership of the calibration list, and `SEG`, `SLC`, `SET`
or `REP` from its position in the loop. The sampling routines return plain
Python and NumPy values; which calibration views are played first, and whether
the support is reordered before acquisition, is decided by the application.
{meth}`~pypulseqpp.Sequence.evaluate_labels` recovers the labels actually
written, per acquisition.

## See also

* {doc}`../../api/sampling`: the routines, their inputs and returns.
* {doc}`sequence-application`: the loop and kernel that consume them.
