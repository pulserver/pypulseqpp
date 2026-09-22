# Sampling and ordering

Which views a scan acquires, the order it plays them in, the angles a
non-Cartesian trajectory turns through, and the schedules that vary a pulse
from one repetition to the next.

```{eval-rst}
.. currentmodule:: pypulseqpp
```

None of this is part of a `.seq` file or of a sequence module: a module solves
the events of one repetition, and a scan loop decides which view each
repetition encodes. These are the routines that decide. Every shipped sequence
is written against them, which is what settled the set.

## Cartesian views

{func}`calc_sampled_lines` chooses the views of one encoded axis and
{func}`calc_sampled_pairs` the `(line, partition)` pairs of two. Both return
the calibration region and the rest separately, because a scan plays the
calibration first — so a reconstruction can estimate coil sensitivities while
the rest is still arriving — and marks it differently. Both keep the centre of
k-space whatever the undersampling asks for.

| Object | Description |
| --- | --- |
| {obj}`~pypulseqpp.calc_sampled_lines` | The calibration views of one encoded axis, and the others acquired. |
| {obj}`~pypulseqpp.calc_sampled_pairs` | The calibration pairs of two encoded axes, and the others acquired. |
| {obj}`~pypulseqpp.make_caipirinha_mask` | A CAIPIRINHA lattice as a boolean mask. |
| {obj}`~pypulseqpp.make_poisson_disc_mask` | A variable-density Poisson-disc draw as a boolean mask. |
| {obj}`~pypulseqpp.make_random_mask` | A random draw as a boolean mask. |

## Traversal and echo trains

{func}`calc_traversal_order` is the order one axis is traversed in. The others
deal a set of views into echo trains, which is what a train's contrast follows
from: the view a train acquires at its first echo is the one that carries it.

| Object | Description |
| --- | --- |
| {obj}`~pypulseqpp.calc_traversal_order` | A permutation of one axis: sequential, reversed, interleaved, centre-out, outside-in or random. |
| {obj}`~pypulseqpp.calc_epi_order` | The phase-encode offsets one EPI shot steps through. |
| {obj}`~pypulseqpp.make_linear_order` | Echo trains that step through the views in order. |
| {obj}`~pypulseqpp.make_centric_order` | Echo trains that start at the centre of k-space. |
| {obj}`~pypulseqpp.make_radial_order` | Echo trains along radial spokes of the view plane. |
| {obj}`~pypulseqpp.make_radial_adaptive_order` | Radial trains whose length follows the local density. |
| {obj}`~pypulseqpp.make_shuffling_order` | Echo trains that vary which view each echo index acquires. |

## Angles

Angles are in radians. A projection turns through them; a stack of them turns
its in-plane trajectory.

| Object | Description |
| --- | --- |
| {obj}`~pypulseqpp.calc_uniform_angles` | Angles spread evenly over a span. |
| {obj}`~pypulseqpp.calc_golden_angles` | The golden-angle increment, whose every prefix is nearly uniform. |
| {obj}`~pypulseqpp.calc_tiny_golden_angles` | A smaller golden increment, for a gentler step between neighbours. |
| {obj}`~pypulseqpp.calc_raga_angles` | Rational approximation of the golden angle, which repeats exactly. |
| {obj}`~pypulseqpp.calc_projection_shell` | Unit spoke directions over a sphere, and the turn that places each shot. |

## Schedules

One value per repetition, for what varies from one to the next.

| Object | Description |
| --- | --- |
| {obj}`~pypulseqpp.make_rf_spoiling_schedule` | The quadratically increasing transmit phase of a spoiled sequence. |
| {obj}`~pypulseqpp.make_phase_cycling_schedule` | A repeating phase cycle, as a balanced sequence alternates. |
| {obj}`~pypulseqpp.make_traps_schedule` | A refocusing-flip train that settles into a pseudo-steady state. |
