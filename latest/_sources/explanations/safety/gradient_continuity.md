# Gradient continuity

Blocks are played back to back with no gap. When a gradient ends one block at a
nonzero amplitude and the next block begins it at a different amplitude, the
file prescribes an instantaneous step, and the amplifier has to slew across it
as it would across any other change of amplitude.
{func}`~pypulseqpp.safety.check_grad_continuity` finds those steps and
establishes that every axis ends the sequence at zero amplitude.

## Continuity criterion

For each block boundary and each physical axis, the check compares the
amplitude the previous block left the axis at with the amplitude the next block
begins it at. Both endpoints are taken in physical coordinates, each under its
own block's rotation, and an absent gradient contributes zero. The step is
reported as a discontinuity when

$$
|\Delta G| \;>\; \texttt{max\_slew} \times \texttt{grad\_raster\_time},
$$

which is the slew-rate inequality with one raster period as $\Delta t$: a step
is legal exactly when it could have been played as a ramp over a single raster
sample.

```{figure} ../../generated/figures/continuity_seam.png
The same pair of blocks with two different endpoint amplitudes on the first.
The left step could have been played as a ramp over one gradient raster period
and is not reported; the right one could not.
```

A discontinuity is therefore not a second physical constraint alongside the
slew limit; it is the slew limit applied where the file provides no waveform
over which the change could occur. The report states each discontinuity as a
step, naming the block, the axis, the amplitudes on either side and the slew
rate the step implies. Those fields identify the pair of blocks to correct,
which a single worst-case number would not.

## Final gradient amplitude

The report also states `ends_at_zero`: whether the last block ends every
physical axis at zero amplitude. A sequence that ends with a gradient still at
a nonzero amplitude begins the next acquisition from an unknown state, and the
verdict is false whether or not any boundary step was found.

## Rotation of the endpoint amplitudes

Each block's `ROTATIONS` extension applies to that block's gradients. Two
consecutive blocks under different rotations can contain identical logical
waveforms and still present a step to the amplifiers, because the same logical
vector resolves onto different physical axes on either side of the boundary.

This is the cost of expressing a rotated trajectory as one interleaf plus a
rotation per shot: the shape library stays small, but continuity has to be
judged after the rotations, on the instances, rather than once per shape. A
non-Cartesian readout that returns to zero between shots is unaffected; one
that runs continuously across the boundary has to be designed so that the
rotated endpoints meet.

## Contents of the report

The blocks a discontinuity names are 1-based indices into the sequence, which
is the index {meth}`~pypulseqpp.Sequence.get_block` takes and the one
{meth}`~pypulseqpp.Sequence.paper_plot` and
{func}`~pypulseqpp.plot.plot_kspace` accept as a `block_range`, so a reported
boundary can be looked at directly.

## See also

* {func}`~pypulseqpp.safety.check_grad_continuity` — the call and its report.
* {doc}`slew_rate` — the same inequality inside a waveform.
* {func}`~pypulseqpp.make_extended_trapezoid` — a gradient stated by its
  corner points, which is how a waveform that starts or ends at a nonzero
  amplitude is written deliberately.
