# Gradient continuity

Blocks are played back to back with no gap. A gradient that ends one block at a
nonzero amplitude and is followed by a block whose gradient starts at a
different amplitude is asking the amplifier for a step, and the hardware has to
slew it like any other change.
{func}`~pypulseqpp.safety.check_grad_continuity` finds those steps, and
establishes that the sequence leaves every axis at zero.

## The criterion

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
slew limit. It is the slew limit applied where no waveform exists to carry the
change, and the report states it as a step, naming the block, the axis, the
amplitudes on either side and the slew rate the step implies — the form that
says which pair of blocks to fix rather than which number to lower.

## Ending at zero

The report also carries `ends_at_zero`: whether the last block leaves every
physical axis at zero amplitude. A sequence that ends with a gradient still on
leaves the amplifier in a state the next scan does not expect, and the verdict
is false whether or not any boundary step was found.

## Why both endpoints need their own rotation

Each block's `ROTATIONS` extension applies to that block's gradients. Two
consecutive blocks under different rotations can hold identical logical
waveforms and still present a step to the amplifiers, because the same logical
vector resolves onto different physical axes on either side of the boundary.

This is the cost of expressing a rotated trajectory as one interleaf plus a
rotation per shot: the shape library stays small, but continuity has to be
judged after the rotations, on the instances, rather than once per shape. A
non-Cartesian readout that returns to zero between shots is unaffected; one
that runs continuously across the boundary has to be designed so that the
rotated endpoints meet.

## Reading the report

The blocks a discontinuity names are 1-based indices into the sequence, which
is what {meth}`~pypulseqpp.Sequence.get_block` takes and what
{meth}`~pypulseqpp.Sequence.paper_plot` and
{func}`~pypulseqpp.plot.plot_kspace` accept as a `block_range`, so a reported
boundary can be looked at directly.

## Related

* {func}`~pypulseqpp.safety.check_grad_continuity` — the call and its report.
* {doc}`slew_rate` — the same inequality inside a waveform.
* {func}`~pypulseqpp.make_extended_trapezoid` — a gradient stated by its
  corner points, which is how a waveform that starts or ends at a nonzero
  amplitude is written deliberately.
