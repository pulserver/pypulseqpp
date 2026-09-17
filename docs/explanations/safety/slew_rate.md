# Slew rate

The rate at which a gradient amplifier can change its output is bounded by the
voltage available across the coil's inductance.
{func}`~pypulseqpp.safety.check_max_slew` establishes whether the sequence ever
asks any axis to change faster than `max_slew`.

## What is compared

The check evaluates the slew rate **within each block**, on the physical axes
after that block's rotation, and compares the largest per-axis value with the
limit:

$$
\max_{t}\;\max_{a \in \{x,y,z\}} \left| \frac{\mathrm{d}G_a}{\mathrm{d}t} \right|
\;\le\; \texttt{max\_slew}.
$$

For a trapezoid the quantity is the amplitude divided by the rise or fall time.
For an arbitrary gradient it is the difference between neighbouring waveform
samples divided by the gradient raster period, so the raster the check is given
is part of the criterion rather than an implementation detail: the same sample
array on a finer raster asks for a proportionally higher slew rate.

As with {doc}`gradient_amplitude`, the report carries the largest simultaneous
vector slew rate and the peak of each axis on its own, and only the per-axis
quantity is compared with the limit. A nonpositive `max_slew` disables the
comparison.

## Within blocks only

The check looks inside blocks. A gradient that ends one block at a nonzero
amplitude and is followed by a block starting at a different amplitude is a step
across one raster period, which no waveform in either block contains. That case
belongs to {doc}`gradient_continuity`, which applies the same inequality to the
pair of samples straddling the boundary.

Separating them is not a redundancy. The interior question is a property of a
**shape**: the normalised waveform's own sample-to-sample differences, scaled by
whatever amplitude an instance plays it at, so it is answered once per distinct
shape and reused by every playout. The boundary question can only be answered
where two neighbours meet, at the amplitudes and rotations they each actually
run, so it is a walk over the block table.

Most Cartesian sequences satisfy the boundary condition trivially, because
trapezoids and simple arbitrary gradients begin and end at zero. It becomes the
binding constraint on families whose readout does not return to zero between
blocks — zero echo time, spirals joined by bridges, a trajectory whose
rewinders were solved per rotation angle.

## The slew rate a trapezoid costs

Amplitude and slew rate together bound how quickly a gradient can deliver a
zeroth moment. In the file format's units, where an area is in 1/m and a slew
rate in Hz/m/s, the shortest waveform delivering an area $A$ at slew rate $S$ is
the triangular one, whose peak amplitude is $\sqrt{AS}$ and whose duration is

$$
T_{\min} = 2\sqrt{A/S}.
$$

No amplitude solves the same area in less time, and once $\sqrt{AS}$ exceeds
`max_grad` the waveform has a flat top and takes longer still. A prewinder or a
phase-encode blip shortened by a factor of two therefore costs a factor of four
in slew rate, which is why echo spacing in an echo-planar train is bounded by
the slew limit rather than by the amplitude limit. The gradient factories solve
this relation from the system limits, which is why
{func}`~pypulseqpp.make_trapezoid` refuses an `area` and a `duration` that
cannot be met together rather than returning a waveform that would fail this
check.

## Derating

Sites commonly run the gradient chain below its nameplate limits — for thermal
headroom, for acoustic reasons, or because a stimulation model bounds it.
{func}`~pypulseqpp.apply_system_derates` returns a copy of the system limits
scaled by a fraction, and {func}`~pypulseqpp.cap_system` a copy lowered to a
stated ceiling; both retain the base limits, so repeated derating does not
compound. Passing the result to the check re-answers the question against the
system the sequence will be played on, without redesigning anything.

A sequence that fails only under a derate is not repaired by the check. Its
gradients have to be redesigned at the lower limit, which lengthens every ramp
and, through the ramps, the echo spacing.

## Related

* {func}`~pypulseqpp.safety.check_max_slew` — the call and its report.
* {doc}`gradient_continuity` — the same inequality across a block boundary.
* {doc}`pns` — the physiological bound on switching, which is usually reached
  first.
