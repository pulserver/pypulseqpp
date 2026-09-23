# Slew rate

```{admonition} TL;DR
:class: tldr

- {func}`~pypulseqpp.safety.check_max_slew` compares the largest per-axis slew
  rate within each block, on the physical axes after that block's rotation,
  with `max_slew` from the system limits. A nonpositive `max_slew` disables the
  comparison.
- For an arbitrary gradient, the slew rate is the difference between
  neighbouring waveform corners divided by their spacing on the gradient raster
  of the system limits, so the same samples on a finer raster imply a
  proportionally higher slew rate.
- The report also states the largest simultaneous vector slew rate and the peak
  of each axis; only the per-axis quantity is compared with the limit. A step
  across a block boundary is evaluated with the same limit by the gradient
  continuity check.
- For an area $A$ (1/m) at slew rate $S$ (Hz/m/s), the shortest waveform is a
  triangle of duration $T_{\min} = 2\sqrt{A/S}$ while its peak $\sqrt{AS}$ does
  not exceed `max_grad`. Halving the duration of a prewinder or phase-encode
  blip requires four times the slew rate.
- {func}`~pypulseqpp.apply_system_derates` and {func}`~pypulseqpp.cap_system`
  return copies of the system limits with reduced `max_grad` and `max_slew`,
  and repeated derating does not compound. A sequence that fails under reduced
  limits must be redesigned at the lower limit, which lengthens its ramps.
```

The rate of change of a gradient amplifier's output is bounded by the voltage
available across the coil inductance.
{func}`~pypulseqpp.safety.check_max_slew` compares the largest per-axis slew
rate within blocks with `max_slew` from the system limits.

## Quantity compared with the limit

The slew rate is evaluated **within each block**, on the physical axes after
that block's rotation:

$$
\max_{t}\;\max_{a \in \{x,y,z\}} \left| \frac{\mathrm{d}G_a}{\mathrm{d}t} \right|
\;\le\; \texttt{max\_slew}.
$$

For a trapezoid this is the amplitude divided by the rise or fall time. For an
arbitrary gradient it is the difference between neighbouring waveform corners
divided by their spacing on the gradient raster given by the system limits, so
the same samples on a finer raster imply a proportionally higher slew rate.

As in {doc}`gradient_amplitude`, the report also states the largest
simultaneous vector slew rate and the peak of each axis; only the per-axis
quantity is compared with the limit. A nonpositive `max_slew` disables the
comparison.

## Within-block and boundary evaluation

A gradient that ends one block at a nonzero amplitude, followed by a block that
starts at a different amplitude, is a step over one raster period that neither
block's waveform contains. {doc}`gradient_continuity` evaluates that step with
the same limit. Trapezoids begin and end at zero, so the boundary condition
constrains mainly readouts that do not return to zero between blocks, such as
zero-echo-time and joined spiral readouts.

## Minimum-duration gradient lobe

For an area $A$ (1/m) at slew rate $S$ (Hz/m/s), the shortest waveform is the
triangle with peak amplitude $\sqrt{AS}$ and duration

$$
T_{\min} = 2\sqrt{A/S}.
$$

When $\sqrt{AS}$ exceeds `max_grad`, the waveform acquires a flat top and is
longer. Halving the duration of a prewinder or phase-encode blip requires four
times the slew rate. {func}`~pypulseqpp.make_trapezoid` raises for an `area`
and `duration` that cannot be satisfied together under the system limits.

## Derating

{func}`~pypulseqpp.apply_system_derates` returns a copy of the system limits
with `max_grad` and `max_slew` scaled from their base values, which the copy
retains, so repeated derating does not compound.
{func}`~pypulseqpp.cap_system` returns a copy with the limits lowered to stated
ceilings. Passing either to the check evaluates the sequence against the reduced
limits; a sequence that fails under them must be redesigned at the lower limit,
which lengthens its ramps.

## See also

* {func}`~pypulseqpp.safety.check_max_slew` — the call and its report.
* {doc}`gradient_continuity` — the same limit across a block boundary.
* {doc}`pns` — the nerve-stimulation estimate for switching gradients.
