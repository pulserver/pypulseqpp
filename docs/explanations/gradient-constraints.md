# Gradient amplitude, slew rate and continuity

```{admonition} TL;DR
:class: tldr

- {func}`~pypulseqpp.safety.check_max_grad` and
  {func}`~pypulseqpp.safety.check_max_slew` compare the largest per-axis
  amplitude and the largest per-axis slew rate within each block, after that
  block's rotation, with `max_grad` and `max_slew` from the system limits. A
  nonpositive limit disables its comparison.
- The reports also state each axis peak and the largest simultaneous vector
  magnitude. The axis peaks are in general attained at different times, so
  their root-sum-square is an upper bound on the vector magnitude.
- A rotation preserves the vector magnitude and changes its per-axis
  components, so a sequence within `max_grad` unrotated can exceed it at an
  oblique prescription.
- {func}`~pypulseqpp.safety.check_grad_continuity` applies the one-raster slew
  criterion $|\Delta G| \leq \mathtt{max\_slew}\cdot\mathtt{grad\_raster\_time}$
  to each block boundary and requires every axis to end at zero.
- For an area $A$ at slew rate $S$, the shortest lobe is a triangle of
  duration $2\sqrt{A/S}$; halving the duration of a prewinder requires four
  times the slew rate.
```

The amplitude, slew-rate and continuity checks read only the sequence and its
system limits. The frames they evaluate in, and the units of their reports, are
in {doc}`constraint-checks`.

## Gradient amplitude

A gradient amplifier has a maximum output current, and therefore a maximum
gradient amplitude on the axis it drives.
{func}`~pypulseqpp.safety.check_max_grad` compares the largest per-axis
amplitude of the sequence with `max_grad` from the system limits.

### Amplitude compared with the limit

The check reconstructs the gradient waveforms after each block's rotation and
evaluates

$$
\max_{t}\;\max_{a \in \{x,y,z\}} |G_a(t)| \;\le\; \mathtt{max\_grad}.
$$

The report also states the peak of each axis, with its 1-based block, and the
largest simultaneous vector magnitude

$$
\max_{t}\;\sqrt{G_x(t)^2 + G_y(t)^2 + G_z(t)^2}.
$$

Only the per-axis quantity is compared with a limit, because `max_grad` in a
Pulseq system description is a per-axis limit. Whether a system also constrains
the vector magnitude is a property of its gradient chain; the reported vector
peak permits that comparison outside the check. A nonpositive `max_grad`
disables the comparison, and the report is returned regardless.

### Simultaneous vector magnitude and per-axis peaks

The three axis peaks are in general attained at different times, so their
root-sum-square is an upper bound on the vector magnitude, and usually a loose
one. The reported vector peak is evaluated at each instant.

```{figure} ../generated/figures/axis_peaks_against_vector.png
A radial gradient echo. The slice-selection lobe reaches 40 mT/m on z while x
and y are idle, and the rotated readout reaches 22 mT/m on each of x and y at
angles the slice lobe is not played at. The root-sum-square of the three axis
peaks is 50 mT/m; the largest simultaneous vector magnitude is 40 mT/m.
```

### Dependence on rotation

A rotation preserves the vector magnitude at every instant and changes its
per-axis components. In a plane, a vector of magnitude $|G| \le$ `max_grad` is
within the per-axis limit at every orientation; a longer one exceeds it at some
orientations. A sequence within `max_grad` unrotated can therefore exceed it at
an oblique prescription.

```{figure} ../generated/figures/rotation_against_per_axis_limit.png
The in-plane gradient vector of a Cartesian gradient echo at the instant its
magnitude is largest, drawn at prescription rotations 15 degrees apart. Left,
a design solved against `max_grad`: its prewinder and its
rewinder-and-spoiler block play the readout and phase-encode axes together, and
each reaches the limit, so the vector is $\sqrt2$ times that amplitude. Only
the orientations that leave it on a diagonal of the box keep both components
inside. Right, the same prescription solved against
`max_grad` divided by $\sqrt2$, which fits the vector inside the circle. Below,
the largest per-axis amplitude over the whole scan against the prescription
angle.
```

The check reads the rotations the sequence holds. A prescription is evaluated
by applying it with {class}`~pypulseqpp.TransformFOV` and checking the result.

### Readout amplitude and resolution

A readout traversing $\Delta k = N/\mathrm{FOV}$ in an acquisition window of
duration $T$ at constant amplitude requires

$$
G = \frac{\Delta k}{T}
$$

in Hz/m, or $\Delta k/(\gamma T)$ in T/m. At fixed field of view and matrix
size, halving $T$ doubles both the receiver bandwidth and the required
amplitude.

## Slew rate

The rate of change of a gradient amplifier's output is bounded by the voltage
available across the coil inductance.
{func}`~pypulseqpp.safety.check_max_slew` compares the largest per-axis slew
rate within blocks with `max_slew` from the system limits.

### Slew rate compared with the limit

The slew rate is evaluated **within each block**, on the axes after that
block's rotation:

$$
\max_{t}\;\max_{a \in \{x,y,z\}} \left| \frac{\mathrm{d}G_a}{\mathrm{d}t} \right|
\;\le\; \mathtt{max\_slew}.
$$

For a trapezoid this is the amplitude divided by the rise or fall time. For an
arbitrary gradient it is the difference between neighbouring waveform corners
divided by their spacing on the gradient raster given by the system limits, so
the same samples on a finer raster imply a proportionally higher slew rate.

As for the amplitude, the report also states the largest
simultaneous vector slew rate and the peak of each axis; only the per-axis
quantity is compared with the limit. A nonpositive `max_slew` disables the
comparison.

### Minimum-duration gradient lobe

For an area $A$ (1/m) at slew rate $S$ (Hz/m/s), the shortest waveform is the
triangle with peak amplitude $\sqrt{AS}$ and duration

$$
T_{\min} = 2\sqrt{A/S}.
$$

When $\sqrt{AS}$ exceeds `max_grad`, the waveform acquires a flat top and is
longer. Halving the duration of a prewinder or phase-encode blip requires four
times the slew rate. {func}`~pypulseqpp.make_trapezoid` raises for an `area`
and `duration` that cannot be satisfied together under the system limits.

(gradient-continuity)=

## Gradient continuity

The slew-rate check evaluates the waveform within each block. A gradient that
ends one block at a nonzero amplitude, followed by a block that starts at a
different amplitude, is a step over one raster period that neither block's
waveform contains. Trapezoids begin and end at zero, so the boundary condition
constrains mainly readouts that do not return to zero between blocks, such as
zero-echo-time and joined spiral readouts.

Adjacent Pulseq blocks have no implicit gap. A change on each axis between the
endpoint of one block and the initial amplitude of the next, after the blocks'
rotations, must satisfy the one-raster slew criterion

$$
|\Delta G| \leq \mathtt{max\_slew} \cdot
                   \mathtt{grad\_raster\_time}.
$$

```{figure} ../generated/figures/continuity_seam.png
A boundary step below the one-raster limit passes; a larger step is reported.
The dashed line marks the block boundary and the red arrow marks $\Delta G$.
```

{func}`~pypulseqpp.safety.check_grad_continuity` evaluates every boundary and
reports the 1-based block index, axis, endpoint amplitudes, and implied slew
rate. The final gradient amplitude on every axis must also be zero; otherwise
`ends_at_zero` and the overall verdict are false.

Each block's rotation is applied before the comparison. Endpoints equal on the
channel axes of two blocks with different rotations can therefore differ after
them. Returning every interleaf to zero avoids this dependence.

{meth}`~pypulseqpp.Sequence.check_timing` runs the same evaluation and reports
its findings as `GRADIENT_DISCONTINUITY` and `GRADIENT_NOT_RAMPED_DOWN`; see
{doc}`timing-and-rasters`.

## Derating

{func}`~pypulseqpp.apply_system_derates` returns a copy of the system limits
with `max_grad` and `max_slew` scaled from their base values, which the copy
retains, so repeated derating does not compound.
{func}`~pypulseqpp.cap_system` returns a copy with the limits lowered to stated
ceilings. Passing either to a check evaluates the sequence against the reduced
limits; a sequence that fails under them must be redesigned at the lower limit,
which lengthens its ramps.

## See also

* {func}`~pypulseqpp.safety.check_max_grad`,
  {func}`~pypulseqpp.safety.check_max_slew` and
  {func}`~pypulseqpp.safety.check_grad_continuity` — the calls and their
  reports.
* {func}`~pypulseqpp.apply_system_derates` and {func}`~pypulseqpp.cap_system` —
  limits other than the ones the sequence was designed under.
* {doc}`constraint-checks` — frames, report units and evaluation intervals.
* {doc}`pns` — the nerve-stimulation estimate for switching gradients.
