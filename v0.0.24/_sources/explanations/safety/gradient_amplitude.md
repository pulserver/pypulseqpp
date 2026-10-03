# Gradient amplitude

```{admonition} TL;DR
:class: tldr

- {func}`~pypulseqpp.safety.check_max_grad` compares the largest per-axis
  amplitude, after each block's rotation, with `max_grad` from the system
  limits. A nonpositive `max_grad` disables the comparison.
- The report also states each axis peak with its 1-based block and the largest
  simultaneous vector magnitude. Only the per-axis quantity is compared with a
  limit, because `max_grad` in a Pulseq system description is a per-axis limit.
- The axis peaks are in general attained at different times, so their
  root-sum-square is an upper bound on the vector magnitude, and usually a
  loose one.
- A rotation preserves the vector magnitude and changes its per-axis
  components, so a sequence within `max_grad` unrotated can exceed it at an
  oblique prescription. A prescription is evaluated by applying it with
  {class}`~pypulseqpp.TransformFOV` and checking the result.
- A readout traversing $\Delta k = N/\mathrm{FOV}$ at constant amplitude in a
  window of duration $T$ requires $G = \Delta k/T$ in Hz/m. At fixed field of
  view and matrix size, halving $T$ doubles both the receiver bandwidth and the
  required amplitude.
```

A gradient amplifier has a maximum output current, and therefore a maximum
gradient amplitude on the axis it drives.
{func}`~pypulseqpp.safety.check_max_grad` compares the largest per-axis
amplitude of the sequence with `max_grad` from the system limits.

## Quantity compared with the limit

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

## Simultaneous vector magnitude and per-axis peaks

The three axis peaks are in general attained at different times, so their
root-sum-square is an upper bound on the vector magnitude, and usually a loose
one. The reported vector peak is evaluated at each instant.

```{figure} ../../generated/figures/axis_peaks_against_vector.png
A radial gradient echo. The slice-selection lobe reaches 40 mT/m on z while x
and y are idle, and the rotated readout reaches 22 mT/m on each of x and y at
angles the slice lobe is not played at. The root-sum-square of the three axis
peaks is 50 mT/m; the largest simultaneous vector magnitude is 40 mT/m.
```

## Dependence on rotation

A rotation preserves the vector magnitude at every instant and changes its
per-axis components. In a plane, a vector of magnitude $|G| \le$ `max_grad` is
within the per-axis limit at every orientation; a longer one exceeds it at some
orientations. A sequence within `max_grad` unrotated can therefore exceed it at
an oblique prescription.

```{figure} ../../generated/figures/rotation_against_per_axis_limit.png
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
by applying it with {class}`~pypulseqpp.TransformFOV` and checking the result;
{func}`~pypulseqpp.apply_system_derates` returns reduced limits to design
against.

## Readout amplitude and resolution

A readout traversing $\Delta k = N/\mathrm{FOV}$ in an acquisition window of
duration $T$ at constant amplitude requires

$$
G = \frac{\Delta k}{T}
$$

in Hz/m, or $\Delta k/(\gamma T)$ in T/m. At fixed field of view and matrix
size, halving $T$ doubles both the receiver bandwidth and the required
amplitude.

## See also

* {func}`~pypulseqpp.safety.check_max_grad` — the call and its report.
* {func}`~pypulseqpp.apply_system_derates` and {func}`~pypulseqpp.cap_system` —
  limits other than the ones the sequence was designed under.
* {doc}`slew_rate` — the limit on the rate of change of amplitude.
