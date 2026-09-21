# Gradient amplitude

A gradient amplifier has a maximum output current, and therefore a maximum
gradient amplitude on the axis it drives. {func}`~pypulseqpp.safety.check_max_grad`
establishes whether any axis exceeds `max_grad` at any point in the sequence.

## Quantity compared with the limit

The check expands the block table into physical-axis gradient waveforms,
applies each block's rotation, and compares the **largest per-axis amplitude**
with the limit:

$$
\max_{t}\;\max_{a \in \{x,y,z\}} |G_a(t)| \;\le\; \texttt{max\_grad}.
$$

The report also states the largest simultaneous **vector magnitude**,

$$
\max_{t}\;\sqrt{G_x(t)^2 + G_y(t)^2 + G_z(t)^2},
$$

and the peak of each axis on its own, with the block each was found in. Only
the per-axis quantity is compared with a limit. A nonpositive `max_grad`
disables the comparison, and the report is returned regardless.

Which of the two quantities a system's limit applies to is a property of the
gradient chain, and vendors differ. The verdict is the per-axis one, because
`max_grad` in a Pulseq system description is a per-axis limit; the vector
magnitude is reported beside it so that a caller whose system constrains the
vector can apply that criterion itself.

## Simultaneous vector magnitude against per-axis peaks

The three axis peaks in the report are in general attained at different times.
Their root-sum-square is therefore an upper bound on the vector magnitude the
sequence actually plays, and usually a loose one: a Cartesian gradient echo
reaches its readout peak on x while y is idle, and its phase-encode peak on y
while x is idle, so the norm of the axis peaks describes an instant the
sequence never plays.

```{figure} ../../generated/figures/axis_peaks_against_vector.png
A radial gradient echo. The slice-selection lobe reaches 40 mT/m on z while x
and y are idle, and the rotated readout reaches 22 mT/m on each of x and y at
angles the slice lobe is not played at. The root-sum-square of the three axis
peaks is 50 mT/m; the largest magnitude the three amplifiers are simultaneously
asked for is 40 mT/m.
```

The reported vector peak is computed sample by sample and is the largest
magnitude the amplifiers are *simultaneously* asked for. Reading it as the norm
of independent maxima, or computing it that way from the axis entries,
overstates the demand.

## Dependence on block rotation

A rotation redistributes one logical waveform over the physical axes. Because a
rotation is an isometry, the vector magnitude at every instant is unchanged;
the per-axis values are not, because the components change.

A radial or spiral trajectory therefore reaches its largest per-axis amplitude
at some particular set of angles and not at others, and a sequence that is
within `max_grad` when checked unrotated can exceed it at a prescribed
orientation. This is why the check applies the block rotations rather than
reading the logical waveforms, and why a prescription rotation is worth passing
where the scan will be prescribed obliquely.

```{figure} ../../generated/figures/rotation_against_per_axis_limit.png
A Cartesian gradient echo whose logical readout, phase-encode and slice axes
each stay within `max_grad`. Its prewinder block and its rewinder-and-spoiler
block play two logical axes at once, so the vector magnitude there exceeds the
per-axis limit. Under a double-oblique prescription that vector is
redistributed over the physical axes and one of them exceeds `max_grad`, while
the vector magnitude, which the rotation leaves unchanged, is the same in both
frames.
```

The vector magnitude is therefore the bound on what any prescription can place
on a single physical axis at that instant: a rotation that aligns the vector
with an axis puts its whole magnitude there. A sequence whose logical per-axis
peaks are within `max_grad` but whose vector peak is not has orientations at
which it fails the check, and the check reports both quantities so that margin
is readable before an orientation is chosen.

## Relationship between gradient amplitude and spatial resolution

The amplitude a readout needs follows from the resolution and the acquisition
duration. Traversing a k-space extent $\Delta k = N/\mathrm{FOV}$ in a window
of duration $T$ at constant amplitude requires

$$
G = \frac{\Delta k}{T}
$$

in the file format's Hz/m, or $\Delta k/(\gamma T)$ in T/m. At a fixed field of
view and matrix size, halving the acquisition window doubles the receiver
bandwidth and doubles the gradient amplitude required. A readout that exceeds
`max_grad` therefore encodes its resolution faster than the amplifier allows,
and lengthening the acquisition window is the direct remedy.

## See also

* {func}`~pypulseqpp.safety.check_max_grad` — the call and its report.
* {func}`~pypulseqpp.apply_system_derates` and {func}`~pypulseqpp.cap_system` —
  checking against limits other than the ones the sequence was designed under.
* {doc}`slew_rate` — the limit on how fast that amplitude may be reached.
