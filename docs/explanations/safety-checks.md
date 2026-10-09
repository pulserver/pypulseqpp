# Safety checks

```{admonition} TL;DR
:class: tldr

- pypulseqpp checks a finished sequence against the limits of a scanner:
  gradient amplitude, slew rate and continuity, peripheral nerve stimulation,
  mechanical resonance, sound pressure and SAR.
- The gradient checks read the waveforms as they are played, after each
  block's rotation and, where you pass one, the prescription rotation.
- Every check returns a verdict and a report, whether it passes or not. The
  report says what was found, where, and against which limit.
- Each check is an estimate under the limits and models you supply. PNS,
  mechanical resonance, sound pressure and SAR need data no `.seq` file holds:
  a nerve model, forbidden bands, an acoustic transfer function, VOPs.
```

A sequence that passes {meth}`~pypulseqpp.Sequence.check_timing` can be
played block by block. Whether it stays within the scanner's limits is a
question about the whole scan, and about how it is played: an oblique slice
puts two logical axes on one physical gradient coil, and a train of readouts
can drive the coil at one of its mechanical resonances. The functions in
{mod}`pypulseqpp.safety` answer these questions from the sequence and a
description of the scanner.

You run them while you design, on the sequence your function returns, as in
lesson 7 of the {doc}`Course <../examples/course>`. A passing check does not
establish that a sequence is safe to run on a scanner or on a subject.

## What pypulseqpp does

:::{container} capabilities

- **Checks gradient amplitude and slew rate on each axis, after each block's rotation.**

  Code: {func}`~pypulseqpp.safety.check_max_grad`, {func}`~pypulseqpp.safety.check_max_slew`. Tests: *a gradient stronger than the scanner allows is refused* (`test_safety.py`); *a rotated sequence is weighed on the axes it plays* (`test_safety.py`).
- **Reports the largest simultaneous vector amplitude beside the axis peaks.** It is the largest value a rotation could put on one axis.

  Code: the `vector` field of the report. Tests: *the vector peak is what a rotation could put on one axis* (`test_safety.py`); *the vector peak is taken where the axes slew at once* (`test_safety.py`).
- **Checks the amplitude step across every block boundary, and that every axis ends at zero.**

  Code: {func}`~pypulseqpp.safety.check_grad_continuity`. Tests: *a gradient that starts where the last did not end is a jump* (`test_safety.py`); *a sequence that leaves a gradient on has not ramped down* (`test_safety.py`).
- **Checks a sequence against another scanner's limits, derated or capped.**

  Code: the `system` argument, {func}`~pypulseqpp.apply_system_derates`, {func}`~pypulseqpp.cap_system`. Test: *a sequence can be weighed against another scanner* (`test_safety.py`).
- **Estimates peripheral nerve stimulation with a chronaxie or a SAFE model, over the whole sequence.**

  Code: {func}`~pypulseqpp.safety.check_pns`, {class}`~pypulseqpp.safety.ChronaxieModel`, {func}`~pypulseqpp.safety.read_safe_model`. Tests: *a rectangular slew follows the strength-duration curve* (`test_pns.py`); *the SAFE response is upstream's* (`test_pns.py`); *a response over threshold is refused* (`test_pns.py`).
- **Applies a prescription rotation after each block's own rotation.**

  Code: the `rotation` argument of {func}`~pypulseqpp.safety.check_pns`, {func}`~pypulseqpp.safety.check_mech_resonance` and {func}`~pypulseqpp.safety.check_spl`. Tests: *the prescription is turned after the block's own rotation* (`test_pns.py`, `test_mech_resonance.py`).
- **Compares the windowed gradient spectrum with the coil's forbidden bands.**

  Code: {func}`~pypulseqpp.safety.check_mech_resonance`, {class}`~pypulseqpp.safety.ForbiddenBand`, {func}`~pypulseqpp.safety.read_forbidden_bands`. Tests: *a sustained sinusoid reads its own amplitude* (`test_mech_resonance.py`); *a band guards only the axis it names* (`test_mech_resonance.py`).
- **Returns the spectrum of any window, on every axis.**

  Code: {func}`~pypulseqpp.safety.mech_resonance_spectrum`. Test: *the spectrum reads the amplitude the check reports* (`test_mech_resonance.py`).
- **Computes the peak and A-weighted average sound pressure of the loudest repetition, played without end, and holds them to 140 dB and 99 dB(A).**

  Code: {func}`~pypulseqpp.safety.check_spl`, {func}`~pypulseqpp.safety.read_acoustic_response`. Tests: *the repetition of most gradient energy is evaluated* (`test_acoustic.py`); *a level above its limit fails the check* (`test_acoustic.py`).
- **Estimates local and global SAR from virtual observation points (VOPs), for single-channel and multichannel transmission.**

  Code: {func}`~pypulseqpp.safety.check_sar`, {class}`~pypulseqpp.safety.VopModel`, {func}`~pypulseqpp.safety.read_vops`. Tests: *the worst repetition decides* (`test_sar.py`); *SAR goes with the square of the drive* (`test_sar.py`); *the global SAR is the largest over the body models* (`test_sar.py`).
- **Applies the VOP file's safety factor to local SAR only.**

  Code: `VopModel.metadata`, the `safety_factor` argument. Test: *the safety factor multiplies local SAR and its ratio, not global* (`test_sar.py`).
- **Compares a sequence's SAR with that of a reference pulse, a ratio that needs no drive calibration.**

  Code: the `reference` argument of {func}`~pypulseqpp.safety.check_sar`. Tests: *the ratios do not depend on the drive calibration* (`test_sar.py`); *the global ratio is taken body by body* (`test_sar.py`).
- **Reads the VOP files mariepy writes, and MARIE's and MATLAB's layouts.**

  Code: {func}`~pypulseqpp.safety.read_vops`. Tests: *a one-body npz file as mariepy writes it is read* (`test_sar.py`); *a mat file stack is read batch first* (`test_sar.py`).
- **Plays the gradients as audio, as MATLAB Pulseq's `sound` does.**

  Code: {meth}`~pypulseqpp.Sequence.sound`. Test: *sound follows MATLAB Pulseq's definition* (`test_sound.py`).

:::

## What the scanner does

- **Its own checks before the scan, and its monitoring during it.** The
  scanner judges a sequence with its own models and limits before playing it,
  and monitors the gradient and RF hardware while it plays. Pulserver runs the
  checks on this page on every design, with the limits the scanner sends.
- **SAR, RF coil heating and gradient heating.** On a scanner these are
  computed by the vendor's routines, under the scanner's own calibration.
  {func}`~pypulseqpp.safety.check_sar` gives an estimate under the VOPs and the
  drive you supply; on a scanner, VOPs supply the input the vendor's routine
  needs for pTx.
- **Resampling onto the scanner's rasters.** The scanner plays a waveform on
  its own rasters. pypulseqpp checks timing on the rasters the sequence
  declares ({doc}`timing-and-rasters`).

## How it works

### Logical and physical axes

A block's `ROTATIONS` extension turns its gradients from the axes they are
stored on to the **logical axes** of the design. A prescription rotation turns
the logical axes to the **physical axes**, the three gradient coils, and is
composed after each block's own rotation. Every gradient check reads the
waveforms after the blocks' rotations. {func}`~pypulseqpp.safety.check_pns`,
{func}`~pypulseqpp.safety.check_mech_resonance` and
{func}`~pypulseqpp.safety.check_spl` take the prescription as `rotation`; for
the amplitude, slew-rate and continuity checks you apply it with
{class}`~pypulseqpp.TransformFOV` and check the result.

Reports are in the units of the sequence: gradient amplitude in Hz/m and slew
rate in Hz/m/s, which `system.gamma` converts to mT/m and T/m/s.
Mechanical-resonance amplitudes are in mT/m, PNS responses are fractions of the
model's threshold, sound pressure is in dB and dB(A), and SAR is in W/kg.

### Gradient amplitude and slew rate

`max_grad` and `max_slew` in a Pulseq system description are per-axis limits,
and the checks compare the largest per-axis value with them. The report also
gives the largest simultaneous vector value. The three axes peak at different
times, so the vector peak is in general smaller than the root-sum-square of
the axis peaks.

```{figure} ../generated/figures/axis_peaks_against_vector.png
A radial gradient echo. The slice-selection lobe reaches 40 mT/m on z while x
and y are idle, and the rotated readout reaches 22 mT/m on each of x and y at
angles the slice lobe is not played at. The root-sum-square of the three axis
peaks is 50 mT/m; the largest simultaneous vector magnitude is 40 mT/m.
```

A rotation preserves the vector magnitude and changes its per-axis components.
A design that reaches `max_grad` on two axes at once is within the limit
unrotated and exceeds it at some oblique prescriptions.

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

The slew rate is evaluated within each block. For a trapezoid it is the
amplitude over the ramp time; for an arbitrary gradient it is the difference
between neighbouring samples over their spacing, so the same samples on a finer
raster imply a higher slew rate. For an area $A$ at slew rate $S$, the shortest
lobe is a triangle of duration $2\sqrt{A/S}$: halving the duration of a
prewinder needs four times the slew rate.

(gradient-continuity)=

### Gradient continuity

Adjacent blocks have no gap between them. A gradient that ends one block at
one amplitude and starts the next at another is a step that neither block's
waveform contains. {func}`~pypulseqpp.safety.check_grad_continuity` holds the
change on each axis, after the blocks' rotations, to what the slew rate allows
in one raster period,

$$
|\Delta G| \leq \mathtt{max\_slew} \cdot \mathtt{grad\_raster\_time},
$$

and requires every axis to end the sequence at zero. Trapezoids start and end
at zero, so the condition constrains readouts that carry a gradient across
blocks, such as zero-echo-time and joined spiral readouts. Two blocks with
different rotations can end and start at the same stored amplitude and still
differ on the played axes. {meth}`~pypulseqpp.Sequence.check_timing` reports
the same findings as `GRADIENT_DISCONTINUITY` and `GRADIENT_NOT_RAMPED_DOWN`.

```{figure} ../generated/figures/continuity_seam.png
A boundary step below the one-raster limit passes; a larger step is reported.
The dashed line marks the block boundary and the red arrow marks $\Delta G$.
```

### Peripheral nerve stimulation

A switching gradient induces an electric field that can stimulate peripheral
nerves. For a rectangular stimulus of duration $\tau$, the chronaxie model
gives the threshold

$$
S(\tau) = S_{\mathrm{rh}}\left(1 + \frac{c}{\tau}\right),
$$

with rheobase $S_{\mathrm{rh}}$ and chronaxie $c$:[^irnich] the shorter the
ramp, the larger the slew rate it takes to stimulate.
{class}`~pypulseqpp.safety.ChronaxieModel` shares one chronaxie between the
axes and takes the rheobase and the geometry factor `alpha` for all three or
per physical axis. A SAFE model, read from a Siemens `.asc` hardware
description by {func}`~pypulseqpp.safety.read_safe_model`, has three
exponential terms per axis.

```{figure} ../generated/figures/strength_duration.png
Slew rate at unity response for a demonstration chronaxie model and the
PyPulseq SAFE example hardware description.
```

Each physical axis's slew rate drives that axis's response, and the axes
combine as $R(t) = \sqrt{R_x^2 + R_y^2 + R_z^2}$; the check fails where $R$
exceeds 1, or the fraction of it you allow. The response depends on the slew
history before each instant, so the check runs over the whole sequence from
rest. With the same coefficients on every axis, a rotation leaves $R$
unchanged; with coefficients per axis, the prescription changes it.

```{figure} ../generated/figures/pns_response.png
Axis responses, combined response, threshold and reported peak for one EPI
acquisition under a synthetic chronaxie model, as returned by
`check_pns(..., trace=True)`.
```

### Mechanical resonance

The Lorentz force on a gradient coil is proportional to its current, and the
coil assembly has narrow mechanical resonances. Gradient drive held at one of
them produces acoustic output and mechanical stress far above other
frequencies,[^hedeen] so vendors forbid frequency bands, each with a tolerance
on the amplitude it may carry.

{func}`~pypulseqpp.safety.check_mech_resonance` slides a window of
`window_width` (40 ms by default) along the physical-axis waveforms, half a
window at a time. Each window is mean-subtracted, Hann-tapered, zero-padded and
transformed, and scaled so that a sustained sinusoid of amplitude $A$ reads
$A$ in mT/m. The window length is the time over which drive counts as
sustained. A band is violated by each window whose largest amplitude inside it
exceeds the band's tolerance, or `min_threshold` (10 mT/m) for a band without
one, on any axis it applies to. The report keeps each band's worst window
whether it violates or not.

A train of readouts of alternating polarity at echo spacing $\Delta t$ has its
fundamental at $f = 1/(2\,\Delta t)$, with an amplitude between $8/\pi^2$ and
$4/\pi$ of the plateau, and harmonics at odd multiples. For an echo-planar
readout, the echo spacing is what a forbidden band constrains.

```{figure} ../generated/figures/gradient_spectra.png
The readout-axis spectrum of a 40 ms window at the middle of two 64 x 64
sequences. The spoiled gradient echo spreads its power below a few hundred
hertz; the single-shot echo-planar train concentrates it in one line at the
reciprocal of twice its echo spacing.
```

The criterion is applied to the commanded waveform. It does not model the
coil's transfer function or its acoustic output.

### Sound pressure

{func}`~pypulseqpp.safety.check_spl` takes the repetition of the sequence with
the most gradient energy, filters it through the coil's acoustic transfer
function on each physical axis as a periodic waveform, and reads the peak level
in dB and the A-weighted average in dB(A) of the steady state that repetition
reaches when played back to back. The limits default to 140 dB and 99 dB(A),
the values of IEC 60601-2-33. The transfer function is measured on the coil
and read by {func}`~pypulseqpp.safety.read_acoustic_response`.

### SAR from virtual observation points

For a transmit array driven by a vector $\mathbf{v}$ of channel phasors, local
SAR at a position is $\mathbf{v}^{\mathsf H} Q \mathbf{v}$, with one matrix $Q$
per position from an electromagnetic simulation on a body model. Virtual
observation points compress these matrices into a few, $\{Q_k\}$, whose
largest value bounds the largest over the body model.[^eichfelder]
{class}`~pypulseqpp.safety.VopModel` holds them, with optional global matrices,
one per body model, and the file's `safety_factor` $M \geq 1$, which
multiplies local SAR for what the model does not carry.

Channel $c$ is driven with $v_c(t) = d_c\,s_c\,b_c(t)$: the RF waveform in Hz,
the block's RF shim (or `default_shim` for a single-channel pulse, played on
every channel), and `drive_per_hz`, the transmit chain's conversion from Hz to
the VOPs' drive unit. In each averaging window $W$ of duration $T_W$,

$$
\mathrm{SAR}_{\mathrm{local}}(W) = \frac{M}{T_W} \max_k \int_W
\mathbf{v}(t)^{\mathsf H} Q_k\, \mathbf{v}(t)\,\mathrm{d}t,
$$

and global SAR is the same integral with the global matrix, the largest over
the body models. The windows are the repetitions
{meth}`~pypulseqpp.Sequence.repetition` detects, or the whole sequence when it
does not repeat. The limits default to 10 W/kg local and 3.2 W/kg global, the
IEC 60601-2-33 normal-mode head values. The check does not average over a
regulatory interval such as six minutes.

`drive_per_hz` is a calibration you rarely have. With `reference`, the check
also compares the sequence with a pulse the scanner knows, such as a hard pulse
in the coil's circularly polarised mode, played under the same model and
drive. The scanner holds that pulse at the global limit $L_G$ at its shortest
repetition time, which fixes the drive scale. The report gives

$$
r_{\mathrm{local}} = \max_W \frac{\mathrm{SAR}_{\mathrm{local}}(W)}{\min_b \mathrm{SAR}^{\mathrm{ref}}_{G_b}},
\qquad
r_{\mathrm{global}} = \max_{W,b} \frac{\mathrm{SAR}_{G_b}(W)}{\mathrm{SAR}^{\mathrm{ref}}_{G_b}},
$$

as `local_to_head` and `global_sar_ratio`, with $G_b$ the global matrix of body
model $b$, and the same ratios of energy as `local_to_head_energy` and
`global_energy_ratio`. At that drive scale a window's local SAR is at most
$r_{\mathrm{local}} L_G$. The scale of `drive_per_hz` and of the matrices
cancels in both ratios; the relative channel gains and the safety factor do
not.

{func}`~pypulseqpp.safety.example_vops` is a synthetic eight-channel model of a
loop array around a uniform cylinder. Its numbers say nothing about any coil or
any subject.

## See it run

- {doc}`../generated/gallery/01-course/07_hardware_and_safety_checks`: every
  check on the radial MPRAGE, with its PNS trace and gradient spectrum.
- {doc}`../api/safety`: the functions and their reports.

## References

[^irnich]: Irnich W, Schmitt F. Magnetostimulation in MRI. *Magnetic Resonance in Medicine*. 1995;33(5):619–623. [doi:10.1002/mrm.1910330418](https://doi.org/10.1002/mrm.1910330418).

[^hedeen]: Hedeen RA, Edelstein WA. Characterization and prediction of gradient acoustic noise in MR imagers. *Magnetic Resonance in Medicine*. 1997;37(1):7–10. [doi:10.1002/mrm.1910370103](https://doi.org/10.1002/mrm.1910370103).

[^eichfelder]: Eichfelder G, Gebhardt M. Local specific absorption rate control for parallel transmission by virtual observation points. *Magnetic Resonance in Medicine*. 2011;66(5):1468–1476. [doi:10.1002/mrm.22927](https://doi.org/10.1002/mrm.22927).
