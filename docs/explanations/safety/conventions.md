# Check conventions

```{admonition} TL;DR
:class: tldr

- A `.seq` file records the system limits the sequence was designed against,
  not the system it will be played on. The two differ when the file is used at
  another site, when a site derates its gradient limits, or when a raster is
  finer in the design script than on the amplifier.
- The gradient-derived checks evaluate the gradient axes after each block's
  `ROTATIONS` extension, which are the logical axes of a design; the SAR check
  evaluates the RF waveforms and RF shims. The physical axes also need the
  prescription rotation: it is passed as `rotation` to
  {func}`~pypulseqpp.safety.check_pns` and
  {func}`~pypulseqpp.safety.check_mech_resonance`, and applied with
  {class}`~pypulseqpp.TransformFOV` before the other gradient checks.
- Every check returns a boolean verdict and a report, whether or not it passes.
  Gradient amplitude, slew-rate and continuity reports are in Hz/m and Hz/m/s,
  mechanical-resonance amplitudes in mT/m, PNS responses as fractions of the
  model threshold, and SAR in W/kg.
- The gradient amplitude, slew-rate and continuity checks read only the
  sequence and its system limits; the PNS, mechanical-resonance and SAR checks
  require site or coil data supplied as arguments. Timing is checked
  separately, by {meth}`~pypulseqpp.Sequence.check_timing`.
- Gradient amplitude, slew rate and continuity are evaluated pointwise, PNS
  over the whole sequence from rest, and mechanical resonance and SAR over
  windows. Any window exceeding its threshold or limit makes the verdict false.
```

The conventions shared by the checks in {mod}`pypulseqpp.safety`: which system
a check evaluates against, the frame it reads gradients in, the units of its
report, and the interval it evaluates over.

## Designed and played systems

A `.seq` file records the system limits the sequence was **designed** against,
in its `[DEFINITIONS]` and in the `system` its writer used. It does not record
the system it will be **played** on. The two are identical when the designer
entered the limits of the scanner the sequence will run on, and differ when the
file is used at another site, when a site derates its gradient limits, or when
a raster is finer in the design script than on the amplifier.

## Frames, rotations and reports

The gradient-derived checks — amplitude, slew rate, continuity, PNS and
mechanical resonance — evaluate the gradient axes after each block's
`ROTATIONS` extension has been applied: the **logical axes** of a design. The
SAR check is RF-derived: it evaluates the RF waveforms and RF shims, and no
gradient axis or rotation enters it.

The **physical gradient axes** are reached by a prescription rotation, composed
after each block's own. {func}`~pypulseqpp.safety.check_pns` and
{func}`~pypulseqpp.safety.check_mech_resonance` take one directly, as their
`rotation` argument. The gradient amplitude, slew-rate and continuity checks
read the rotations the sequence holds, so a prescription is applied to them by
transforming the sequence first with {class}`~pypulseqpp.TransformFOV`, which
composes it into the rotation extensions, and checking the result.

Every check returns a boolean verdict and a report. The report is returned
whether or not the check passes, and states the values found, where they were
found and the limit or threshold they were compared with. Gradient amplitude,
slew-rate and continuity reports are in Hz/m and Hz/m/s; `gamma` on the system
limits converts them to mT/m and T/m/s. Mechanical-resonance amplitudes are in
mT/m, PNS responses are fractions of the model threshold, and SAR is in W/kg.

## Required data

The first three checks read only the sequence and its system limits. The last
three require site or coil data that no sequence file contains — a nerve
model's coefficients, a gradient assembly's forbidden bands, a body model's
VOPs and the transmit chain's calibration — supplied as arguments.

Timing is checked separately, by
{meth}`~pypulseqpp.Sequence.check_timing`, which establishes that every event
time is addressable on the raster its event is played on and that the transmit
and receive dead times are respected. A sequence whose gradients are within
every limit can still be unplayable because one delay is off the raster.

## Evaluation interval

Gradient amplitude, slew rate and continuity are pointwise: every sample of the
scan is either within the limit or outside it, and the largest value determines
the verdict.

The nerve models are dynamic. Their response at one sample depends on the
preceding slew history, so the stimulation check is evaluated over the whole
sequence in one pass, starting from rest, rather than over a representative
repetition.

The mechanical-resonance and SAR checks are evaluated over **windows** rather
than samples. The resonance check slides a window of a stated length along the
sequence and transforms each; the SAR check averages RF energy over each
repetition detected from the block definitions. Any window exceeding its
threshold or limit makes the verdict false.
