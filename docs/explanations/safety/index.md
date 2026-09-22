# Gradient, PNS and SAR constraints

A `.seq` file records the system limits the sequence was **designed** against,
in its `[DEFINITIONS]` and in the `system` its writer used. It does not record
the system it will be **played** on. The two are identical when the designer
entered the limits of the scanner the sequence will run on, and differ when the
file is used at another site, when a site derates its gradient limits, or when
a raster is finer in the design script than on the amplifier.

The checks in {mod}`pypulseqpp.safety` evaluate a finished sequence against
stated limits and models. They are **design-time estimates**. They do not
replace the scanner's own gate before download, nor its hardware monitor during
the scan, and they do not establish patient safety.

## Common conventions

The gradient-derived checks — amplitude, slew rate, continuity, PNS and
mechanical resonance — evaluate the **physical gradient axes**, after each
block's `ROTATIONS` extension has been applied. The SAR check is RF-derived: it
evaluates the RF waveforms and RF shims, and no gradient axis or rotation enters
it.

A prescription rotation is a further rotation, composed after each block's own.
{func}`~pypulseqpp.safety.check_pns` and
{func}`~pypulseqpp.safety.check_mech_resonance` take one directly, as their
`rotation` argument. The gradient amplitude, slew-rate and continuity checks
read the rotations the sequence holds, so a prescription is applied to them by
transforming the sequence first with {class}`~pypulseqpp.TransformFOV` and
checking the result.

Every check returns a boolean verdict and a report. The report is returned
whether or not the check passes, and states the values found, where they were
found and the limit or threshold they were compared with. Gradient amplitude,
slew-rate and continuity reports are in Hz/m and Hz/m/s; `gamma` on the system
limits converts them to mT/m and T/m/s. Mechanical-resonance amplitudes are in
mT/m, PNS responses are fractions of the model threshold, and SAR is in W/kg.

## Checks and their required data

| Check | Compares | Needs |
| --- | --- | --- |
| {doc}`gradient_amplitude` | the largest per-axis gradient amplitude | `max_grad` from the system limits |
| {doc}`slew_rate` | the largest per-axis slew rate within a block | `max_slew` and the gradient raster from the system limits |
| {doc}`gradient_continuity` | the amplitude step across each block boundary | `max_slew` and the gradient raster from the system limits |
| {doc}`pns` | a nerve model's response to the slew of each axis | a SAFE or chronaxie model |
| {doc}`mechanical_resonance` | the windowed gradient amplitude spectrum | a forbidden-band table |
| {doc}`sar` | window-averaged local and global SAR | virtual observation points, a drive calibration and SAR limits |

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

```{toctree}
:hidden:
:maxdepth: 1

gradient_amplitude
slew_rate
gradient_continuity
pns
mechanical_resonance
sar
```
