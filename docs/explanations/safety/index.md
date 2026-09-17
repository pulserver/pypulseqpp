# Gradient, PNS and SAR constraints

A `.seq` file records the system limits the sequence was **designed** against,
in its `[DEFINITIONS]` and in the `system` its writer used. It does not record
the system it will be **played** on. The two are identical when the designer
entered the limits of the scanner the sequence will run on, and differ when the
file is used at another site, when a site derates its gradient limits, or when
a raster is finer in the design script than on the amplifier.

The checks in {mod}`pypulseqpp.safety` evaluate a finished sequence against a
stated set of limits, which is the sequence's own
{class}`~pypulseqpp.Opts` or one passed to the check. They are **design-time
estimates**. They do not replace the scanner's own gate before download, nor
its hardware monitor during the scan, and they do not establish patient safety.

## Common conventions

Every check works on the **physical gradient axes**, after each block's
`ROTATIONS` extension has been applied. The amplifier driving an axis produces
the physical waveform, and a nerve or a mechanical mode responds to the physical
field, so a limit on either applies in the physical frame. A prescription
rotation, given as the `rotation` argument, is applied after the block's own, so
a sequence designed in a logical frame can be checked in the orientation it will
be prescribed at.

Every check returns the same pair: a boolean verdict and a report naming where
the extreme was found, with the extreme itself and the limit it was compared
with. The report is returned whether or not the check passes, so a passing
sequence's margin is readable from the same call.

Amplitudes in a report are in the file format's own units, Hz/m and Hz/m/s;
`gamma` on the system limits converts them to mT/m and T/m/s.

## Checks and their required data

| Check | Compares | Needs |
| --- | --- | --- |
| {doc}`gradient_amplitude` | the largest per-axis gradient amplitude | `max_grad` |
| {doc}`slew_rate` | the largest per-axis slew rate within a block | `max_slew`, the gradient raster |
| {doc}`gradient_continuity` | the amplitude step across each block boundary | `max_slew`, the gradient raster |
| {doc}`pns` | a nerve model's response to the slew of each axis | a coil response model |
| {doc}`mechanical_resonance` | the windowed gradient amplitude spectrum | a forbidden-band table |
| {doc}`sar` | window-averaged local and global SAR | virtual observation points and a drive calibration |

The first three checks read only the sequence and its system limits, so they
can always run. The last three additionally require site or coil data that no
sequence file contains — a nerve model's coefficients, a magnet's forbidden
bands, a subject's VOP model and the transmit chain's calibration — and this
data must be supplied explicitly.

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

The mechanical-resonance and SAR checks are defined over a **window** rather
than over a sample. Both bound a sustained drive — the acoustic response to a
periodic gradient waveform, and the energy deposited per unit time — and
neither is determined by any single sample. The resonance check slides a window
of a stated length along the sequence; the SAR check averages over each
repetition the block definitions repeat with, and the largest window average
determines the verdict.

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
