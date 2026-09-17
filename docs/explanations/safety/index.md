# Safety and hardware checks

A `.seq` file carries the system limits the sequence was **designed** against,
in its `[DEFINITIONS]` and in the `system` its writer used. It does not carry
the system it will be **played** on. The two coincide when the designer typed
in the numbers of the scanner in front of them, and part company when the file
travels, when a site derates its gradient limits, or when a raster is finer in
the script than on the amplifier.

The checks in {mod}`pypulseqpp.safety` evaluate a finished sequence against a
stated set of limits, which is the sequence's own
{class}`~pypulseqpp.Opts` or one passed to the check. They are **design-time
estimates**. They do not replace the scanner's own gate before download, nor
its hardware monitor during the scan, and they do not establish patient safety.

## What the checks share

Every check works on the **physical gradient axes**, after each block's
`ROTATIONS` extension has been applied, because it is the amplifier driving an
axis that has to produce the waveform and the nerve or the mechanical mode that
responds to it. A prescription rotation, given as the `rotation` argument, is
applied after the block's own, so a sequence designed in a logical frame can be
checked in the orientation it will be prescribed at.

Every check returns the same pair: a boolean verdict and a report naming where
the extreme was found, with the extreme itself and the limit it was compared
with. A report is returned whether or not the check passes, so the margin of a
sequence that passes is readable from the same call that would have refused it.

Amplitudes in a report are in the file format's own units, Hz/m and Hz/m/s;
`gamma` on the system limits converts them to mT/m and T/m/s.

## The six checks

| Check | Compares | Needs |
| --- | --- | --- |
| {doc}`gradient_amplitude` | the largest per-axis gradient amplitude | `max_grad` |
| {doc}`slew_rate` | the largest per-axis slew rate within a block | `max_slew`, the gradient raster |
| {doc}`gradient_continuity` | the amplitude step across each block boundary | `max_slew`, the gradient raster |
| {doc}`pns` | a nerve model's response to the slew of each axis | a coil response model |
| {doc}`mechanical_resonance` | the windowed gradient amplitude spectrum | a forbidden-band table |
| {doc}`sar` | window-averaged local and global SAR | virtual observation points and a drive calibration |

The first three need nothing the sequence and the system limits do not already
carry, so they can always run. The last three need site or coil data no
sequence file carries — a nerve model's coefficients, a magnet's forbidden
bands, a subject's VOP model and the transmit chain's calibration — and are
supplied explicitly rather than guessed.

Timing is a separate question, and a separate call:
{meth}`~pypulseqpp.Sequence.check_timing` establishes that every event time is
addressable on the raster its event is played on, and that the transmit and
receive dead times are respected. A sequence whose gradients are within every
limit can still be unplayable because one delay is off the raster.

## Which repetition a check is defined over

Gradient amplitude, slew rate and continuity are pointwise: every sample of the
scan is either within the limit or is not, and the worst sample decides.

Peripheral nerve stimulation carries the nerve model's memory, so it is
evaluated over the whole sequence in one pass, from rest, rather than over a
representative repetition.

The mechanical-resonance and SAR checks are defined over a **window** rather
than over a sample, because what they bound is a sustained drive: the acoustic
response to a periodic gradient waveform and the energy deposited per unit
time. The resonance check slides a window of a stated length along the
sequence; the SAR check averages over each repetition the block definitions
repeat with, and the worst repetition decides.

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
