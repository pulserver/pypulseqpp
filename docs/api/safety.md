# Hardware safety

```{eval-rst}
.. currentmodule:: pypulseqpp.safety
```

Safety checks inspect what the complete sequence asks of the scanner. They
apply block rotations, evaluate all three physical axes together, and may use
either the sequence's own limits or a different {class}`pypulseqpp.Opts`.

{func}`check_max_grad` checks the peak played gradient amplitude;
{func}`check_max_slew` checks slew within blocks; and
{func}`check_grad_continuity` checks the transitions between blocks and the
end of the sequence.

{func}`check_mech_resonance` slides a Hann-tapered window along each physical
axis and compares the amplitude spectrum of every window with the forbidden
bands that guard that axis. Bands come as {class}`ForbiddenBand` entries or
from a vendor table through {func}`read_forbidden_bands`. The transform is
MKL's when the optional `mkl` extra is installed, and the compiled-in pocketfft
otherwise.

These checks estimate what the gradients ask of the hardware. They are not a
complete scanner or patient-safety assessment, and they do not evaluate SAR or
PNS.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   check_max_grad
   check_max_slew
   check_grad_continuity
   check_mech_resonance
   read_forbidden_bands
   ForbiddenBand
```
