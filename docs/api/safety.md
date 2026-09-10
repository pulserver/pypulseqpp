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

These are gradient-limit checks, not a complete scanner or patient-safety
assessment. They do not evaluate SAR, PNS, or mechanical resonance.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   check_max_grad
   check_max_slew
   check_grad_continuity
```
