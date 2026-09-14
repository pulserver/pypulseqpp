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

{func}`check_pns` runs a nerve model over the slew of each physical axis and
compares the root-sum-square response with its threshold. The model is either
SAFE, as upstream PyPulseq computes it, from a description shaped like
upstream's `safe_example_hw()` or a Siemens `.asc` file read by
{func}`read_safe_model`; or a rheobase-chronaxie {class}`ChronaxieModel`.

{func}`check_sar` averages SAR from virtual observation points over each real
repetition, and compares the worst repetition's largest VOP SAR with a local
limit and, given a global matrix, its global SAR with a global limit. VOPs come
as a {class}`VopModel`, read from a `.mat` or `.npz` file by {func}`read_vops`,
or from {func}`example_vops`, a synthetic model for demonstration only. RF power
in Pulseq's Hz units is {func}`pypulseqpp.calc_rf_power` for one pulse and
`Sequence.calc_rf_power` for a sequence.

These checks estimate what the gradients and RF ask of the hardware and of the
subject. They are not a complete scanner or patient-safety assessment.

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
   check_pns
   read_safe_model
   ChronaxieModel
   check_sar
   read_vops
   example_vops
   VopModel
```
