# Hardware safety

`pypulseqpp.safety`: what a complete sequence asks of the scanner and of the
subject. The checks apply block rotations, judge the three physical axes
together, and use the sequence's own limits or another
{class}`pypulseqpp.Opts`. They are estimates, not a complete scanner or
patient-safety assessment.

```{eval-rst}
.. currentmodule:: pypulseqpp.safety
```

## Gradient limits

{func}`check_max_grad` checks the peak played amplitude,
{func}`check_max_slew` the slew within blocks, and
{func}`check_grad_continuity` the transitions between blocks and the end of
the sequence.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   check_max_grad
   check_max_slew
   check_grad_continuity
```

## Mechanical resonance

{func}`check_mech_resonance` slides a Hann-tapered window along each physical
axis and compares every window's amplitude spectrum with the forbidden bands
guarding that axis. Bands come as {class}`ForbiddenBand` entries or from a
vendor table through {func}`read_forbidden_bands`. The transform is MKL's
when the optional `mkl` extra is installed, and the compiled-in pocketfft
otherwise.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   check_mech_resonance
   read_forbidden_bands
   ForbiddenBand
```

## Nerve stimulation

{func}`check_pns` runs a nerve model over the slew of each physical axis and
compares the root-sum-square response with its threshold. The model is either
SAFE, as upstream PyPulseq computes it, from a description shaped like
upstream's `safe_example_hw()` or a Siemens `.asc` file read by
{func}`read_safe_model`; or a rheobase-chronaxie {class}`ChronaxieModel`.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   check_pns
   read_safe_model
   ChronaxieModel
```

## SAR

{func}`check_sar` averages SAR from virtual observation points over each real
repetition, and compares the worst repetition's largest VOP SAR with a local
limit and, given a global matrix, its global SAR with a global limit. VOPs come
as a {class}`VopModel`, read from a `.mat` or `.npz` file by
{func}`read_vops`, or from {func}`example_vops`, a synthetic model for
demonstration only. RF power in Pulseq's Hz units is
{func}`pypulseqpp.calc_rf_power` for one pulse and `Sequence.calc_rf_power`
for a sequence.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   check_sar
   read_vops
   example_vops
   VopModel
```
