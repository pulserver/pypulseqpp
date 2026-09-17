# Gradient, PNS and SAR checks

`pypulseqpp.safety`: checks of a complete sequence against gradient hardware
limits, vendor forbidden gradient bands, a peripheral-nerve-stimulation model
and a VOP SAR model. Every check applies each block's rotation and evaluates
the three physical gradient axes, and takes its limits from the sequence's own
{class}`pypulseqpp.Opts` or from one passed to it. They are estimates, not a
complete scanner or patient-safety assessment.
{doc}`../explanations/safety/index` covers what each one computes and the
criterion it applies.

```{eval-rst}
.. currentmodule:: pypulseqpp.safety
```

## Gradient limits

{func}`check_max_grad` checks the peak gradient amplitude,
{func}`check_max_slew` the slew rate within blocks, and
{func}`check_grad_continuity` the gradient amplitude across block boundaries
and at the end of the sequence.

The amplitude and slew-rate checks compare the largest **per-axis** peak with
`max_grad` and `max_slew`. They also report the largest simultaneous vector
magnitude over the three physical axes, but do not compare it with a limit;
that magnitude is not the norm of independently attained axis peaks.

```{eval-rst}
.. autosummary::
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
   :nosignatures:

   check_sar
   read_vops
   example_vops
   VopModel
```
