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

| Object | Description |
| --- | --- |
| {obj}`~pypulseqpp.safety.check_max_grad` | Check the peak gradient amplitude against ``max_grad``. |
| {obj}`~pypulseqpp.safety.check_max_slew` | Check the within-block slew rate against ``max_slew``. |
| {obj}`~pypulseqpp.safety.check_grad_continuity` | Check gradient amplitude continuity across block boundaries. |

## Mechanical resonance

{func}`check_mech_resonance` slides a Hann-tapered window along each physical
axis and compares every window's amplitude spectrum with the forbidden bands
guarding that axis. Bands come as {class}`ForbiddenBand` entries or from a
vendor table through {func}`read_forbidden_bands`. The transform is MKL's
when the optional `mkl` extra is installed, and the compiled-in pocketfft
otherwise.

| Object | Description |
| --- | --- |
| {obj}`~pypulseqpp.safety.check_mech_resonance` | Check the gradient amplitude spectrum against forbidden gradient bands. |
| {obj}`~pypulseqpp.safety.read_forbidden_bands` | Read forbidden bands from a vendor table. |
| {obj}`~pypulseqpp.safety.ForbiddenBand` | A forbidden gradient band: a frequency range on one physical axis. |

## Nerve stimulation

{func}`check_pns` runs a nerve model over the slew of each physical axis and
compares the root-sum-square response with its threshold. The model is either
SAFE, as upstream PyPulseq computes it, from a description shaped like
upstream's `safe_example_hw()` or a Siemens `.asc` file read by
{func}`read_safe_model`; or a rheobase-chronaxie {class}`ChronaxieModel`.

| Object | Description |
| --- | --- |
| {obj}`~pypulseqpp.safety.check_pns` | Check peripheral nerve stimulation against a SAFE or chronaxie model. |
| {obj}`~pypulseqpp.safety.read_safe_model` | Read a SAFE nerve model from a Siemens ``.asc`` hardware description. |
| {obj}`~pypulseqpp.safety.ChronaxieModel` | Rheobase-chronaxie nerve model, one coefficient set for every physical axis. |

## SAR

{func}`check_sar` averages SAR from virtual observation points over each real
repetition, and compares the worst repetition's largest VOP SAR with a local
limit and, given a global matrix, its global SAR with a global limit. VOPs come
as a {class}`VopModel`, read from a `.mat` or `.npz` file by
{func}`read_vops`, or from {func}`example_vops`, a synthetic model for
demonstration only. RF power in Pulseq's Hz units is
{func}`pypulseqpp.calc_rf_power` for one pulse and `Sequence.calc_rf_power`
for a sequence.

| Object | Description |
| --- | --- |
| {obj}`~pypulseqpp.safety.check_sar` | Check window-averaged local and global SAR against their limits. |
| {obj}`~pypulseqpp.safety.read_vops` | Read VOPs and a global SAR matrix from a ``.mat`` or ``.npz`` file. |
| {obj}`~pypulseqpp.safety.example_vops` | Build a synthetic VOP model, for demonstrations and tests only. |
| {obj}`~pypulseqpp.safety.VopModel` | Virtual observation points, and an optional global SAR matrix. |
