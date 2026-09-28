# Gradient, PNS and SAR checks

Checks of a complete sequence against the gradient hardware limits, the
forbidden gradient bands of a gradient coil, a peripheral-nerve-stimulation
model and a VOP SAR model. They are estimates, not a complete scanner or
patient-safety assessment. {doc}`../explanations/safety/index` covers what each
one computes and the criterion it applies.

```{eval-rst}
.. currentmodule:: pypulseqpp.safety
```

## Gradient limits

Gradient-derived: the checks evaluate the three gradient axes after each
block's rotation, against the limits of `system` or of the sequence's own
{class}`pypulseqpp.Opts`. They take no prescription rotation, so these are the
physical axes only once {class}`~pypulseqpp.TransformFOV` has composed one into
the rotation extensions. Only the largest per-axis peak is compared with the
limit; the simultaneous vector magnitude is reported but not compared.

| Object | Input | Returns | Purpose |
| --- | --- | --- | --- |
| {obj}`~pypulseqpp.safety.check_max_grad` | Sequence, system limits (`max_grad`) | `(is_ok, report)`; peaks in Hz/m | Peak gradient amplitude. |
| {obj}`~pypulseqpp.safety.check_max_slew` | Sequence, system limits (`max_slew`, gradient raster) | `(is_ok, report)`; peaks in Hz/m/s | Within-block slew rate. |
| {obj}`~pypulseqpp.safety.check_grad_continuity` | Sequence, system limits (`max_slew`, gradient raster) | `(is_ok, report)`; steps in Hz/m and Hz/m/s | Amplitude continuity across block boundaries and at the end. |

## Mechanical resonance

Gradient-derived: the check evaluates the physical gradient axes after the block
rotations and the `rotation` argument, sampled on the sequence's gradient
raster. The forbidden bands are an argument; only `gamma` is read from
`system`.

| Object | Input | Returns | Purpose |
| --- | --- | --- | --- |
| {obj}`~pypulseqpp.safety.check_mech_resonance` | Sequence, forbidden bands, window width (s), `rotation` | `(is_ok, report)`; amplitudes in mT/m | Windowed gradient spectrum against forbidden bands. |
| {obj}`~pypulseqpp.safety.mech_resonance_spectrum` | Sequence, window index, window width (s), `rotation` | `frequency` (Hz) and `(3, bins)` `amplitude` (mT/m) | Gradient amplitude spectrum of one window. |
| {obj}`~pypulseqpp.safety.read_forbidden_bands` | Siemens `.asc` path | List of `ForbiddenBand` | Forbidden bands from a hardware description. |
| {obj}`~pypulseqpp.safety.ForbiddenBand` | Axis, `f_min`, `f_max` (Hz), tolerance (mT/m) | Band record | Forbidden band on one or every physical axis. |

## Nerve stimulation

Gradient-derived: the check evaluates the slew of each physical gradient axis
after the block rotations and the `rotation` argument, sampled on the
sequence's gradient raster. The nerve model is an argument; only `gamma` is
read from `system`.

| Object | Input | Returns | Purpose |
| --- | --- | --- | --- |
| {obj}`~pypulseqpp.safety.check_pns` | Sequence, SAFE or chronaxie model, `rotation`, `trace` | `(is_ok, report)`; responses as fractions of threshold | Peripheral nerve stimulation estimate. |
| {obj}`~pypulseqpp.safety.read_safe_model` | Siemens `.asc` path | SAFE model, per-axis coefficients | SAFE model from a hardware description. |
| {obj}`~pypulseqpp.safety.ChronaxieModel` | Chronaxie (s), rheobase (T/m/s), `alpha` | Model for `check_pns` | Rheobase–chronaxie nerve model, all axes alike. |

## SAR

RF-derived: the check evaluates the RF waveforms and RF shims, and reads no
gradient, rotation or `system`. It takes the VOPs as a {class}`VopModel`, the
channel drive calibration `drive_per_hz`, and `local_limit` and
`global_limit` in W/kg.

| Object | Input | Returns | Purpose |
| --- | --- | --- | --- |
| {obj}`~pypulseqpp.safety.check_sar` | Sequence, `VopModel`, `drive_per_hz`, limits (W/kg) | `(is_ok, report)`; window-averaged SAR in W/kg | Local and global SAR per window. |
| {obj}`~pypulseqpp.safety.read_vops` | `.mat` or `.npz` path | `VopModel` | VOPs and global SAR matrix from a file. |
| {obj}`~pypulseqpp.safety.example_vops` | Channel count | `model`, `drive_per_hz` (V/Hz), `cp_shim` | Synthetic VOP model, for demonstrations and tests only. |
| {obj}`~pypulseqpp.safety.VopModel` | VOPs `(N, Nc, Nc)`, optional global matrix `(Nc, Nc)` | Model for `check_sar`, W/kg per unit drive squared | Virtual observation points. |
