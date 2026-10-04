# Sequence and system limits

The sequence container, the system limits a sequence is designed under, and
the geometry transforms applied to a finished sequence. Gradient amplitude
limits are in Hz/m, slew-rate limits in Hz/m/s and rasters in s;
{doc}`../explanations/pulseq/index` describes the format these objects
represent.

```{eval-rst}
.. currentmodule:: pypulseqpp
```

## Sequence container

| Object | Input | Returns | Purpose |
| --- | --- | --- | --- |
| {obj}`~pypulseqpp.Sequence` | System limits | `Sequence` holding libraries, blocks, definitions, system limits | Block construction, file I/O, waveform, k-space and timing analysis. |

## RF instances

{meth}`~pypulseqpp.Sequence.rf_instances` returns the RF events a sequence
plays, one per block that contains one, and the distinct RF pulses they play.
A waveform is normalised to unit peak magnitude, with its peak held beside it
in Hz; sample times are in s from the start of the RF event and flip angles in
degrees.

| Object | Input | Returns | Purpose |
| --- | --- | --- | --- |
| {obj}`~pypulseqpp.RfInstances` | — | `definitions`; `definition` and `amplitude` per instance | The RF events of a sequence in block order, each with the pulse it plays and its amplitude relative to that pulse's base instance. |
| {obj}`~pypulseqpp.RfDefinition` | — | `waveform`, `peak_hz`, `time`, `delay`, `center`, `use`, `flip_deg` | One RF pulse at unit peak magnitude, with its timing, use and flip angle. |

## Field-of-view transforms

| Object | Input | Returns | Purpose |
| --- | --- | --- | --- |
| {obj}`~pypulseqpp.TransformFOV` | Rotation, translation (m), per-axis gradient scale, or 4-by-4 transform | `TransformFOV`; `apply_to_sequence` returns the transformed sequence | Prescription geometry: a rotation composed after each block's own, a translation along the channel or logical axes. |

## System limits

{func}`apply_system_derates` and {func}`cap_system` return copies and do not
modify their argument.

| Object | Input | Returns | Purpose |
| --- | --- | --- | --- |
| {obj}`~pypulseqpp.Opts` | Gradient and slew limits with units, rasters, dead times, `gamma`, `B0` | `Opts` holding limits in Hz/m, Hz/m/s and rasters in s | System limits and rasters. |
| {obj}`~pypulseqpp.default_system` | `Opts` or `None` | The argument, or the shared default `Opts` | Default-system resolution. |
| {obj}`~pypulseqpp.apply_system_derates` | `Opts`, gradient and slew fractions | `Opts` copy, limits scaled from base values | Derated design limits. |
| {obj}`~pypulseqpp.cap_system` | `Opts`, gradient and slew ceilings with their units | `Opts` copy, limits lowered to the ceilings | Capped design limits. |
| {obj}`~pypulseqpp.MAX_GRAD_DERATE` | — | `float` | Default `grad_derate` of `apply_system_derates`. |
| {obj}`~pypulseqpp.MAX_SLEW_DERATE` | — | `float` | Default `slew_derate` of `apply_system_derates`. |
