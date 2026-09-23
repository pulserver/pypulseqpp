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

## Field-of-view transforms

| Object | Input | Returns | Purpose |
| --- | --- | --- | --- |
| {obj}`~pypulseqpp.TransformFOV` | Rotation, translation (m), per-axis gradient scale, or 4-by-4 transform | `TransformFOV`; `apply_to_sequence` returns the transformed sequence | Prescription geometry in the logical frame. |

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
