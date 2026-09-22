# Sequence and system limits

The sequence container, the system limits a sequence is designed under, and
the geometry transforms applied to a finished one.

```{eval-rst}
.. currentmodule:: pypulseqpp
```

## Sequence container

{doc}`../explanations/pulseq/index` describes the format these objects represent.

{class}`Sequence` holds the event libraries, block table, definitions and
system limits of one Pulseq sequence. It builds, reads and writes the sequence,
expands its gradient and RF waveforms and its k-space trajectory, inspects its
structure and checks its timing, over the compiled core.

| Object | Description |
| --- | --- |
| {obj}`~pypulseqpp.Sequence` | A Pulseq sequence containing events, blocks and definitions. |

## Field-of-view transforms

{class}`TransformFOV` applies a prescription to an existing sequence:
per-axis gradient amplitude scaling in the logical frame, a rotation composed
after each block's own rotation, and a translation in logical metres. Field of
view scales inversely with gradient amplitude, so halving an axis's amplitude
doubles the field of view along it.

| Object | Description |
| --- | --- |
| {obj}`~pypulseqpp.TransformFOV` | Geometry prescription applied to an existing sequence. |

## System limits

{class}`Opts` holds the gradient amplitude, slew-rate, RF and ADC limits
together with the RF, gradient, ADC and block duration rasters.
{func}`apply_system_derates` and {func}`cap_system` return adjusted copies and
leave the caller's limits unchanged. `MAX_GRAD_DERATE` and `MAX_SLEW_DERATE`
are the fractions {func}`apply_system_derates` applies by default, so that a
design solved one axis at a time stays within the limit when more than one axis
plays.

| Object | Description |
| --- | --- |
| {obj}`~pypulseqpp.Opts` | PyPulseq system limits with shared raster defaults. |
| {obj}`~pypulseqpp.default_system` | Return ``system``, or the shared default system when it is ``None``. |
| {obj}`~pypulseqpp.apply_system_derates` | Return a copy with gradient and slew limits scaled from their base values. |
| {obj}`~pypulseqpp.cap_system` | Return a copy with gradient and slew limits lowered to the specified ceilings. |
| {obj}`~pypulseqpp.MAX_GRAD_DERATE` | Fraction of the gradient amplitude limit a designed waveform may reach. |
| {obj}`~pypulseqpp.MAX_SLEW_DERATE` | Fraction of the slew-rate limit a designed waveform may reach. |
