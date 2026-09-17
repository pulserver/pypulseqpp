# Sequence and system limits

`pypulseqpp`: the sequence container, the system limits a sequence is designed
under, and the geometry transforms applied to it.

```{eval-rst}
.. currentmodule:: pypulseqpp
```

## Sequence container

{doc}`../explanations/pulseq/index` describes the format these objects represent.

{class}`Sequence` holds the event libraries, block table, definitions and
system limits of one Pulseq sequence. It builds, reads and writes the sequence,
expands its gradient and RF waveforms and its k-space trajectory, inspects its
structure and checks its timing, over the compiled core.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:
   :template: autosummary/sequence.rst

   Sequence
```

## Field-of-view transforms

{class}`TransformFOV` applies a prescription to an existing sequence:
per-axis gradient amplitude scaling in the logical frame, a rotation composed
after each block's own rotation, and a translation in logical metres. Field of
view scales inversely with gradient amplitude, so halving an axis's amplitude
doubles the field of view along it.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   TransformFOV
```

## System limits

{class}`Opts` holds the gradient amplitude, slew-rate, RF and ADC limits
together with the RF, gradient, ADC and block duration rasters.
{func}`apply_system_derates` and {func}`cap_system` return adjusted copies and
leave the caller's limits unchanged.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   Opts
   default_system
   apply_system_derates
   cap_system
   MAX_GRAD_DERATE
   MAX_SLEW_DERATE
```
