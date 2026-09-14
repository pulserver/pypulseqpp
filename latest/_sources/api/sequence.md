# Sequence and system

`pypulseqpp`: the container a sequence is written into, the scanner limits it is
designed under, and the transforms and reports applied to it.

```{eval-rst}
.. currentmodule:: pypulseqpp
```

## Sequence container

{class}`Sequence` holds the event libraries, block table, definitions and
scanner limits of one Pulseq sequence. It authors, reads and writes the
sequence, expands its waveforms and k-space, inspects its structure and checks
its timing over the compiled core.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:
   :template: autosummary/sequence.rst

   Sequence
```

## Field-of-view transforms

{class}`TransformFOV` translates, rotates and scales the logical frame of the
events a sequence plays; scaling the gradient amplitude scales the FOV
inversely.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   TransformFOV
```

## Scanner limits

{class}`Opts` holds the gradient, slew, RF and ADC limits and rasters.
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

## Timing reports

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   check_timing
   print_error_report
```
