# Sequence and system

```{eval-rst}
.. currentmodule:: pypulseqpp
```

{class}`Sequence` holds the event libraries, block table, definitions, and
scanner limits for one Pulseq sequence. It supplies PyPulseq-compatible
authoring methods and performs reading, writing, waveform expansion, k-space
calculation, structural inspection, and timing checks over the compiled core.
{meth}`Sequence.paper_plot` draws one repetition as a publication-style diagram,
with the others underneath.

{class}`Opts` describes the scanner limits on gradient, slew, RF, and ADC
rasters. {func}`apply_system_derates` and {func}`cap_system` return adjusted
copies, leaving the limits supplied by the caller unchanged.

{class}`TransformFOV` applies logical-frame translation, rotation, and gradient
amplitude scaling to the events a sequence plays. Amplitude scaling changes
FOV size inversely.

## Sequence container

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   Sequence
   TransformFOV
```

## Scanner limits

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   Opts
   default_system
   apply_system_derates
   cap_system
```

```{eval-rst}
.. autodata:: pypulseqpp.MAX_GRAD_DERATE
.. autodata:: pypulseqpp.MAX_SLEW_DERATE
```

## Reports

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   print_error_report
```
