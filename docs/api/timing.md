# Timing utilities

```{eval-rst}
.. currentmodule:: pypulseqpp
```

Pulseq events must land on the scanner's RF, gradient, ADC, or block raster.
These helpers choose legal ADC sampling and readout timing or quantize a time
to the requested raster.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   calc_adc_segments
   calc_adc_timing
   quantize_readout_timing
   round_to_raster
   ceil_to_raster
```
