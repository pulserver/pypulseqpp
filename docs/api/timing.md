# Timing and rasterization

`pypulseqpp`: ADC dwell and acquisition duration compatible with both the ADC
and gradient rasters, and times quantized to a raster. Pulseq quantizes event
timing to the RF, gradient, ADC and block duration rasters.

Receiver bandwidth is the full sampling bandwidth, `1 / dwell` in Hz;
bandwidth per pixel is `1 / (num_samples * dwell)`.

```{eval-rst}
.. currentmodule:: pypulseqpp
```

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
