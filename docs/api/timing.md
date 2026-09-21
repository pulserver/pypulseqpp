# Timing and rasterization

`pypulseqpp`: ADC dwell and acquisition duration compatible with both the ADC
and gradient rasters, and times quantized to a raster. Pulseq quantizes event
timing to the RF, gradient, ADC and block duration rasters.

Receiver bandwidth is the full sampling bandwidth, `1 / dwell` in Hz;
bandwidth per pixel is `1 / (num_samples * dwell)`.

```{eval-rst}
.. currentmodule:: pypulseqpp
```

| Object | Description |
| --- | --- |
| {obj}`~pypulseqpp.calc_adc_segments` | Calculate splitting of the ADC in segments with equal samples. |
| {obj}`~pypulseqpp.calc_adc_timing` | Choose an ADC dwell whose acquisition duration is a multiple of the gradient raster. |
| {obj}`~pypulseqpp.quantize_readout_timing` | Choose an ADC dwell from a requested receiver bandwidth. |
| {obj}`~pypulseqpp.round_to_raster` | Round seconds to the nearest multiple of ``raster_s``, with ties to even. |
| {obj}`~pypulseqpp.ceil_to_raster` | Round seconds up to the next raster multiple, within a 1e-10 relative tolerance. |
