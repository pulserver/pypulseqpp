# Timing and rasterization

ADC dwell times and acquisition durations compatible with both the ADC and the
gradient raster, and the quantization of a time to a raster. Times are in s.
Receiver bandwidth is `1 / dwell` in Hz; bandwidth per pixel is
`1 / (num_samples * dwell)`. {doc}`../explanations/pulseq/timing-and-rasterization`
describes the four rasters and the coupling between the ADC and gradient
rasters.

```{eval-rst}
.. currentmodule:: pypulseqpp
```

## ADC timing

| Object | Input | Returns | Purpose |
| --- | --- | --- | --- |
| {obj}`~pypulseqpp.calc_adc_segments` | Samples, dwell, system limits, `shorten` or `lengthen` | Number of segments; samples per segment | Equal-sample ADC segmentation. |
| {obj}`~pypulseqpp.calc_adc_timing` | Samples, target dwell, gradient and ADC rasters | Dwell (s); acquisition duration (s) | Dwell on the ADC raster, duration on the gradient raster. |
| {obj}`~pypulseqpp.quantize_readout_timing` | Samples, receiver bandwidth (Hz), rasters, minimum flat time | Dwell (s); flat time (s) | Dwell from a requested receiver bandwidth. |

## Raster quantization

| Object | Input | Returns | Purpose |
| --- | --- | --- | --- |
| {obj}`~pypulseqpp.round_to_raster` | Time (s), raster (s) | Nearest raster multiple (s), ties to even | Round to nearest. |
| {obj}`~pypulseqpp.ceil_to_raster` | Time (s), raster (s) | Next raster multiple (s), 1e-10 relative tolerance | Round up. |
