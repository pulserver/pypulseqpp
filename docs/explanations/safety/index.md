# Gradient, PNS and SAR constraints

The checks in {mod}`pypulseqpp.safety` evaluate a finished sequence against
stated limits and models. They are **design-time estimates**. They do not
replace the scanner's own gate before download, nor its hardware monitor during
the scan, and they do not establish patient safety.

## Checks

| Check | Compares | Needs |
| --- | --- | --- |
| {doc}`gradient_amplitude` | the largest per-axis gradient amplitude | `max_grad` from the system limits |
| {doc}`slew_rate` | the largest per-axis slew rate within a block | `max_slew` and the gradient raster from the system limits |
| {doc}`gradient_continuity` | the amplitude step across each block boundary | `max_slew` and the gradient raster from the system limits |
| {doc}`pns` | a nerve model's response to the slew of each axis | a SAFE or chronaxie model |
| {doc}`mechanical_resonance` | the windowed gradient amplitude spectrum | a forbidden-band table |
| {doc}`sar` | window-averaged local and global SAR | virtual observation points, a drive calibration and SAR limits |
The first five checks are gradient-derived and evaluate the physical gradient
axes; the SAR check is RF-derived and evaluates the RF waveforms and shims. The
frames, report units and evaluation intervals common to the checks are in
{doc}`conventions`.

```{toctree}
:hidden:
:maxdepth: 1

gradient_amplitude
slew_rate
gradient_continuity
pns
mechanical_resonance
sar
conventions
```
