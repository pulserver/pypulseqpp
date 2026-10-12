# API reference

Exact interfaces, units and defaults, grouped by the layer they belong to.
Concepts are in {doc}`../explanations/index` and complete workflows in the
{doc}`examples <../examples/index>`.

## Sequence representation

| Page | Module | Contents |
| --- | --- | --- |
| {doc}`sequence` | `pypulseqpp` | Sequence container, system limits and transforms of a finished sequence. |
| {doc}`io` | `pypulseqpp.io` | Reading and writing `.seq` files, text and binary. |
| {doc}`events` | `pypulseqpp` | ADC, delay, trigger, label, rotation and RF-shim events; block operations; PyPulseq interoperability. |

## Event design

| Page | Module | Contents |
| --- | --- | --- |
| {doc}`gradients` | `pypulseqpp` | Trapezoid, arbitrary and wave-encoding gradients; scaling, splitting and combining; gradient audio. |
| {doc}`rf` | `pypulseqpp` | RF pulses from basic to SLR, B1-selective, multidimensional and pTx designs; RF analysis and Bloch simulation. |
| {doc}`trajectories` | `pypulseqpp` | Radial, spiral and rosette trajectories and their gradient waveforms. |
| {doc}`timing` | `pypulseqpp` | ADC dwell times and durations on the ADC and gradient rasters; raster rounding. |

## Sequence construction

| Page | Module | Contents |
| --- | --- | --- |
| {doc}`sampling` | `pypulseqpp` | Cartesian masks, view ordering, EPI shot offsets, non-Cartesian orientations and RF schedules. |
| {doc}`modules` | `pypulseqpp.sequences` | Excitation, preparation and readout modules; non-Cartesian interleaves. |
| {doc}`sequence-functions` | `pypulseqpp.sequences` | Protocol parsing, labels, writing and timing of sequence functions. |
| {doc}`../sequences` | `pypulseqpp.sequences` | Every shipped sequence, by family, with its reference page. |

## Analysis and tools

| Page | Module | Contents |
| --- | --- | --- |
| {doc}`safety` | `pypulseqpp.safety` | Gradient, slew-rate, continuity, mechanical-resonance, PNS and SAR checks. |
| {doc}`plotting` | `pypulseqpp.plot` | SeqEyes view, publication diagram, k-space and RF-profile figures. |
| {doc}`cli` | `pypulseqpp.cli` | Running sequence functions and scripts from the shell. |

```{toctree}
:hidden:

sequence
io
events
gradients
rf
trajectories
timing
sampling
modules
sequence-functions
safety
plotting
cli
../sequences
```
