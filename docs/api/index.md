# API reference

Conceptual background is in {doc}`../explanations/index` and complete workflows
in the {doc}`examples <../examples/index>`.

| Page | Module | What it documents |
| --- | --- | --- |
| {doc}`sequence` | `pypulseqpp` | The sequence container, the system limits a sequence is designed under, and the transforms applied to a finished one. |
| {doc}`events` | `pypulseqpp` | ADC, delay, trigger, label, rotation and RF-shim events, the operations that act on a block, and the conversions an upstream PyPulseq function is called through. |
| {doc}`gradients` | `pypulseqpp` | Gradient events on one logical axis, from trapezoids and phase encodes to arbitrary and wave-encoding waveforms, and the operations that scale, split and combine them. |
| {doc}`rf` | `pypulseqpp` | RF events, from the basic factories to SLR, slice-encoding, B1-selective, multidimensional and parallel-transmit designs, and the analysis and Bloch simulation of what they produce. |
| {doc}`trajectories` | `pypulseqpp` | Radial, spiral and rosette k-space trajectories, and the gradient waveforms that trace them. |
| {doc}`timing` | `pypulseqpp` | ADC dwell times and acquisition durations solved against both the ADC and the gradient raster, and the quantization of a time to a raster. |
| {doc}`modules` | `pypulseqpp.sequences` | The excitation, preparation and readout modules, and the non-Cartesian interleaves a readout plays. |
| {doc}`apps` | `pypulseqpp.sequences` | {class}`~pypulseqpp.sequences.SequenceApp`, which holds a prescription, the sampling order derived from it, and the kernel one repetition plays. |
| {doc}`safety` | `pypulseqpp.safety` | The gradient amplitude, slew-rate, continuity, mechanical-resonance, nerve-stimulation and SAR checks, and the readings they return beside a verdict. |
| {doc}`plotting` | `pypulseqpp.plot` | The SeqEyes view, the publication diagram, and the k-space and RF-profile figures. |
| {doc}`cli` | `pypulseqpp.cli` | Running a sequence application from the shell, and writing the file it produces. |
| {doc}`../sequences` | `pypulseqpp.sequences` | Every complete sequence the package ships, grouped by family, each with its own reference page. |

```{toctree}
:hidden:

sequence
events
gradients
rf
trajectories
timing
modules
apps
safety
plotting
cli
../sequences
```
