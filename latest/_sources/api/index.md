# API reference

Conceptual background is in {doc}`../explanations/index` and complete workflows
in the {doc}`examples <../examples>`.

| Page | Contents |
| --- | --- |
| {doc}`sequence` | `pypulseqpp`: the sequence container, system limits and field-of-view transforms |
| {doc}`events` | `pypulseqpp`: ADC, delay, trigger, label, rotation and RF-shim events, block and event operations, interoperability with PyPulseq |
| {doc}`gradients` | `pypulseqpp`: gradient factories, from trapezoids and phase encodes to arbitrary and wave-encoding waveforms, and operations on them |
| {doc}`rf` | `pypulseqpp`: RF pulse factories, SLR, slice-encoding, B1-selective, multidimensional and parallel-transmit design, analysis and Bloch simulation |
| {doc}`trajectories` | `pypulseqpp`: radial, spiral and rosette k-space trajectories, and the gradient waveforms that trace them |
| {doc}`timing` | `pypulseqpp`: ADC dwell and duration on both the ADC and gradient rasters, and times quantized to a raster |
| {doc}`modules` | `pypulseqpp.sequences`: excitation, preparation and readout modules, and the non-Cartesian interleaves the readouts play |
| {doc}`apps` | `pypulseqpp.sequences`: {class}`~pypulseqpp.sequences.SequenceApp`, the base class of the shipped example sequences |
| {doc}`safety` | `pypulseqpp.safety`: gradient amplitude, slew-rate, continuity, mechanical-resonance, PNS and SAR checks |
| {doc}`plotting` | `pypulseqpp.plot`: the SeqEyes view, the publication diagram, and k-space and RF-profile figures |
| {doc}`cli` | `pypulseqpp.cli`: running a sequence from the shell and writing its file |
| {doc}`../sequences` | Every complete sequence the package ships, grouped by family, each with its own reference page |

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
