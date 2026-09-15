# API reference

| Page | Contents |
| --- | --- |
| {doc}`sequence` | `pypulseqpp`: the sequence container, scanner limits and field-of-view transforms |
| {doc}`events` | `pypulseqpp`: ADC, delay, trigger, label, rotation and RF-shim events, block and event operations, interoperability with PyPulseq |
| {doc}`gradients` | `pypulseqpp`: gradient factories, from trapezoids and phase encodes to arbitrary and wave-CAIPI waveforms, and operations on them |
| {doc}`rf` | `pypulseqpp`: RF pulse factories, SLR, slice-encoding, B1-selective, multidimensional and parallel-transmit design, analysis and Bloch simulation |
| {doc}`trajectories` | `pypulseqpp`: radial, spiral and rosette k-space paths, and the gradients that play them |
| {doc}`timing` | `pypulseqpp`: legal ADC timing, and times quantized to the scanner's rasters |
| {doc}`modules` | `pypulseqpp.sequences`: excitation, preparation and readout modules, and the complete sequences built from them |
| {doc}`safety` | `pypulseqpp.safety`: gradient, mechanical-resonance, nerve-stimulation and SAR checks |
| {doc}`plotting` | `pypulseqpp.plot`: the SeqEyes view, the publication diagram, and k-space and RF-profile figures |
| {doc}`cli` | `pypulseqpp.cli`: running a sequence from the shell and writing its file |

```{toctree}
:hidden:

sequence
events
gradients
rf
trajectories
timing
modules
safety
plotting
cli
```
