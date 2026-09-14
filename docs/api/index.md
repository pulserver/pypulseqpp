# API Reference

PyPulseq++ exposes the vocabulary used to design a Pulseq sequence from one
top-level namespace. RF pulse design and gradient waveform design have their
own pages; sequence construction and reusable modules are grouped
by the job they perform.

{doc}`sequence`
: The sequence container, scanner limits, field-of-view transformations, file
  I/O, timing checks, waveforms, k-space, and reports.

{doc}`events`
: ADC, delay, trigger, label, rotation, and RF-shim events, together with
  operations on blocks and events.

{doc}`gradients`
: Gradient factories and operations, from trapezoids and phase encodes to
  arbitrary waveforms.

{doc}`rf`
: RF pulse factories, SLR and multidimensional pulse design, bandwidth
  calculation, and Bloch simulation.

{doc}`trajectories`
: Designing radial, spiral and rosette k-space paths and turning them into
  playable gradients.

{doc}`timing`
: Quantizing ADC dwell, readout duration, and other times to scanner rasters.

{doc}`modules`
: Composable excitation, preparation, and readout modules.

{doc}`safety`
: Sequence-level gradient amplitude, slew-rate, continuity, mechanical-resonance,
  peripheral nerve stimulation and SAR checks.

{doc}`plotting`
: The SeqEyes view, the publication diagram, and k-space and RF-profile
  figures.

{doc}`cli`
: Running a sequence function from the shell and writing its output for a file
  or scanner.

```{toctree}
:hidden:
:maxdepth: 1

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
