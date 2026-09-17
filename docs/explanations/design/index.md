# Sequence design in pypulseqpp

The Pulseq representation described in {doc}`../pulseq/index` is a flat list of
blocks. These pages describe the two abstractions the package places above it:
a reusable block layout whose timing and gradient waveforms are solved once,
and an application that adds a prescription, a sampling order and a scan loop.

Neither abstraction is part of the file format. A sequence written through them
is an ordinary `.seq` file, and a sequence written without them is equally
valid.

{doc}`sequence-module`
: What a module solves and what it leaves to the caller, the published event
  templates and the `center` timing reference, and what the excitation,
  preparation and readout families contribute.

{doc}`sequence-application`
: The separation of prescription, sampling order and repetition; class
  attributes as design settings; encoding indices as labels; and prescan
  chains.

```{toctree}
:hidden:

sequence-module
sequence-application
```
