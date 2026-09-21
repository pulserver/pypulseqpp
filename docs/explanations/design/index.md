# Sequence design in pypulseqpp

The Pulseq representation described in {doc}`../pulseq/index` is a flat list of
blocks. The package places two abstractions above it: a reusable block layout
whose timing and gradient waveforms are solved once, and an application that
adds a prescription, a sampling order and a scan loop.

Neither abstraction is part of the file format. A sequence written through them
is an ordinary `.seq` file, and a sequence written without them is equally
valid.

| Concept | What it covers |
| --- | --- |
| {doc}`sequence-module` | What a module publishes, the `center` reference its timing is measured from, and what excitation, preparation and readout modules are each responsible for. |
| {doc}`sequence-application` | How a prescription becomes a sampling order and a repetition kernel, and how encoding labels and a prescan chain are written. |

```{toctree}
:hidden:

sequence-module
sequence-application
```
