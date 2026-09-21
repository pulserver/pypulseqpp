# Sequence design in pypulseqpp

The Pulseq representation described in {doc}`../pulseq/index` is a flat list of
blocks. The package places two abstractions above it: a reusable block layout
whose timing and gradient waveforms are solved once, and an application that
adds a prescription, a sampling order and a scan loop.

Neither abstraction is part of the file format. A sequence written through them
is an ordinary `.seq` file, and a sequence written without them is equally
valid.

| Concept | Scope |
| --- | --- |
| {doc}`sequence-module` | Reusable block layouts, published event templates, the `center` timing reference, and excitation, preparation and readout responsibilities. |
| {doc}`sequence-application` | Prescription, sampling order, repetition kernels, design settings, encoding labels and prescan chains. |

```{toctree}
:hidden:

sequence-module
sequence-application
```
