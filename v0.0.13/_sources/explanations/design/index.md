# Sequence design in pypulseqpp

The abstractions the package places above the flat block list of
{doc}`../pulseq/index`: a reusable block layout whose timing and gradient
waveforms are solved once, an application that adds a prescription, a sampling
order and a scan loop, and the sampling routines that feed it. None of them is
part of the file format; a sequence written through them is an ordinary `.seq`
file.

| Concept | What it covers |
| --- | --- |
| {doc}`sequence-module` | What a module publishes, the `center` reference its timing is measured from, and what excitation, preparation and readout modules are each responsible for. |
| {doc}`sequence-application` | How a prescription becomes a sampling order and a repetition kernel, and how encoding labels and a prescan chain are written. |
| {doc}`sampling` | Acquired support and temporal ordering, the kinds of value the sampling routines exchange, and where encoding labels are created. |

```{toctree}
:hidden:

sequence-module
sequence-application
sampling
```
