# Internals

The rules and derivations behind the explanation pages: what each computation
evaluates, in which frame and units, and over which interval. Each page is
linked from the explanation it supports.

| Page | Scope |
| --- | --- |
| {doc}`pulseq-file` | Sections, libraries, deduplication, signatures, revisions, binary form and rasters. |
| {doc}`cpp-core` | Native hot paths, snapshot ownership, revision counter and facade contracts. |
| {doc}`rf-and-gradient-design` | Pulse designers, pTx layout, Bloch simulation and the gradient solver. |
| {doc}`sequence-modules` | Module structure, event publication, protocol parsing, labels and chains. |
| {doc}`sampling-and-ordering` | Supports, orderings, EPI offsets, angle increments and RF schedules. |
| {doc}`sequence-analysis` | Waveform reconstruction, k-space integrals, repetition and FOV transforms. |
| {doc}`safety-checks` | Gradient, PNS, mechanical-resonance, sound-pressure and SAR computations. |

```{toctree}
:hidden:

pulseq-file
cpp-core
rf-and-gradient-design
sequence-modules
sampling-and-ordering
sequence-analysis
safety-checks
```
