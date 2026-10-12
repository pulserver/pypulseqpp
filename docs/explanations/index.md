# Explanations

The concepts behind pypulseqpp, one topic per page.

| Page | Contents |
| --- | --- |
| {doc}`pulseq-file` | The `.seq` format: storage, deduplication, signatures and rasters. |
| {doc}`cpp-core` | The PyPulseq-compatible facade over the C++ core. |
| {doc}`rf-and-gradient-design` | SLR, adiabatic, Bloch-Siegert and pTx pulses; Bloch simulation; gradients from a k-space path. |
| {doc}`sequence-modules` | Sequence modules, sequence functions and protocols. |
| {doc}`sampling-and-ordering` | Undersampling masks, view ordering, golden angles and RF spoiling schedules. |
| {doc}`sequence-analysis` | Waveforms, k-space, repetition detection, RF power and FOV transforms. |
| {doc}`safety-checks` | Gradient, PNS, mechanical-resonance, acoustic and SAR checks. |

The rules, formulas and derivations behind each page are in the
{doc}`internals <../developer-guide/internals/index>` of the developer guide.

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
