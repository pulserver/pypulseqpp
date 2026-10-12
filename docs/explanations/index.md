# Explanations

What pypulseqpp does and how, one topic per page. Each page opens with a
summary, lists what pypulseqpp does with the code and tests behind each item,
and shows the mechanism in one figure; the full rules and formulas are in the
{doc}`internals <../developer-guide/internals/index>` of the developer guide.

| Explanation | What it covers |
| --- | --- |
| {doc}`pulseq-file` | What a `.seq` file holds, how it is stored, deduplicated and signed, and the rasters its times lie on. |
| {doc}`cpp-core` | The PyPulseq-compatible facade over a native core: what runs in C++ and what is left to PyPulseq. |
| {doc}`rf-and-gradient-design` | SLR, adiabatic, Bloch-Siegert and pTx pulses, the Bloch simulator, and gradients from a k-space path. |
| {doc}`sequence-modules` | Sequence modules, sequence functions and the protocol they declare. |
| {doc}`sampling-and-ordering` | Undersampling masks, view ordering, golden angles and RF spoiling schedules. |
| {doc}`sequence-analysis` | Waveforms, k-space, repetition detection, RF power and field-of-view transforms. |
| {doc}`safety-checks` | Gradient, PNS, mechanical-resonance, sound-pressure and SAR checks. |

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
