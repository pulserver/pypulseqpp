# Sequence modules

A {class}`~pypulseqpp.sequences.SequenceModule` solves a reusable block layout
independently of the acquisition loop. The module contains block tuples, named
mutable event templates, and a timing reference.

```text
SequenceModule
├── blocks: [(event, ...), ...]
├── events: named mutable event templates
└── center: timing reference (s)
```

## Timing centre

The `center` attribute gives the module timing reference in seconds from its
start, usually an RF pulse centre or an echo. For an inversion module followed
by an excitation module, the recovery delay required for inversion time
\(T_I\) is

$$
t_{\mathrm{delay}} = T_I
- (t_{\mathrm{inv}} - t_{\mathrm{centre,inv}})
- t_{\mathrm{centre,exc}}.
$$

This definition composes pulse-centre timing without depending on block
boundaries.

## Module contents

| Module family | Blocks and events | Responsibility |
| --- | --- | --- |
| Excitation | RF event, selection gradient, rephaser | Excitation or refocusing geometry and pulse timing. |
| Preparation | RF and gradient preparation blocks | Inversion, T2 preparation, saturation, diffusion, or magnetisation transfer. |
| Readout | Prephasing, ADC event, readout gradients, rewinding or spoiling | Echo timing, acquisition bandwidth, and sampling trajectory. |

Named events are published both as attributes and through `events`. An
acquisition loop may change an event template before `add_block`, for example
by setting an RF phase offset or scaling a phase-encode gradient. Previously
registered blocks and the module's internal sequence remain unchanged.

A readout constructed with `te=None` uses its shortest realizable echo time.
Requested bandwidths and times are rasterized, and the achieved values are
reported by the module.

## Representation boundary

Module construction separates fixed waveform and timing design from per-view
encoding. This mirrors the Pulseq distinction between event definitions and
playout instances; see {doc}`../pulseq/libraries-and-shapes`. Sampling order
and repetition structure belong to {doc}`sequence-application`.

## See also

* {doc}`../../api/modules` — module interfaces and parameters.
* {doc}`../../examples/custom-modules` — custom module implementation.
