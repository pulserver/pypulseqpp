# Bloch simulation

Isochromats, and the Bloch simulation of a sequence's blocks played on them.
Positions are in m, relaxation times in s, off-resonance and fields in Hz, and
event times in s from the start of their block;
{doc}`../explanations/simulation` states the rotation, RF and receiver
conventions. {meth}`Sequence.simulate <pypulseqpp.Sequence.simulate>` plays a
sequence's blocks on isochromats and returns the demodulated ADC samples;
{meth}`Isochromats.play <pypulseqpp.Isochromats.play>` plays the events of one
block given as waveforms.

```{eval-rst}
.. currentmodule:: pypulseqpp
```

## Isochromats

| Object | Input | Returns | Purpose |
| --- | --- | --- | --- |
| {obj}`~pypulseqpp.Isochromats` | Positions (m), proton density, T1 and T2 (s), off-resonance (Hz), transmit and receive sensitivities | `Isochromats` holding their magnetisation from one call to the next | Bloch simulation with relaxation of a sequence played block by block. |
