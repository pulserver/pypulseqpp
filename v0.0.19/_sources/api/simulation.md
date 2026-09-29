# Bloch simulation

Isochromats, and the Bloch simulation of a sequence's blocks played on them.
Positions are in m, relaxation times in s, off-resonance and fields in Hz, and
event times in s from the start of their block;
{doc}`../explanations/simulation` states the rotation, RF and receiver
conventions. {meth}`Sequence.simulate <pypulseqpp.Sequence.simulate>` plays a
sequence's blocks on isochromats and returns the demodulated ADC samples;
{meth}`Isochromats.play <pypulseqpp.Isochromats.play>` plays the events of one
block: gradients as corner points, under a rotation where one is given, and RF
and ADC events or the field and sample times they amount to.
{meth}`Isochromats.repetitions <pypulseqpp.Isochromats.repetitions>` plays one
sequence of blocks many times, each time with its RF and ADC phase offsets and
its phase encoding changed, from the affine map one repetition applies to each
isochromat.

```{eval-rst}
.. currentmodule:: pypulseqpp
```

## Isochromats

| Object | Input | Returns | Purpose |
| --- | --- | --- | --- |
| {obj}`~pypulseqpp.Isochromats` | Positions (m), proton density, T1 and T2 (s), off-resonance (Hz), transmit and receive sensitivities | `Isochromats` holding their magnetisation from one call to the next | Bloch simulation with relaxation of a sequence played block by block. |
| {obj}`~pypulseqpp.Repetitions` | One repetition's blocks; each repetition's RF and ADC phase offsets (rad) and phase-encoding areas (1/m); a tolerance | Each repetition's samples, demodulated as `Isochromats.play` demodulates them | The repetitions `Isochromats.repetitions` makes, played in turn. |
