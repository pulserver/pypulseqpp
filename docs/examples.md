# Examples

Executable sequence-design studies. Each page states a design question, varies
the parameters that decide it, and reports a result that can be read off a
figure or a table.

The conceptual background these pages rely on is in
{doc}`explanations/index`: the Pulseq representation, the sequence-design
abstractions, and the gradient, stimulation and SAR constraints. The interfaces
they call are documented in {doc}`api/index`.

The complete sequences the package ships are listed in the {doc}`sequences`,
and each has a reference page carrying a representative configuration and its
sequence diagrams. An example here demonstrates a result rather than
cataloguing an interface.

## Cartesian imaging

Echo-planar and fast-spin-echo acquisitions. The readout lies on a regular
grid, so the design freedom is in how the phase-encoded axes are covered and in
how long the echo train that covers them lasts.

```{eval-rst}
.. minigallery:: ../gallery/01-cartesian/*.py
```

## Non-Cartesian trajectories

Readouts that do not lie on a grid. The achievable trajectory is bounded by the
gradient amplitude and slew rate over the whole readout, and by the amplitude
at which the receiver's dwell time still satisfies the sampling criterion.

```{eval-rst}
.. minigallery:: ../gallery/02-non-cartesian/*.py
```

## RF pulse design

Pulse design against a simulated response. The profile a selective pulse
produces follows from a Bloch simulation of the pulse itself, and the pulses
that can be played are bounded by the gradient amplitude their selection
requires.

```{eval-rst}
.. minigallery:: ../gallery/03-rf-pulses/*.py
```
