# Gradient design

`pypulseqpp`: gradient events on one logical gradient axis, and the operations
on them. Amplitudes are in Hz/m, slew rates in Hz/m/s and gradient areas in
1/m, except where a docstring states otherwise.

```{eval-rst}
.. currentmodule:: pypulseqpp
```

## Gradient factories

Trapezoidal, extended-trapezoid and arbitrary gradient events, and wave-encoding gradients superposed on a readout plateau for a wave-CAIPI
trajectory.

| Object | Description |
| --- | --- |
| {obj}`~pypulseqpp.make_arbitrary_grad` | Create a gradient from amplitudes sampled at raster centres. |
| {obj}`~pypulseqpp.make_crusher` | Create a crusher winding ``dephasing_cycles`` of phase across a voxel. |
| {obj}`~pypulseqpp.make_extended_trapezoid` | Create an extended trapezoid from amplitude and time points. |
| {obj}`~pypulseqpp.make_extended_trapezoid_area` | Create the shortest extended trapezoid for an area and endpoint amplitudes. |
| {obj}`~pypulseqpp.make_hexagon_gradient_area` | Design a minimum-duration gradient with specified area and endpoint amplitudes. |
| {obj}`~pypulseqpp.make_phase_blip` | Create a phase-encode blip of area ``steps / fov`` for an echo train. |
| {obj}`~pypulseqpp.make_phase_encoding` | Create a positive phase-encode template of area ``1 / (2 * resolution)``. |
| {obj}`~pypulseqpp.make_trapezoid` | Create a trapezoidal gradient event. |
| {obj}`~pypulseqpp.make_wave_gradients` | Create self-balanced wave-encoding gradient events for one readout flat top. |

## Gradient operations

Concatenate, superpose, split, scale or rotate gradient events. Each docstring
states how the operation changes the event's timing.

| Object | Description |
| --- | --- |
| {obj}`~pypulseqpp.add_gradients` | Superpose gradient events on one logical axis. |
| {obj}`~pypulseqpp.calc_ramp` | Connect two three-dimensional k-space points subject to gradient and slew-rate limits. |
| {obj}`~pypulseqpp.concatenate_gradients` | Concatenate gradients on one channel without modifying the inputs. |
| {obj}`~pypulseqpp.points_to_waveform` | Interpolate amplitude and time points onto the gradient raster. |
| {obj}`~pypulseqpp.rotate_3d` | Rotate and sum gradient projections onto the output axes. |
| {obj}`~pypulseqpp.scale_grad` | Scale a gradient event by a scalar. |
| {obj}`~pypulseqpp.split_gradient` | Split a trapezoid into ramp-up, plateau and ramp-down events. |
| {obj}`~pypulseqpp.split_gradient_at` | Split a trapezoid into two extended trapezoids at a specified time. |
