# Gradient design

Gradient events on one logical gradient axis, and the operations on them.
Amplitudes are in Hz/m, slew rates in Hz/m/s, gradient areas in 1/m and times
in s, except where a docstring states otherwise;
{doc}`../explanations/pulseq/events-and-blocks` describes the trapezoid and
arbitrary gradient events.

```{eval-rst}
.. currentmodule:: pypulseqpp
```

## Gradient factories

| Object | Input | Returns | Purpose |
| --- | --- | --- | --- |
| {obj}`~pypulseqpp.make_arbitrary_grad` | Channel, amplitudes at raster centres, endpoint amplitudes | Arbitrary gradient event | Sampled gradient waveform. |
| {obj}`~pypulseqpp.make_crusher` | Dephasing cycles, voxel size (m), channel, endpoint amplitudes | Extended trapezoid; vertex times and amplitudes | Crusher of `dephasing_cycles / voxel_size` area. |
| {obj}`~pypulseqpp.make_extended_trapezoid` | Channel, amplitude and time points | Extended trapezoid (arbitrary gradient event) | Piecewise-linear gradient waveform. |
| {obj}`~pypulseqpp.make_extended_trapezoid_area` | Area, channel, endpoint amplitudes | Extended trapezoid; vertex times and amplitudes | Shortest extended trapezoid for an area. |
| {obj}`~pypulseqpp.make_hexagon_gradient_area` | Channel, endpoint amplitudes, area | Gradient event; corner times and amplitudes | Minimum-duration gradient for an area. |
| {obj}`~pypulseqpp.make_phase_blip` | Channel, FOV (m), steps, duration | Trapezoid of area `steps / fov` | Echo-train phase-encode blip. |
| {obj}`~pypulseqpp.make_phase_encoding` | Channel, resolution (m), duration | Trapezoid of area `1 / (2 * resolution)` | Largest phase encode, scaled per view. |
| {obj}`~pypulseqpp.make_trapezoid` | Channel; area, flat area or amplitude; timing | Trapezoid event | Trapezoidal gradient. |
| {obj}`~pypulseqpp.make_wave_gradients` | Flat time, cycles, amplitude (T/m), channels | Sine and cosine arbitrary gradients; optional amplitude (T/m) | Wave-CAIPI wave-encoding gradients. |

## Gradient operations

Each docstring states how the operation changes the event's timing.

| Object | Input | Returns | Purpose |
| --- | --- | --- | --- |
| {obj}`~pypulseqpp.add_gradients` | Gradient events on one axis | Gradient event | Superposition. |
| {obj}`~pypulseqpp.calc_ramp` | Two k-space point pairs `(3, 2)` (1/m), limits | Connecting k-space points; success flag | Minimum-time 3D k-space connection. |
| {obj}`~pypulseqpp.concatenate_gradients` | Gradient events on one channel, in play order | One gradient event | Concatenation. |
| {obj}`~pypulseqpp.points_to_waveform` | Amplitude and time points, gradient raster | Waveform on the gradient raster | Interpolation onto the raster. |
| {obj}`~pypulseqpp.rotate_3d` | Rotation matrix, quaternion or angles; block events | Non-gradient events, then rotated gradients | 3D rotation of a block's gradients. |
| {obj}`~pypulseqpp.scale_grad` | Gradient event, scale factor | Scaled gradient event | Amplitude scaling. |
| {obj}`~pypulseqpp.split_gradient` | Trapezoid | Ramp-up, plateau and ramp-down events | Trapezoid decomposition. |
| {obj}`~pypulseqpp.split_gradient_at` | Gradient event, time (s) | Two gradient events whose sum is the input | Split at a time point. |
