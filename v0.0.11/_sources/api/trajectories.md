# Trajectory design

k-space trajectories of one base interleaf, and their conversion to gradient
waveforms. k-space coordinates are in 1/m, that is cycles per metre, gradient
amplitudes in Hz/m and slew rates in Hz/m/s, on the channel axes;
{doc}`../explanations/pulseq/libraries-and-shapes` describes how one interleaf
is reused for every shot through rotation extensions.

```{eval-rst}
.. currentmodule:: pypulseqpp
```

## Trajectories and gradient conversion

The trajectory functions return geometry as a polyline, with no time
parameterization. {func}`traj_to_grad` assigns the timing within the vector
gradient amplitude and slew-rate limits of the system.

| Object | Input | Returns | Purpose |
| --- | --- | --- | --- |
| {obj}`~pypulseqpp.calc_radial_trajectory` | FOV (m), matrix, number of points | k-space points `(num_points, 2)` | Full radial spoke through the k-space centre. |
| {obj}`~pypulseqpp.calc_spiral_trajectory` | FOV (m), matrix, design interleaves, density | k-space points `(num_points, 2)` from the origin | Spiral-out interleaf. |
| {obj}`~pypulseqpp.calc_rosette_trajectory` | FOV (m), matrix, petals, frequency ratio | k-space points `(num_points, 2)`, closed at the origin | Rosette interleaf. |
| {obj}`~pypulseqpp.traj_to_grad` | k-space trajectory, time last `(n,)`, `(2, n)`, `(3, n)` | Gradient waveform (Hz/m); slew rate (Hz/m/s) | Time-optimal trajectory-to-gradient conversion. |
