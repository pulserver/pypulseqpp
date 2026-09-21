# Trajectory design

`pypulseqpp`: k-space trajectories in 1/m, that is cycles per metre, and the
gradient waveforms that trace them. Each function returns the geometry of one base interleaf as a polyline without
a time parameterization;
{func}`traj_to_grad` assigns the timing within the gradient amplitude and slew
limits and returns the gradient and slew-rate waveforms.

```{eval-rst}
.. currentmodule:: pypulseqpp
```

| Object | Description |
| --- | --- |
| {obj}`~pypulseqpp.calc_radial_trajectory` | Return one full radial spoke through the centre of k-space. |
| {obj}`~pypulseqpp.calc_spiral_trajectory` | Return one spiral-out interleaf, from the centre to ``kmax``. |
| {obj}`~pypulseqpp.calc_rosette_trajectory` | Return one rosette interleaf, petals through the centre of k-space. |
| {obj}`~pypulseqpp.traj_to_grad` | Convert a k-space trajectory to gradient and slew-rate waveforms. |
