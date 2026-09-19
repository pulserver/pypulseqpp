# Trajectory design

`pypulseqpp`: k-space trajectories in 1/m, that is cycles per metre, and the
gradient waveforms that trace them. Each function returns the geometry of one base interleaf as a polyline without
a time parameterization;
{func}`traj_to_grad` assigns the timing within the gradient amplitude and slew
limits and returns the gradient and slew-rate waveforms.

```{eval-rst}
.. currentmodule:: pypulseqpp
```

```{eval-rst}
.. autosummary::
   :nosignatures:

   calc_radial_trajectory
   calc_spiral_trajectory
   calc_rosette_trajectory
   traj_to_grad
```
