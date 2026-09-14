# Trajectory design

```{eval-rst}
.. currentmodule:: pypulseqpp
```

The trajectory functions design a k-space path in cycles per metre: a radial
spoke, a spiral interleave or a rosette. {func}`traj_to_grad` re-parameterizes
a path within the scanner's gradient and slew limits and returns the gradient
and slew waveforms that play it.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   calc_radial_trajectory
   calc_spiral_trajectory
   calc_rosette_trajectory
   traj_to_grad
```
