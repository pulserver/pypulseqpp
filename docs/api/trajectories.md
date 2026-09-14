# Trajectory design

`pypulseqpp`: k-space paths in cycles per metre, and the gradients that play
them. {func}`traj_to_grad` re-parameterizes a path within the scanner's
gradient and slew limits and returns the gradient and slew waveforms.

```{eval-rst}
.. currentmodule:: pypulseqpp
```

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   calc_radial_trajectory
   calc_spiral_trajectory
   calc_rosette_trajectory
   traj_to_grad
```
