# Trajectory design

```{eval-rst}
.. currentmodule:: pypulseqpp
```

{func}`traj_to_grad` re-parameterizes a k-space path within the scanner's
gradient and slew limits and returns the gradient and slew waveforms that play
it. Angle generators provide the rotations used to repeat a radial path in a
plane or distribute projections over a sphere.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   traj_to_grad
   calc_golden_angles
   calc_tiny_golden_angles
   calc_raga_angles
   calc_uniform_angles
   calc_projection_shell
```
