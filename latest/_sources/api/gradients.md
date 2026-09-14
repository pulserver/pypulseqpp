# Gradient waveform design

```{eval-rst}
.. currentmodule:: pypulseqpp
```

Gradient factories produce trapezoidal, extended, or arbitrary events on one
logical axis. The operations below concatenate, superpose, split, scale, or
rotate those events; their individual contracts specify how timing changes.

## Gradient factories

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   make_arbitrary_grad
   make_crusher
   make_extended_trapezoid
   make_extended_trapezoid_area
   make_hexagon_gradient_area
   make_phase_blip
   make_phase_encoding
   make_trapezoid
```

## Gradient operations

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   add_gradients
   calc_ramp
   concatenate_gradients
   points_to_waveform
   rotate_3d
   scale_grad
   split_gradient
   split_gradient_at
```
