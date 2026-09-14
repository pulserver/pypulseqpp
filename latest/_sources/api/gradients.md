# Gradient waveform design

`pypulseqpp`: gradient events on one logical axis, and the operations on them.

```{eval-rst}
.. currentmodule:: pypulseqpp
```

## Gradient factories

Trapezoidal, extended-trapezoid and arbitrary events, and the wave-CAIPI
corkscrew played under a readout's flat top.

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
   make_wave_gradients
```

## Gradient operations

Concatenate, superpose, split, scale or rotate events; each contract states
how timing changes.

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
