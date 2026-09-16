# Gradient design

`pypulseqpp`: gradient events on one logical gradient axis, and the operations
on them. Amplitudes are in Hz/m, slew rates in Hz/m/s and gradient areas in
1/m, except where a docstring states otherwise.

```{eval-rst}
.. currentmodule:: pypulseqpp
```

## Gradient factories

Trapezoidal, extended-trapezoid and arbitrary gradient events, and the
wave-encoding gradients played under a readout's flat top, which trace the
wave-CAIPI corkscrew trajectory.

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

Concatenate, superpose, split, scale or rotate gradient events. Each docstring
states how the operation changes the event's timing.

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
