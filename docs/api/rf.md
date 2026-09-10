# RF pulse design

```{eval-rst}
.. currentmodule:: pypulseqpp
```

The basic RF factories construct hard, arbitrary, Gaussian, sinc, and
adiabatic pulses. The design functions add SLR, simultaneous-multislice,
spectral-spatial, and two-dimensional selective pulses. Depending on the
factory and options, results include an RF event, accompanying gradients,
and design metadata. Add the returned events to {class}`Sequence` blocks.

## Pulse factories

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   make_adiabatic_pulse
   make_arbitrary_rf
   make_block_pulse
   make_gauss_pulse
   make_sinc_pulse
```

## SLR and multidimensional design

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   make_slr_pulse
   make_sms_pulse
   make_spsp_pulse
   make_2d_selective_pulse
```

## Analysis and simulation

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   calc_rf_bandwidth
   calc_rf_center
   bloch
   sim_rf
```
