# RF pulse design

`pypulseqpp`: RF pulses, from basic factories to designed pulses, and their
analysis. Design functions return an RF event and, depending on their options,
the gradients it plays under and design metadata.

```{eval-rst}
.. currentmodule:: pypulseqpp
```

## Pulse factories

Hard, arbitrary, Gaussian, sinc and adiabatic pulses, and the adiabatic
half-passage pair.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   make_adiabatic_pulse
   make_arbitrary_rf
   make_block_pulse
   make_gauss_pulse
   make_half_passages
   make_sinc_pulse
```

## SLR and multidimensional design

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   make_slr_pulse
   make_recursive_slr_pulses
   make_sms_pulse
   make_spsp_pulse
   make_2d_selective_pulse
```

## Slice encoding

Slab pulses whose sub-slices are encoded across repeated acquisitions, by
gSlider phases or Hadamard signs, and PINS pulses that excite a comb of slices.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   make_gslider_pulse
   make_hadamard_pulse
   make_pins_pulse
```

## B1 selection

Pulses selective in the transmit field's amplitude rather than in position, and
the adiabatic Bloch-Siegert pulse that encodes B1 into phase.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   make_b1_selective_pulse
   make_b1_gslider_pulse
   make_b1_hadamard_pulse
   make_bloch_siegert_pulse
```

## Parallel transmit

A dynamic pTx pulse holds every transmit channel's waveform in one RF event.
{func}`calc_rf_shim` computes static channel weights and
{func}`make_spokes_pulse` designs spokes.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   make_ptx_pulse
   split_ptx_pulse
   calc_rf_shim
   make_spokes_pulse
```

## Analysis and simulation

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   calc_rf_bandwidth
   calc_rf_center
   calc_rf_power
   sim_bloch
   sim_rf
```
