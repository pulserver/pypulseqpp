# RF pulse design

`pypulseqpp`: RF events, from the basic factories to designed pulses, and their
analysis. A design function returns an RF event and, depending on its options, a slice-selection gradient, rephasing gradient and
design metadata.
RF amplitudes are in Hz, frequency offsets in Hz and phase offsets in radians.

```{eval-rst}
.. currentmodule:: pypulseqpp
```

## Pulse factories

Hard, arbitrary, Gaussian, sinc and adiabatic pulses, and the adiabatic
half-passage pair.

| Object | Description |
| --- | --- |
| {obj}`~pypulseqpp.make_adiabatic_pulse` | Construct an adiabatic RF pulse: a frequency sweep whose rotation is insensitive to B1. |
| {obj}`~pypulseqpp.make_arbitrary_rf` | Create an RF pulse with the given pulse shape. |
| {obj}`~pypulseqpp.make_block_pulse` | Create a block (RECT or hard) pulse. |
| {obj}`~pypulseqpp.make_gauss_pulse` | Create a [optionally slice selective] Gauss pulse. |
| {obj}`~pypulseqpp.make_half_passages` | Build the adiabatic half-passage pair that tips magnetisation down and back. |
| {obj}`~pypulseqpp.make_sinc_pulse` | Create a sinc RF event with optional slice-selection and rephasing gradients. |

## SLR and multidimensional design

| Object | Description |
| --- | --- |
| {obj}`~pypulseqpp.make_slr_pulse` | Design an RF pulse using the Shinnar-Le Roux algorithm. |
| {obj}`~pypulseqpp.make_recursive_slr_pulses` | Design SLR pulses that each excite the same transverse magnetisation. |
| {obj}`~pypulseqpp.make_sms_pulse` | Modulate one RF pulse into equispaced spectral bands. |
| {obj}`~pypulseqpp.make_spsp_pulse` | Design a spectral-spatial pulse on an alternating slice gradient. |
| {obj}`~pypulseqpp.make_2d_selective_pulse` | Design a small-tip 2D-selective pulse on a spiral excitation trajectory. |

## Slice encoding

Slab pulses whose sub-slices are encoded across repeated acquisitions, by
gSlider phases or Hadamard signs, and PINS pulses that excite a comb of slices.

| Object | Description |
| --- | --- |
| {obj}`~pypulseqpp.make_gslider_pulse` | Design a gSlider slab pulse with ``subslice`` at ``subslice_phase``. |
| {obj}`~pypulseqpp.make_hadamard_pulse` | Design a slab pulse whose sub-bands are signed by row ``row`` of a Hadamard matrix. |
| {obj}`~pypulseqpp.make_pins_pulse` | Design a PINS pulse exciting a slice every ``slice_separation`` along z. |

## B1 selection

Pulses selective in the transmit field's amplitude rather than in position, and
the adiabatic Bloch-Siegert pulse that encodes B1 into phase.

| Object | Description |
| --- | --- |
| {obj}`~pypulseqpp.make_b1_selective_pulse` | Design a B1-selective RF pulse, excited only where the transmit amplitude lies in a band. |
| {obj}`~pypulseqpp.make_b1_gslider_pulse` | Design a B1-selective gSlider pulse with one B1 sub-band at ``subslice_phase``. |
| {obj}`~pypulseqpp.make_b1_hadamard_pulse` | Design a B1-selective pulse whose B1 sub-bands are signed by a Hadamard row. |
| {obj}`~pypulseqpp.make_bloch_siegert_pulse` | Design an adiabatic Bloch-Siegert encoding pulse. |

## Parallel transmit

A dynamic pTx pulse holds every transmit channel's waveform in one RF event,
one channel after another over a shared time base.
{func}`calc_rf_shim` computes static channel weights and
{func}`make_spokes_pulse` designs spokes.

| Object | Description |
| --- | --- |
| {obj}`~pypulseqpp.make_ptx_pulse` | Make a dynamic pTx pulse from one waveform per transmit channel. |
| {obj}`~pypulseqpp.split_ptx_pulse` | Return a pulse's waveforms, one row per transmit channel, in Hz. |
| {obj}`~pypulseqpp.calc_rf_shim` | Return per-channel weights whose combined B1 field has magnitude ``target``. |
| {obj}`~pypulseqpp.make_spokes_pulse` | Design a spokes pTx pulse: one slice excited at several in-plane k positions. |

## Analysis and simulation

| Object | Description |
| --- | --- |
| {obj}`~pypulseqpp.calc_rf_bandwidth` | Estimate RF bandwidth from the envelope's Fourier magnitude. |
| {obj}`~pypulseqpp.calc_rf_center` | Return the RF event's effective rotation time and corresponding sample index. |
| {obj}`~pypulseqpp.calc_rf_power` | Return an RF event's energy, peak power and RMS amplitude, as MATLAB Pulseq's ``calcRfPower``. |
| {obj}`~pypulseqpp.sim_bloch` | Simulate the Bloch equation without relaxation, in hard-pulse steps. |
| {obj}`~pypulseqpp.sim_rf` | Simulate an RF pulse versus off-resonance without relaxation. |
