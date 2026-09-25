# RF pulse design

RF events, from the basic factories to designed pulses, and their analysis.
Flip angles are in rad, RF amplitudes and frequency offsets in Hz, phase
offsets in rad and durations in s. With `return_gz`, a slice-selective design
also returns its slice-selection and rephasing gradients;
{doc}`../explanations/pulseq/events-and-blocks` describes the RF event.

```{eval-rst}
.. currentmodule:: pypulseqpp
```

## Pulse factories

| Object | Input | Returns | Purpose |
| --- | --- | --- | --- |
| {obj}`~pypulseqpp.make_adiabatic_pulse` | Sweep type, adiabaticity, bandwidth, duration | RF event; optional slice-select and rephasing gradients | Adiabatic (B1-insensitive) frequency sweep. |
| {obj}`~pypulseqpp.make_arbitrary_rf` | Complex waveform, flip angle | RF event; optional slice-select gradient | RF event from a given waveform. |
| {obj}`~pypulseqpp.make_block_pulse` | Flip angle, duration or bandwidth | RF event | Hard (rectangular) pulse. |
| {obj}`~pypulseqpp.make_gauss_pulse` | Flip angle, duration, time-bandwidth product | RF event; optional slice-select and rephasing gradients | Gaussian pulse. |
| {obj}`~pypulseqpp.make_half_passages` | Duration, adiabaticity, sweep type | Two RF events, `down` and `up` | Adiabatic half-passage pair. |
| {obj}`~pypulseqpp.make_sinc_pulse` | Flip angle, duration, time-bandwidth product | RF event; optional slice-select and rephasing gradients | Sinc pulse. |

## SLR and multidimensional design

| Object | Input | Returns | Purpose |
| --- | --- | --- | --- |
| {obj}`~pypulseqpp.make_slr_pulse` | Flip angle, duration, time-bandwidth product, pulse and filter type | RF event; optional slice-select and rephasing gradients | Shinnar-Le Roux design. |
| {obj}`~pypulseqpp.make_recursive_slr_pulses` | Number of segments, time-bandwidth product, T1, segment TR | One RF event per segment; optional refocusing pulse | Equal transverse magnetisation per segment. |
| {obj}`~pypulseqpp.make_sms_pulse` | RF event, number of bands, band spacing (Hz) | Modulated RF event; band offsets (Hz); complex weights | Simultaneous multi-slice modulation. |
| {obj}`~pypulseqpp.make_spsp_pulse` | Flip angle, slice thickness, spectral bandwidth | RF event; alternating gradient; optional rephaser | Spectral-spatial pulse. |
| {obj}`~pypulseqpp.make_2d_selective_pulse` | Flip angle, FOV, matrix, target or size | RF event; per-axis gradients; rephasers | Small-tip 2D-selective pulse on a spiral. |

## Slice encoding

| Object | Input | Returns | Purpose |
| --- | --- | --- | --- |
| {obj}`~pypulseqpp.make_gslider_pulse` | Flip angle, sub-slice count, sub-slice index, phase (rad) | RF event; optional slice-select and rephasing gradients | gSlider slab encoding. |
| {obj}`~pypulseqpp.make_hadamard_pulse` | Flip angle, Hadamard order, row | RF event; optional slice-select and rephasing gradients | Hadamard slab encoding. |
| {obj}`~pypulseqpp.make_pins_pulse` | Flip angle, slice thickness, slice separation (m) | RF event; blip gradient; rephaser | PINS multiband excitation. |

## B1 selection

B1 amplitudes of these designs are inputs in T; the returned RF events are in
Hz.

| Object | Input | Returns | Purpose |
| --- | --- | --- | --- |
| {obj}`~pypulseqpp.make_b1_selective_pulse` | Flip angle, amplitude (T), relative B1 passband | RF event | B1-selective excitation. |
| {obj}`~pypulseqpp.make_b1_gslider_pulse` | Flip angle, amplitude (T), sub-band count and index | RF event | B1-selective gSlider encoding. |
| {obj}`~pypulseqpp.make_b1_hadamard_pulse` | Flip angle, amplitude (T), Hadamard order, row | RF event | B1-selective Hadamard encoding. |
| {obj}`~pypulseqpp.make_bloch_siegert_pulse` | Amplitude (T), duration, sweep shape, side of resonance | RF event | Adiabatic Bloch-Siegert B1 encoding. |

## Parallel transmit

A dynamic pTx pulse is one RF event holding each transmit channel's waveform
in turn over a shared time base.

| Object | Input | Returns | Purpose |
| --- | --- | --- | --- |
| {obj}`~pypulseqpp.make_ptx_pulse` | Complex waveforms `(channels, samples)` (Hz) | pTx RF event | Dynamic pTx pulse. |
| {obj}`~pypulseqpp.split_ptx_pulse` | pTx RF event | Complex waveforms `(channels, samples)` (Hz) | Per-channel waveforms of a pTx pulse. |
| {obj}`~pypulseqpp.calc_rf_shim` | Complex B1+ maps per channel, mask, target | Complex weights `(channels,)` | Magnitude least-squares RF shim. |
| {obj}`~pypulseqpp.make_spokes_pulse` | Flip angle, B1+ maps, FOV, slice thickness | pTx RF event; gradients; rephasers | Spokes pTx excitation. |

## Analysis and simulation

| Object | Input | Returns | Purpose |
| --- | --- | --- | --- |
| {obj}`~pypulseqpp.calc_rf_bandwidth` | RF event, cutoff | Bandwidth (Hz); optional spectrum and frequency axis | Bandwidth from the envelope spectrum. |
| {obj}`~pypulseqpp.calc_rf_center` | RF event | Centre time (s); sample index | RF pulse centre. |
| {obj}`~pypulseqpp.calc_rf_power` | RF event, time step | Energy (Hz² s), peak power (Hz²), RMS amplitude (Hz) | RF power, as MATLAB Pulseq `calcRfPower`. |
| {obj}`~pypulseqpp.sim_bloch` | Complex B1 (Hz), Bz (Hz), time step | Final magnetisation `(P, 3)` | Hard-pulse Bloch simulation, no relaxation. |
| {obj}`~pypulseqpp.sim_rf` | RF event, rephase factor | Mz and Mxy profiles, frequency axis (Hz), refocusing efficiency | Off-resonance profile, no relaxation. |
