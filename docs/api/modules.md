# Sequence modules

Reusable block layouts with their timing and gradient waveforms solved, and
the non-Cartesian base interleaves the readout modules play. Every module takes
the system limits, a {class}`pypulseqpp.Opts`, as its first argument; flip
angles are in degrees, times in s, lengths in m and frequencies in Hz. The
published encoding events are templates the scan loop scales before adding the
blocks, as {doc}`../explanations/design/sequence-module` describes.

```{eval-rst}
.. currentmodule:: pypulseqpp.sequences
```

## Base classes

| Object | Input | Publishes | Purpose |
| --- | --- | --- | --- |
| {obj}`~pypulseqpp.sequences.SequenceModule` | Arguments forwarded to `init_module` | `blocks`, named events on `events`, `center` (s) | Base class of reusable block layouts. |
| {obj}`~pypulseqpp.sequences.RfModule` | Arguments forwarded to `init_module` | As `SequenceModule`, plus `sim_rf` of its RF event | Base class of every excitation and preparation module. |

## Excitation and refocusing

| Object | Input | Publishes | Purpose |
| --- | --- | --- | --- |
| {obj}`~pypulseqpp.sequences.NonSelectiveExcitation` | Flip angle, duration, frequency and phase offsets | `rf` | Rectangular non-selective excitation. |
| {obj}`~pypulseqpp.sequences.NonSelectiveRefocusing` | Flip angle, duration, crusher cycles and axis | `rf_ref`, crusher `gz_spoil` | Rectangular refocusing between two identical crushers. |
| {obj}`~pypulseqpp.sequences.SpatialSelectiveExcitation` | Flip angle, thickness, duration, time-bandwidth product | `rf`, `gz`, `gz_reph`, `selection_amplitude` (Hz/m) | SLR slice or slab excitation. |
| {obj}`~pypulseqpp.sequences.SpatialSelectiveRefocusing` | Thickness, flip angle, duration, crusher cycles | `rf_ref`, crusher–plateau–crusher `gz`, `selection_amplitude` (Hz/m) | SLR refocusing with crushers joined to the plateau. |
| {obj}`~pypulseqpp.sequences.SpatialSelective2DExcitation` | Flip angle, excitation FOV and matrix, disc or target | `rf`, one spiral gradient per axis | Small-tip spiral excitation selective in two dimensions. |
| {obj}`~pypulseqpp.sequences.FrequencySelectiveExcitation` | Flip angle, bandwidth, offset (ppm and Hz) | `rf`, `duration_s` | SLR excitation of a spectral band. |
| {obj}`~pypulseqpp.sequences.SpspExcitation` | Flip angle, thickness, spectral bandwidth, subpulse count | `rf`, alternating `gz`, `gz_reph` | Spectral-spatial excitation. |
| {obj}`~pypulseqpp.sequences.SmsExcitation` | Flip angle, thickness, slice gap, band count | `rf`, `gz`, `gz_reph`, band offsets (Hz) and positions (m) | Simultaneous multi-slice SLR excitation. |
| {obj}`~pypulseqpp.sequences.MultibandExcitation` | Flip angle, duration, sideband offset, band count | `rf`, band offsets (Hz), band power fractions | Non-selective excitation with off-resonance sidebands. |

## Magnetisation preparation

| Object | Input | Publishes | Purpose |
| --- | --- | --- | --- |
| {obj}`~pypulseqpp.sequences.InversionPreparation` | Duration, adiabatic pulse type, bandwidth, crusher cycles | `rf_prep`, `gz_spoil`, `prep_labels` | Non-selective adiabatic inversion. |
| {obj}`~pypulseqpp.sequences.BlochSiegertPreparation` | Frequency offset, duration, peak B1 (Hz) | `rf_prep`, spoilers, `kbs` (rad/Hz²) | Off-resonance Fermi pulse for Bloch-Siegert B1 mapping. |
| {obj}`~pypulseqpp.sequences.DiffusionPreparation` | b-value (s/mm²), gradient duration and separation | `rf_prep`, `rf_ref`, `rf_store`, `g_diff`, spoilers | Non-selective diffusion preparation. |
| {obj}`~pypulseqpp.sequences.FatSaturation` | Offset (ppm), flip angle, bandwidth, optional thickness | `rf_prep`, spoilers, `gz` when selective | SLR fat saturation. |
| {obj}`~pypulseqpp.sequences.IhMtPreparation` | Flip angle, frequency offset, duration, pulse count | `rf_prep`, spoilers, `band_offsets_hz` | Dual-offset saturation for ihMT. |
| {obj}`~pypulseqpp.sequences.MtPreparation` | Flip angle, frequency offset, duration, pulse count | `rf_prep`, spoilers | Off-resonance SLR saturation for MT. |
| {obj}`~pypulseqpp.sequences.OffResonanceSaturation` | RF event, pulse count, spoiler cycles | `rf_prep`, spoilers | Repeated off-resonance saturation pulse. |
| {obj}`~pypulseqpp.sequences.T1T2Preparation` | Preparation echo time, refocusing count | `rf_prep`, `rf_ref`, `rf_store`, spoilers | T2 preparation stored on −z. |
| {obj}`~pypulseqpp.sequences.T2Preparation` | Preparation echo time, final tip, refocusing count | `rf_prep`, `rf_ref`, `rf_store`, spoilers | Adiabatic T2 preparation. |

## Cartesian readouts

Each readout takes the excitation RF event with its selection gradient events,
and a field of view and matrix per encoded axis, readout first; a fast
spin-echo readout also takes the refocusing events.

| Object | Input | Publishes | Purpose |
| --- | --- | --- | --- |
| {obj}`~pypulseqpp.sequences.LineReadout2D` | Excitation, FOV, matrix, TE, TR, bandwidth | `adc`, `gx`, `gx_pre`, `gy_pre`, rewinders, spoiler | Cartesian line, phase-encoded along y. |
| {obj}`~pypulseqpp.sequences.LineReadout3D` | Excitation, FOV, matrix, TE, TR, bandwidth | `adc`, `gx`, `gx_pre`, `gy_pre`, `gz_pre`, rewinders | Cartesian line of a 3D slab. |
| {obj}`~pypulseqpp.sequences.EpiReadout2D` | Excitation, FOV, matrix, ETL, scheme, acceleration | `adc`, `gx` lobes, `gy_blips`, prephasers, `echo_times` (s) | Single- or multi-shot EPI train. |
| {obj}`~pypulseqpp.sequences.EpiReadout3D` | Excitation, FOV, matrix, ETL, scheme, acceleration | `adc`, `gx` lobes, `gy_blips`, `gz_blips`, prephasers | Slab-selective EPI train, partition-encoded along z. |
| {obj}`~pypulseqpp.sequences.FseReadout2D` | Excitation, refocusing, FOV, matrix, ETL, ESP | `rf_ref`, `adc`, `gx`, phase encodes, `echo_times` (s) | Slice-selective CPMG echo train. |
| {obj}`~pypulseqpp.sequences.FseReadout3D` | Excitation, refocusing, FOV, matrix, ETL, ESP | `rf_ref`, `adc`, `gx`, `gy_pre`, `gz_pre`, `echo_times` (s) | Single-slab CPMG echo train. |
| {obj}`~pypulseqpp.sequences.BssfpReadout2D` | Excitation, FOV, matrix, TR, half-flip preparation | `adc`, `gx`, balanced phase encodes and rewinders | Slice-selective balanced SSFP. |
| {obj}`~pypulseqpp.sequences.BssfpReadout3D` | Excitation, FOV, matrix, TR, half-flip preparation | `adc`, `gx`, phase and partition encodes, rewinders | Slab-selective balanced SSFP. |
| {obj}`~pypulseqpp.sequences.PropellerReadout2D` | Excitation, FOV, matrix, blade width, blade count | EPI train events, `blade_angles` (rad) | 2D PROPELLER blade set. |
| {obj}`~pypulseqpp.sequences.PropellerStackReadout` | As `PropellerReadout2D`, plus `fov_z`, `matrix_z` | EPI train events, `blade_angles` (rad), partition encodes | Stack of 2D PROPELLER blade sets. |

## Non-Cartesian readouts

{class}`NonCartesianReadout` takes a solved two-channel interleaf from
{ref}`non-cartesian-interleaves`; the spiral and rosette readouts design theirs
from the prescription.

| Object | Input | Publishes | Purpose |
| --- | --- | --- | --- |
| {obj}`~pypulseqpp.sequences.NonCartesianReadout` | Excitation, a `NonCartesianGradient`, TE, TR | `adc`, `gx`, `gy`, prewinders, rewinders, `gz_spoil` | Readout of a solved interleaf with its TR budget. |
| {obj}`~pypulseqpp.sequences.RadialReadout2D` | Excitation, FOV, matrix, TE, TR, bandwidth | `adc`, `gx`, `gy`, `gz_spoil` | Full radial spoke through the centre of a plane. |
| {obj}`~pypulseqpp.sequences.RadialStackReadout` | As `RadialReadout2D`, plus `fov_z`, `matrix_z` | `adc`, `gx`, `gy`, `gz_pre`, `gz_rew` | Radial spokes with Cartesian partitions (stack of stars). |
| {obj}`~pypulseqpp.sequences.RadialProjectionReadout` | Excitation, FOV, matrix, TE, TR, bandwidth | `adc`, `gx`, `gy`, `gz_spoil` | Radial spokes of a 3D projection acquisition. |
| {obj}`~pypulseqpp.sequences.SpiralReadout2D` | Excitation, FOV, matrix, design interleaves, density | `adc`, `gx`, `gy`, prewinders, rewinders, `trajectory` | One spiral arm in a plane. |
| {obj}`~pypulseqpp.sequences.SpiralStackReadout` | As `SpiralReadout2D`, plus `fov_z`, `matrix_z` | As `SpiralReadout2D`, plus `gz_pre`, `gz_rew` | Spiral arms with Cartesian partitions along z. |
| {obj}`~pypulseqpp.sequences.SpiralProjectionReadout` | As `SpiralReadout2D` | As `SpiralReadout2D` | Spiral arms rotated over a sphere. |
| {obj}`~pypulseqpp.sequences.SpiralNavigator` | FOV, matrix, slab thickness, flip angle, navigator TR | `excitation`, `readout`, per-plane `rotations` | Three orthogonal thick-slab spiral navigators. |
| {obj}`~pypulseqpp.sequences.RosetteReadout2D` | Excitation, FOV, matrix, petals, frequency ratio | `adc`, `gx`, `gy`, `trajectory` | One multi-petal rosette interleaf in a plane. |
| {obj}`~pypulseqpp.sequences.RosetteStackReadout` | As `RosetteReadout2D`, plus `fov_z`, `matrix_z` | As `RosetteReadout2D`, plus `gz_pre`, `gz_rew` | Rosette petals with Cartesian partitions along z. |
| {obj}`~pypulseqpp.sequences.RosetteProjectionReadout` | As `RosetteReadout2D` | As `RosetteReadout2D` | Rosette petals rotated over a sphere. |
| {obj}`~pypulseqpp.sequences.ZteReadout` | Non-selective `rf`, FOV, matrix, spoke directions or count | `g_ramp`, `g_hold`, `g_read`, `adc`, `shot_rotations` | Continuous-gradient ZTE shell with its initial and final ramps. |

(non-cartesian-interleaves)=

## Non-Cartesian interleaves

Each object holds one base interleaf: its readout `gradients`, its `adc`, and
the `prewinders` and `rewinders` to and from k = 0 at zero gradient amplitude.
`trajectory` is in 1/m. The scan loop rotates the interleaf per shot;
`rotated` returns a copy rotated in its plane.

| Object | Input | Publishes | Purpose |
| --- | --- | --- | --- |
| {obj}`~pypulseqpp.sequences.NonCartesianGradient` | Designed gradient events, ADC event, k-space path | The interleaf, as given | Base interleaf from events designed elsewhere. |
| {obj}`~pypulseqpp.sequences.Arbitrary` | k-space path `(n, 2)` or `(n, 3)` (1/m), matrix | Solved interleaf, with prewinders and rewinders as needed | Base interleaf from a caller-supplied path. |
| {obj}`~pypulseqpp.sequences.Spiral` | FOV, matrix, design interleaves, direction, density | Solved spiral interleaf | Constant-, variable- or dual-density spiral. |
| {obj}`~pypulseqpp.sequences.Rosette` | FOV, matrix, petals, frequency ratio, echo spacing | Solved interleaf from and to k = 0 | Multi-petal rosette. |
