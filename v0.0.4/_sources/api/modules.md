# Sequence modules

Reusable block layouts, with their timing and gradient waveforms solved and
their encoding events left as templates for the scan loop to scale, together
with the non-Cartesian interleaves the readout modules play.

```{eval-rst}
.. currentmodule:: pypulseqpp.sequences
```

## Base classes

A module is a reusable block layout with named, mutable event templates. The
scan loop chooses the views, scales the encoding events and adds the blocks.
A {class}`SequenceModule` lays its blocks out in `init_module`; an
{class}`RfModule` is a module built around one RF event, which
{meth}`~RfModule.sim_rf` simulates against off-resonance.

| Object | Description |
| --- | --- |
| {obj}`~pypulseqpp.sequences.SequenceModule` | A reusable block layout with named, mutable event templates. |
| {obj}`~pypulseqpp.sequences.RfModule` | Sequence module with off-resonance simulation of an individual RF pulse. |

## Excitation and refocusing

| Object | Description |
| --- | --- |
| {obj}`~pypulseqpp.sequences.NonSelectiveExcitation` | Rectangular RF excitation without a selection gradient. |
| {obj}`~pypulseqpp.sequences.NonSelectiveRefocusing` | Rectangular refocusing pulse between two identical crushers. |
| {obj}`~pypulseqpp.sequences.SpatialSelectiveExcitation` | SLR excitation with a selection gradient and optional rephasing. |
| {obj}`~pypulseqpp.sequences.SpatialSelectiveRefocusing` | SLR refocusing with matched crushers joined to the selection plateau. |
| {obj}`~pypulseqpp.sequences.SpatialSelective2DExcitation` | Small-tip spiral excitation selective in two dimensions. |
| {obj}`~pypulseqpp.sequences.FrequencySelectiveExcitation` | SLR excitation of a spectral band without spatial selection. |
| {obj}`~pypulseqpp.sequences.SpspExcitation` | Spectral-spatial excitation using SLR subpulses on an alternating gradient. |
| {obj}`~pypulseqpp.sequences.SmsExcitation` | SLR slice excitation modulated into simultaneous spectral bands. |
| {obj}`~pypulseqpp.sequences.MultibandExcitation` | Non-spatially-selective RF excitation with off-resonance sidebands. |

## Magnetisation preparation

| Object | Description |
| --- | --- |
| {obj}`~pypulseqpp.sequences.InversionPreparation` | Non-selective adiabatic inversion, followed by an optional crusher. |
| {obj}`~pypulseqpp.sequences.BlochSiegertPreparation` | Off-resonance Fermi pulse for Bloch-Siegert B1 mapping. |
| {obj}`~pypulseqpp.sequences.DiffusionPreparation` | Non-selective diffusion preparation with hard tip-down and storage pulses. |
| {obj}`~pypulseqpp.sequences.FatSaturation` | SLR fat saturation with a three-axis spoiler and optional spatial selection. |
| {obj}`~pypulseqpp.sequences.IhMtPreparation` | Simultaneous dual-offset saturation for ihMT measurements. |
| {obj}`~pypulseqpp.sequences.MtPreparation` | Off-resonance SLR saturation followed by an optional spoiler. |
| {obj}`~pypulseqpp.sequences.OffResonanceSaturation` | Repeat one off-resonance pulse, followed by an optional three-axis spoiler. |
| {obj}`~pypulseqpp.sequences.T1T2Preparation` | T2 preparation with storage on -z to initiate T1 recovery. |
| {obj}`~pypulseqpp.sequences.T2Preparation` | Adiabatic T2 preparation with tip-down, refocusing and storage pulses. |

## Cartesian readouts

| Object | Description |
| --- | --- |
| {obj}`~pypulseqpp.sequences.LineReadout2D` | One Cartesian line, frequency-encoded along x and phase-encoded along y. |
| {obj}`~pypulseqpp.sequences.LineReadout3D` | One Cartesian line of a 3D slab, phase-encoded along y and z. |
| {obj}`~pypulseqpp.sequences.EpiReadout2D` | A single- or multi-shot EPI train, frequency-encoded along x. |
| {obj}`~pypulseqpp.sequences.EpiReadout3D` | A slab-selective EPI train, phase-encoded along y and partition-encoded along z. |
| {obj}`~pypulseqpp.sequences.FseReadout2D` | A slice-selective CPMG train, frequency-encoded along x. |
| {obj}`~pypulseqpp.sequences.FseReadout3D` | Single-slab CPMG train, phase-encoded along y and partition-encoded along z. |
| {obj}`~pypulseqpp.sequences.BssfpReadout2D` | Slice-selective balanced SSFP, phase-encoded along y. |
| {obj}`~pypulseqpp.sequences.BssfpReadout3D` | Slab-selective balanced SSFP, encoded along y and z. |
| {obj}`~pypulseqpp.sequences.PropellerReadout2D` | EPI blades turned about the centre of k-space, as a 2D PROPELLER set. |
| {obj}`~pypulseqpp.sequences.PropellerStackReadout` | A stack of 2D PROPELLER blade sets, partition-encoded along z. |

## Non-Cartesian readouts

{class}`NonCartesianReadout` plays any interleaf from
{ref}`non-cartesian-interleaves` as a whole repetition.
The spiral and rosette readouts design theirs from the prescription.

| Object | Description |
| --- | --- |
| {obj}`~pypulseqpp.sequences.NonCartesianReadout` | A solved interleaf with its prewinder, rewinder and repetition-time budget. |
| {obj}`~pypulseqpp.sequences.RadialReadout2D` | A full radial spoke through the centre of a plane. |
| {obj}`~pypulseqpp.sequences.RadialStackReadout` | Radial spokes in-plane with Cartesian partitions along z, a stack of stars. |
| {obj}`~pypulseqpp.sequences.RadialProjectionReadout` | Radial spokes for a spherical projection acquisition. |
| {obj}`~pypulseqpp.sequences.SpiralReadout2D` | One spiral arm in a plane. |
| {obj}`~pypulseqpp.sequences.SpiralStackReadout` | Spiral arms in-plane, Cartesian partitions along z. |
| {obj}`~pypulseqpp.sequences.SpiralProjectionReadout` | Spiral arms turned over a sphere. |
| {obj}`~pypulseqpp.sequences.SpiralNavigator` | Three orthogonal thick-slab spiral navigators. |
| {obj}`~pypulseqpp.sequences.RosetteReadout2D` | One multi-petal rosette interleaf in a plane. |
| {obj}`~pypulseqpp.sequences.RosetteStackReadout` | Rosette petals in-plane, Cartesian partitions along z. |
| {obj}`~pypulseqpp.sequences.RosetteProjectionReadout` | Rosette petals turned over a sphere. |
| {obj}`~pypulseqpp.sequences.ZteReadout` | Continuous-gradient ZTE shell, including its initial and final ramps. |

(non-cartesian-interleaves)=

## Non-Cartesian interleaves

One base interleaf each, with its ADC and the prewinding and rewinding
gradients to and from k = 0. The scan loop rotates it per shot.

| Object | Description |
| --- | --- |
| {obj}`~pypulseqpp.sequences.NonCartesianGradient` | One canonical non-Cartesian base interleaf, independent of its acquisition schedule. |
| {obj}`~pypulseqpp.sequences.Arbitrary` | Base interleaf from a caller-supplied 2D or 3D k-space path. |
| {obj}`~pypulseqpp.sequences.Spiral` | Constant-, variable- or dual-density spiral base interleaf. |
| {obj}`~pypulseqpp.sequences.Rosette` | Multi-petal rosette base interleaf, starting and ending at k = 0. |
