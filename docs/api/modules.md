# Sequence modules

`pypulseqpp.sequences`: reusable block layouts with solved timing and gradient
waveforms and named event templates for scan-loop encoding, together
with the non-Cartesian interleaves used by the readout modules.

```{eval-rst}
.. currentmodule:: pypulseqpp.sequences
```

## Base classes

A module is a reusable block layout with named, mutable event templates. The
scan loop chooses the views, scales the encoding events and adds the blocks.
A {class}`SequenceModule` lays its blocks out in `init_module`; an
{class}`RfModule` is a module built around one RF event, which
{meth}`~RfModule.sim_rf` simulates against off-resonance.

```{eval-rst}
.. autosummary::
   :nosignatures:

   SequenceModule
   RfModule
```

## Excitation and refocusing

```{eval-rst}
.. autosummary::
   :nosignatures:

   NonSelectiveExcitation
   NonSelectiveRefocusing
   SpatialSelectiveExcitation
   SpatialSelectiveRefocusing
   SpatialSelective2DExcitation
   FrequencySelectiveExcitation
   SpspExcitation
   SmsExcitation
   MultibandExcitation
```

## Magnetization preparation

```{eval-rst}
.. autosummary::
   :nosignatures:

   InversionPreparation
   BlochSiegertPreparation
   DiffusionPreparation
   FatSaturation
   IhMtPreparation
   MtPreparation
   OffResonanceSaturation
   T1T2Preparation
   T2Preparation
```

## Cartesian readouts

```{eval-rst}
.. autosummary::
   :nosignatures:

   LineReadout2D
   LineReadout3D
   EpiReadout2D
   EpiReadout3D
   FseReadout2D
   FseReadout3D
   BssfpReadout2D
   BssfpReadout3D
   PropellerReadout2D
   PropellerStackReadout
```

## Non-Cartesian readouts

{class}`NonCartesianReadout` plays any interleaf from
{ref}`non-cartesian-interleaves` as a whole repetition.
The spiral and rosette readouts design theirs from the prescription.

```{eval-rst}
.. autosummary::
   :nosignatures:

   NonCartesianReadout
   RadialReadout2D
   RadialStackReadout
   RadialProjectionReadout
   SpiralReadout2D
   SpiralStackReadout
   SpiralProjectionReadout
   SpiralNavigator
   RosetteReadout2D
   RosetteStackReadout
   RosetteProjectionReadout
   ZteReadout
```

(non-cartesian-interleaves)=

## Non-Cartesian interleaves

One base interleaf each, with its ADC and the prewinding and rewinding
gradients to and from k = 0. The scan loop rotates it per shot.

```{eval-rst}
.. autosummary::
   :nosignatures:

   NonCartesianGradient
   Arbitrary
   Spiral
   Rosette
```
