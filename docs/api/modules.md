# Sequence modules

`pypulseqpp.sequences`: modules that solve the timing and gradients of a group
of blocks once and expose the events for a scan loop, and the non-Cartesian
interleaves the readouts play.

```{eval-rst}
.. currentmodule:: pypulseqpp.sequences
```

## Base classes

Modules stay plain Pulseq building blocks: the scan loop chooses the views,
scales the encoding events and adds the blocks. A {class}`SequenceModule` lays
its blocks out in `init_module`; an {class}`RfModule` is one built around a
pulse, which {meth}`~RfModule.sim_rf` simulates.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   SequenceModule
   RfModule
```

## Excitation and refocusing

```{eval-rst}
.. autosummary::
   :toctree: ../generated
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
   :toctree: ../generated
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
   :toctree: ../generated
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

{class}`NonCartesianReadout` plays any interleave from
{ref}`non-cartesian-interleaves` as a whole repetition.
The spiral and rosette readouts design theirs from the prescription.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
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

One base interleave each, with its ADC and the moment bridges to and from
k = 0. The scan loop rotates it per shot.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   NonCartesianGradient
   Arbitrary
   Spiral
   Rosette
```
