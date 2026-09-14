# Sequence modules

`pypulseqpp.sequences`: modules that solve the timing and gradients of a group
of blocks once and expose the events for a scan loop, and the complete
sequences built from them.

```{eval-rst}
.. currentmodule:: pypulseqpp.sequences
```

## Complete sequences

A {class}`SequenceApp` designs its events and sampling order from the
prescription its `init_sequence` takes; {meth}`~SequenceApp.design` runs its
scan `loop`, one `kernel` call per repetition. Prescans listed by
{meth}`~SequenceApp.prescans` are written by {meth}`~SequenceApp.write` as
files linked through `NextSequence`. Each zoo script reached as
`sequences.<name>` defines one.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   SequenceApp
```

## Base classes

Modules stay plain Pulseq building blocks: the scan loop chooses the views,
scales the encoding events and adds the blocks.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   SequenceModule
   RfModule
   OffResonanceSaturation
   NonCartesianReadout
```

## Excitation and refocusing

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   NonSelectiveExcitation
   NonSelectiveRefocusing
   Inversion
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

   BlochSiegertPreparation
   DiffusionPreparation
   FatSaturation
   IhMtPreparation
   InversionPreparation
   MtPreparation
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

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

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

## Shared readout contracts

The 2D, stack and projection variants inherit their constructor contracts from
these implementation classes: reference pages for shared parameters and event
attributes. Instantiate the named variants above.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   ~readout.line._LineReadout
   ~readout.epi._EpiReadout
   ~readout.fse._FseReadout
   ~readout.bssfp._BssfpReadout
   ~readout.propeller._PropellerReadout
   ~readout.noncartesian._RadialReadout
   ~readout.noncartesian._SpiralReadout
   ~readout.noncartesian._RosetteReadout
```
