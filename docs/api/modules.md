# Sequence modules

```{eval-rst}
.. currentmodule:: pypulseqpp.sequences
```

A {class}`SequenceModule` solves the timing and gradients of a group of blocks
once, then exposes the resulting events for a sequence's scan loop. Modules
remain plain Pulseq building blocks: the loop chooses the views, scales the
encoding events, and adds the blocks.

## Complete sequences

A {class}`SequenceApp` is a whole sequence. Constructing it designs the events
and the sampling order from the prescription its `init_sequence` accepts.
{meth}`~SequenceApp.design` then runs its `loop`, the scan loop, which plays
one `kernel` call per repetition. Settings that are not prescribed are class
attributes, so a subclass that overrides one is the same sequence under
another setting. Each zoo script reached as `sequences.<name>` defines one,
and its `main` builds and designs it.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   SequenceApp
```

## Base classes

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

The 2D, stack and projection variants inherit their constructor contracts
from the implementation classes below. These are reference pages for shared
parameters and event attributes; instantiate the named variants above.

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
