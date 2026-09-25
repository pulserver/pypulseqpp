======================
3D echo-planar imaging
======================

:Family: Echo-planar imaging
:Dimensionality: 3D, slab-selective
:Sampling: Cartesian, skipped-CAIPI when undersampled
:Readout: One blipped echo train per (shot, shell)

3D gradient-echo EPI: one train per ``(shot, shell)``, skipped-CAIPI sampled.

Prescription
------------

.. currentmodule:: pypulseqpp.sequences

.. autofunction:: epi3D_sequence

.. minigallery:: pypulseqpp.sequences.epi3D_sequence
   :add-heading: Designed and drawn

See also
--------

Other echo-planar imaging sequences: :doc:`epi2D_sequence <epi2D_sequence>`.

Every shipped sequence is listed in the :doc:`catalogue </sequences>`.
