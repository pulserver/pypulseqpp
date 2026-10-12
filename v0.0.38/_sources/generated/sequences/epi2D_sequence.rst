======================
2D echo-planar imaging
======================

:Family: Echo-planar imaging
:Dimensionality: 2D, multi-slice
:Sampling: Cartesian
:Readout: One blipped echo train per excitation

Multi-slice 2D gradient-echo EPI: single-shot or segmented, optionally multiband.

Prescription
------------

.. currentmodule:: pypulseqpp.sequences

.. autofunction:: epi2D_sequence

.. minigallery:: pypulseqpp.sequences.epi2D_sequence
   :add-heading: Designed and drawn

See also
--------

Other echo-planar imaging sequences: :doc:`epi3D_sequence <epi3D_sequence>`.

The same sampling in another family: :doc:`gre2D_sequence <gre2D_sequence>`, :doc:`gre3D_sequence <gre3D_sequence>`, :doc:`gre_multiecho2D_sequence <gre_multiecho2D_sequence>`, :doc:`gre_multiecho3D_sequence <gre_multiecho3D_sequence>`, :doc:`se2D_sequence <se2D_sequence>`, :doc:`se3D_sequence <se3D_sequence>`, :doc:`fse3D_sequence <fse3D_sequence>`, :doc:`mprage3D_sequence <mprage3D_sequence>`, :doc:`bssfp2D_sequence <bssfp2D_sequence>`, :doc:`bssfp3D_sequence <bssfp3D_sequence>`.

Every shipped sequence is listed in the :doc:`catalogue </sequences>`.
