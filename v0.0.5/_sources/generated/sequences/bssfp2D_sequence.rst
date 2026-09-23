================
2D balanced SSFP
================

:Family: Balanced SSFP
:Dimensionality: 2D, multi-slice
:Sampling: Cartesian
:Readout: One phase-encode line per repetition, every gradient axis balanced

Balanced SSFP 2D Cartesian: one complete train per slice, optionally cardiac-gated.

Prescription
------------

.. currentmodule:: pypulseqpp.sequences

.. autofunction:: bssfp2D_sequence

.. minigallery:: pypulseqpp.sequences.bssfp2D_sequence
   :add-heading: Designed and drawn

See also
--------

Other balanced ssfp sequences: :doc:`bssfp3D_sequence <bssfp3D_sequence>`.

The same sampling in another family: :doc:`gre2D_sequence <gre2D_sequence>`, :doc:`gre3D_sequence <gre3D_sequence>`, :doc:`gre_multiecho2D_sequence <gre_multiecho2D_sequence>`, :doc:`gre_multiecho3D_sequence <gre_multiecho3D_sequence>`, :doc:`se2D_sequence <se2D_sequence>`, :doc:`se3D_sequence <se3D_sequence>`, :doc:`fse3D_sequence <fse3D_sequence>`, :doc:`mprage3D_sequence <mprage3D_sequence>`, :doc:`epi2D_sequence <epi2D_sequence>`.

Every shipped sequence is listed in the :doc:`catalogue </sequences>`.
