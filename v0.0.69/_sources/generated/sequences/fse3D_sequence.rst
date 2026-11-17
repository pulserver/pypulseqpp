=================
3D fast spin echo
=================

:Family: Fast spin echo
:Dimensionality: 3D, slab-selective
:Sampling: Cartesian
:Readout: One CPMG train per excitation, one (line, partition) view per echo

3D Cartesian fast spin echo: one CPMG train per excitation over a (ky, kz) grid.

Prescription
------------

.. currentmodule:: pypulseqpp.sequences

.. autofunction:: fse3D_sequence

.. minigallery:: pypulseqpp.sequences.fse3D_sequence
   :add-heading: Designed and drawn

See also
--------

The same sampling in another family: :doc:`gre2D_sequence <gre2D_sequence>`, :doc:`gre3D_sequence <gre3D_sequence>`, :doc:`gre_multiecho2D_sequence <gre_multiecho2D_sequence>`, :doc:`gre_multiecho3D_sequence <gre_multiecho3D_sequence>`, :doc:`se2D_sequence <se2D_sequence>`, :doc:`se3D_sequence <se3D_sequence>`, :doc:`mprage3D_sequence <mprage3D_sequence>`, :doc:`bssfp2D_sequence <bssfp2D_sequence>`, :doc:`bssfp3D_sequence <bssfp3D_sequence>`, :doc:`epi2D_sequence <epi2D_sequence>`.

Every shipped sequence is listed in the :doc:`catalogue </sequences>`.
