======================
2D Cartesian spin echo
======================

:Family: Spin echo
:Dimensionality: 2D, multi-slice
:Sampling: Cartesian
:Readout: One phase-encode line per excitation, read at a refocused echo

Multi-slice 2D Cartesian spin echo: one line per excitation.

Prescription
------------

.. currentmodule:: pypulseqpp.sequences

.. autofunction:: se2D_sequence

.. minigallery:: pypulseqpp.sequences.se2D_sequence
   :add-heading: Designed and drawn

See also
--------

Other spin echo sequences: :doc:`se3D_sequence <se3D_sequence>`, :doc:`se_radial2D_sequence <se_radial2D_sequence>`, :doc:`se_stack_of_stars3D_sequence <se_stack_of_stars3D_sequence>`, :doc:`se_spiral2D_sequence <se_spiral2D_sequence>`, :doc:`se_stack_of_spirals3D_sequence <se_stack_of_spirals3D_sequence>`, :doc:`se_propeller2D_sequence <se_propeller2D_sequence>`, :doc:`se_epi_propeller2D_sequence <se_epi_propeller2D_sequence>`, :doc:`se_stack_of_blades3D_sequence <se_stack_of_blades3D_sequence>`.

The same sampling in another family: :doc:`gre2D_sequence <gre2D_sequence>`, :doc:`gre3D_sequence <gre3D_sequence>`, :doc:`gre_multiecho2D_sequence <gre_multiecho2D_sequence>`, :doc:`gre_multiecho3D_sequence <gre_multiecho3D_sequence>`, :doc:`fse3D_sequence <fse3D_sequence>`, :doc:`mprage3D_sequence <mprage3D_sequence>`, :doc:`bssfp2D_sequence <bssfp2D_sequence>`, :doc:`bssfp3D_sequence <bssfp3D_sequence>`, :doc:`epi2D_sequence <epi2D_sequence>`.

Every shipped sequence is listed in the :doc:`catalogue </sequences>`.
