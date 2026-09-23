======================
2D PROPELLER spin echo
======================

:Family: Spin echo
:Dimensionality: 2D, multi-slice
:Sampling: PROPELLER blades
:Readout: One line of one blade per excitation

Multi-slice 2D PROPELLER spin echo: one blade line per excitation.

Prescription
------------

.. currentmodule:: pypulseqpp.sequences

.. autofunction:: se_propeller2D_sequence

.. minigallery:: pypulseqpp.sequences.se_propeller2D_sequence
   :add-heading: Designed and drawn

See also
--------

Other spin echo sequences: :doc:`se2D_sequence <se2D_sequence>`, :doc:`se3D_sequence <se3D_sequence>`, :doc:`se_radial2D_sequence <se_radial2D_sequence>`, :doc:`se_stack_of_stars3D_sequence <se_stack_of_stars3D_sequence>`, :doc:`se_spiral2D_sequence <se_spiral2D_sequence>`, :doc:`se_stack_of_spirals3D_sequence <se_stack_of_spirals3D_sequence>`, :doc:`se_epi_propeller2D_sequence <se_epi_propeller2D_sequence>`, :doc:`se_stack_of_blades3D_sequence <se_stack_of_blades3D_sequence>`.

The same sampling in another family: :doc:`gre_propeller2D_sequence <gre_propeller2D_sequence>`.

Every shipped sequence is listed in the :doc:`catalogue </sequences>`.
