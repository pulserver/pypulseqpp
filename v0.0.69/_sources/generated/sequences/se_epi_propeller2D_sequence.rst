==============================================
2D PROPELLER spin echo with echo-planar blades
==============================================

:Family: Spin echo
:Dimensionality: 2D, multi-slice
:Sampling: PROPELLER blades, echo-planar within a blade
:Readout: One whole blade per excitation, read as an echo-planar train

Multi-slice 2D PROPELLER spin echo: one EPI blade per excitation.

Prescription
------------

.. currentmodule:: pypulseqpp.sequences

.. autofunction:: se_epi_propeller2D_sequence

.. minigallery:: pypulseqpp.sequences.se_epi_propeller2D_sequence
   :add-heading: Designed and drawn

See also
--------

Other spin echo sequences: :doc:`se2D_sequence <se2D_sequence>`, :doc:`se3D_sequence <se3D_sequence>`, :doc:`se_radial2D_sequence <se_radial2D_sequence>`, :doc:`se_stack_of_stars3D_sequence <se_stack_of_stars3D_sequence>`, :doc:`se_spiral2D_sequence <se_spiral2D_sequence>`, :doc:`se_stack_of_spirals3D_sequence <se_stack_of_spirals3D_sequence>`, :doc:`se_propeller2D_sequence <se_propeller2D_sequence>`, :doc:`se_stack_of_blades3D_sequence <se_stack_of_blades3D_sequence>`.

Every shipped sequence is listed in the :doc:`catalogue </sequences>`.
