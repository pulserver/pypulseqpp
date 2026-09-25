=======================
2D spiral gradient echo
=======================

:Family: Gradient echo
:Dimensionality: 2D, multi-slice
:Sampling: Spiral
:Readout: One interleaf per repetition, RF-spoiled

RF-spoiled, multi-slice 2D spiral gradient echo: one interleaf per repetition.

Prescription
------------

.. currentmodule:: pypulseqpp.sequences

.. autofunction:: gre_spiral2D_sequence

.. minigallery:: pypulseqpp.sequences.gre_spiral2D_sequence
   :add-heading: Designed and drawn

See also
--------

Other gradient echo sequences: :doc:`gre2D_sequence <gre2D_sequence>`, :doc:`gre3D_sequence <gre3D_sequence>`, :doc:`gre_multiecho2D_sequence <gre_multiecho2D_sequence>`, :doc:`gre_multiecho3D_sequence <gre_multiecho3D_sequence>`, :doc:`gre_radial2D_sequence <gre_radial2D_sequence>`, :doc:`gre_stack_of_stars3D_sequence <gre_stack_of_stars3D_sequence>`, :doc:`gre_stack_of_spirals3D_sequence <gre_stack_of_spirals3D_sequence>`, :doc:`gre_propeller2D_sequence <gre_propeller2D_sequence>`, :doc:`gre_stack_of_blades3D_sequence <gre_stack_of_blades3D_sequence>`.

The same sampling in another family: :doc:`se_spiral2D_sequence <se_spiral2D_sequence>`.

Every shipped sequence is listed in the :doc:`catalogue </sequences>`.
