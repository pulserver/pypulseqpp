================================
3D stack-of-blades gradient echo
================================

:Family: Gradient echo
:Dimensionality: 3D, slab-selective
:Sampling: PROPELLER in-plane, Cartesian partitions
:Readout: One blade line at one partition per repetition, RF-spoiled

RF-spoiled 3D stack of blades: PROPELLER blades in-plane, Cartesian partitions along z.

Prescription
------------

.. currentmodule:: pypulseqpp.sequences

.. autofunction:: gre_stack_of_blades3D_sequence

.. minigallery:: pypulseqpp.sequences.gre_stack_of_blades3D_sequence
   :add-heading: Designed and drawn

See also
--------

Other gradient echo sequences: :doc:`gre2D_sequence <gre2D_sequence>`, :doc:`gre3D_sequence <gre3D_sequence>`, :doc:`gre_multiecho2D_sequence <gre_multiecho2D_sequence>`, :doc:`gre_multiecho3D_sequence <gre_multiecho3D_sequence>`, :doc:`gre_radial2D_sequence <gre_radial2D_sequence>`, :doc:`gre_stack_of_stars3D_sequence <gre_stack_of_stars3D_sequence>`, :doc:`gre_spiral2D_sequence <gre_spiral2D_sequence>`, :doc:`gre_stack_of_spirals3D_sequence <gre_stack_of_spirals3D_sequence>`, :doc:`gre_propeller2D_sequence <gre_propeller2D_sequence>`.

The same sampling in another family: :doc:`se_stack_of_blades3D_sequence <se_stack_of_blades3D_sequence>`.

Every shipped sequence is listed in the :doc:`catalogue </sequences>`.
