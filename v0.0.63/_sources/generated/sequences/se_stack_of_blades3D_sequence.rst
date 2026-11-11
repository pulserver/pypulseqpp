============================
3D stack-of-blades spin echo
============================

:Family: Spin echo
:Dimensionality: 3D, slab-selective
:Sampling: PROPELLER in-plane, Cartesian partitions
:Readout: One blade line at one partition per excitation

3D stack-of-blades spin echo: one blade line at one partition per excitation.

Prescription
------------

.. currentmodule:: pypulseqpp.sequences

.. autofunction:: se_stack_of_blades3D_sequence

.. minigallery:: pypulseqpp.sequences.se_stack_of_blades3D_sequence
   :add-heading: Designed and drawn

See also
--------

Other spin echo sequences: :doc:`se2D_sequence <se2D_sequence>`, :doc:`se3D_sequence <se3D_sequence>`, :doc:`se_radial2D_sequence <se_radial2D_sequence>`, :doc:`se_stack_of_stars3D_sequence <se_stack_of_stars3D_sequence>`, :doc:`se_spiral2D_sequence <se_spiral2D_sequence>`, :doc:`se_stack_of_spirals3D_sequence <se_stack_of_spirals3D_sequence>`, :doc:`se_propeller2D_sequence <se_propeller2D_sequence>`, :doc:`se_epi_propeller2D_sequence <se_epi_propeller2D_sequence>`.

The same sampling in another family: :doc:`gre_stack_of_blades3D_sequence <gre_stack_of_blades3D_sequence>`.

Every shipped sequence is listed in the :doc:`catalogue </sequences>`.
