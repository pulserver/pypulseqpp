===================
2D spiral spin echo
===================

:Family: Spin echo
:Dimensionality: 2D, multi-slice
:Sampling: Spiral
:Readout: One interleaf per excitation, read at a refocused echo

Multi-slice 2D spiral spin echo: one interleaf per excitation.

Prescription
------------

.. currentmodule:: pypulseqpp.sequences

.. autofunction:: se_spiral2D_sequence

Representative configuration for documentation
----------------------------------------------

The configuration below is chosen to make the structure of the sequence
legible at the size of this page: the matrix is small, delays that would
otherwise dominate the diagram are short, and the figures are drawn from
the sequence this call designs. It is not a protocol recommendation.

.. code-block:: python

    from pypulseqpp import sequences

    seq = sequences.se_spiral2D_sequence(
        n=64,
        n_shots=8,
        n_slices=1,
        te=None,
        tr=0.022,
        n_dummy=0,
    )

.. figure:: /generated/sequences/se_spiral2D_sequence-repetition.png
   :width: 100%

   One repetition. The remaining repetitions are drawn underneath in grey, so an event that changes between them appears as a band.

.. figure:: /generated/sequences/se_spiral2D_sequence-kspace.png
   :width: 100%

   Sampling locations in k-space, coloured by shot.

Related sequences
-----------------

Other spin echo sequences: :doc:`se2D_sequence <se2D_sequence>`, :doc:`se3D_sequence <se3D_sequence>`, :doc:`se_radial2D_sequence <se_radial2D_sequence>`, :doc:`se_stack_of_stars3D_sequence <se_stack_of_stars3D_sequence>`, :doc:`se_stack_of_spirals3D_sequence <se_stack_of_spirals3D_sequence>`, :doc:`se_propeller2D_sequence <se_propeller2D_sequence>`, :doc:`se_epi_propeller2D_sequence <se_epi_propeller2D_sequence>`, :doc:`se_stack_of_blades3D_sequence <se_stack_of_blades3D_sequence>`.

The same sampling in another family: :doc:`gre_spiral2D_sequence <gre_spiral2D_sequence>`.

Every shipped sequence is listed in the :doc:`catalogue </sequences>`.
