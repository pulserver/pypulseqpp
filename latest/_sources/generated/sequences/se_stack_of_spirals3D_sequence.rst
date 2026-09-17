=============================
3D stack-of-spirals spin echo
=============================

:Family: Spin echo
:Dimensionality: 3D, slab-selective
:Sampling: Spiral in-plane, Cartesian partitions
:Readout: One interleaf at one partition per excitation

3D stack-of-spirals spin echo: one interleaf at one partition per excitation.

Prescription
------------

.. currentmodule:: pypulseqpp.sequences

.. autofunction:: se_stack_of_spirals3D_sequence

Representative configuration for documentation
----------------------------------------------

The configuration below is chosen to make the structure of the sequence
legible at the size of this page: the matrix is small, delays that would
otherwise dominate the diagram are short, and the figures are drawn from
the sequence this call designs. It is not a protocol recommendation.

.. code-block:: python

    from pypulseqpp import sequences

    seq = sequences.se_stack_of_spirals3D_sequence(
        n=64,
        n_z=4,
        n_shots=8,
        te=None,
        tr=0.019,
        n_dummy=0,
    )

.. figure:: /generated/sequences/se_stack_of_spirals3D_sequence-repetition.png
   :width: 100%

   One repetition. The remaining repetitions are drawn underneath in grey, so an event that changes between them appears as a band.

.. figure:: /generated/sequences/se_stack_of_spirals3D_sequence-kspace.png
   :width: 100%

   Sampling locations in k-space, coloured by shot.

Related sequences
-----------------

Other spin echo sequences: :doc:`se2D_sequence <se2D_sequence>`, :doc:`se3D_sequence <se3D_sequence>`, :doc:`se_radial2D_sequence <se_radial2D_sequence>`, :doc:`se_stack_of_stars3D_sequence <se_stack_of_stars3D_sequence>`, :doc:`se_spiral2D_sequence <se_spiral2D_sequence>`, :doc:`se_propeller2D_sequence <se_propeller2D_sequence>`, :doc:`se_epi_propeller2D_sequence <se_epi_propeller2D_sequence>`, :doc:`se_stack_of_blades3D_sequence <se_stack_of_blades3D_sequence>`.

The same sampling in another family: :doc:`gre_stack_of_spirals3D_sequence <gre_stack_of_spirals3D_sequence>`, :doc:`mprage_stack_of_spirals3D_sequence <mprage_stack_of_spirals3D_sequence>`.

Every shipped sequence is listed in the :doc:`catalogue </sequences>`.
