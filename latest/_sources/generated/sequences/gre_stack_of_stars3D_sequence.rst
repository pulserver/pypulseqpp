===============================
3D stack-of-stars gradient echo
===============================

:Family: Gradient echo
:Dimensionality: 3D, slab-selective
:Sampling: Radial in-plane, Cartesian partitions
:Readout: One spoke at one partition per repetition, RF-spoiled

RF-spoiled 3D stack of stars: radial spokes in-plane, Cartesian partitions along z.

Prescription
------------

.. currentmodule:: pypulseqpp.sequences

.. autofunction:: gre_stack_of_stars3D_sequence

Representative configuration for documentation
----------------------------------------------

The configuration below is chosen to make the structure of the sequence
legible at the size of this page: the matrix is small, delays that would
otherwise dominate the diagram are short, and the figures are drawn from
the sequence this call designs. It is not a protocol recommendation.

.. code-block:: python

    from pypulseqpp import sequences

    seq = sequences.gre_stack_of_stars3D_sequence(
        n=64,
        ry=8,
        n_z=4,
        tr=None,
        n_dummy=0,
        readout_oversampling=1.0,
    )

.. figure:: /generated/sequences/gre_stack_of_stars3D_sequence-repetition.png
   :width: 100%

   One repetition. The remaining repetitions are drawn underneath in grey, so an event that changes between them appears as a band.

.. figure:: /generated/sequences/gre_stack_of_stars3D_sequence-kspace.png
   :width: 100%

   Sampling locations in k-space, coloured by shot.

Related sequences
-----------------

Other gradient echo sequences: :doc:`gre2D_sequence <gre2D_sequence>`, :doc:`gre3D_sequence <gre3D_sequence>`, :doc:`gre_multiecho2D_sequence <gre_multiecho2D_sequence>`, :doc:`gre_multiecho3D_sequence <gre_multiecho3D_sequence>`, :doc:`gre_radial2D_sequence <gre_radial2D_sequence>`, :doc:`gre_spiral2D_sequence <gre_spiral2D_sequence>`, :doc:`gre_stack_of_spirals3D_sequence <gre_stack_of_spirals3D_sequence>`, :doc:`gre_propeller2D_sequence <gre_propeller2D_sequence>`, :doc:`gre_stack_of_blades3D_sequence <gre_stack_of_blades3D_sequence>`.

The same sampling in another family: :doc:`se_stack_of_stars3D_sequence <se_stack_of_stars3D_sequence>`, :doc:`mprage_stack_of_stars3D_sequence <mprage_stack_of_stars3D_sequence>`.

Every shipped sequence is listed in the :doc:`catalogue </sequences>`.
