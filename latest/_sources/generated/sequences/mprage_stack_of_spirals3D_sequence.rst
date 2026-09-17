==========================
3D stack-of-spirals MPRAGE
==========================

:Family: MPRAGE
:Dimensionality: 3D, slab-selective
:Sampling: Spiral in-plane, Cartesian partitions
:Readout: One inversion per shot, followed by the spiral interleaves of one partition as a spoiled gradient-echo train

3D MPRAGE on a stack of spirals: one inversion per partition, then its interleaves.

Each shot applies one inversion and then acquires every sampled in-plane view of a single partition, so the partition encode is constant within a shot and the number of shots equals the number of sampled partitions. The inversion time is measured from the centre of the inversion pulse to the centre of the first excitation of the train, and the repetition time is the interval between successive inversions.

Prescription
------------

.. currentmodule:: pypulseqpp.sequences

.. autofunction:: mprage_stack_of_spirals3D_sequence

Representative configuration for documentation
----------------------------------------------

The configuration below is chosen to make the structure of the sequence
legible at the size of this page: the matrix is small, delays that would
otherwise dominate the diagram are short, and the figures are drawn from
the sequence this call designs. It is not a protocol recommendation.

.. code-block:: python

    from pypulseqpp import sequences

    seq = sequences.mprage_stack_of_spirals3D_sequence(
        n=64,
        n_z=4,
        n_shots=8,
        ti=0.015,
        tr=0.11,
        n_dummy=0,
    )

.. figure:: /generated/sequences/mprage_stack_of_spirals3D_sequence-shot.png
   :width: 100%

   One shot: the inversion, the inversion delay, the train of interleaves, and the recovery.

.. figure:: /generated/sequences/mprage_stack_of_spirals3D_sequence-kspace.png
   :width: 100%

   Sampling locations in k-space, coloured by shot.

Related sequences
-----------------

Other mprage sequences: :doc:`mprage3D_sequence <mprage3D_sequence>`, :doc:`mprage_stack_of_stars3D_sequence <mprage_stack_of_stars3D_sequence>`.

The same sampling in another family: :doc:`gre_stack_of_spirals3D_sequence <gre_stack_of_spirals3D_sequence>`, :doc:`se_stack_of_spirals3D_sequence <se_stack_of_spirals3D_sequence>`.

Every shipped sequence is listed in the :doc:`catalogue </sequences>`.
