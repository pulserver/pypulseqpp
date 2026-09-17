===================
3D Cartesian MPRAGE
===================

:Family: MPRAGE
:Dimensionality: 3D, slab-selective
:Sampling: Cartesian
:Readout: One inversion per shot, followed by the phase-encode lines of one partition as a spoiled gradient-echo train

3D MPRAGE: one inversion per partition, then a train of spoiled low-flip lines.

Each shot applies one inversion and then acquires every sampled in-plane view of a single partition, so the partition encode is constant within a shot and the number of shots equals the number of sampled partitions. The inversion time is measured from the centre of the inversion pulse to the centre of the first excitation of the train, and the repetition time is the interval between successive inversions.

Prescription
------------

.. currentmodule:: pypulseqpp.sequences

.. autofunction:: mprage3D_sequence

Representative configuration for documentation
----------------------------------------------

The configuration below is chosen to make the structure of the sequence
legible at the size of this page: the matrix is small, delays that would
otherwise dominate the diagram are short, and the figures are drawn from
the sequence this call designs. It is not a protocol recommendation.

.. code-block:: python

    from pypulseqpp import sequences

    seq = sequences.mprage3D_sequence(
        n_x=64,
        n_y=16,
        n_z=8,
        ti=0.015,
        tr=0.11,
        n_dummy=0,
        readout_oversampling=1.0,
    )

.. figure:: /generated/sequences/mprage3D_sequence-shot.png
   :width: 100%

   One shot: the inversion pulse and its crusher, the inversion delay, the gradient-echo train, and the recovery that completes the inversion-to-inversion interval.

.. figure:: /generated/sequences/mprage3D_sequence-echoes.png
   :width: 100%

   The first repetitions of the train, at the raster the events are played on.

Related sequences
-----------------

Other mprage sequences: :doc:`mprage_stack_of_stars3D_sequence <mprage_stack_of_stars3D_sequence>`, :doc:`mprage_stack_of_spirals3D_sequence <mprage_stack_of_spirals3D_sequence>`.

The same sampling in another family: :doc:`gre2D_sequence <gre2D_sequence>`, :doc:`gre3D_sequence <gre3D_sequence>`, :doc:`gre_multiecho2D_sequence <gre_multiecho2D_sequence>`, :doc:`gre_multiecho3D_sequence <gre_multiecho3D_sequence>`, :doc:`se2D_sequence <se2D_sequence>`, :doc:`se3D_sequence <se3D_sequence>`, :doc:`fse3D_sequence <fse3D_sequence>`, :doc:`bssfp2D_sequence <bssfp2D_sequence>`, :doc:`bssfp3D_sequence <bssfp3D_sequence>`, :doc:`epi2D_sequence <epi2D_sequence>`.

Every shipped sequence is listed in the :doc:`catalogue </sequences>`.
