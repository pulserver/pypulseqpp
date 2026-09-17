================
2D balanced SSFP
================

:Family: Balanced SSFP
:Dimensionality: 2D, multi-slice
:Sampling: Cartesian
:Readout: One phase-encode line per repetition, every gradient axis balanced

Balanced SSFP 2D Cartesian: one complete train per slice, optionally cardiac-gated.

Prescription
------------

.. currentmodule:: pypulseqpp.sequences

.. autofunction:: bssfp2D_sequence

Representative configuration for documentation
----------------------------------------------

The configuration below is chosen to make the structure of the sequence
legible at the size of this page: the matrix is small, delays that would
otherwise dominate the diagram are short, and the figures are drawn from
the sequence this call designs. It is not a protocol recommendation.

.. code-block:: python

    from pypulseqpp import sequences

    seq = sequences.bssfp2D_sequence(
        n_x=64,
        n_y=16,
        n_slices=1,
        n_phases=1,
        tr=None,
        n_dummy=4,
        readout_oversampling=1.0,
        readout_bandwidth_hz=80000.0,
    )

.. figure:: /generated/sequences/bssfp2D_sequence-repetition.png
   :width: 100%

   One repetition. The remaining repetitions are drawn underneath in grey, so an event that changes between them appears as a band.

.. figure:: /generated/sequences/bssfp2D_sequence-train.png
   :width: 100%

   Consecutive repetitions. Every gradient axis returns to zero moment within each repetition.

Related sequences
-----------------

Other balanced ssfp sequences: :doc:`bssfp3D_sequence <bssfp3D_sequence>`.

The same sampling in another family: :doc:`gre2D_sequence <gre2D_sequence>`, :doc:`gre3D_sequence <gre3D_sequence>`, :doc:`gre_multiecho2D_sequence <gre_multiecho2D_sequence>`, :doc:`gre_multiecho3D_sequence <gre_multiecho3D_sequence>`, :doc:`se2D_sequence <se2D_sequence>`, :doc:`se3D_sequence <se3D_sequence>`, :doc:`fse3D_sequence <fse3D_sequence>`, :doc:`mprage3D_sequence <mprage3D_sequence>`, :doc:`epi2D_sequence <epi2D_sequence>`.

Every shipped sequence is listed in the :doc:`catalogue </sequences>`.
