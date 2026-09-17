======================
2D echo-planar imaging
======================

:Family: Echo-planar imaging
:Dimensionality: 2D, multi-slice
:Sampling: Cartesian
:Readout: One blipped echo train per excitation

Multi-slice 2D gradient-echo EPI: single-shot or segmented, optionally multiband.

Prescription
------------

.. currentmodule:: pypulseqpp.sequences

.. autofunction:: epi2D_sequence

Representative configuration for documentation
----------------------------------------------

The configuration below is chosen to make the structure of the sequence
legible at the size of this page: the matrix is small, delays that would
otherwise dominate the diagram are short, and the figures are drawn from
the sequence this call designs. It is not a protocol recommendation.

.. code-block:: python

    from pypulseqpp import sequences

    seq = sequences.epi2D_sequence(
        n_x=32,
        n_y=32,
        n_slices=1,
        tr=None,
        n_dummy=0,
    )

.. figure:: /generated/sequences/epi2D_sequence-train.png
   :width: 100%

   One echo train: the excitation, the prewinders, and the alternating readout lobes with the phase-encode blips between them.

.. figure:: /generated/sequences/epi2D_sequence-echoes.png
   :width: 100%

   The opening of one echo train, at the raster the events are played on.

Related sequences
-----------------

Other echo-planar imaging sequences: :doc:`epi3D_sequence <epi3D_sequence>`.

The same sampling in another family: :doc:`gre2D_sequence <gre2D_sequence>`, :doc:`gre3D_sequence <gre3D_sequence>`, :doc:`gre_multiecho2D_sequence <gre_multiecho2D_sequence>`, :doc:`gre_multiecho3D_sequence <gre_multiecho3D_sequence>`, :doc:`se2D_sequence <se2D_sequence>`, :doc:`se3D_sequence <se3D_sequence>`, :doc:`fse3D_sequence <fse3D_sequence>`, :doc:`mprage3D_sequence <mprage3D_sequence>`, :doc:`bssfp2D_sequence <bssfp2D_sequence>`, :doc:`bssfp3D_sequence <bssfp3D_sequence>`.

Every shipped sequence is listed in the :doc:`catalogue </sequences>`.
