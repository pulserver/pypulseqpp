======================
3D echo-planar imaging
======================

:Family: Echo-planar imaging
:Dimensionality: 3D, slab-selective
:Sampling: Cartesian, skipped-CAIPI when undersampled
:Readout: One blipped echo train per (shot, shell)

3D gradient-echo EPI: one train per ``(shot, shell)``, skipped-CAIPI sampled.

Prescription
------------

.. currentmodule:: pypulseqpp.sequences

.. autofunction:: epi3D_sequence

Representative configuration for documentation
----------------------------------------------

The configuration below is chosen to make the structure of the sequence
legible at the size of this page: the matrix is small, delays that would
otherwise dominate the diagram are short, and the figures are drawn from
the sequence this call designs. It is not a protocol recommendation.

.. code-block:: python

    from pypulseqpp import sequences

    seq = sequences.epi3D_sequence(
        n_x=32,
        n_y=32,
        n_z=4,
        tr=None,
        n_dummy=0,
    )

.. figure:: /generated/sequences/epi3D_sequence-train.png
   :width: 100%

   One echo train, with the partition encode applied before it.

.. figure:: /generated/sequences/epi3D_sequence-echoes.png
   :width: 100%

   The opening of one echo train, at the raster the events are played on.

Related sequences
-----------------

Other echo-planar imaging sequences: :doc:`epi2D_sequence <epi2D_sequence>`.

Every shipped sequence is listed in the :doc:`catalogue </sequences>`.
