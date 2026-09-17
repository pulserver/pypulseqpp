=================
3D fast spin echo
=================

:Family: Fast spin echo
:Dimensionality: 3D, slab-selective
:Sampling: Cartesian
:Readout: One CPMG train per excitation, one (line, partition) view per echo

3D Cartesian fast spin echo: one CPMG train per excitation over a (ky, kz) grid.

Prescription
------------

.. currentmodule:: pypulseqpp.sequences

.. autofunction:: fse3D_sequence

Representative configuration for documentation
----------------------------------------------

The configuration below is chosen to make the structure of the sequence
legible at the size of this page: the matrix is small, delays that would
otherwise dominate the diagram are short, and the figures are drawn from
the sequence this call designs. It is not a protocol recommendation.

.. code-block:: python

    from pypulseqpp import sequences

    seq = sequences.fse3D_sequence(
        n_x=64,
        n_y=16,
        n_z=8,
        etl=8,
        te=None,
        tr=0.1,
        n_dummy=0,
        readout_oversampling=1.0,
    )

.. figure:: /generated/sequences/fse3D_sequence-train.png
   :width: 100%

   One echo train: the excitation, then one refocusing pulse and one acquisition per echo.

.. figure:: /generated/sequences/fse3D_sequence-echoes.png
   :width: 100%

   The opening of one echo train, at the raster the events are played on.

Related sequences
-----------------

The same sampling in another family: :doc:`gre2D_sequence <gre2D_sequence>`, :doc:`gre3D_sequence <gre3D_sequence>`, :doc:`gre_multiecho2D_sequence <gre_multiecho2D_sequence>`, :doc:`gre_multiecho3D_sequence <gre_multiecho3D_sequence>`, :doc:`se2D_sequence <se2D_sequence>`, :doc:`se3D_sequence <se3D_sequence>`, :doc:`mprage3D_sequence <mprage3D_sequence>`, :doc:`bssfp2D_sequence <bssfp2D_sequence>`, :doc:`bssfp3D_sequence <bssfp3D_sequence>`, :doc:`epi2D_sequence <epi2D_sequence>`.

Every shipped sequence is listed in the :doc:`catalogue </sequences>`.
